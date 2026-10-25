"""
Chunk load lifecycle: tickets in, loaded chunks out.

`TicketManager` (world/chunk_ticket.py) says *what level every chunk has*; this
module is the driver that turns that level map into chunks in memory and back
out again. It answers the three questions the ticket model deliberately does not:

* **Which chunks must exist right now?** — every chunk with level <= 33
  (`loaded_positions()`), i.e. the union of the ticket squares. Nothing else is
  allowed to stay in memory, which is the whole point: without an unload pass a
  long session grows the chunk manager until the LRU cap throws live chunks
  away, and the LRU cap has no idea which of them a client is looking at.
* **Who loads them?** — `on_missing_chunks`, the same callback the streamer
  uses, so a chunk requested because it is in a player's view and one requested
  because a ticket covers it become one generation, not two (`claim_chunk` is
  the dedupe).
* **Who unloads them, and how does the data survive that?** — chunks that lost
  their ticket are dropped after a short grace period, and they are handed to
  the save pipeline *first*: `save_chunk` takes ownership of the `Chunk` object
  and only then does the coordinate leave the manager. That ordering is what
  makes "unload" safe; the reverse (drop, then save) has nothing left to save.

Threading
---------
Constructed by `World` and driven by the owning `DimensionActor` thread — the
same thread that owns the `ChunkManager` and the streamer. Every read and write
here is therefore a plain one, and there is no lock anywhere in this file. The
one boundary is the save callback, which hands the object across to the saver
threads (world/chunk_saver.py).

A world with no actor (tools, tests) drives `tick()` by hand; the class does not
care who calls it, only that one thread does.

Player tickets
--------------
`set_player_view` is called from every movement packet, so the hot path is the
early return: the player's chunk, view distance and simulation distance are
compared against the anchor the tickets were placed at, and nothing happens
until one of them changes. A player standing still costs three integer
comparisons per tick.

Two tickets are placed per player, and the split is the point:

* `PLAYER_LOADING` at level 31 with radius = view distance — loads terrain.
* `PLAYER_SIMULATION` at level 31 with radius = simulation distance — loads
  *and* simulates, and keeps the dimension active.

Because only the simulation ticket carries the SIMULATION flag, a large view
distance never widens the area where entities tick: `ticks_entities` answers
from the simulation map alone. That is the same separation the wiki describes
between "player loading" and "player simulation", and it is why a player with
view-distance 10 / simulation-distance 5 has entities ticking in an 11x11
square while 25x25 chunks are in memory.
"""

import heapq
import math
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

from loguru import logger

from world.chunk import Chunk, ChunkPos, ChunkManager
from world.chunk_saver import should_persist
from world.chunk_ticket import (
    FORCED,
    PLAYER_LOADING,
    PLAYER_SIMULATION,
    LoadType,
    Ticket,
    TicketManager,
    TicketType,
)
from world.dimension import Dimension, DimensionType

#: Ticks a chunk that lost its ticket is kept before it is unloaded. Walking
#: back and forth over a chunk border would otherwise save and re-read the same
#: chunks twice a second; one second of hysteresis costs a few chunks of memory
#: and removes that churn entirely.
UNLOAD_GRACE_TICKS = 20

#: How many load requests (generation claims) one tick may fire. A player
#: crossing a border on the far side of a radius-32 view can reveal ~130 chunks
#: at once; without a budget the actor would post all of them into the
#: generator queue and stall the tick behind the bookkeeping.
REQUEST_BUDGET_PER_TICK = 64

#: How long a coordinate waits before it is requested again. Two jobs: a request
#: the generator refused (queue full) is retried instead of being forgotten,
#: and a claim whose worker died is re-requested instead of blocking the view
#: forever. The second one is a known gap in the streamer's own bookkeeping
#: (`view.requested` is permanent), which this pass deliberately does not share.
REQUEST_RETRY_TICKS = 40

#: Ticks a blank claim (present, `is_generated` False) may stay blank before it
#: is treated as stalled and re-requested. The wait backs off exponentially from
#: here (see `claim_stall_max_ticks`).
CLAIM_STALL_TICKS = 200

#: Ceiling on the stall backoff (5 minutes). A request the pipeline *dropped*
#: has to be recoverable, but a request that is merely queued behind a deep view
#: must not be re-queued once per stall timeout: the vanilla backend generates
#: one chunk per second per core, so a radius-32 view legitimately takes minutes
#: to drain, and re-requesting every ten seconds would multiply that work.
CLAIM_STALL_MAX_TICKS = 6000

#: Priority handed to the generator for a request with no player near it, so
#: force-loaded chunks still get generated while players' terrain goes first.
IDLE_PRIORITY = 8


@dataclass
class PlayerTickets:
    """Where one player's tickets sit, and how wide they are."""

    player_id: str
    anchor: Optional[ChunkPos] = None
    view_distance: int = 0
    simulation_distance: int = 0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @property
    def chunk(self) -> ChunkPos:
        return ChunkPos(math.floor(self.x / 16), math.floor(self.z / 16))


def _chebyshev(a: ChunkPos, b: ChunkPos) -> int:
    return max(abs(a.x - b.x), abs(a.z - b.z))


@dataclass(frozen=True)
class TicketMsg:
    """A load/unload request from a thread that does not own the dimension.

    Ticket state, the chunk manager and the lifecycle all belong to the
    dimension actor thread (that is what makes this file lock-free). The console
    command and the world API run on *other* threads, so they cannot call
    `add_ticket`/`unload_chunk` directly: on a free-threaded build, mutating the
    ticket table while the actor iterates it raises "dictionary changed size
    during iteration" inside the tick, and on a GIL build it silently corrupts
    the same structures. This message is the only supported way across.

    Frozen and self-describing, so posting one is a queue put and applying it is
    a method call on the owner thread.
    """

    action: str  # "force" | "unforce" | "unload"
    x: int = 0
    z: int = 0
    save: bool = True
    force: bool = False

    def apply(self, lifecycle: "ChunkLifecycle") -> object:
        """Run on the owning actor thread. Returns whatever the action answered."""
        if self.action == "force":
            return lifecycle.force_load(self.x, self.z, True)
        if self.action == "unforce":
            return lifecycle.force_load(self.x, self.z, False)
        if self.action == "unload":
            return lifecycle.unload_chunk(self.x, self.z, save=self.save, force=self.force)
        raise ValueError(f"unknown ticket action {self.action!r}")


@dataclass
class StatsMsg:
    """Ask the owning thread for a consistent snapshot of its bookkeeping.

    `lifecycle.stats()` iterates several sets the actor mutates every tick, so a
    console reading it directly is the read half of the same race `TicketMsg`
    closes for writes. The answering thread fills `box` before setting `done`.
    """

    box: Dict[str, object] = field(default_factory=dict)
    done: threading.Event = field(default_factory=threading.Event)


class ChunkLifecycle:
    """Keeps a dimension's loaded chunks in step with its tickets."""

    def __init__(
        self,
        dimension: Dimension,
        *,
        on_missing_chunks: Optional[Callable[[int, int, int], bool]] = None,
        save_chunk: Optional[Callable[[Chunk], bool]] = None,
        load_chunk: Optional[Callable[[int, int], Optional[Chunk]]] = None,
        unload_grace_ticks: int = UNLOAD_GRACE_TICKS,
        request_budget: int = REQUEST_BUDGET_PER_TICK,
        request_retry_ticks: int = REQUEST_RETRY_TICKS,
        claim_stall_ticks: int = CLAIM_STALL_TICKS,
        claim_stall_max_ticks: int = CLAIM_STALL_MAX_TICKS,
        sync_fallback: bool = True,
        enabled: bool = True,
        light=None,
    ):
        """
        Args:
            dimension: The dimension this lifecycle owns.
            on_missing_chunks: `(cx, cz, priority) -> accepted`, the world's
                claim/request entry point. When None (tools, or a server whose
                tick pipeline is down) and `sync_fallback` is set, chunks are
                generated on the calling thread instead.
            save_chunk: `(chunk) -> accepted`. Called with ownership of an
                unloaded chunk before it is dropped; False means "could not
                take it", and the chunk is then *kept* rather than lost.
            load_chunk: `(cx, cz) -> Chunk or None`, the *disk-first* load path
                used by `sync_fallback`. It must be this and not
                `Dimension.load_chunk`: that one regenerates from the seed and
                never looks at the region file, so a chunk a player edited in a
                previous session would come back as pristine terrain -- and then
                be written back over the edit on its next unload.
            unload_grace_ticks: See `UNLOAD_GRACE_TICKS`.
            request_budget: Load requests per tick.
            request_retry_ticks: Re-request interval for a coordinate that is
                still missing.
            claim_stall_ticks: When a blank claim counts as stalled.
            sync_fallback: Generate in-process when no handler is installed.
            enabled: When False, `tick()` does nothing (a dimension that is
                not being driven).
            light: The dimension's `DimensionLight`, if one is running. The
                lifecycle's only interest in it is *when a chunk becomes
                readable*: this pass is what knows a coordinate went from
                claimed to generated, and light has to exist before the
                streamer (or the spawner) reads the chunk. None means no light
                engine -- tools, and any test that builds a lifecycle by hand.
        """
        self.dimension = dimension
        self.dimension_key: DimensionType = dimension.dimension_type
        self.manager: ChunkManager = dimension.chunk_manager
        self.tickets: TicketManager = dimension.tickets

        self._on_missing = on_missing_chunks
        self._save_chunk = save_chunk
        self._load_chunk = load_chunk
        self.unload_grace_ticks = max(0, int(unload_grace_ticks))
        self.request_budget = max(1, int(request_budget))
        self.request_retry_ticks = max(1, int(request_retry_ticks))
        self.claim_stall_ticks = max(1, int(claim_stall_ticks))
        self.claim_stall_max_ticks = max(
            self.claim_stall_ticks, int(claim_stall_max_ticks)
        )
        self.sync_fallback = sync_fallback
        self.enabled = enabled
        #: The dimension's light engine, or None. See the `light` argument.
        self.light = light

        self._players: Dict[str, PlayerTickets] = {}

        #: Chunks with level <= 33 right now, and the chunks this lifecycle is
        #: responsible for unloading. `_owned` is deliberately narrower than
        #: "everything in the manager": chunks another subsystem loaded
        #: explicitly (`World.load_chunk` from a tool or a command) are not
        #: this pass's to drop.
        self._desired: Set[ChunkPos] = set()
        self._owned: Set[ChunkPos] = set()

        #: Desired but absent (or present and not generated yet).
        self._untracked: Set[ChunkPos] = set()
        #: coord -> tick of the last load request; coord -> (tick the blank
        #: claim was first seen, retries so far).
        self._requested: Dict[ChunkPos, int] = {}
        self._claims: Dict[ChunkPos, Tuple[int, int]] = {}
        #: coord -> tick it left the desired set (grace period bookkeeping).
        self._left: Dict[ChunkPos, int] = {}

        self._ticks = 0
        self._fingerprint: Tuple = ()
        self._last_pass = 0
        self._counters = {
            "loaded": 0,
            "unloaded": 0,
            "saved_on_unload": 0,
            "skipped_unchanged": 0,
            "kept_unsaved": 0,
            "requests": 0,
            "request_refused": 0,
            "stall_retries": 0,
        }

        # The manager's LRU is a safety valve for chunks nothing asked for; a
        # chunk a ticket covers must never be evicted, or it would be evicted
        # and reloaded every tick.
        self.manager.set_pin_predicate(self._is_pinned)

    # ---------------- wiring ----------------

    def set_missing_handler(
        self, on_missing_chunks: Optional[Callable[[int, int, int], bool]]
    ) -> None:
        """Install (or clear) the load-request entry point."""
        self._on_missing = on_missing_chunks

    def set_save_handler(self, save_chunk: Optional[Callable[[Chunk], bool]]) -> None:
        """Install (or clear) the hand-off to the save pipeline."""
        self._save_chunk = save_chunk

    def set_load_handler(
        self, load_chunk: Optional[Callable[[int, int], Optional[Chunk]]]
    ) -> None:
        """Install (or clear) the disk-first load path used without a server."""
        self._load_chunk = load_chunk

    # ---------------- player tickets (the automatic half) ----------------

    def set_player_view(
        self,
        player_id: str,
        x: float,
        y: float,
        z: float,
        view_distance: int,
        simulation_distance: int,
    ) -> None:
        """Place or move a player's loading/simulation tickets.

        Called for every movement packet. The common case — the player is still
        inside the chunk their tickets were placed for — returns after three
        comparisons.
        """
        if not self.enabled:
            return

        player = self._players.get(player_id)
        if player is None:
            player = PlayerTickets(player_id=player_id)
            self._players[player_id] = player

        player.x, player.y, player.z = x, y, z

        view = max(1, int(view_distance))
        # The wiki caps the ticking squares at min(2s+1, 63) and min(2s+3, 65),
        # which is exactly "s at most 31" in the level model: the cap belongs to
        # the caller, because the level map is a level map and not a region.
        # Clamping here is what makes a `simulation-distance = 32` config behave
        # like vanilla instead of ticking seven chunks further out.
        simulation = max(1, min(int(simulation_distance), 31))
        target = player.chunk

        if (
            player.anchor == target
            and player.view_distance == view
            and player.simulation_distance == simulation
        ):
            return

        anchor = player.anchor
        player.anchor = target
        player.view_distance = view
        player.simulation_distance = simulation

        loading_key = f"player-loading:{player_id}"
        simulation_key = f"player-simulation:{player_id}"

        if anchor is None:
            self.tickets.add(PLAYER_LOADING, target, radius=view, key=loading_key)
            self.tickets.add(
                PLAYER_SIMULATION, target, radius=simulation, key=simulation_key
            )
            logger.debug(
                f"{self.dimension.name}: player {player_id} tickets at {target} "
                f"(view {view}, simulation {simulation})"
            )
            return

        # Same anchor, different radius: re-add in place (add() refreshes).
        if anchor == target:
            self.tickets.add(PLAYER_LOADING, target, radius=view, key=loading_key)
            self.tickets.add(
                PLAYER_SIMULATION, target, radius=simulation, key=simulation_key
            )
            return

        # Moved. `move` keeps the ticket's identity (and, for a simulation
        # ticket, its remaining behaviour); the radius is refreshed by an
        # explicit re-add so a client that lowered its render distance does not
        # keep the old one.
        self.tickets.move(PLAYER_LOADING, anchor, target, key=loading_key)
        self.tickets.move(PLAYER_SIMULATION, anchor, target, key=simulation_key)
        self.tickets.add(PLAYER_LOADING, target, radius=view, key=loading_key)
        self.tickets.add(
            PLAYER_SIMULATION, target, radius=simulation, key=simulation_key
        )

    def remove_player(self, player_id: str) -> None:
        """Drop a player's tickets. Their chunks unload after the grace period."""
        player = self._players.pop(player_id, None)
        if player is None:
            return
        self.tickets.remove_key(f"player-loading:{player_id}")
        self.tickets.remove_key(f"player-simulation:{player_id}")
        logger.debug(f"{self.dimension.name}: player {player_id} tickets removed")

    @property
    def players(self) -> Dict[str, PlayerTickets]:
        return self._players

    # ---------------- explicit load/unload interfaces ----------------

    def add_ticket(
        self,
        ttype: TicketType,
        x: int,
        z: int,
        *,
        level: Optional[int] = None,
        radius: Optional[int] = None,
        key=None,
    ) -> Ticket:
        """Keep a chunk (and its neighbourhood) loaded at a chosen level."""
        ticket = self.tickets.add(
            ttype, ChunkPos(x, z), level=level, radius=radius, key=key
        )
        self._sync_now()
        return ticket

    def remove_ticket(self, ttype: TicketType, x: int, z: int, key=None) -> bool:
        removed = self.tickets.remove(ttype, ChunkPos(x, z), key)
        if removed:
            self._sync_now()
        return removed

    def force_load(self, x: int, z: int, loaded: bool = True) -> bool:
        """`/forceload` semantics: a persisted, dimension-activating ticket.

        Identity is `(type, position)`, so asking for the same chunk twice is
        one ticket and removing it needs no token from the caller.
        """
        pos = ChunkPos(x, z)
        if loaded:
            self.tickets.add(FORCED, pos)
            self._sync_now()
            return True
        removed = self.tickets.remove(FORCED, pos)
        self._sync_now()
        return removed

    def force_loaded_positions(self) -> List[ChunkPos]:
        return [
            ticket.pos
            for ticket in self.tickets
            if ticket.type is FORCED
        ]

    def load_level(self, x: int, z: int) -> int:
        return self.tickets.level(ChunkPos(x, z))

    def load_type(self, x: int, z: int) -> LoadType:
        return self.tickets.load_type(ChunkPos(x, z))

    def is_loaded(self, x: int, z: int) -> bool:
        """Whether the chunk is loaded *or* on its way (claimed) already."""
        if self.manager.peek(ChunkPos(x, z)) is not None:
            return True
        return self.tickets.is_loaded(ChunkPos(x, z))

    def ticks_entities(self, x: int, z: int) -> bool:
        return self.tickets.ticks_entities(ChunkPos(x, z))

    def ticks_blocks(self, x: int, z: int) -> bool:
        return self.tickets.ticks_blocks(ChunkPos(x, z))

    def unload_chunk(self, x: int, z: int, *, save: bool = True, force: bool = False) -> bool:
        """Take a chunk out of memory now.

        Refuses when a ticket still holds it (level <= 33): the ticket would
        simply reload it on the next tick, and dropping it in the meantime is
        how a chunk a client is looking at disappears. `force=True` is for
        tests and operators.
        """
        pos = ChunkPos(x, z)
        chunk = self.manager.peek(pos)
        if chunk is None:
            return False
        if not force and self.tickets.is_loaded(pos):
            return False
        if save and self._should_save(chunk):
            if not self._hand_over(chunk):
                return False
        self.manager.remove(pos)
        self._forget(pos)
        self._counters["unloaded"] += 1
        return True

    def unload_all(self, *, save: bool = True, force: bool = True) -> int:
        """Drop every chunk this lifecycle owns (shutdown path)."""
        count = 0
        for pos in list(self._owned):
            chunk = self.manager.peek(pos)
            if chunk is None:
                self._forget(pos)
                continue
            if not force and self.tickets.is_loaded(pos):
                continue
            if save and self._should_save(chunk):
                if not self._hand_over(chunk):
                    continue
            self.manager.remove(pos)
            self._forget(pos)
            count += 1
        self._counters["unloaded"] += count
        return count

    # ---------------- the tick ----------------

    def tick(self, tick_no: Optional[int] = None) -> None:
        """Advance one tick: load what is required, drop what is not.

        Ticket *expiry* is not handled here -- `Dimension.tick` owns it, one
        line before this runs. Purging in both places would age every timeout
        at twice the rate, and a one-tick `UNKNOWN` ticket would never be
        observable at all. All this pass needs is the resulting ticket set,
        which it picks up through the fingerprint.
        """
        if not self.enabled:
            return

        self._ticks = self._ticks + 1 if tick_no is None else max(self._ticks + 1, tick_no)
        now = self._ticks

        fingerprint = self._ticket_fingerprint()
        changed = fingerprint != self._fingerprint
        if changed:
            self._fingerprint = fingerprint
            self._rebuild_desired()
            self._prune_bookkeeping()

        self._load_missing(now)

        # The unload pass walks every chunk this lifecycle owns, so it runs when
        # something could have changed (a ticket moved) or once per grace
        # period, not every tick.
        if changed or now - self._last_pass >= self.unload_grace_ticks:
            self._last_pass = now
            self._unload_left(now)

    # ---------------- internals ----------------

    def _ticket_fingerprint(self) -> Tuple:
        """Everything about the ticket set that can change the level map.

        Recomputed every tick and compared instead of caching a "dirty" flag on
        the manager: the ticket set is a handful of entries, the desired set can
        be thousands of chunks, and this way a ticket added by anything (the
        lifecycle, a command, a dragon fight) is picked up without a
        notification protocol.
        """
        return tuple(
            sorted(
                (t.type.name, t.pos.x, t.pos.z, t.level, t.radius)
                for t in self.tickets
            )
        )

    def _rebuild_desired(self) -> None:
        desired = self.tickets.loaded_positions()
        self._desired = desired
        self._untracked = {
            pos for pos in desired if self.manager.peek(pos) is None
        }

    def _prune_bookkeeping(self) -> None:
        """Drop request/claim/left entries that are no longer relevant."""
        for pos in list(self._requested):
            if pos not in self._desired:
                del self._requested[pos]
        for pos in list(self._claims):
            if pos not in self._desired:
                del self._claims[pos]
        for pos in list(self._left):
            if pos in self._desired:
                del self._left[pos]

    def _sync_now(self) -> None:
        """Recompute the desired set outside the tick (an explicit interface call).

        Cheap and safe because the closed-form level model has no propagation
        state to keep consistent, but the *requests* stay on the tick: a caller
        adding a hundred tickets should not fire a hundred claims from the
        caller's thread.
        """
        self._fingerprint = self._ticket_fingerprint()
        self._rebuild_desired()

    def _is_pinned(self, pos: ChunkPos) -> bool:
        return pos in self._desired

    def _load_missing(self, now: int) -> None:
        if not self._untracked:
            return

        ready: List[ChunkPos] = []
        candidates: List[ChunkPos] = []

        for pos in self._untracked:
            chunk = self.manager.peek(pos)
            if chunk is None:
                last = self._requested.get(pos)
                if last is None or now - last >= self.request_retry_ticks:
                    candidates.append(pos)
                continue
            if chunk.is_generated:
                ready.append(pos)
                continue
            # Present but blank: a worker owns it. Normally that resolves on its
            # own; if it does not, the claim is stalled (a worker died, or the
            # request was dropped) and the coordinate has to be asked for again
            # or the view waits on it forever. The wait doubles per retry: with
            # a deep generation queue a chunk can legitimately be blank for
            # minutes, and re-queueing it every stall timeout would multiply the
            # work by the length of the queue.
            evidence = self._claims.get(pos)
            if evidence is None:
                self._claims[pos] = (now, 0)
            else:
                seen_at, retries = evidence
                timeout = min(
                    self.claim_stall_ticks * (2 ** min(retries, 8)),
                    self.claim_stall_max_ticks,
                )
                if now - seen_at >= timeout:
                    self._claims[pos] = (now, retries + 1)
                    self._counters["stall_retries"] += 1
                    candidates.append(pos)

        for pos in ready:
            self._untracked.discard(pos)
            self._claims.pop(pos, None)
            self._request_light(pos)

        if not candidates:
            return

        if len(candidates) > self.request_budget:
            candidates = heapq.nsmallest(
                self.request_budget, candidates, key=self._priority_key
            )

        for pos in candidates:
            self._request(pos, now)

    def _request(self, pos: ChunkPos, now: int) -> None:
        accepted = False
        if self._on_missing is not None:
            try:
                accepted = bool(self._on_missing(pos.x, pos.z, self._priority(pos)))
            except Exception:
                logger.exception(
                    f"{self.dimension.name}: load request for {pos} failed"
                )
                return
        elif self.sync_fallback:
            accepted = self._dimension_load(pos)
        else:
            # No handler and no synchronous fallback: leave the coordinate
            # untracked so it is retried when the pipeline comes up.
            return

        if not accepted:
            self._counters["request_refused"] += 1
            # Left out of `_requested` on purpose: refused requests are retried
            # on the next tick instead of being forgotten (the generator queue
            # being full is transient).
            return

        self._requested[pos] = now
        self._owned.add(pos)
        self._counters["requests"] += 1

    def _dimension_load(self, pos: ChunkPos) -> bool:
        """Load one coordinate on the calling thread (no generation pipeline).

        Prefers the disk-first loader; `Dimension.load_chunk` is only the
        fallback for a dimension with no storage behind it (a scratch dimension
        in a test), because it regenerates from the seed and would silently
        replace saved terrain -- and player edits -- with a fresh copy.
        """
        try:
            if self._load_chunk is not None:
                chunk = self._load_chunk(pos.x, pos.z)
            else:
                chunk = self.dimension.load_chunk(pos.x, pos.z)
        except Exception:
            logger.exception(f"{self.dimension.name}: synchronous load of {pos} failed")
            return False
        return chunk is not None

    def _priority_key(self, pos: ChunkPos):
        return (self._nearest_player_distance(pos), pos.x, pos.z)

    def _nearest_player_distance(self, pos: ChunkPos) -> int:
        if not self._players:
            return 0
        best = 1 << 30
        for player in self._players.values():
            if player.anchor is None:
                continue
            distance = _chebyshev(pos, player.anchor)
            if distance < best:
                best = distance
        return 0 if best == (1 << 30) else best

    def _priority(self, pos: ChunkPos) -> int:
        """Nearer chunks outrank farther ones in the generation queue."""
        if not self._players:
            return IDLE_PRIORITY
        return max(0, 64 - self._nearest_player_distance(pos))

    def _unload_left(self, now: int) -> None:
        if not self._owned:
            return

        for pos in list(self._owned):
            if pos in self._desired:
                self._left.pop(pos, None)
                continue

            chunk = self.manager.peek(pos)
            if chunk is None:
                self._forget(pos)
                continue

            since = self._left.get(pos)
            if since is None:
                self._left[pos] = now
                continue
            if now - since < self.unload_grace_ticks:
                continue

            if self._should_save(chunk):
                if not self._hand_over(chunk):
                    # The save pipeline could not take it. Keeping the chunk is
                    # the only lossless answer; the next pass tries again.
                    self._counters["kept_unsaved"] += 1
                    continue
                self._counters["saved_on_unload"] += 1
            else:
                # On disk already and untouched: dropping it is free. This is
                # the counter that shows the skip policy working.
                self._counters["skipped_unchanged"] += 1

            self.manager.remove(pos)
            self._forget(pos)
            self._counters["unloaded"] += 1

    def _should_save(self, chunk: Chunk) -> bool:
        """Whether an unloaded chunk is worth writing.

        The policy itself lives in `chunk_saver.should_persist` because the
        manager's LRU safety valve has to answer it the same way; this is only
        the call site that knows the chunk is being unloaded.
        """
        return should_persist(chunk)

    def _hand_over(self, chunk: Chunk) -> bool:
        """Give a chunk to the save pipeline. True when it accepted ownership."""
        if self._save_chunk is None:
            return False
        try:
            return bool(self._save_chunk(chunk))
        except Exception:
            logger.exception(
                f"{self.dimension.name}: handing chunk {chunk.position} to the "
                f"save pipeline failed"
            )
            return False

    def _forget(self, pos: ChunkPos) -> None:
        self._owned.discard(pos)
        self._untracked.discard(pos)
        self._left.pop(pos, None)
        self._requested.pop(pos, None)
        self._claims.pop(pos, None)

    # ---------------- notifications from the actor thread ----------------

    def adopt_claim(self, pos: ChunkPos) -> None:
        """Take unload responsibility for a claim someone else created.

        The shared missing-chunk callback is reached from two callers: this
        class's own `_request` (which adds to `_owned` itself) and the
        streamer's view pump, which has no bookkeeping of its own. A claim the
        streamer created first would otherwise be invisible to every unload
        path -- `_rebuild_desired` reads a present claim as "on its way", so
        the coordinate never enters `_untracked`, `_request` never runs for
        it, and on the player's departure the unload pass walks only `_owned`
        and leaves it in memory forever. Adoption makes "requested by anyone
        while the lifecycle is the load authority" imply "unloaded by the
        lifecycle".

        Idempotent with `_request`: a coordinate claimed by both callers is
        added once.
        """
        self._owned.add(pos)

    def on_chunk_ready(self, pos: ChunkPos) -> None:
        """A generated chunk arrived: it is no longer waiting to be loaded.

        It is also queued for its first light pass, and that line is load
        bearing: a chunk a worker built carries `ChunkSection`'s **default**
        light arrays (`world/chunk_gen_worker.py` packs blocks and non-air
        counts, never light), so between this moment and the patch that fills
        them the encoder has nothing real to send and falls back to the
        heightmap approximation -- 15 above the surface, 0 below, no block
        light. Measured on a live probe (real server, real generator pool, one
        viewer): without this call the engine's queue stayed empty for the whole
        session (`pending_chunks=0`, `chunks_lit=0`), all 49 chunk packets went
        out with `light_populated=False`, and the world the client drew was the
        approximation. The engine is the only thing that can light it, and this
        is one of the two places that learn a coordinate became readable (the
        other is `_load_missing`, where a chunk read back from a region file is
        noticed -- and which this method would otherwise shortcut, since it
        discards the coordinate from `_untracked` before that pass can see it).
        """
        self._untracked.discard(pos)
        self._claims.pop(pos, None)
        self._request_light(pos)

    def _request_light(self, pos: ChunkPos) -> None:
        """Queue a chunk for its first light pass, if an engine is attached.

        Called from the two places that learn a coordinate became readable:
        `on_chunk_ready` (the actor's `ChunkReadyMsg`) and `_load_missing`,
        which is where a chunk loaded from disk or generated by the
        synchronous fallback is noticed. Both are needed -- a chunk read from a
        region file never goes through `on_chunk_ready`, and it is exactly the
        chunk whose stored light may be missing (`light_populated` False).

        The engine keeps the coordinate queued until all nine of its neighbours
        are in memory (`LIGHT_PLAN.md` §4.5), so queueing early is free.
        """
        if self.light is not None:
            self.light.request_chunk(pos)

    def discard_request(self, x: int, z: int) -> None:
        """Forget a load request the pipeline dropped (a failed generation)."""
        pos = ChunkPos(x, z)
        self._requested.pop(pos, None)
        self._claims.pop(pos, None)

    # ---------------- persistence of force-loaded chunks ----------------

    def persistable_tickets(self) -> List[dict]:
        return self.tickets.persistable()

    def restore_tickets(self, records) -> int:
        restored = self.tickets.restore(records)
        if restored:
            self._sync_now()
        return restored

    # ---------------- introspection ----------------

    @property
    def desired_chunks(self) -> int:
        return len(self._desired)

    @property
    def owned_chunks(self) -> int:
        return len(self._owned)

    def owned_positions(self) -> List[ChunkPos]:
        """The coordinates this lifecycle is holding, as a snapshot list.

        `owned_chunks` answers "how many", which is all the streamer's stats
        need. The natural spawner needs the coordinates themselves: it walks
        them once a round to find the chunks that are entity-ticking, and
        `spawnable_chunk_count` is the divisor of the population cap.

        Returned as a new list rather than the internal set, because the
        caller is a different thread from the one that mutates `_owned` only
        in the sense that it *copies* - a set being mutated during iteration
        would raise, and a copy costs a few hundred small tuples once every
        twenty ticks. This is deliberately NOT the residency test: `_owned`
        gains a coordinate when a load is *requested*, and the chunk leaves
        memory later, after the grace period. `NaturalSpawner._reconcile_residency`
        asks `manager.peek()` for that.
        """
        return list(self._owned)

    def stats(self) -> Dict[str, object]:
        loading = 0
        for pos in self._owned:
            chunk = self.manager.peek(pos)
            if chunk is not None and not chunk.is_generated:
                loading += 1
        return {
            "dimension": self.dimension.name,
            "ticks": self._ticks,
            "tickets": len(self.tickets),
            "players": len(self._players),
            "desired": len(self._desired),
            "owned": len(self._owned),
            "untracked": len(self._untracked),
            "in_memory": self.manager.get_chunk_count(),
            "loading": loading,
            "pending_requests": len(self._requested),
            "waiting_unload": len(self._left),
            "ticket_types": self.tickets.types_in_use(),
            **self._counters,
        }
