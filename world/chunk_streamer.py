"""
Per-player chunk streaming.

A `ChunkStreamer` owns the set of chunks each connected player has been sent,
and tops that set up as the player moves. One streamer lives on each
`DimensionActor` thread, which is the same thread that owns the `Dimension` —
so reading a chunk here is a plain read with no lock, and the view bookkeeping
never races the tick that mutates it.

The thread boundary that actually matters is the socket. `send_packet` is an
asyncio coroutine owned by the event loop, while this module runs on an actor
thread. The only crossing point is `ChunkSink.send`, which hands a finished
list of packets to the loop and lets the loop write them. Nothing here ever
touches the socket directly.

Per tick, per player, the streamer emits:

    [UpdateViewDistance]              once, when the radius is first known
    [UpdateViewPosition]              whenever the player's chunk changes
    [ForgetLevelChunk] x n            whatever left the view disc
    LightUpdate x n                   light that changed in a chunk they have
    ChunkBatchStart
    LevelChunkWithLight x n           nearest-first, throttled to a budget
    ChunkBatchFinished(n)

`LightUpdate` is the one thing here that does not come from the view: a chunk's
light changes on its own (a torch placed, a hole dug), and the only thing that
knows is the dimension's `DimensionLight`. Its dirty set is drained **once per
tick**, before the views are walked, so every player is offered the same
changes; one goes only to a player who already has that chunk, because for
anyone else the light travels inside the chunk packet. Light updates and chunk
sends draw on the same `chunks_per_tick` budget.

The view is a *disc*, not a square. A square view draws its own edges in the
fog, which reads as a wall of missing terrain a few chunks out.
"""

import asyncio
import math
from concurrent.futures import Future
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from loguru import logger

from network.packet import Packet
from network.packets import (
    CChunkBatchFinished,
    CChunkBatchStart,
    CLevelChunkWithLight,
    CLightUpdate,
    CSetChunkCacheCenter,
    CSetChunkCacheRadius,
    CUnloadChunk,
)
from world.chunk_codec import ChunkCodec, default_biome_id
from world.dimension import Dimension

if TYPE_CHECKING:  # pragma: no cover - the engine imports numba; nothing here needs it
    from world.light_engine import DimensionLight

ChunkCoord = Tuple[int, int]


def _log_send_failure(fut: "Future") -> None:
    """Surface a failed send without letting the exception die silently.

    `run_coroutine_threadsafe` hands the failure to a Future on the actor
    thread; nobody awaits it, so without this the connection would just stop
    receiving chunks and look like a hung client.
    """
    if fut.cancelled():
        return
    exc = fut.exception()
    if exc is not None:
        logger.warning(f"Chunk batch send failed: {exc}")


class ChunkSink:
    """One player's socket, as seen from the actor thread.

    Built on the event loop (it captures the running loop), driven from an
    actor thread. `send` is the whole point where those two worlds meet.
    """

    def __init__(self, send: Callable[[Sequence[Packet]], "asyncio.Future"], loop: asyncio.AbstractEventLoop):
        # send: async (Sequence[Packet]) -> None
        self._send = send
        self._loop = loop
        self.closed = False

    def send(self, packets: Sequence[Packet]) -> None:
        """Queue `packets` for writing on the event loop. Safe from any thread."""
        if self.closed or not packets:
            return
        if self._loop.is_closed():
            self.closed = True
            return
        asyncio.run_coroutine_threadsafe(self._send(list(packets)), self._loop) \
            .add_done_callback(_log_send_failure)

    def close(self) -> None:
        """Stop accepting sends. Chunks already handed to the loop still go out."""
        self.closed = True


@dataclass(frozen=True)
class PlayerViewMsg:
    """Attach or move a player's view. Handled on the dimension actor thread.

    `sink=None` means "remove this player entirely"; the fields are then
    ignored. Keeping attach/move/remove on one message type means the actor
    inbox stays a single ordered stream per dimension — a player cannot be
    moved after being removed if the messages arrive in order.

    One message now feeds three consumers on the actor thread: the chunk
    streamer (terrain in view), the entity tracker (who can see whom) and the
    player movement validator (`physics.player`, "is this path possible").
    The entity fields below are ignored by the streamer and the chunk
    fields are re-derived by the tracker; sharing the message keeps the
    inbox one ordered stream instead of three interleaved ones that could
    disagree about ordering - in particular a player's leave is one message,
    so none of the three can miss it.

    `entity_id` is the player's runtime entity id — the tracker's key.
    `player_id` remains the handler-identity string the streamer has always
    keyed by; the two are different on purpose (see
    client_handler._publish_view). `uuid` is the profile UUID, carried
    because Add Entity must send the same one Player Info Add did (the
    client binds skin and tab-list entry by it). `spawn_blob` carries the
    prebuilt Set Entity Data payload used the first time another client sees
    this player; only the tracker reads it.
    """

    player_id: str
    sink: Optional[ChunkSink] = None
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    radius: int = 0
    #: Simulation distance, in chunks. Separate from `radius` (the client's
    #: render distance) because the two answer different questions: `radius`
    #: says how far terrain is *sent*, this says how far it is *ticked*. Only
    #: the lifecycle reads it, to place the simulation ticket; the streamer's
    #: disc stays render-distance sized.
    simulation_radius: int = 0
    entity_id: int = 0
    uuid: str = ""
    yaw: float = 0.0
    pitch: float = 0.0
    on_ground: bool = True
    spawn_blob: Optional[bytes] = None
    #: Spectator mode. Only the natural spawner reads it, and it is the one
    #: place the flag has to survive the trip: vanilla skips spectators in
    #: both the "is a mob too close to you" test and the local population
    #: cap, so a spectator standing in a field must not suppress spawning
    #: there - and must not have their own cap counted either.
    spectator: bool = False
    #: How many poses this handler has published for this player, counting
    #: from 1. `physics.player` uses it to recognise a message that is not
    #: newer than the one it already judged (the inbox is ordered, so this is
    #: insurance rather than a load-bearing guarantee); 0 means "publisher did
    #: not sequence this message", which is what every pre-existing
    #: construction site in tools/ sends and is treated as always-fresh.
    sequence: int = 0
    #: The newest teleport id this player has been told about, i.e.
    #: `ClientHandler._pending_teleport_id` at publish time. `physics.player`
    #: seeds its own correction ids above it so that a correction can never
    #: reuse an id the client is still waiting to confirm - the handshake is
    #: one sequence shared with the join teleport, not a second one.
    teleport_id: int = 0


class PlayerChunkView:
    """What one player has been sent, and what they still need.

    Owned by the actor thread. Never touched from the event loop.
    """

    def __init__(self, player_id: str, sink: ChunkSink, radius: int):
        self.player_id = player_id
        self.sink = sink

        # Chunks already streamed, as (cx, cz).
        self.sent: Set[ChunkCoord] = set()

        # Chunks in the view that are not yet sent, nearest-first. Rebuilt
        # whenever the player crosses a chunk border, because "nearest" is
        # relative to the centre.
        self.queued: List[ChunkCoord] = []

        # Chunks we have already asked the generator for. Re-requesting every
        # tick would flood the priority queue with duplicates.
        self.requested: Set[ChunkCoord] = set()

        # Position as last reported by the player, in blocks.
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

        self.radius = max(1, radius)

        # Framing state. None means "never told the client yet"; the values
        # are compared each tick so a change re-sends, not just the first.
        self.sent_radius: Optional[int] = None
        self.center: Optional[ChunkCoord] = None

    @property
    def chunk(self) -> ChunkCoord:
        return (math.floor(self.x / 16), math.floor(self.z / 16))

    def pending(self) -> int:
        return len(self.queued)


def _chunk_disc(cx: int, cz: int, radius: int) -> Set[ChunkCoord]:
    """Every chunk whose centre is within `radius` of (cx, cz)."""
    r2 = radius * radius
    return {
        (cx + dx, cz + dz)
        for dx in range(-radius, radius + 1)
        for dz in range(-radius, radius + 1)
        if dx * dx + dz * dz <= r2
    }


def _dist_sq(pos: ChunkCoord, center: ChunkCoord) -> int:
    return (pos[0] - center[0]) ** 2 + (pos[1] - center[1]) ** 2


class ChunkStreamer:
    """Streams chunks to every player in one dimension.

    Constructed and driven exclusively on the owning `DimensionActor` thread.
    """

    def __init__(
        self,
        dimension: Dimension,
        on_missing_chunks: Optional[Callable[[int, int, int], bool]] = None,
        light: Optional["DimensionLight"] = None,
    ):
        self.dimension = dimension

        #: The dimension's light engine, or None for a streamer that only moves
        #: terrain (every tool that predates lighting, and the fixtures). The
        #: actor owns the engine and hands it over here -- the same
        #: assignment-after-construction shape `ChunkLifecycle.light` uses, so
        #: the two consumers cannot end up with different engines. Nothing else
        #: in this class reads it.
        self.light = light

        #: Light changes taken from the engine that could not be delivered
        #: inside `chunks_per_tick`. This is *delivery debt*, not a second copy
        #: of "what changed": entries only ever come from `take_dirty()`, they
        #: are re-checked against `view.sent` every tick, and a coord nobody
        #: can see any more simply falls out.
        self._light_outstanding: Set[ChunkCoord] = set()

        #: Light updates skipped because the chunk left memory between the
        #: light write and the send -- `LIGHT_PLAN.md` §4.7's dropped item.
        self.light_dropped_unloaded = 0

        # One codec per dimension: it precomputes a few lookup tables, so
        # building one per packet would be wasteful in the hot path. The height
        # comes from the dimension type, which `world.dimension` reads out of
        # `assets/dimension.json` - a codec built from a hand-copied height
        # encodes the wrong number of sections (the nether used to send 8).
        self.codec = ChunkCodec(
            min_y=dimension.settings.min_y,
            height=dimension.settings.height,
            default_biome=default_biome_id(),
        )

        # Called as on_missing_chunks(cx, cz, priority) for chunks in view that
        # are not loaded yet. The server points this at background generation.
        # Returns False if the request was refused (e.g. the generation queue is
        # full); the caller then leaves the chunk un-requested so it is retried.
        self._on_missing = on_missing_chunks

        self.views: Dict[str, PlayerChunkView] = {}

        # How many chunks a single player may cost in one tick.
        #
        # TODO: this is a fixed constant, and it is the throttle policy. The
        # trade-off is join speed against tick budget: at ~1.13 ms per chunk
        # encode, 16/tick spends ~18 ms of the 50 ms budget on one player,
        # and that cost is per-player, so N players cost N x 18 ms and the
        # overworld actor starts missing deadlines. Vanilla 1.20.2+ instead
        # runs an accumulator the client steers — the client sends back
        # `chunk_batch_received` (a float the server clamps to [0.1, 500],
        # starting at 9.0, max 10 batches in flight) meaning "I processed
        # that many chunks per tick, send me that many more". So the client
        # absorbs a fast server and a slow one paces itself down.
        #
        # That serverbound packet does not exist in this codebase yet. The
        # choice worth making: keep the constant and just cap it lower, or
        # add the ack path and let the client drive. The accumulator is a
        # small addition once the packet exists, and it is what makes view
        # distance survive more than a couple of players.
        #
        # One budget covers two kinds of send: a chunk and a `LightUpdate` cost
        # this player one slot each. Light goes first within the tick, because
        # it corrects terrain the player is already looking at while a queued
        # chunk is not on screen yet (and the chunk packet carries its own light
        # when it does go). In the steady state the light side is a handful of
        # coords, so chunks still get nearly the whole budget.
        self.chunks_per_tick: int = 16

    # ---------------- player lifecycle (actor thread) ----------------

    def attach(self, msg: PlayerViewMsg) -> None:
        """Add or move a player's view."""
        if msg.sink is None:
            self.detach(msg.player_id)
            return

        view = self.views.get(msg.player_id)
        if view is None:
            view = PlayerChunkView(msg.player_id, msg.sink, msg.radius)
            self.views[msg.player_id] = view
            logger.info(
                f"Streamer attaching {msg.player_id} at radius {view.radius}"
            )

        view.sink = msg.sink
        view.radius = max(1, msg.radius)
        view.x, view.y, view.z = msg.x, msg.y, msg.z

    def detach(self, player_id: str) -> None:
        """Drop a player's view. Their chunks stay client-side; we just stop
        tracking them, and tell the client to forget what fell outside it."""
        view = self.views.pop(player_id, None)
        if view is None:
            return
        view.sink.close()
        logger.info(
            f"Streamer detached {player_id} "
            f"({len(view.sent)} chunks sent, {view.pending()} pending)"
        )

    def detach_all(self) -> None:
        for player_id in list(self.views):
            self.detach(player_id)

    # ---------------- the tick ----------------

    def pump(self) -> None:
        """Advance every view by one tick. Runs on the actor thread.

        The light changes are taken from the engine **once**, before any view is
        walked: `take_dirty()` returns and clears the dimension's set, so a
        per-view take would hand the second player an empty one. What the budget
        could not deliver is carried to the next tick rather than dropped, and
        `Chunk.is_light_dirty` is cleared only after every view has had its turn
        -- inside the loop it would hide the change from the next player, and
        the same flag is what `chunk_saver.should_persist` reads.
        """
        light_work = sorted(self._take_light_dirty() | self._light_outstanding)
        delivered: Set[ChunkCoord] = set()
        owed: Set[ChunkCoord] = set()

        for view in list(self.views.values()):
            try:
                self._pump_view(view, light_work, delivered, owed)
            except Exception:
                # One broken view must not stop the others, and must not
                # kill the tick — the Ticker would start discarding deadlines.
                logger.exception(f"Chunk streaming failed for {view.player_id}")

        self._light_outstanding = owed
        for coord in delivered:
            chunk = self.dimension.get_chunk(*coord)
            if chunk is not None:
                chunk.is_light_dirty = False

    def _take_light_dirty(self) -> Set[ChunkCoord]:
        """The chunks whose light changed since the last tick, or an empty set.

        `DimensionLight.take_dirty()` is the only thing that knows a chunk's
        light bytes changed -- the streamer keeps no second copy of that fact,
        only the delivery debt for changes it has not finished sending
        (`_light_outstanding`). A streamer with no engine attached returns
        nothing and costs one attribute test per tick.
        """
        if self.light is None:
            return set()
        return set(self.light.take_dirty())

    def _pump_view(
        self,
        view: PlayerChunkView,
        light_work: Sequence[ChunkCoord],
        delivered: Set[ChunkCoord],
        owed: Set[ChunkCoord],
    ) -> None:
        if view.sink.closed:
            self.detach(view.player_id)
            return

        center = view.chunk
        target = _chunk_disc(center[0], center[1], view.radius)

        # --- framing ---
        out: List[Packet] = []

        if view.sent_radius != view.radius:
            out.append(CSetChunkCacheRadius(view.radius))
            view.sent_radius = view.radius

        if view.center != center:
            out.append(CSetChunkCacheCenter(center[0], center[1]))
            view.center = center

        # --- unload what left the disc ---
        left = view.sent - target
        if left:
            for coord in left:
                view.sent.discard(coord)
                view.requested.discard(coord)
            out.extend(CUnloadChunk(cx, cz) for cx, cz in sorted(left))

        # --- light that changed in a chunk this player already has ---
        #
        # Before the chunk sends, and out of the same budget: a light update
        # fixes a picture the player is looking at, while a queued chunk has not
        # been drawn yet. `view.sent` is the "who has seen it" test and it is
        # current here -- the unload pass above just pruned it.
        remaining = self.chunks_per_tick
        for coord in light_work:
            if coord not in view.sent:
                # Not sent to this player: the light will ride inside their
                # chunk packet, so there is nothing to correct.
                continue
            if remaining <= 0:
                # Budget spent. Owed, never lost: the engine's set has already
                # been drained, so the debt is carried by `_light_outstanding`.
                owed.add(coord)
                continue
            chunk = self.dimension.get_chunk(*coord)
            if chunk is None:
                # Still in view but evicted: nothing to encode. §4.7's "drop the
                # item" answer for a coord with no chunk behind it.
                self.light_dropped_unloaded += 1
                continue
            out.append(CLightUpdate(
                coord[0], coord[1], self.codec.encode_light(chunk)
            ))
            delivered.add(coord)
            remaining -= 1

        # --- queue what entered the disc, nearest first ---
        missing = target - view.sent
        if missing:
            # Rebuild rather than append: the player may have moved, and
            # "nearest" is relative to a centre that just changed.
            view.queued = sorted(missing, key=lambda c: (_dist_sq(c, center), c))
        self._request_missing(view, center)

        # --- send, within what the light updates left of the budget ---
        batch: List[Packet] = []
        for _ in range(remaining):
            if not view.queued:
                break
            coord = view.queued[0]
            chunk = self.dimension.get_chunk(*coord)
            if chunk is None or not chunk.is_generated:
                # `None` means a worker has not created it yet; not
                # `is_generated` means a worker created the blank Chunk and is
                # filling it in right now. Both mean "not ready".
                #
                # Strict ordering: if the *nearest* queued chunk is still
                # generating, hold everything. This is what guarantees the
                # player's own chunk is the first thing the client ever sees,
                # and it makes terrain grow outward from under the player
                # instead of popping in at the far edge.
                #
                # The second condition is not optional. Sending a blank chunk
                # would put an all-air cube on the client and, because the
                # coord is marked `sent` below, nothing would ever replace it
                # -- the view would look finished and be permanently wrong.
                break
            view.queued.pop(0)
            batch.append(CLevelChunkWithLight(self.codec.encode(chunk)))
            view.sent.add(coord)
            view.requested.discard(coord)

        if not out and not batch:
            return

        out.append(CChunkBatchStart())
        out.extend(batch)
        out.append(CChunkBatchFinished(len(batch)))
        view.sink.send(out)

    def _request_missing(self, view: PlayerChunkView, center: ChunkCoord) -> None:
        """Ask for background generation of queued chunks that aren't loaded."""
        if not self._on_missing:
            return
        for coord in view.queued:
            if coord in view.requested:
                continue
            # Present but not `is_generated` is not a miss -- a worker created
            # the blank Chunk and is filling it in, so asking again would
            # only queue a duplicate that regenerates work already in flight.
            if self.dimension.get_chunk(*coord) is not None:
                continue
            # Nearer chunks get a higher priority so the generator pool works
            # on what the player is about to see.
            priority = max(0, 64 - math.isqrt(_dist_sq(coord, center)))
            if self._on_missing(coord[0], coord[1], priority):
                view.requested.add(coord)
            # A refused request stays out of `requested`, so the next pump
            # retries it. Marking it anyway would strand the chunk: `queued` is
            # only rebuilt when the disc changes, and nothing else re-requests
            # a coord until it leaves and re-enters the player's view.

    def discard_request(self, cx: int, cz: int) -> None:
        """Forget an in-flight generation request for one coord.

        Called when the claim for (cx, cz) is resolved one way or the other:
        the chunk finished (its presence in the manager now does the dedupe
        the `requested` mark was doing) or generation failed (the coord must
        be re-requestable, or the view stalls behind it forever). Runs on the
        owning actor thread, like everything else in this class.
        """
        coord: ChunkCoord = (cx, cz)
        for view in self.views.values():
            view.requested.discard(coord)

    # ---------------- introspection ----------------

    def stats(self) -> Dict[str, Dict[str, int]]:
        return {
            pid: {
                "sent": len(v.sent),
                "queued": v.pending(),
                "requested": len(v.requested),
                "radius": v.radius,
            }
            for pid, v in self.views.items()
        }
