"""
Chunk tickets: which chunks are loaded, and what may be done in them.

A chunk is not loaded because someone decided to load it; it is loaded because
some *source* holds a ticket on it -- a player, a portal, `/forceload`, the
dragon fight, a chunk that a falling sand entity is about to need. This module
is that ticket table plus the level model derived from it, and nothing else:
no I/O, no locks, no dependency on `Dimension` or on any thread. Callers own
the instance and the threading story around it. See `Chunk.mdx` for the
vanilla semantics this mirrors (levels, load types, ticket types, idle
timeout).

Why a closed form instead of a flood fill
-----------------------------------------
Vanilla propagates levels by relaxation: a ticket pins its load level on its
own chunk, each step outward adds exactly 1, and the result is capped at 44.
The obvious implementation is a BFS per ticket, or an incremental graph that
is patched every time a ticket moves. Both are wrong for this codebase. The
hot event is a player crossing a chunk border, which happens several times a
second per player, and a flood fill at that moment costs more than everything
else on the dimension's tick put together.

The propagation metric is Chebyshev (a chunk is a neighbour of the 8 around
it) and a ticket is a *square* of constant level with a uniform +1 per step
outside it. That makes the fixpoint available in closed form:

    level(c) = min over tickets t of  t.level + max(0, chebyshev(c, t.pos) - t.radius)

clamped to MAX_LEVEL. This is exact, and the two properties that make it
exact are worth writing down, because they are what a future rewrite would
have to preserve:

*Source.* If `c` is inside `t`'s square then `chebyshev - radius <= 0`, so the
term is exactly `t.level` and never `t.level - something`. The `max(0, ...)`
is load-bearing: without it a ticket would keep *lowering* the level toward
its own centre, which is the opposite of what a square of constant level means.

*Propagation.* Let `m = min(level(c-1), level(c+1))` over the 8 neighbours,
one step closer to whichever ticket achieves `level(c)`. If `c` is already
inside that ticket's square, both `c` and its neighbours there are pinned at
`t.level`, so `level(c) == t.level == 1 + t.level - 1 = 1 + m` after the -1/+1
cancel; if `c` is outside, the neighbour one step closer is one step closer in
Chebyshev too, so its term is exactly one smaller and is itself the minimum
over the neighbours (nothing else can undercut it, or that other ticket would
have been the minimum at `c` as well). Either way `level(c) == 1 + m`. The
clamp preserves the same pair of equations: both sides saturate at
`MAX_LEVEL` together, so a level above 44 is reported as 44 and propagates as
44, which is exactly vanilla's "levels above 44 are instantly unloaded".

So `level(c)` needs one Chebyshev distance per ticket and no propagation
state at all. Nothing has to be invalidated when a ticket is added, moved or
removed, and a player crossing a border costs one dictionary lookup, which is
the whole point.

Two levels, not one
-------------------
`level()` is the *loading* model and `simulation_level()` is the *ticking*
model: the same closed form over only the SIMULATION-flagged tickets. Keeping
them separate is what makes view distance and simulation distance independent
-- a player's `player_loading` ticket decides how far chunks are held in
memory, and their `player_simulation` ticket decides how far entities and
blocks actually run. A chunk can be held in memory at level 31 by a loading
ticket and still not tick a single entity, and `ticks_entities` is False
there. Folding the two ticket kinds into one model would silently widen the
ticking area to the view distance, which is the bug this split exists to
prevent.

Positions
---------
Every position here is a `ChunkPos`, and every distance is Chebyshev (chunk
steps, not blocks, and diagonal counts as one). `ChunkPos` defines `__eq__`
and `__hash__` but no ordering, and it is a mutable dataclass, so it must
never be used as a dict key in a structure that outlives the caller's
intentions -- this module keys its own tables by `(x, z)` tuples for that
reason and only stores `ChunkPos` values as payload.
"""

from collections.abc import Hashable, Iterable, Iterator
from dataclasses import dataclass
from enum import IntEnum, IntFlag
from typing import Dict, List, Optional, Set, Tuple

from world.chunk import ChunkPos

#: Levels above this are instantly unloaded in vanilla. It is also the
#: propagation cap, which is why `level()` never returns more than this.
MAX_LEVEL = 44

#: The lowest level reachable without mods. Nothing here enforces it (a
#: hand-built ticket may use any level, and lower is "more loaded"), it is
#: here so callers do not have to invent their own constant for it.
MIN_LEVEL = 22

#: `level <= 31`: everything runs, entities included.
ENTITY_TICKING_LEVEL = 31
#: `level == 32`: everything but entities, mob spawning and chunk ticks.
BLOCK_TICKING_LEVEL = 32
#: `level == 33`: nothing runs, but blocks and entities are still readable.
BORDER_LEVEL = 33
#: `level >= 34`: not even readable; world generation still happens.
INACCESSIBLE_LEVEL = 34

#: `level <= 33` means "held in memory". Above this the chunk is gone as far
#: as gameplay is concerned, which is the cut-off `loaded_positions()` uses.
MAX_LOADED_LEVEL = BORDER_LEVEL

#: Game ticks without a dimension-activating ticket after which the dimension
#: stops processing entity and block entity ticks (see `Chunk.mdx`).
DIMENSION_IDLE_TIMEOUT = 300


class TicketFlags(IntFlag):
    """Vanilla's five ticket flags, as a bit set.

    The numbers are not free: `int(flags)` is what `load_type_for_level`-style
    serialisation and any future protocol or config surface would carry, so
    they are pinned to the order the wiki lists them in rather than chosen for
    convenience.
    """

    NONE = 0
    #: Restored from disk when the world starts again.
    PERSIST = 1
    #: Chunks are held in memory. Set on every ticket type vanilla defines;
    #: a ticket without it would be a pure record with no loading effect.
    LOADING = 2
    #: Contributes to the simulation (`simulation_level`) model.
    SIMULATION = 4
    #: Keeps the owning dimension from going idle and stopping its ticks.
    KEEP_DIMENSION_ACTIVE = 8
    #: Counts its timeout down even when its own chunk is not loaded yet.
    #: Only `unknown` carries this in vanilla.
    CAN_EXPIRE_IF_UNLOADED = 16


class LoadType(IntEnum):
    """What a chunk's level lets the game do with it."""

    ENTITY_TICKING = 0
    BLOCK_TICKING = 1
    BORDER = 2
    INACCESSIBLE = 3


def load_type_for_level(level: int) -> LoadType:
    """The load type a chunk at `level` is in.

    Only 31/32/33 have their own type; everything else falls off one end or
    the other, and vanilla reports all of it as `INACCESSIBLE` -- including
    levels below 22, which are "more loaded than possible" rather than
    something with a name.
    """
    if level <= ENTITY_TICKING_LEVEL:
        return LoadType.ENTITY_TICKING
    if level == BLOCK_TICKING_LEVEL:
        return LoadType.BLOCK_TICKING
    if level == BORDER_LEVEL:
        return LoadType.BORDER
    return LoadType.INACCESSIBLE


def is_loaded_level(level: int) -> bool:
    """Whether a level means the chunk is held in memory at all."""
    return level <= MAX_LOADED_LEVEL


def chebyshev(a: ChunkPos, b: ChunkPos) -> int:
    """Chunk steps between two positions, diagonals counting as one.

    Exposed because callers reasoning about "is this chunk within N of the
    player" (spawn guards, mob spawning, the checks in `tools/`) need the same
    metric the level model uses, and a second definition of it drifting from
    this one would be invisible until a chunk on a diagonal stopped ticking.
    """
    return max(abs(a.x - b.x), abs(a.z - b.z))


@dataclass(frozen=True)
class TicketType:
    """One row of vanilla's ticket table.

    Frozen because these are shared, module-level singletons: two ticket types
    with the same name and different levels would make `restore()` load a
    different area than the original session had, so nothing may edit one.
    """

    name: str
    #: Load level pinned on the ticket's own chunk (and its radius).
    level: int
    #: Chebyshev radius held at `level`. 0 is a one-chunk ticket.
    radius: int
    #: Ticks until the ticket expires; 0 means it never does.
    timeout: int
    flags: int

    @property
    def does_load(self) -> bool:
        return bool(self.flags & TicketFlags.LOADING)

    @property
    def does_simulate(self) -> bool:
        return bool(self.flags & TicketFlags.SIMULATION)

    @property
    def keeps_dimension_active(self) -> bool:
        return bool(self.flags & TicketFlags.KEEP_DIMENSION_ACTIVE)

    @property
    def persists(self) -> bool:
        return bool(self.flags & TicketFlags.PERSIST)

    @property
    def can_expire_if_unloaded(self) -> bool:
        return bool(self.flags & TicketFlags.CAN_EXPIRE_IF_UNLOADED)

    def __str__(self) -> str:
        return self.name


# --------------------------------------------------------------------------
# The ticket table. Values are from Chunk.mdx "Ticket types"; the comments
# record the one number per row that the wiki states in words rather than
# digits, so a future edit can be checked against the source of truth.
# --------------------------------------------------------------------------

#: Where a (re)joining player lands. 1 second, refreshed while the spawn area
#: is prepared -- hence the short timeout and the level-30 square.
PLAYER_SPAWN = TicketType("player_spawn", level=30, radius=3, timeout=20,
                          flags=TicketFlags.LOADING)

#: Placed while a respawn position is searched for. 1 tick, border level.
SPAWN_SEARCH = TicketType("spawn_search", level=33, radius=0, timeout=1,
                          flags=TicketFlags.LOADING)

#: The ender dragon fight, at (0,0) of the end, for as long as a player can
#: see the boss bar. Level 24 with radius 9 is the largest simulated square
#: any ticket in the game pins. Notably it does *not* keep the end active.
DRAGON = TicketType("dragon", level=24, radius=9, timeout=0,
                    flags=TicketFlags.LOADING | TicketFlags.SIMULATION)

#: The player's own view distance, as a loading (not simulating) square. The
#: radius is supplied by the caller: it is the configured view distance, and
#: it deliberately has no constant here.
PLAYER_LOADING = TicketType("player_loading", level=31, radius=0, timeout=0,
                            flags=TicketFlags.LOADING)

#: The player's simulation distance: the square where entities and blocks
#: actually run. Radius supplied by the caller, same reasoning.
PLAYER_SIMULATION = TicketType("player_simulation", level=31, radius=0,
                               timeout=0,
                               flags=(TicketFlags.LOADING
                                      | TicketFlags.SIMULATION
                                      | TicketFlags.KEEP_DIMENSION_ACTIVE))

#: `/forceload add`, one ticket per chunk, persisted.
FORCED = TicketType("forced", level=31, radius=0, timeout=0,
                    flags=(TicketFlags.LOADING | TicketFlags.SIMULATION
                           | TicketFlags.KEEP_DIMENSION_ACTIVE
                           | TicketFlags.PERSIST))

#: A portal teleport destination. 15 seconds, persisted, keeps the target
#: dimension active -- this is what stops a nether trip from arriving in an
#: idle dimension with frozen entities.
PORTAL = TicketType("portal", level=30, radius=3, timeout=300,
                    flags=(TicketFlags.LOADING | TicketFlags.SIMULATION
                           | TicketFlags.KEEP_DIMENSION_ACTIVE
                           | TicketFlags.PERSIST))

#: A thrown ender pearl's landing area. 2 seconds, refreshed by the pearl both
#: on a timer and whenever it crosses into a new chunk. Not persisted: it is
#: re-created from the pearl's own saved data when its owner rejoins.
ENDER_PEARL = TicketType("ender_pearl", level=31, radius=0, timeout=40,
                         flags=(TicketFlags.LOADING | TicketFlags.SIMULATION
                                | TicketFlags.KEEP_DIMENSION_ACTIVE))

#: Given out automatically when the game has to touch a chunk that has not
#: reached the required generation step. The only type that may expire before
#: its chunk is loaded -- otherwise a chunk that never finishes generating
#: would hold its own ticket forever.
UNKNOWN = TicketType("unknown", level=31, radius=0, timeout=1,
                     flags=TicketFlags.LOADING
                     | TicketFlags.CAN_EXPIRE_IF_UNLOADED)

#: Every ticket type there is, by name. `restore()` resolves persisted records
#: through this, so a name that is not here cannot be loaded back.
TICKET_TYPES: Dict[str, TicketType] = {
    t.name: t for t in (
        PLAYER_SPAWN, SPAWN_SEARCH, DRAGON, PLAYER_LOADING, PLAYER_SIMULATION,
        FORCED, PORTAL, ENDER_PEARL, UNKNOWN,
    )
}

#: The identity of a ticket that is *not* identified by its source: the type,
#: plus the chunk it sits on, plus whatever `key` the caller supplied.
TicketKey = Tuple[str, int, int, Hashable]


@dataclass
class Ticket:
    """One live ticket.

    Mutable on purpose. `add()` on an existing identity refreshes this object
    in place and hands the same instance back, so a caller that keeps the
    return value of `add()` keeps a handle on the ticket for as long as it
    lives, instead of holding a copy that silently stops being the ticket.
    """

    type: TicketType
    pos: ChunkPos
    level: int
    radius: int
    #: Ticks left, or -1 for "never expires".
    ticks_remaining: int
    #: Identity of the source: a player id, an entity id, ... `None` means the
    #: ticket is identified by its position alone, which is what `/forceload`
    #: and portal tickets want.
    key: Hashable = None
    created_tick: int = 0

    @property
    def expires(self) -> bool:
        """Whether this ticket counts down at all."""
        return self.ticks_remaining >= 0

    @property
    def does_load(self) -> bool:
        return self.type.does_load

    @property
    def does_simulate(self) -> bool:
        return self.type.does_simulate

    @property
    def keeps_dimension_active(self) -> bool:
        return self.type.keeps_dimension_active

    @property
    def persists(self) -> bool:
        return self.type.persists

    def ticket_key(self) -> TicketKey:
        """This ticket's identity, in the form the manager's table is keyed by."""
        return (self.type.name, self.pos.x, self.pos.z, self.key)

    def __str__(self) -> str:
        return (f"{self.type.name}@({self.pos.x},{self.pos.z}) L{self.level} "
                f"r{self.radius} ttl={self.ticks_remaining}")


class TicketManager:
    """The tickets of one dimension, and the levels they produce.

    Not thread safe, and not trying to be: the tick, the level queries and the
    ticket edits that answer them all belong on one owner thread (in this
    server, the dimension's actor). The level model is read far more often
    than it is written -- the streamer asks about chunks every tick, while a
    ticket changes when a player crosses a border -- so the storage is shaped
    for the read: a per-position index of the tickets standing there, and one
    arithmetic term per ticket per query.

    `idle_ticks` is the count the wiki's idle timeout is about. It is bumped
    by `tick()` while no ticket keeps the dimension active, and reset the
    moment one does; a dimension is allowed to stop ticking its entities and
    block entities once it reaches `DIMENSION_IDLE_TIMEOUT`. This module only
    counts -- deciding what "stopping" means for a dimension is the dimension
    owner's business.
    """

    def __init__(self, name: str = ""):
        self.name = name
        #: Identity -> ticket. The authority on what exists.
        self._by_key: Dict[TicketKey, Ticket] = {}
        #: `(x, z)` -> the tickets standing on that chunk. Derived state, kept
        #: in step by add/move/remove/remove_key/clear, never written from
        #: outside. Two jobs: `level()` hashes two plain ints instead of a
        #: `ChunkPos` (whose `__hash__` is a Python-level call, on the hottest
        #: query in the dimension), and dropping a ticket finds the tickets it
        #: could be without scanning the whole table.
        self._by_pos: Dict[Tuple[int, int], List[Ticket]] = {}
        self.idle_ticks: int = 0

    # ------------------------------------------------------------------
    # mutate
    # ------------------------------------------------------------------

    def add(self, ttype: TicketType, pos: ChunkPos, *, level: Optional[int] = None,
            radius: Optional[int] = None, key: Hashable = None,
            tick: int = 0) -> Ticket:
        """Add or refresh a ticket, returning the live one.

        The same `(type, pos, key)` triple *is* one ticket: re-adding refreshes
        its level, radius and timeout in place rather than stacking a second
        entry. That is not a convenience -- it is the contract the two moving
        tickets in the game rely on. A player's simulation ticket is re-placed
        every time they cross a chunk border, and an ender pearl re-places its
        ticket whenever it moves or its timeout is about to run out; if either
        stacked, an afternoon of walking would leave thousands of tickets
        behind and the level map would never drop a chunk.

        `tick` is the current game tick, recorded as `created_tick`. It is not
        used for expiry -- tickets count down by `tick()` calls, not by
        timestamp -- it is there so a caller can tell how long a ticket has
        existed.
        """
        tkey = (ttype.name, pos.x, pos.z, key)
        existing = self._by_key.get(tkey)
        if existing is not None:
            # Refresh in place: identity, and any handle a caller is holding,
            # survive the update.
            existing.level = ttype.level if level is None else level
            existing.radius = ttype.radius if radius is None else radius
            existing.ticks_remaining = self._initial_ttl(ttype)
            existing.created_tick = tick
            return existing
        return self._insert(ttype, pos, level, radius, key, tick, None)

    def _insert(self, ttype: TicketType, pos: ChunkPos, level: Optional[int],
                radius: Optional[int], key: Hashable, tick: int,
                ticks_remaining: Optional[int]) -> Ticket:
        """Create a ticket nobody has added before. Shared by add and restore.

        `ticks_remaining=None` means "start the type's timeout"; an explicit
        value is what `restore()` passes so a persisted ticket resumes with the
        countdown it was saved with.
        """
        ticket = Ticket(
            type=ttype,
            pos=pos,
            level=ttype.level if level is None else level,
            radius=ttype.radius if radius is None else radius,
            ticks_remaining=self._initial_ttl(ttype) if ticks_remaining is None
            else ticks_remaining,
            key=key,
            created_tick=tick,
        )
        self._by_key[ticket.ticket_key()] = ticket
        self._by_pos.setdefault((pos.x, pos.z), []).append(ticket)
        return ticket

    @staticmethod
    def _initial_ttl(ttype: TicketType) -> int:
        """The `ticks_remaining` a freshly placed ticket starts with.

        No timeout is stored as -1 rather than 0 so that "never expires" and
        "about to expire" are different numbers; a 0 that meant both would
        make every permanent ticket (a player's, `/forceload`) expire on its
        first tick.
        """
        return -1 if ttype.timeout <= 0 else ttype.timeout

    def remove(self, ttype: TicketType, pos: ChunkPos, key: Hashable = None) -> bool:
        """Drop one ticket. False if there was nothing to drop."""
        tkey = (ttype.name, pos.x, pos.z, key)
        ticket = self._by_key.pop(tkey, None)
        if ticket is None:
            return False
        self._drop_from_pos(ticket)
        return True

    def remove_key(self, key: Hashable) -> int:
        """Drop every ticket owned by `key`; returns how many went.

        Pass the same object the tickets were added with. `None` is a legal
        argument and drops every position-only ticket (all `/forceload` and
        portal tickets in the dimension) -- supported because that is
        occasionally what a shutdown path wants, but it is rarely what a
        caller means, so it is spelled out here rather than guarded against.
        """
        doomed = [t for t in self._by_key.values() if t.key == key]
        for ticket in doomed:
            del self._by_key[ticket.ticket_key()]
            self._drop_from_pos(ticket)
        return len(doomed)

    def move(self, ttype: TicketType, old_pos: ChunkPos, new_pos: ChunkPos,
             key: Hashable = None) -> bool:
        """Move a ticket to another chunk, keeping its identity.

        The identity includes the position, so the table row is re-keyed --
        but the `Ticket` object itself is moved rather than replaced. A caller
        holding the return value of `add` (the streamer holds one per player,
        the pearl holds its own) keeps a live handle across the move, which is
        what "same ticket, new chunk" has to mean for a caller.

        Only the position changes. Level, radius and the timeout in progress
        are carried over, so a portal ticket that has been counting down keeps
        counting after a move instead of getting a fresh 300 ticks.
        """
        tkey = (ttype.name, old_pos.x, old_pos.z, key)
        ticket = self._by_key.pop(tkey, None)
        if ticket is None:
            return False
        self._drop_from_pos(ticket)
        ticket.pos = new_pos
        self._by_key[ticket.ticket_key()] = ticket
        self._by_pos.setdefault((new_pos.x, new_pos.z), []).append(ticket)
        return True

    def clear(self) -> None:
        """Forget every ticket. The level map goes back to all-`MAX_LEVEL`."""
        self._by_key.clear()
        self._by_pos.clear()

    def _drop_from_pos(self, ticket: Ticket) -> None:
        """Remove one ticket from the per-position index."""
        pk = (ticket.pos.x, ticket.pos.z)
        bucket = self._by_pos.get(pk)
        if bucket is None:
            return
        # Matched by identity, not by equality: `Ticket` is a mutable dataclass
        # with value equality, so `==` would happily delete a different ticket
        # that happens to hold the same numbers.
        for index, other in enumerate(bucket):
            if other is ticket:
                del bucket[index]
                break
        if not bucket:
            # Empty buckets are dropped rather than kept: a query that misses
            # the position index walks every bucket in it, so an index that
            # grows with every chunk the player has ever stood on would turn a
            # closed form back into a scan of the whole dimension's history.
            del self._by_pos[pk]

    # ------------------------------------------------------------------
    # query: the level model
    # ------------------------------------------------------------------

    def level(self, pos: ChunkPos) -> int:
        """The chunk's load level; `MAX_LEVEL` when nothing reaches it.

        The closed form from the module docstring, one term per ticket. There
        is no shortcut for "a ticket stands on this chunk" and there cannot be
        one: a ticket whose square covers `pos` contributes exactly its own
        level, and `add` accepts a level override, so a nearby covering ticket
        may legitimately be at a *lower* level than the one standing here and
        win the minimum. A player's chunk is not necessarily at the player's
        level -- a portal or a `/forceload` square overlapping it can pull it
        down -- and loading the player's ticket as "the" answer is a real,
        reachable bug rather than a theoretical one.

        The scan is over the tickets, never over the area, which is the
        property that has to survive any future optimisation here.
        """
        px, pz = pos.x, pos.z
        best = MAX_LEVEL
        for bucket in self._by_pos.values():
            for ticket in bucket:
                term = ticket.level + max(
                    0, max(abs(px - ticket.pos.x), abs(pz - ticket.pos.z))
                    - ticket.radius)
                if term < best:
                    best = term
        return best

    def load_type(self, pos: ChunkPos) -> LoadType:
        return load_type_for_level(self.level(pos))

    def is_loaded(self, pos: ChunkPos) -> bool:
        return is_loaded_level(self.level(pos))

    def simulation_level(self, pos: ChunkPos) -> int:
        """`level()` restricted to SIMULATION tickets.

        Returns `MAX_LEVEL` when no simulating ticket covers `pos`, which
        reads as "nothing ticks here" through `ticks_entities`/`ticks_blocks`.
        Loading tickets are invisible to it by construction: that is the whole
        separation between view distance and simulation distance.

        Unlike `level()` there is no per-position fast path, because the two
        ticket kinds sit on the same chunk: a player's chunk carries both a
        `player_loading` ticket at the view radius (which must not widen the
        ticking area) and a `player_simulation` ticket at the simulation
        radius (which defines it), so the position index cannot answer this
        one on its own.
        """
        best = MAX_LEVEL
        px, pz = pos.x, pos.z
        for bucket in self._by_pos.values():
            for ticket in bucket:
                if not ticket.does_simulate:
                    continue
                term = ticket.level + max(
                    0, max(abs(px - ticket.pos.x), abs(pz - ticket.pos.z))
                    - ticket.radius)
                if term < best:
                    best = term
        return best

    def ticks_entities(self, pos: ChunkPos) -> bool:
        """Whether entities are ticked here (`level <= 31`)."""
        return self.simulation_level(pos) <= ENTITY_TICKING_LEVEL

    def ticks_blocks(self, pos: ChunkPos) -> bool:
        """Whether blocks, block entities and chunk ticks run here (<= 32)."""
        return self.simulation_level(pos) <= BLOCK_TICKING_LEVEL

    def loaded_positions(self) -> Set[ChunkPos]:
        """Every chunk at or below `MAX_LOADED_LEVEL`.

        Enumerated as the union of the tickets' *qualifying squares* rather
        than by scanning a bounding box and calling `level()` per cell: the
        cells a ticket can pin at a loaded level are exactly the Chebyshev
        square of radius `ticket.radius + (MAX_LOADED_LEVEL - ticket.level)`,
        and a ticket already above the cut-off (SPAWN_SEARCH at 33 with
        anything but radius 0, an UNKNOWN pinned on an unloaded chunk)
        contributes none. Cost is the size of the answer, which is the best
        any caller could ask for -- the streamer iterates this set every tick.
        """
        out: Set[ChunkPos] = set()
        for ticket in self._by_key.values():
            reach = MAX_LOADED_LEVEL - ticket.level
            if reach < 0:
                continue
            radius = ticket.radius + reach
            cx, cz = ticket.pos.x, ticket.pos.z
            for x in range(cx - radius, cx + radius + 1):
                for z in range(cz - radius, cz + radius + 1):
                    out.add(ChunkPos(x, z))
        return out

    def ticking_positions(self) -> Set[ChunkPos]:
        """Every chunk where blocks tick (`simulation_level <= 32`).

        Same enumeration over the simulating tickets only, with the cut-off at
        `BLOCK_TICKING_LEVEL`. Entity-ticking chunks are the subset of this
        with a threshold of 31; a caller that needs exactly those should filter
        with `ticks_entities`, since the difference is one ring of squares.
        """
        out: Set[ChunkPos] = set()
        for ticket in self._by_key.values():
            if not ticket.does_simulate:
                continue
            reach = BLOCK_TICKING_LEVEL - ticket.level
            if reach < 0:
                continue
            radius = ticket.radius + reach
            cx, cz = ticket.pos.x, ticket.pos.z
            for x in range(cx - radius, cx + radius + 1):
                for z in range(cz - radius, cz + radius + 1):
                    out.add(ChunkPos(x, z))
        return out

    def activates_dimension(self) -> bool:
        """Whether any ticket keeps this dimension from going idle."""
        return any(t.keeps_dimension_active for t in self._by_key.values())

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    @property
    def tickets(self) -> Tuple[Ticket, ...]:
        """The live tickets, as a tuple.

        A copy, and not a `list` of the internal storage: callers that iterate
        while a tick expires something would otherwise mutate the sequence
        under themselves, and a `list` would invite them to try.
        """
        return tuple(self._by_key.values())

    def types_in_use(self) -> Dict[str, int]:
        """Ticket count per type name. Cheap enough to log every few seconds."""
        counts: Dict[str, int] = {}
        for ticket in self._by_key.values():
            counts[ticket.type.name] = counts.get(ticket.type.name, 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self._by_key)

    def __iter__(self) -> Iterator[Ticket]:
        return iter(tuple(self._by_key.values()))

    def __str__(self) -> str:
        label = self.name or "dimension"
        return (f"TicketManager({label}: {len(self)} tickets, "
                f"{len(self._by_pos)} positions, idle {self.idle_ticks})")

    # ------------------------------------------------------------------
    # tick
    # ------------------------------------------------------------------

    def tick(self, tick_no: int = 0) -> List[Ticket]:
        """Advance every timeout by one tick; return the tickets that expired.

        Two rules, both from the wiki:

        * A type with a timeout counts down once per call and is dropped at
          zero. A ticket whose timeout is `-1` (no timeout defined) is never
          touched -- it has no countdown to advance.
        * A ticket whose own chunk is not loaded does *not* count down, or a
          portal ticket in a chunk nobody has generated yet would expire
          before the chunk it is waiting for ever arrives. The exception is
          `CAN_EXPIRE_IF_UNLOADED` (only `unknown`), which exists precisely so
          that a chunk that never finishes generating cannot hold its own
          ticket open forever.

        Loadedness is sampled once, for every position with a countdown, from
        the level map as it stands *before* this tick removes anything. A
        ticket that loads its own chunk therefore counts down -- its own chunk
        is loaded, by itself -- which is the intended behaviour and the reason
        the sampling is not done per ticket as it is dropped.
        """
        positions = {t.pos for t in self._by_key.values() if t.ticks_remaining > 0}
        loaded = {p: self.is_loaded(p) for p in positions}

        expired: List[Ticket] = []
        for ticket in tuple(self._by_key.values()):
            if ticket.ticks_remaining <= 0:
                continue
            if not loaded[ticket.pos] and not ticket.type.can_expire_if_unloaded:
                continue
            ticket.ticks_remaining -= 1
            if ticket.ticks_remaining == 0:
                del self._by_key[ticket.ticket_key()]
                self._drop_from_pos(ticket)
                expired.append(ticket)

        if self.activates_dimension():
            self.idle_ticks = 0
        else:
            self.idle_ticks += 1
        return expired

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def persistable(self) -> List[dict]:
        """The PERSIST-flagged tickets as plain records.

        Dicts of primitives, not `Ticket` objects: this is the shape that goes
        to disk (`/forceload` chunks and open portals have to survive a
        restart), and it deliberately carries no `ChunkPos`, no `TicketType`
        and no enum -- a record that only survives as long as this process
        does is not persistence.
        """
        return [
            {
                "type": t.type.name,
                "x": t.pos.x,
                "z": t.pos.z,
                "level": t.level,
                "radius": t.radius,
                "ticks_remaining": t.ticks_remaining,
                "key": t.key,
                "created_tick": t.created_tick,
            }
            for t in self._by_key.values() if t.persists
        ]

    def restore(self, records: Iterable[dict]) -> int:
        """Re-add persisted records; returns how many were restored.

        Skipped, not fatal: a record naming a ticket type this build does not
        know (a world saved by a newer version, a type that was removed), or
        one whose type is not persistable, or one with unreadable coordinates.
        Refusing to boot over one bad row would be worse than booting without
        it, and the count is returned so the caller can log the difference.

        The saved level, radius and remaining ticks win over the type's
        defaults, so a restored area is the area that was saved.
        """
        restored = 0
        for record in records:
            if not isinstance(record, dict):
                continue
            ttype = TICKET_TYPES.get(record.get("type"))
            if ttype is None or not ttype.persists:
                continue
            try:
                pos = ChunkPos(int(record["x"]), int(record["z"]))
            except (KeyError, TypeError, ValueError):
                continue
            self._insert(
                ttype,
                pos,
                record.get("level"),
                record.get("radius"),
                record.get("key"),
                int(record.get("created_tick") or 0),
                record.get("ticks_remaining"),
            )
            restored += 1
        return restored

    def stats(self) -> dict:
        """A log-shaped summary of what this dimension is holding.

        One pass over the tickets, and deliberately no `level()` call: this is
        meant to be cheap enough to log on a timer with a few hundred players
        online, and asking the level map a question per ticket would make it
        quadratic in the ticket count for no gain. The load type reported is
        therefore the ticket's *own* level -- the level it pins where it
        stands -- not the chunk's effective level where another ticket may be
        lower. That is the useful reading anyway: it answers "what are these
        tickets for", and the effective map is what `level()` is for.
        """
        by_type: Dict[str, int] = {}
        by_load_type: Dict[str, int] = {}
        expiring = 0
        for ticket in self._by_key.values():
            by_type[ticket.type.name] = by_type.get(ticket.type.name, 0) + 1
            name = load_type_for_level(ticket.level).name
            by_load_type[name] = by_load_type.get(name, 0) + 1
            if ticket.expires:
                expiring += 1
        return {
            "name": self.name,
            "tickets": len(self._by_key),
            "positions": len(self._by_pos),
            "by_type": by_type,
            "by_load_type": by_load_type,
            "expiring": expiring,
            "activates_dimension": self.activates_dimension(),
            "idle_ticks": self.idle_ticks,
        }
