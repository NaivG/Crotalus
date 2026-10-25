"""
Vanilla's natural mob spawner, at one round per second.

The algorithm is Pumpkin's `crates/pumpkin-core/src/world/natural_spawner.rs`,
which is a faithful port of vanilla's `NaturalSpawner`. The shape of a round:

    for every chunk that is entity-ticking:
        for every category that is under its cap:
            pick a random column in the chunk, and a random y under the surface
            walk up to 4 positions, each nudged up to 5 blocks from the last
                is it far enough from every player, and from the world spawn?
                is the ground a spawn floor, the space empty, the light right?
                spawn, until the chunk's per-type limit

**Three deliberate departures from Pumpkin**, each because following it would
produce a world that does not spawn mobs, or spawns the wrong ones:

1. **`spawnable_chunk_count` is computed, not left at zero.** Pumpkin has a
   setter for it and never calls it, so its global cap test compares against
   zero and its spawner never places anything at all. Vanilla's value is the
   number of entity-ticking chunks in the level, which is what
   `_refresh_chunk_count` counts here. At the default view distances that is
   121 chunks, so a `CREATURE` category caps at `10 * 121 // 289 = 4`.

2. **The spawner row is drawn by weight.** `get_random_spawn_mob_at` ends in
   `.choose(&mut rng())`, which is uniform, and cannot do otherwise because
   Pumpkin's generated `Spawner` struct has no `weight` field. The weights in
   the report are hand-tuned, so `BiomeSpawns.draw` honours them.

3. **Same-type spawns keep 24 blocks between them.** Vanilla's `State.mobs`
   rule, absent from Pumpkin. Without it one round can drop two identical
   herds side by side. It is a *same-type* rule: a herd of four cows standing
   together is correct, and only a second, separate herd is refused.

**Counters are recomputed each round instead of kept incrementally.** Vanilla
and Pumpkin both keep `mob_category_counts`, the per-player local cap and the
spawn potential, mutated from entity add/remove callbacks. Here they are
derived from `dimension.entities` at the top of every round. The list is at
most a few hundred mobs and this runs three times a second, so recomputing
costs less than the bookkeeping would - and it cannot drift. An incremental
counter that misses one removal is a world that slowly stops spawning
anything, with no error to notice it by.

**The RNG is one stream for the level, carried across the round.**
`get_random_pos_within` builds `Xoroshiro::from_seed(get_seed())` fresh on
every call, so every chunk in every round draws the same column offsets and
herds land in an identical pattern everywhere. Vanilla uses a per-level
`WorldgenRandom` that carries state; this does too, and `rng` is injectable
so the checks can pin a seed.

**The interval is a cadence, not a work factor.** Vanilla calls
`spawn_for_chunk` from `tick_chunk` - every tick, for every entity-ticking
chunk - so it samples a column 2420 times a second at the default view
distances. Running the same walk once per 20 ticks instead would sample 121
times a second, and since only about 1% of a column's Y values are a valid
spawn height, that is the difference between a world that populates and one
that stays empty for minutes. So a round here is the same *amount* of work
vanilla does over the interval, spread across it: the round's counters are
built on the first tick of the interval and the candidate list is walked in
`interval` slices, one per tick. Twenty slices of a fifth of the work each is
cheaper per tick than vanilla by 20x, and the population moves on a 20-tick
cadence exactly as before.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

from entity.biome_spawns import BiomeSpawns
from entity.mob import Mob, check_spawn_rules, create_mob, mob_ai_context
from entity.mob_category import CATEGORY_LIST, UNLIMITED, MobCategory
from entity.spawn_placement import is_loaded, is_space_empty, is_spawn_position_ok
from world.vanilla.biome import SpawnCost, biome_table
from world.vanilla.random_source import Xoroshiro

#: Vanilla's `MAGIC_NUMBER`. The population cap is quoted per 289 = 17*17
#: chunks and interpolated for however much world is actually loaded, so a
#: server with a small view distance caps lower rather than spawning a
#: full-size population into a fraction of the world.
MAGIC_NUMBER = 289

#: 24 blocks, squared. A spawn must be further than this from every player and
#: from the world spawn, so nothing appears in your face as you arrive.
MIN_SPAWN_DISTANCE_SQ = 576.0

#: 128 blocks, squared. The radius within which a player's own cap applies;
#: `Chunk.mdx` calls it the 240x240 area passive mobs are limited to.
PLAYER_NEAR_SQ = 16384.0

#: The exclusive bound of vanilla's `nextInt(6)`, giving a nudge of -5..+5.
SPAWN_NUDGE = SPAWN_RADIUS = 5

#: Attempts per walk: `ceil(rand * 4)`, so 1 to 4.
MAX_ATTEMPTS = 4

#: Rounds of the walk per chunk per category.
POSITION_ROUNDS = 3

#: Ticks between rounds. Vanilla ticks its spawner every game tick but spawns
#: into only a random subset of chunks each time; a 20-tick cadence over all
#: candidate chunks is the same average at a tenth of the cost. Configurable.
DEFAULT_INTERVAL_TICKS = 20

_ZERO_COST = SpawnCost(0.0, 0.0)


class SpawnState:
    """The per-level bookkeeping one round reads and writes.

    Rebuilt at the top of each round from the dimension's entity list, then
    extended as the round places mobs. `potential` is the exception that
    proves the rule: it has to survive *within* a round, because the energy
    test for attempt N must see the charge attempt N-1 deposited.
    """

    __slots__ = ("category_counts", "player_counts", "potential", "players",
                 "spawnable_chunk_count")

    def __init__(self) -> None:
        #: category id -> how many mobs of it are loaded, over every chunk.
        self.category_counts: List[int] = [0] * len(CATEGORY_LIST)
        #: player_id -> category id -> how many that player can see.
        self.player_counts: Dict[str, List[int]] = {}
        #: `(x, y, z, charge)` of every mob that deposits charge. Empty for
        #: every type whose biome records no `spawn_costs` - see
        #: `can_spawn_by_energy`.
        self.potential: List[Tuple[int, int, int, float]] = []
        #: `(player_id, x, y, z)` for each non-spectator this round consults.
        self.players: List[Tuple[str, float, float, float]] = []
        #: How many chunks are entity-ticking. Set once per round, before any
        #: cap is consulted - the global cap is meaningless without it, and
        #: computing it lazily per query would walk the ticket table dozens of
        #: times a round.
        self.spawnable_chunk_count: int = 0

    def reset_counts(self) -> None:
        """Zero every counter, keeping the dictionaries' shape."""
        for i in range(len(self.category_counts)):
            self.category_counts[i] = 0
        for counts in self.player_counts.values():
            for i in range(len(counts)):
                counts[i] = 0
        self.potential = []

    # ----- population caps -----

    def can_spawn_for_category_global(self, category: MobCategory) -> bool:
        """The level-wide cap, scaled by how much world is loaded.

        `MISC` stores -1 for "no limit", which is not a number to multiply:
        a negative cap would let everything through. It is answered `True`
        here and refused later by `_is_valid_spawn`, because MISC is a
        summon-only category that no biome's spawner list ever names.
        """
        if category.max_count == UNLIMITED:
            return True
        return self.category_counts[_index_of(category)] < (
            category.max_count * self.spawnable_chunk_count // MAGIC_NUMBER
        )

    def can_spawn_for_category_local(self, category: MobCategory,
                                     chunk_x: int, chunk_z: int) -> bool:
        """Whether any player near this chunk is under their own cap.

        The per-player cap is the flat `category.max_count` - 10 for
        `CREATURE` - with no area scaling, measured from the chunk's centre
        (`(x<<4)+8`) rather than from the block. A chunk with no player
        within 128 blocks spawns nothing at all, which is what keeps herds
        tied to where somebody is standing.
        """
        index = _index_of(category)
        cx, cz = (chunk_x << 4) + 8, (chunk_z << 4) + 8
        for player_id, px, _, pz in self.players:
            dx, dz = cx - px, cz - pz
            if dx * dx + dz * dz >= PLAYER_NEAR_SQ:
                continue
            counts = self.player_counts.get(player_id)
            # A player with no entry has placed nothing and is under the cap
            # for every category. This is the common case early on, and
            # short-circuiting here matches Pumpkin's `can_spawn`.
            if counts is None or counts[index] < category.max_count:
                return True
        return False

    # ----- spawn potential -----

    def potential_energy_change(self, x: int, y: int, z: int,
                                charge: float) -> float:
        """The energy a mob of this type would find at this position.

        Each existing charge contributes `its_charge / distance` and the sum
        is scaled by the newcomer's own charge. A mob at the exact position of
        an existing charge gets infinity, which is how a stack self-limits.
        The falloff is `1/d`, not `1/d^2` - reading it as a squared inverse
        over-thins the field and lets more mobs through than vanilla.
        """
        if charge == 0.0:
            return 0.0
        total = 0.0
        for cx, cy, cz, other in self.potential:
            dx, dy, dz = cx - x, cy - y, cz - z
            dist_sq = dx * dx + dy * dy + dz * dz
            if dist_sq == 0:
                return math.inf
            total += other / math.sqrt(dist_sq)
        return total * charge

    def can_spawn_by_energy(self, x: int, y: int, z: int, energy_budget: float,
                            charge: float) -> bool:
        """Whether this position is within the type's per-block energy budget.

        A biome recording no `spawn_costs` for the type spends no energy, and
        the answer is unconditionally yes. That is vanilla's own default, not
        a shortcut around a missing feature: it is why 63 of the 65 biomes
        spawn monsters off an empty cost table. Only `soul_sand_valley` and
        one other record costs in this report, and neither holds a type this
        server builds.
        """
        if energy_budget == 0.0 and charge == 0.0:
            return True
        return self.potential_energy_change(x, y, z, charge) <= energy_budget

    def deposit_charge(self, x: int, y: int, z: int, charge: float) -> None:
        """Record a placed mob's charge, so the next attempt sees the field."""
        if charge != 0.0:
            self.potential.append((x, y, z, charge))

    def after_spawn(self, category: MobCategory, player_ids: Sequence[str],
                    x: int, y: int, z: int, charge: float) -> None:
        """Count a mob that has just been placed."""
        index = _index_of(category)
        self.category_counts[index] += 1
        for player_id in player_ids:
            counts = self.player_counts.get(player_id)
            if counts is not None:
                counts[index] += 1
        self.deposit_charge(x, y, z, charge)


def _index_of(category: MobCategory) -> int:
    return CATEGORY_LIST.index(category)


class NaturalSpawner:
    """Runs vanilla's spawn loop over one dimension, on that dimension's thread.

    Owns its RNG, its per-chunk entity index and the round counter. Holds no
    locks: like the streamer and the tracker beside it, everything it touches
    belongs to the one actor thread.
    """

    def __init__(self, dimension, lifecycle=None, tracker=None,
                 spawns: Optional[BiomeSpawns] = None,
                 rng: Optional[Xoroshiro] = None,
                 seed: int = 0,
                 interval_ticks: int = DEFAULT_INTERVAL_TICKS,
                 enabled: bool = True,
                 physics=None) -> None:
        self.dimension = dimension
        self.lifecycle = lifecycle
        self.tracker = tracker
        #: The dimension's `PhysicsSolver`, or None in a tool that runs the
        #: spawner without one. When it is set, every mob this spawner admits
        #: gets a body, and every mob it drops loses one -- both in this file,
        #: because this is the only place that creates and destroys mobs, and a
        #: body that outlives its mob is a ghost the solver would keep stepping.
        self.physics = physics
        #: The world handle every mob this spawner places is given. Built once
        #: here and shared, because it holds references rather than per-mob
        #: state - see `entity/ai/context.py`. `None` collaborators stay `None`:
        #: a tool that runs the spawner without a tracker or a solver gets
        #: mobs whose AI can see the world but cannot move, and the count that
        #: says so is `MobAI.no_body_ticks`.
        self.ai_context = mob_ai_context(dimension, physics, tracker)
        self.spawns = spawns if spawns is not None else BiomeSpawns(
            ("minecraft:chicken", "minecraft:cow", "minecraft:sheep")
        )
        self.rng = rng if rng is not None else Xoroshiro.from_seed(seed)
        self.interval_ticks = max(1, interval_ticks)
        self.enabled = enabled

        #: Chunk position -> the entity ids of the mobs standing in it. The
        #: residency pass walks this and nothing else, so the cost of noticing
        #: an unload is proportional to the chunks that actually have mobs -
        #: not to the six hundred that are loaded.
        self.chunk_entities: Dict[Tuple[int, int], Set[int]] = {}
        #: Entity id -> the chunk it is currently indexed under. The inverse
        #: of `chunk_entities`, and the reason a mob that *walks* can be moved
        #: between entries in O(1): without it, `rekey_movers` would have to
        #: search every chunk's set to find the one it is in.
        self.entity_chunks: Dict[int, Tuple[int, int]] = {}

        self.state = SpawnState()
        #: The candidate chunks and categories the current round is walking.
        #: Set together by `_begin_round` and read by every slice; empty means
        #: no round is open, which is the state between construction and the
        #: first interval boundary.
        self._candidates: List[Tuple[int, int]] = []
        self._categories: List[MobCategory] = []
        self._stats = {
            "rounds": 0,
            "ticks_spawned": 0,
            "spawned": 0,
            "removed_on_unload": 0,
            "candidate_chunks": 0,
            "skipped_over_cap": 0,
            "rejected_position": 0,
        }

    # ----- player tracking -----

    def attach(self, msg) -> None:
        """Follow a `PlayerViewMsg`, when no tracker is available to ask.

        The tracker is the authority on who is attached and `sync_players`
        reads it once per round; this path is for a spawner constructed
        without one (the checks, and any dimension with no players online).
        Both write the same two fields, so a spawner that has a tracker never
        runs this.

        The key is `msg.entity_id`, not `msg.player_id`, and that has to
        match `viewer_positions` exactly: it is the same table the per-player
        caps are indexed by, so using the integration-side handle here would
        give one player two unrelated identities in one round - one that the
        distance guard counts and one the cap calculator does not.
        """
        player_id = msg.entity_id
        self._forget_player(player_id)
        if msg.sink is None or getattr(msg, "spectator", False):
            # A spectator is not a spawn target and holds no local cap:
            # vanilla skips them in both the distance test and the cap
            # calculator. Forgetting rather than skipping matters - a player
            # who toggles into spectator mode is still one entity id, and
            # the stale entry would keep their old caps counted.
            return
        self.state.players.append((player_id, msg.x, msg.y, msg.z))
        self.state.player_counts.setdefault(
            player_id, [0] * len(CATEGORY_LIST)
        )

    def sync_players(self) -> None:
        """Re-read every player's position from the tracker's viewers.

        Once per round rather than per `PlayerViewMsg`: a player walking one
        block a tick must not also cost a list rebuild a tick, and the
        positions a round uses only have to be right to within the round. The
        tracker owns the authoritative list; this mirrors it instead of
        duplicating the message stream, so the two cannot disagree about who
        is present.
        """
        if self.tracker is not None:
            self.state.players = [
                (viewer_id, x, y, z)
                for viewer_id, x, y, z, spectator in self.tracker.viewer_positions()
                if not spectator
            ]
        live = {p[0] for p in self.state.players}
        for player_id in list(self.state.player_counts):
            if player_id not in live:
                self.state.player_counts.pop(player_id, None)
        for player_id, *_ in self.state.players:
            self.state.player_counts.setdefault(
                player_id, [0] * len(CATEGORY_LIST)
            )

    def _forget_player(self, player_id: str) -> None:
        self.state.players = [p for p in self.state.players
                              if p[0] != player_id]
        self.state.player_counts.pop(player_id, None)

    # ----- per-tick -----

    def tick(self, tick_no: int) -> None:
        """Re-key the movers, reconcile residency, then run this tick's slice.

        Residency runs every tick and spawning only on a slice, and the
        asymmetry is what makes this cheap: a mob in an unloading chunk has
        to leave within the tick or a client that still has it in view keeps
        a ghost, while the population caps only need to be approximately
        current. Re-keying runs on the same every-tick footing for the same
        reason - it is a consequence of the same thing (see `rekey_movers`).

        A round begins on the tick that is a multiple of `interval_ticks` -
        that is where the players, the candidate list and the counters are
        read - and its `interval_ticks` slices then take turns walking the
        candidate list. See the module docstring for why the work is spread
        out rather than done in one go.
        """
        # Refill the AI's per-tick pathfinding allowance first: the entity
        # pass that *used* this tick's budget already ran (Dimension.tick
        # precedes the spawner in the actor's tick order), so this refill is
        # the one the next tick's goals spend against. Unconditional on
        # purpose - the spawner with spawning disabled still owns the mobs
        # it placed, and their AI still searches.
        if self.ai_context is not None:
            self.ai_context.begin_tick()
        self.rekey_movers()
        self._reconcile_residency()
        if not self.enabled:
            return
        interval = self.interval_ticks
        if tick_no % interval == 0:
            self._begin_round()
        elif not self._candidates:
            # A tick that lands inside a round which has not been opened -
            # the very first tick, or one after `enabled` was turned back
            # on. Nothing to walk; this is not an error.
            return
        offset = tick_no % interval
        for chunk_x, chunk_z in self._candidates[offset::interval]:
            for category in self._categories:
                # Both caps, per chunk. Vanilla re-tests the *global* cap
                # inside `spawnForChunk` (through
                # `getFilteredSpawningCategories`) and the local one right
                # after; testing only at the round's start would let a round
                # place its whole allowance in the first chunk it reaches
                # and then keep going. Pumpkin hoists the global test out to
                # once per level, which is why this is worth stating.
                if not self.state.can_spawn_for_category_global(category):
                    self._stats["skipped_over_cap"] += 1
                    continue
                if not self.state.can_spawn_for_category_local(
                    category, chunk_x, chunk_z
                ):
                    self._stats["skipped_over_cap"] += 1
                    continue
                self._spawn_for_chunk(category, chunk_x, chunk_z)
        self._stats["ticks_spawned"] += 1

    def rekey_movers(self) -> int:
        """Move any mob that has walked into a different chunk to that chunk's
        entry, and return how many moved.

        **This exists because mobs walk.** The index used to be written once,
        at placement, and read only by `_reconcile_residency` - which was
        correct while a mob never left the block it was placed on. A wandering
        cow makes it false within seconds, and the consequence is not subtle: a
        mob that has walked into a chunk which then unloads is not in the index
        under that chunk, so the residency pass never sees it and it stays in
        `Dimension.entities` and in the tracking table forever, as a ghost in a
        world that has forgotten the ground under it.

        Every tick, and only for mobs this spawner placed: walking
        `self.entity_chunks` rather than `Dimension.entities` is what stops
        the spawner adopting an entity it does not own (a summoned mob, a
        dropped item, anything a future round adds). A mob that has not
        crossed a chunk boundary costs one tuple comparison.
        """
        moved = 0
        for entity_id, previous in list(self.entity_chunks.items()):
            entity = self.dimension.get_entity(entity_id)
            if entity is None:
                # Gone by another path; the removal paths clear the index, so
                # reaching here means something bypassed them. Drop both
                # halves of the index rather than carry a key to a mob nobody
                # can find -- a stale id left in a chunk's set would break the
                # `sum(len(v)) == len(entities)` shape the pipeline check
                # asserts, and a residency pass over a chunk holding only
                # stale ids would iterate ghosts.
                self.entity_chunks.pop(entity_id, None)
                ids = self.chunk_entities.get(previous)
                if ids is not None:
                    ids.discard(entity_id)
                    if not ids:
                        self.chunk_entities.pop(previous, None)
                continue
            current = entity.position.chunk
            if current == previous:
                continue
            ids = self.chunk_entities.get(previous)
            if ids is not None:
                ids.discard(entity_id)
                if not ids:
                    self.chunk_entities.pop(previous, None)
            self.chunk_entities.setdefault(current, set()).add(entity_id)
            self.entity_chunks[entity_id] = current
            moved += 1
        return moved

    def forget_entity(self, entity) -> bool:
        """Drop one mob from the chunk index, the tracker and the dimension.

        The same tables `_reconcile_residency` clears, in the same order and for
        the same reason: the tracker first, while the entity is still reachable
        and its viewers are known, so every client that saw it is told it is
        gone; then the dimension, so nothing is left in a world that has
        forgotten it. The one caller is "this mob fell out of the world"
        (`DimensionActor._on_physics_removed`). Returns False when the dimension
        did not have it, so a double removal is visible rather than silently
        counted.
        """
        if self.dimension.get_entity(entity.entity_id) is None:
            return False
        entity.removed = True
        # Stop the AI before the mob leaves every table, for the same reason
        # the tracker goes first: a goal that is still running holds a
        # navigator destination in a world the mob is about to not be in, and
        # nothing would ever call its `stop`.
        if entity.ai is not None:
            entity.ai.stop()
        if self.tracker is not None:
            self.tracker.remove_entity(entity.entity_id)
        self.dimension.remove_entity(entity.entity_id)
        # The chunk the index says it is in, not the mob's live one: a mob
        # that has walked since the last `rekey_movers` is filed under where
        # the residency pass will find it, and dropping the *live* chunk's
        # entry would leave a stale id in a set the pass may never revisit.
        chunk_pos = self.entity_chunks.pop(entity.entity_id, None)
        if chunk_pos is None:
            chunk_pos = entity.position.chunk
        ids = self.chunk_entities.get(chunk_pos)
        if ids is not None:
            ids.discard(entity.entity_id)
            if not ids:
                self.chunk_entities.pop(chunk_pos, None)
        if self.physics is not None:
            self.physics.remove_body(entity.entity_id)
        return True

    def _reconcile_residency(self) -> None:
        """Drop mobs whose chunk has left memory, from both tables.

        The test is `chunk_manager.peek() is None` - the chunk is genuinely
        gone - rather than the ticket level or the lifecycle's `_owned` set.
        `_owned` gains a coordinate when a load is *requested*, and the chunk
        is only actually removed later, after `unload_grace_ticks`; asking
        the manager is what gives the grace period meaning, and it respects
        chunks a command or a tool loaded directly. One O(1) lookup per chunk
        that holds mobs, and no new callback on `ChunkLifecycle`.

        Both tables are cleared together and in this order: the tracker
        first, while the entity is still reachable and its viewers still
        known, so every client that saw it is told it is gone; then the
        dimension, so nothing is left in a world that has forgotten the chunk.
        """
        for chunk_pos in list(self.chunk_entities):
            if self.dimension.is_chunk_loaded(*chunk_pos):
                continue
            for entity_id in sorted(self.chunk_entities.pop(chunk_pos)):
                self.entity_chunks.pop(entity_id, None)
                entity = self.dimension.get_entity(entity_id)
                if entity is not None:
                    entity.removed = True
                    if entity.ai is not None:
                        entity.ai.stop()
                if self.tracker is not None:
                    self.tracker.remove_entity(entity_id)
                self.dimension.remove_entity(entity_id)
                if self.physics is not None:
                    self.physics.remove_body(entity_id)
                self._stats["removed_on_unload"] += 1

    # ----- one round -----

    def _begin_round(self) -> None:
        """Open a round: read the world, then let the slices walk it.

        Everything the whole round shares is decided here and not recomputed
        per slice - the player list, the candidate chunks, the categories,
        and the population counts. That is a deliberate staleness window of
        one interval: a mob placed in slice 0 is counted against the cap for
        slice 1 (`SpawnState.after_spawn` does that), but a mob placed by
        some other system mid-round is not seen until the next one. Vanilla
        maintains these counters from callbacks and never has the window;
        the trade buys a per-tick cost of `candidates / interval` instead of
        `candidates`.
        """
        self._stats["rounds"] += 1
        self._stats["skipped_over_cap"] = 0
        self._stats["rejected_position"] = 0

        self.sync_players()
        self._candidates = self._candidate_chunks()
        self._stats["candidate_chunks"] = len(self._candidates)
        self._rebuild_state(len(self._candidates))
        self._categories = self._filtered_categories()

    def _filtered_categories(self) -> List[MobCategory]:
        """The categories worth attempting this round.

        Vanilla filters on three gamerule-shaped flags - friendlies, enemies,
        passives - and then the global cap. Only friendlies are on: there is
        no type in `MOB_CLASSES` outside `CREATURE`, and a category whose
        biome rows are all filtered away would be a draw that cannot succeed.
        The `BiomeSpawns.categories()` test is the honest statement of that -
        a category with no surviving row anywhere is not iterated at all,
        rather than iterated and quietly drawing nothing.
        """
        available = set(self.spawns.categories())
        return [
            category for category in CATEGORY_LIST
            if category.is_friendly
            and category.biome_key in available
            and self.state.can_spawn_for_category_global(category)
        ]

    def _candidate_chunks(self) -> List[Tuple[int, int]]:
        """The chunks eligible to spawn this round.

        Entity-ticking is the right filter, and it is a ticket level rather
        than the load level: `Chunk.mdx` puts mob spawning under "chunk tick",
        which is a smaller square than the loaded one. A chunk at load level
        33 is loaded and sendable but not ticking, and a mob placed there
        would sit in a world that is not simulating.
        """
        return [
            (chunk_x, chunk_z)
            for chunk_x, chunk_z in self._owned_positions()
            if self.dimension.ticks_entities(chunk_x, chunk_z)
        ]

    def _owned_positions(self) -> List[Tuple[int, int]]:
        """The chunks the lifecycle is holding, as `(x, z)` pairs.

        Falls back to nothing when there is no lifecycle, which is the right
        answer for a dimension nobody has put a ticket on: a chunk that is in
        memory without a ticket is not ticking, so it is not a candidate.
        """
        if self.lifecycle is None:
            return []
        return [(pos.x, pos.z) for pos in self.lifecycle.owned_positions()]

    def _rebuild_state(self, candidate_count: int) -> None:
        """Recount the population from the entities that are actually loaded.

        The alternative - incrementing on add, decrementing on remove - has
        one failure mode that is very hard to see: a missed removal is a world
        that slowly stops spawning anything, with no error and no crash.
        Rebuilding from `dimension.entities` cannot drift.
        """
        self.state.spawnable_chunk_count = candidate_count
        self.state.reset_counts()

        table = biome_table()
        for entity in list(self.dimension.entities.values()):
            if not isinstance(entity, Mob) or entity.entity_type is None:
                continue
            category = entity.category
            if category is None:
                continue
            index = _index_of(category)
            self.state.category_counts[index] += 1
            ex, ey, ez = entity.block_pos
            for player_id, px, py, pz in self.state.players:
                dx, dy, dz = ex - px, ey - py, ez - pz
                if dx * dx + dy * dy + dz * dz >= PLAYER_NEAR_SQ:
                    continue
                counts = self.state.player_counts.get(player_id)
                if counts is not None:
                    counts[index] += 1
            self.state.deposit_charge(
                ex, ey, ez, self._spawn_cost(table, ex, ey, ez,
                                             entity.entity_type).charge
            )

    # ----- the vanilla walk -----

    def _spawn_for_chunk(self, category: MobCategory, chunk_x: int,
                         chunk_z: int) -> None:
        chunk = self.dimension.get_chunk(chunk_x, chunk_z)
        if chunk is None or not chunk.is_generated:
            return
        x, y, z = self._random_pos_within(chunk, chunk_x, chunk_z)
        if y <= chunk.min_y:
            return
        facts = _block_facts()
        state = chunk.get_block(x & 15, y, z & 15)
        # Vanilla bails when the chosen start block is solid or a full cube:
        # the entire walk happens at this one y, so a bad start y dooms the
        # whole round for this category in this chunk.
        if facts.is_solid(state) or facts.is_full_cube(state):
            return
        self._spawn_category_for_position(category, x, y, z, chunk_x, chunk_z)

    def _random_pos_within(self, chunk, chunk_x: int,
                           chunk_z: int) -> Tuple[int, int, int]:
        """A random column in the chunk and a random y at or below its surface.

        The y is drawn from the whole column rather than pinned to the
        surface, which is what lets a herd appear at a cave mouth as well as
        on a hilltop: the surface is only the *ceiling* of the draw.
        """
        x = (chunk_x << 4) + self.rng.next_bounded_i32(16)
        z = (chunk_z << 4) + self.rng.next_bounded_i32(16)
        top = chunk.get_height_at(x & 15, z & 15) + 1
        bottom = chunk.min_y
        y = bottom if top <= bottom else bottom + self.rng.next_bounded_i32(
            top - bottom + 1
        )
        return (x, y, z)

    def _spawn_category_for_position(self, category: MobCategory, x: int, y: int,
                                     z: int, chunk_x: int,
                                     chunk_z: int) -> None:
        """Three rounds of up to four attempts, each nudged up to 5 blocks.

        The herd size is committed on the first attempt that clears the
        distance guard, and that is what makes a group a *group*: one draw of
        "this is a herd of four cows", then four positions walked out from it,
        rather than four independent draws that would scatter. `attempts` is
        therefore reused as the herd size once the draw happens - the same
        variable, exactly as vanilla uses it.
        """
        y_start = y
        cluster_size = 0
        for _ in range(POSITION_ROUNDS):
            cx, cz = x, z
            spawner = None
            attempts = math.ceil(self.rng.next_f32() * MAX_ATTEMPTS)
            for _ in range(attempts):
                cx += (self.rng.next_bounded_i32(SPAWN_NUDGE)
                       - self.rng.next_bounded_i32(SPAWN_NUDGE))
                cz += (self.rng.next_bounded_i32(SPAWN_NUDGE)
                       - self.rng.next_bounded_i32(SPAWN_NUDGE))

                player_distance_sq = self._nearest_player_distance_sq(
                    cx, y_start, cz
                )
                if not self._is_right_distance(player_distance_sq, cx, y_start,
                                               cz, chunk_x, chunk_z):
                    continue

                if spawner is None:
                    spawner = self._draw_spawner(category, cx, y_start, cz)
                    if spawner is None:
                        break
                    attempts = spawner.min_count + self.rng.next_between(
                        0, max(0, spawner.max_count - spawner.min_count)
                    )

                from entity.entity_type import entity_type_by_name
                entity_type = entity_type_by_name(spawner.type_name)
                if entity_type is None:
                    break
                if not self._is_valid_spawn(category, entity_type, cx, y_start,
                                            cz, player_distance_sq):
                    self._stats["rejected_position"] += 1
                    continue

                entity = self._place(category, entity_type, cx, y_start, cz)
                if entity is None:
                    continue
                cluster_size += 1
                if cluster_size >= entity_type.limit_per_chunk:
                    return

    def _draw_spawner(self, category: MobCategory, x: int, y: int, z: int):
        """One weighted row from the biome at this position.

        None means "this biome spawns nothing of this category here", and the
        caller stops walking. Vanilla commits to a row or gives up on the
        round rather than re-drawing, which is what stops a biome with a thin
        spawner list from over-producing.
        """
        biome = self._biome_at(x, y, z)
        if biome is None:
            return None
        return self.spawns.draw(biome, category.biome_key, self.rng.next_f32())

    def _biome_at(self, x: int, y: int, z: int) -> Optional[str]:
        """The biome's name at a block, or None if its chunk is not loaded."""
        chunk = self.dimension.get_chunk(x >> 4, z >> 4)
        if chunk is None:
            return None
        return biome_table().names.get(chunk.get_biome_at(x, y, z))

    def _is_right_distance(self, player_distance_sq: Optional[float], x: int,
                           y: int, z: int, chunk_x: int,
                           chunk_z: int) -> bool:
        """Far enough from every player, from the world spawn, and in a real chunk.

        Three conditions, all of them about not surprising the player: 24
        blocks from anyone (spectators excluded, matching the cap calculator),
        24 blocks from the world spawn in the overworld, and a chunk that is
        either the one being processed or another loaded one - the walk nudges
        into a neighbour, and that neighbour has to be real or the placement
        check would read air and pass.

        Vanilla also requires the position to be inside the world border.
        This server has no world border, so that condition is absent; it is a
        declared gap, not one that happens to pass.
        """
        if (player_distance_sq is None
                or player_distance_sq <= MIN_SPAWN_DISTANCE_SQ):
            return False

        if self.dimension.name == "minecraft:overworld":
            sx, sy, sz = self.dimension.spawn_point
            dx, dy, dz = sx - x, (sy + 0.5) - y, sz - z
            if dx * dx + dy * dy + dz * dz < MIN_SPAWN_DISTANCE_SQ:
                return False

        if (x >> 4, z >> 4) == (chunk_x, chunk_z):
            return True
        return is_loaded(self.dimension, x, z)

    def _nearest_player_distance_sq(self, x: int, y: int,
                                    z: int) -> Optional[float]:
        """Distance squared to the nearest non-spectator, None if alone.

        Measured from the block *centre* (`x + 0.5`), which is where the mob
        will actually stand and what the 24-block rule is about - a mob
        spawning at a block's corner is not "24 blocks from you".
        """
        best: Optional[float] = None
        cx, cy, cz = x + 0.5, float(y), z + 0.5
        for _, px, py, pz in self.state.players:
            dx, dy, dz = cx - px, cy - py, cz - pz
            dist = dx * dx + dy * dy + dz * dz
            if best is None or dist < best:
                best = dist
        return best

    def _is_valid_spawn(self, category: MobCategory, entity_type,
                        x: int, y: int, z: int,
                        player_distance_sq: float) -> bool:
        """Every rule between "we picked a type" and "we may place it"."""
        if category.max_count == UNLIMITED:
            return False
        # `can_spawn_far_from_player` types ignore the despawn radius; the
        # others are refused beyond it, because vanilla would remove on the
        # same tick what it had just placed.
        if not entity_type.can_spawn_far_from_player:
            reach = category.despawn_distance
            if player_distance_sq > reach * reach:
                return False
        if not entity_type.summonable:
            return False

        biome = self._biome_at(x, y, z)
        if biome is None or not self.spawns.has_type(
                biome, category.biome_key, entity_type.name):
            return False
        if not is_spawn_position_ok(self.dimension, x, y, z, entity_type):
            return False
        if not is_space_empty(self.dimension, entity_type, x + 0.5, float(y),
                              z + 0.5):
            return False
        if not check_spawn_rules(entity_type, self.dimension, (x, y, z)):
            return False
        return self._passes_energy_and_spacing(entity_type, x, y, z)

    def _passes_energy_and_spacing(self, entity_type, x: int, y: int,
                                   z: int) -> bool:
        """The energy budget, then vanilla's 24-block same-type rule.

        The spacing rule is same-type, matching vanilla's `State.mobs`. A herd
        of four cows in one field is what the rule exists to permit; what it
        forbids is a *second* herd of cows appearing beside the first.
        Pumpkin has no such check, so its rounds can stack herds.
        """
        cost = self._spawn_cost(biome_table(), x, y, z, entity_type)
        if not self.state.can_spawn_by_energy(x, y, z, cost.energy_budget,
                                              cost.charge):
            return False
        name = entity_type.name
        for other in self.dimension.entities.values():
            if not isinstance(other, Mob) or other.entity_type is None:
                continue
            if other.entity_type.name != name:
                continue
            ox, oy, oz = other.block_pos
            dx, dy, dz = ox - x, oy - y, oz - z
            if dx * dx + dy * dy + dz * dz < MIN_SPAWN_DISTANCE_SQ:
                return False
        return True

    def _spawn_cost(self, table, x: int, y: int, z: int, entity_type) -> SpawnCost:
        biome = self._biome_at(x, y, z)
        if biome is None:
            return _ZERO_COST
        return table.spawn_cost_of(biome, entity_type.name)

    def _place(self, category: MobCategory, entity_type, x: int, y: int,
               z: int) -> Optional[Mob]:
        """Build the mob, put it in the world, then hand it to the tracker.

        The order is the tracker's own player path: the entity goes into
        `Dimension.entities` first so it has an id and a home, and only then
        into the tracking table, so a client told about it can look it up.
        A mob whose id lands outside the chunk recorded here would be
        invisible to the residency pass and never cleaned up, so the index is
        written from the same coordinates the placement validated.
        """
        entity = create_mob(
            entity_type, x + 0.5, float(y), z + 0.5,
            dimension=self.dimension.name,
            yaw=self.rng.next_f32() * 360.0,
        )
        if entity is None:
            return None
        if not self.dimension.spawn_entity(entity):
            return None
        chunk_pos = (x >> 4, z >> 4)
        self.chunk_entities.setdefault(chunk_pos, set()).add(entity.entity_id)
        self.entity_chunks[entity.entity_id] = chunk_pos
        if self.tracker is not None:
            self.tracker.add_entity(entity)
        if self.physics is not None:
            # The body takes the spawn validation's own verdict on `on_ground`
            # rather than re-deriving it, so a mob that the placement placed on
            # solid ground does not fall a tick before physics agrees.
            self.physics.add_entity(entity)
        # The AI is bound here, and after the body, because a context whose
        # `has_body` says False is a mob that stands still - binding before
        # the body exists would give every mob one tick of standing perfectly
        # still for no reason. A spawner with no context at all (`ai_context`
        # is None - the pipeline check, or a host that wants static mobs)
        # places mobs whose AI never binds, which is inert by construction:
        # `MobAI.tick` returns before any goal runs without a context.
        if entity.ai is not None and self.ai_context is not None:
            entity.ai.bind(self.ai_context)

        cost = self._spawn_cost(biome_table(), x, y, z, entity_type)
        self.state.after_spawn(category, self._players_near(chunk_pos),
                               x, y, z, cost.charge)
        self._stats["spawned"] += 1
        return entity

    def _players_near(self, chunk_pos: Tuple[int, int]) -> List[str]:
        """The player ids whose local cap this chunk's spawn counts against."""
        cx, cz = (chunk_pos[0] << 4) + 8, (chunk_pos[1] << 4) + 8
        return [
            player_id for player_id, px, _, pz in self.state.players
            if (cx - px) ** 2 + (cz - pz) ** 2 < PLAYER_NEAR_SQ
        ]

    # ----- introspection -----

    def stats(self) -> Dict[str, object]:
        result = dict(self._stats)
        result["enabled"] = self.enabled
        result["interval_ticks"] = self.interval_ticks
        result["spawnable_chunk_count"] = self.state.spawnable_chunk_count
        result["tracked_chunks"] = len(self.chunk_entities)
        result["round_open"] = bool(self._candidates)
        result["entities"] = sum(len(v) for v in self.chunk_entities.values())
        result["spawners"] = self.spawns.stats()
        return result

    def __repr__(self) -> str:
        return (f"<NaturalSpawner {self.dimension.name} "
                f"spawned={self._stats['spawned']}>")


def _block_facts():
    from world.vanilla.features import block_facts

    return block_facts()


__all__ = ["DEFAULT_INTERVAL_TICKS", "MAGIC_NUMBER", "MIN_SPAWN_DISTANCE_SQ",
           "PLAYER_NEAR_SQ", "NaturalSpawner", "SpawnState"]
