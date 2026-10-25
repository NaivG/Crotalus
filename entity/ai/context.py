"""
What a goal is allowed to ask about the world, and how it is allowed to move.

Two small things live here, and they are together because they are the two
answers to "what does this mob know about itself":

* `MobAttributes` - the vanilla attribute values a mob is *constructed* with.
  They are per-class constants, not registry data: `assets/entities.json`
  carries no `attributes` section (vanilla sets them in
  `CowEntity.createCowAttributes`), so putting them in the registry would be
  inventing a file the report does not have.
* `AIContext` - the bundle of the three collaborators an AI needs, all three of
  which belong to the dimension actor thread: the `Dimension` (blocks), the
  `PhysicsSolver` (velocity), and the `EntityTracker` (who is watching).

**Why a context object rather than handing the goals the mob and letting them
reach for what they need.** Vanilla hands goals the `Mob`, and the mob's
`getEntity().level()` reaches the world from there. That works because a `Mob`
always lives in a level. Here it does not: a `Mob` built by
`create_mob` exists before `NaturalSpawner._place` has put it in a dimension
or given it a body, and the tests build mobs with no world at all. A goal
reaching through the mob to find a solver would be `None`-checking a
capability that is supposed to be guaranteed by construction.

So the mob holds *no* AI state beyond its goals, and the goals hold an
`AIContext` that the spawner binds once, at placement - the same tick it calls
`physics.add_entity`. Before that binding an unbound mob's AI does nothing,
rather than guessing.

**The block reads go through `spawn_placement.state_at`, not
`dimension.get_block`.** `Dimension.get_block` answers 0 - air - for a
coordinate whose chunk is not in memory, and air passes every predicate a
goal wants to ask ("is this water", "is this solid"). A wandering cow would
therefore happily path into unloaded space and float over the void. The
`None` that `state_at` returns is the rejection, and it is the same rejection
the spawner already trusts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from world.vanilla.features import block_facts

__all__ = ["AIContext", "MobAttributes", "PLAYER_EYE_HEIGHT"]


#: A player's eye height, for the goals that look *at* one. Vanilla reads it
#: off the player entity's type; the tracker hands out positions only, and a
#: player's eye height is 1.62 in every version this server supports.
PLAYER_EYE_HEIGHT = 1.62


#: Node expansions (A* pops) all of a dimension's searches combined may spend
#: in one tick, when config does not say otherwise. Measured on the bench
#: (tools/bench_pathfinding.py): a typical wander leg is 12-40 pops at
#: ~0.09 ms each, and a search that floods a hostile 16-block radius burns
#: its full 560-pop budget for ~48 ms -- one such search is a whole tick, and
#: a herd meeting a river is several of them, which is exactly the discarded
#: tick the /perf stage table showed. 200 pops cap the worst tick at ~20 ms
#: of searching while leaving room for five or so ordinary wander legs.
DEFAULT_PATH_NODES_PER_TICK = 200


def _configured(name: str, default, cast):
    """One `[advanced]` key, clamped to something usable. Config or default.

    Same shape as the light engine's helper (`world/light_engine.py`), which
    reads `[world]`; the pathfinding valves live under `[advanced]` next to
    the other performance valves (`server-authoritative-movement` and co).
    """
    try:
        from core.config import config
    except ImportError:  # pragma: no cover - tools without the config module
        return default
    value = config.get(f"advanced.{name}", default)
    try:
        return cast(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class MobAttributes:
    """The vanilla per-class attribute block.

    Only the ones an AI actually reads this round. Vanilla's cow registers
    `max_health`, `follow_range`, `knockback_resistance` and `movement_speed`;
    the first is already on `Mob`, `knockback_resistance` needs a damage path
    that does not exist, and the other two are here.
    """

    movement_speed: float = 0.25
    """Blocks per tick at `speed_modifier = 1.0`, on ground. See
    `control.GROUND_SPEED_FACTOR` for why the attribute value is the answer
    rather than a fraction of it."""

    follow_range: float = 16.0
    """Blocks a goal may search for something to react to. `LookAtEntityGoal`
    clamps its own range to 6.0, so this is only consulted by goals that do
    not yet exist - it is here so adding one does not mean inventing a number."""

    step_height: float = 0.6
    """Blocks the mob walks up without jumping. Matches the physics solver's
    own `STEP_HEIGHT`, which is the number that will actually stop the mob."""

    jump_power: float = 0.42
    """Upward velocity a jump starts with. Vanilla's `LivingEntity.jumpPower`
    default; none of these three animals override it."""

    max_look_yaw_change: float = 45.0
    max_look_pitch_change: float = 45.0
    """Degrees per tick the look control may turn. Vanilla's
    `getMaxLookYawChange` / `getMaxLookPitchChange` defaults; all three
    animals leave them alone."""


#: Vanilla's movement_speed attributes, per the 1.21.11 data (Pumpkin's
#: generated `entity_type.rs`) and the Speed attribute page alike. Three
#: distinct rows: the cow is the 0.2 row (with goat and mooshroom), the sheep
#: the 0.23 row, and the chicken-pig row is 0.25 (with skeleton). The first
#: three were one-each; the chicken-pig pairing is the first repeat, and what
#: *distinguishes* them on the wire is `escapeDanger`'s speed modifier
#: (1.4 vs 1.5) and `WanderAroundGoal`'s jump-friendly wander. The comment was
#: rewritten when the pig row landed: leaving it as "no two share a row"
#: would let a later reader "un-share" the chicken and the pig and silently
#: break the steady state.
COW_ATTRIBUTES = MobAttributes(movement_speed=0.2, follow_range=16.0)
SHEEP_ATTRIBUTES = MobAttributes(movement_speed=0.23, follow_range=16.0)
CHICKEN_ATTRIBUTES = MobAttributes(movement_speed=0.25, follow_range=16.0)
PIG_ATTRIBUTES = MobAttributes(movement_speed=0.25, follow_range=16.0)
#: Vanilla's rabbit row: 0.3 movement_speed, and a slightly larger follow
#: range (16 -> rabbit spawners in deserts use 8, but the attribute block is
#: 16 in vanilla's default registry, and the goal clamps its own search
#: radius to 6 anyway). Follow_range matches the others here.
RABBIT_ATTRIBUTES = MobAttributes(movement_speed=0.3, follow_range=16.0)


class AIContext:
    """The three collaborators a goal needs, bound once per dimension.

    Immutable and shared by every mob in the dimension: it holds references,
    not per-mob state, so there is exactly one of them per actor and no
    question of who may mutate what. `None` is allowed for all three, because
    `NaturalSpawner` is constructed without a tracker or a physics solver in
    `tools/check_mob_spawning.py`, and a context that insisted on them would
    make those tests build a solver they are not testing.

    The consequence of a missing `physics` is spelled out rather than
    papered over: `set_velocity` returns False, `MobAI` counts it, and the mob
    stands still. It does not fall over, and it does not teleport.

    One piece of per-dimension state rides along: the pathfinding budget
    (`path_nodes_per_tick`), which every search in the dimension spends
    against. It lives here because the finder already reaches the context on
    every search and there is exactly one context per actor, which makes it
    the one place a "how much has this tick already paid" counter can be
    both shared and thread-safe by ownership rather than by locking.
    """

    __slots__ = ("dimension", "physics", "tracker", "_facts", "min_y", "max_y",
                 "path_nodes_per_tick", "_path_nodes_left")

    def __init__(self, dimension=None, physics=None, tracker=None,
                 path_nodes_per_tick=None) -> None:
        self.dimension = dimension
        self.physics = physics
        self.tracker = tracker
        self._facts = None
        settings = getattr(dimension, "settings", None)
        self.min_y = int(settings.min_y) if settings is not None else -64
        self.max_y = (self.min_y + int(settings.height)
                      if settings is not None else 320)
        if path_nodes_per_tick is None:
            path_nodes_per_tick = _configured("path-nodes-per-tick",
                                              DEFAULT_PATH_NODES_PER_TICK, int)
        #: 0 means the budget is off: every search runs to its own
        #: `max_iters`, whatever the tick is doing. The spawner refills the
        #: allowance once per actor tick (`begin_tick`), and every search in
        #: between spends against it (`path_nodes_left` / `spend_path_nodes`).
        self.path_nodes_per_tick = max(0, int(path_nodes_per_tick))
        self._path_nodes_left = self.path_nodes_per_tick

    # ----- pathfinding budget -----

    def begin_tick(self) -> None:
        """Refill the search allowance. Once per actor tick, spawner's job.

        The spawner's `tick` runs on the owning thread every tick (it is also
        what re-keys movers and reconciles residency), after the entity pass
        of the same tick -- so the refill it performs is the one the *next*
        tick's entities spend against. There is no plumbing into
        `Dimension.tick` because there does not need to be.
        """
        self._path_nodes_left = self.path_nodes_per_tick

    def path_nodes_left(self) -> int:
        """Expansions still affordable this tick; -1 when the budget is off.

        -1 rather than a huge number so the finder can tell "no cap" from
        "cap not reached" without a sentinel like `float('inf')` mixing into
        integer `min()`.
        """
        if self.path_nodes_per_tick <= 0:
            return -1
        return self._path_nodes_left

    def spend_path_nodes(self, used: int) -> None:
        """Charge `used` expansions to this tick's allowance."""
        if self.path_nodes_per_tick > 0:
            self._path_nodes_left -= used

    # ----- blocks -----

    def state(self, x: int, y: int, z: int) -> Optional[int]:
        """The block state at a world position, or None if it is not loaded.

        None is the answer a caller must handle, and every caller here does:
        "I cannot see that block" is never "there is nothing there".
        """
        if self.dimension is None:
            return None
        from entity.spawn_placement import state_at

        return state_at(self.dimension, x, y, z)

    @property
    def facts(self):
        """`BlockFacts`, loaded once per context."""
        if self._facts is None:
            self._facts = block_facts()
        return self._facts

    def is_solid(self, x: int, y: int, z: int) -> bool:
        """Whether the block is solid. An unloaded or missing block is False."""
        state = self.state(x, y, z)
        return state is not None and self.facts.is_solid(state)

    def blocks_motion(self, x: int, y: int, z: int) -> bool:
        """Whether the block has any collision at all (fences, grass, cobweb)."""
        state = self.state(x, y, z)
        return state is not None and self.facts.blocks_motion(state)

    def is_water(self, x: int, y: int, z: int) -> bool:
        """Whether the block holds water.

        `fluid_of` rather than `is_liquid`: vanilla's `Fluid.is(FluidTags.WATER)`
        is water specifically, and a cow that swims in lava is a bug, not a
        simplification.
        """
        state = self.state(x, y, z)
        if state is None:
            return False
        return self.facts.fluid_of(state) == "minecraft:water"

    def is_fluid(self, x: int, y: int, z: int) -> bool:
        """Whether the block holds water or lava - what `SwimGoal` asks for.

        Vanilla's `is_in_fluid` is "touching water above the swim line, or
        touching lava at all", so lava counts with no height test. The two are
        one predicate here because the AI's response to both is the same
        impulse, and the damage lava would do has no entry point anyway.
        """
        state = self.state(x, y, z)
        if state is None:
            return False
        return self.facts.fluid_of(state) in ("minecraft:water", "minecraft:lava")

    def is_ground_stable(self, x: int, y: int, z: int) -> bool:
        """Whether there is something to stand on *below* (x, y, z).

        Vanilla's `is_not_stable`, negated: the random-position picker only
        accepts a candidate whose floor is solid, which is what stops a cow
        choosing a spot in mid-air.
        """
        return self.is_solid(x, y - 1, z)

    # ----- movement -----

    def has_body(self, entity_id: int) -> bool:
        """Whether the mob has a physics body to drive."""
        return self.physics is not None and self.physics.has_body(entity_id)

    def set_velocity(self, entity_id: int, vx: float, vy: float,
                     vz: float) -> bool:
        """Write a body's velocity. False when there is no body to write.

        Goes through `PhysicsSolver.set_velocity` rather than
        `EntityPosition.set_velocity` for the reason `physics/solver.py`'s
        docstring spells out at length: `BodyStore` is the state and
        `EntityPosition.velocity_*` is a copy the solver overwrites on its
        next tick. Writing the copy is the classic "the AI set a speed and
        nothing happened" bug.
        """
        if self.physics is None:
            return False
        return self.physics.set_velocity(entity_id, vx, vy, vz)

    # ----- observers -----

    def nearest_viewer(self, x: float, y: float, z: float,
                       max_distance: float):
        """The nearest attached player's feet position, or None.

        Read through `EntityTracker.viewer_positions`, the same snapshot the
        spawner uses to know who is online. Going to the tracker rather than
        to the `World` is deliberate: the tracker is the single authority on
        who is attached, and a goal that asked the world would have a second
        answer to that question.

        Returns `(x, y, z, eye_y)`.
        """
        if self.tracker is None:
            return None
        best = None
        best_sq = max_distance * max_distance
        for _viewer_id, vx, vy, vz, spectator in self.tracker.viewer_positions():
            if spectator:
                continue
            dx = vx - x
            dz = vz - z
            distance_sq = dx * dx + dz * dz
            if distance_sq <= best_sq:
                best_sq = distance_sq
                best = (vx, vy, vz, vy + PLAYER_EYE_HEIGHT)
        return best

    def push_rotation(self, entity_id: int, x: float, y: float, z: float,
                      yaw: float, pitch: float, on_ground: bool) -> bool:
        """Tell the tracker a mob turned without moving.

        The physics solver only pushes bodies that *moved* or that changed
        ground state (`PhysicsSolver._notify_tracker`), so a cow whose head
        turns while it stands still produces no packet on its own - the head
        would visibly lag a tick behind the body forever. The tracker dedups
        on its own baselines (`pick_move` returns `NONE` when nothing changed),
        so calling this every tick is safe.
        """
        if self.tracker is None:
            return False
        self.tracker.update_entity(entity_id, x, y, z, yaw, pitch, on_ground)
        return True

    def viewers(self) -> List[Tuple[int, float, float, float, bool]]:
        """The tracker's viewer snapshot, as `(id, x, y, z, spectator)`."""
        if self.tracker is None:
            return []
        return self.tracker.viewer_positions()

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        name = getattr(self.dimension, "name", None)
        return f"<AIContext {name} physics={self.physics is not None}>"
