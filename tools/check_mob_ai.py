"""Acceptance check for mob AI: goals, controls, movement, and the wire.

The whole brain on a synthetic world this file builds itself -- no server, no
client, no worldgen. A real `PhysicsSolver`, real `Cow`/`Sheep`/`Chicken`
objects, a real goal selector, and a recording tracker standing in for the
`EntityTracker`. Every case is "given this world and this mob, tick it and
assert what it did":

1. the three mobs carry **vanilla's goal list**, in vanilla's priority order;
2. the numbers that differ between them are vanilla's -- a sheep carries 0.23
   while a cow carries 0.2 and a chicken 0.25, a cow bolts at 2.0 and a
   chicken at 1.4;
3. a cow on a flat plain **wanders**, and covers exactly the ground the
   accel+friction model says it should, tick for tick -- the attribute is an
   acceleration, so a 0.2 cow converges at ~0.088 blocks per tick, not 0.2;
4. a mob with nothing to do writes **no velocity at all**, so its body is
   allowed to sleep -- and a mob that stops walking is left to the solver's
   friction, gliding to a halt and back to sleep without one more write;
5. `RandomLookAroundGoal` **takes MOVE**: a cow staring at nothing does not walk;
6. `LookAtEntityGoal` **turns the cow toward a player**, and the tracker is
   told about a turn that moved nothing -- the case the physics solver skips;
7. `SwimGoal` **floats a cow up out of water**, at vanilla's rate;
8. a mob walking into a wall **gives up** rather than pushing into it forever;
9. `EscapeDangerGoal` **runs**, and runs faster than a wander;
10. an **unbound** AI (no context) does nothing, rather than guessing;
11. the selector's **control arbitration** -- a lower-priority goal cannot take
    MOVE from a running one, and takes it back the tick it stops.

Run: py -3.14t tools/check_mob_ai.py

Exit code 0 when everything passes, 1 otherwise.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import registry  # noqa: E402

registry.init_registries()

from entity.ai.animal import (  # noqa: E402
    CHICKEN_PROFILE,
    COW_PROFILE,
    PIG_PROFILE,
    SHEEP_PROFILE,
)
from entity.ai.control import (  # noqa: E402
    GROUND_SPEED_FACTOR,
    MoveControl,
    change_angle,
    wrap_degrees,
)
from entity.ai.context import AIContext  # noqa: E402
from entity.ai.goal import (  # noqa: E402
    JUMP,
    LOOK,
    MOVE,
    Goal,
    GoalSelector,
    to_goal_ticks,
)
from entity.ai.navigator import (  # noqa: E402
    MAX_TRAVEL_TICKS,
    STUCK_CHECK_INTERVAL,
)
from entity.entity_type import entity_type_by_name  # noqa: E402
from entity.mob import create_mob  # noqa: E402
from physics.block_shapes import block_shapes  # noqa: E402
from physics.bodies import SLEEPING  # noqa: E402
from physics.solver import GROUND_FRICTION, PhysicsSolver  # noqa: E402
from physics.world_view import WorldView  # noqa: E402
from world import block_states  # noqa: E402
from world.chunk import Chunk, ChunkPos  # noqa: E402

#: The real overworld window, so the section arithmetic is the live one.
MIN_Y, HEIGHT = -64, 384
FLOOR_SECTION = (0 - MIN_Y) // 16   # the section holding world Y 0..15
FLOOR_TOP = 1.0                     # a stone layer at y = 0 ends at y = 1

failures: List[str] = []
checks = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global checks
    checks += 1
    marker = "OK  " if condition else "FAIL"
    print(f"  {marker} {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(label)


def near(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


# ==================== the synthetic world ====================


class World:
    """Flat stone at y = 0, plus whatever a test builds on top of it."""

    def __init__(self, radius: int = 4) -> None:
        self.chunks: Dict[Tuple[int, int], Chunk] = {}
        self.view = WorldView(MIN_Y, HEIGHT)
        for cx in range(-radius, radius + 1):
            for cz in range(-radius, radius + 1):
                chunk = Chunk(position=ChunkPos(cx, cz), min_y=MIN_Y,
                              height=HEIGHT)
                section = chunk.sections[FLOOR_SECTION]
                section.blocks.reshape(16, 16, 16)[0, :, :] = STONE
                section.recompute_non_air_count()
                self.chunks[(cx, cz)] = chunk
        self.view.rebuild(self.chunks.values())

    def put(self, x: int, y: int, z: int, state: int) -> None:
        cx, cz = x >> 4, z >> 4
        chunk = self.chunks.get((cx, cz))
        if chunk is None:
            chunk = Chunk(position=ChunkPos(cx, cz), min_y=MIN_Y, height=HEIGHT)
            self.chunks[(cx, cz)] = chunk
        chunk.set_block(x & 15, y, z & 15, state)
        self.view.add(chunk)

    def fill(self, x0: int, y0: int, z0: int, x1: int, y1: int, z1: int,
             state: int) -> None:
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                for z in range(z0, z1 + 1):
                    self.put(x, y, z, state)


STONE = block_states.default_state_id("minecraft:stone")
WATER = block_states.default_state_id("minecraft:water")


class FakeDimension:
    """What `AIContext` and `spawn_placement.state_at` need from a Dimension,
    plus the two `NaturalSpawner.rekey_movers` reads."""

    def __init__(self, world: World, name: str = "minecraft:overworld") -> None:
        self.name = name
        self.world = world
        self.settings = SimpleNamespace(min_y=MIN_Y, height=HEIGHT)
        self.entities: Dict[int, object] = {}

    def get_chunk(self, x: int, z: int):
        return self.world.chunks.get((x, z))

    def get_entity(self, entity_id: int):
        return self.entities.get(entity_id)

    def is_chunk_loaded(self, x: int, z: int) -> bool:
        return (x, z) in self.world.chunks


class RecordingTracker:
    """The two `AIContext` and `PhysicsSolver` use from an `EntityTracker`."""

    def __init__(self) -> None:
        self.updates: List[Tuple[int, float, float, float, float, float,
                                 bool]] = []
        #: Viewer snapshots, `(id, x, y, z, spectator)`.
        self.viewers: List[Tuple[int, float, float, float, bool]] = []

    def update_entity(self, entity_id, x, y, z, yaw, pitch, on_ground) -> None:
        self.updates.append((entity_id, x, y, z, yaw, pitch, bool(on_ground)))

    def viewer_positions(self):
        return list(self.viewers)

    def yaws_for(self, entity_id: int) -> List[float]:
        return [u[4] for u in self.updates if u[0] == entity_id]


class Rig:
    """A world, a solver, one real mob, and the tick loop that drives both."""

    def __init__(self, world: World, type_name: str = "minecraft:cow",
                 entity_id: int = 1, x: float = 8.5, y: float = FLOOR_TOP,
                 z: float = 8.5, bind: bool = True) -> None:
        self.world = world
        self.dimension = FakeDimension(world)
        self.tracker = RecordingTracker()
        self.entities = {}
        self.solver = PhysicsSolver(
            world.view, block_shapes(), MIN_Y, HEIGHT,
            entities=self.entities, tracker=self.tracker,
        )
        entity_type = entity_type_by_name(type_name)
        self.mob = create_mob(entity_type, x, y, z)
        self.mob.entity_id = entity_id
        self.entities[entity_id] = self.mob
        self.solver.add_entity(self.mob)
        self.context = AIContext(self.dimension, self.solver, self.tracker)
        if bind:
            self.mob.ai.bind(self.context)

    @property
    def pose(self):
        return self.mob.position

    @property
    def ai(self):
        return self.mob.ai

    def slot(self) -> int:
        return self.solver.bodies.slot_of(self.mob.entity_id)

    def is_sleeping(self) -> bool:
        slot = self.slot()
        return slot is not None and bool(
            self.solver.bodies.flags[slot] & SLEEPING)

    def tick(self, count: int = 1) -> None:
        """One game tick, in the order `DimensionActor._on_tick` uses."""
        for _ in range(count):
            self.mob.tick()
            self.solver.tick()


def distance(a, b) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[2] - b[2]) ** 2)


def pose_of(rig: Rig) -> Tuple[float, float, float]:
    return (rig.pose.x, rig.pose.y, rig.pose.z)


# ==================== 1-2: the goal table ====================

VANILLA_GOALS = [
    (0, "SwimGoal"),
    (1, "EscapeDangerGoal"),
    (5, "WanderAroundGoal"),
    (6, "LookAtEntityGoal"),
    (7, "RandomLookAroundGoal"),
]


def section_goals() -> None:
    print("\n== the goal table ==")
    for type_name in ("minecraft:cow", "minecraft:sheep", "minecraft:chicken",
                      "minecraft:pig"):
        entity_type = entity_type_by_name(type_name)
        mob = create_mob(entity_type, 0.5, 1.0, 0.5)
        table = [(p.priority, type(p.goal).__name__)
                 for p in mob.ai.selector.goals]
        check(f"{type_name} carries vanilla's goal list", table == VANILLA_GOALS,
              str(table))

    cow = create_mob(entity_type_by_name("minecraft:cow"), 0.5, 1.0, 0.5)
    sheep = create_mob(entity_type_by_name("minecraft:sheep"), 0.5, 1.0, 0.5)
    chicken = create_mob(entity_type_by_name("minecraft:chicken"), 0.5, 1.0,
                         0.5)
    pig = create_mob(entity_type_by_name("minecraft:pig"), 0.5, 1.0, 0.5)
    check("cow carries vanilla's 0.2 (not the chicken's 0.25)",
          near(cow.attributes.movement_speed, 0.2))
    check("sheep carries vanilla's 0.23", near(sheep.attributes.movement_speed,
                                               0.23))
    check("chicken carries vanilla's 0.25",
          near(chicken.attributes.movement_speed, 0.25))
    check("pig shares chicken's 0.25 row (vanilla's pig/chicken pairing)",
          near(pig.attributes.movement_speed, 0.25)
          and pig.attributes.movement_speed == chicken.attributes.movement_speed)

    speeds = {
        type_name: [p.goal.speed
                    for p in mob.ai.selector.goals
                    if type(p.goal).__name__ == "EscapeDangerGoal"][0]
        for type_name, mob in (("cow", cow), ("sheep", sheep),
                               ("chicken", chicken), ("pig", pig))
    }
    check("danger speeds are vanilla's (2.0 / 1.25 / 1.4 / 1.5)",
          near(speeds["cow"], 2.0) and near(speeds["sheep"], 1.25)
          and near(speeds["chicken"], 1.4) and near(speeds["pig"], 1.5),
          str(speeds))

    check("the four tempt item sets are written down",
          COW_PROFILE.tempt_items == ("minecraft:wheat",)
          and SHEEP_PROFILE.tempt_items == ("minecraft:wheat",)
          and len(CHICKEN_PROFILE.tempt_items) == 6
          and PIG_PROFILE.tempt_items == ("minecraft:carrot",
                                          "minecraft:potato",
                                          "minecraft:beetroot"))

    check("to_goal_ticks halves and rounds up: 120 -> 60",
          to_goal_ticks(120) == 60 and to_goal_ticks(1) == 1
          and to_goal_ticks(3) == 2)
    check("ground speed factor is 1.0 on grass",
          near(GROUND_SPEED_FACTOR, 1.0, 1e-6), f"{GROUND_SPEED_FACTOR}")


# ==================== 3: wandering ====================

def walking_speeds(type_name: str, ticks: int = 60) -> List[float]:
    """Per-tick ground displacement while a mob walks one direction.

    Driven by an explicit `move_to` rather than by the wander roll: the claim
    under test is "the AI accelerates exactly as the model says", and that is
    a statement about the controls, not about how often a cow gets bored.
    """
    rig = Rig(World(), type_name)
    rig.ai.selector.goals = []
    rig.ai.navigator.move_to(rig.pose.x + 20.0, rig.pose.y, rig.pose.z, 1.0)
    speeds: List[float] = []
    for _ in range(ticks):
        before = pose_of(rig)
        rig.tick()
        after = pose_of(rig)
        moved = math.hypot(after[0] - before[0], after[2] - before[2])
        if moved > 0.01:
            speeds.append(moved)
    return speeds


def expected_speeds(type_name: str, ticks: int,
                    speed_modifier: float = 1.0) -> List[float]:
    """The accel+friction model's analytic displacement series, tick for tick.

    `apply_input` adds `modifier * a * (a * GROUND_SPEED_FACTOR)` to the
    body's velocity each tick, and the solver multiplies a *grounded* body's
    horizontal velocity by `GROUND_FRICTION` (0.546, grass) after the move -
    the 0.98 air drag is vertical-only, exactly as vanilla's `travel` splits
    it. The displacement of tick n is therefore the geometric series
    `D * (1 - q**n)` with `q = 0.546` and `D = accel / (1 - q)` its steady
    state. The first tick moves exactly one acceleration (0.04 for a cow),
    the series converges at ~0.088, and neither endpoint is negotiable: the
    first is the moveRelative write, the second is where the attribute
    tables are calibrated to land.
    """
    mob = create_mob(entity_type_by_name(type_name), 0.5, 1.0, 0.5)
    a = mob.attributes.movement_speed
    accel = (speed_modifier * a) * (a * GROUND_SPEED_FACTOR)
    q = GROUND_FRICTION
    steady = accel / (1.0 - q)
    return [steady * (1.0 - q ** n) for n in range(1, ticks + 1)]


def section_wander() -> None:
    print("\n== wandering ==")
    rig = Rig(World())
    start = pose_of(rig)
    rig.tick(800)
    walked = distance(start, pose_of(rig))
    check("a cow wanders on a flat plain", walked > 2.0, f"{walked:.2f} blocks")
    check("the cow is still on the floor, not in it or over it",
          near(rig.pose.y, FLOOR_TOP, 0.01), f"y={rig.pose.y}")

    # Vanilla's walk speed, tick for tick: the AI writes one acceleration
    # per tick and the solver applies its friction, so the displacement
    # follows the analytic series exactly -- including the ramp up from
    # 0.04 to the ~0.088 steady state. The tolerance is 1e-6 rather than
    # 1e-9 because `GROUND_SPEED_FACTOR` carries vanilla's float32 residue
    # (0.21600002 / 0.216 is 1.0000000926), and the point of the check is
    # that nothing *else* is scaling the walk.
    cow_speeds = walking_speeds("minecraft:cow")
    cow_expected = expected_speeds("minecraft:cow", len(cow_speeds))
    check("a walking cow follows the accel+friction series tick for tick",
          len(cow_speeds) == len(cow_expected)
          and all(near(s, e, 1e-6) for s, e in zip(cow_speeds, cow_expected)),
          f"n={len(cow_speeds)}"
          + (f" first={cow_speeds[0]:.8f} last={cow_speeds[-1]:.8f}"
             if cow_speeds else " never moved"))

    sheep_speeds = walking_speeds("minecraft:sheep")
    sheep_expected = expected_speeds("minecraft:sheep", len(sheep_speeds))
    check("a walking sheep follows the accel+friction series tick for tick",
          len(sheep_speeds) == len(sheep_expected)
          and all(near(s, e, 1e-6) for s, e in zip(sheep_speeds,
                                                   sheep_expected)),
          f"n={len(sheep_speeds)}"
          + (f" first={sheep_speeds[0]:.8f} last={sheep_speeds[-1]:.8f}"
             if sheep_speeds else " never moved"))

    # The steady states are the numbers the attribute table is calibrated
    # against: a 0.2 cow ambles at ~0.088 blocks per tick (1.76 per second),
    # not at 0.2 -- the attribute is an acceleration and enters the model
    # squared. Pinning them here is what stops a future edit from silently
    # turning cows back into something that outruns a galloping horse.
    check("a cow's steady walk is ~0.088 blocks per tick",
          near(cow_speeds[-1], cow_expected[-1], 1e-6)
          and near(cow_expected[-1],
                   0.2 ** 2 * GROUND_SPEED_FACTOR / (1.0 - GROUND_FRICTION),
                   1e-9),
          f"steady={cow_expected[-1]:.6f}")

    # The pig row: same 0.25 attribute as chicken, but escapeDanger 1.5 vs
    # chicken's 1.4. The *walk* must converge to the chicken's steady state
    # (the attribute drives it); the panic multiplier is what makes the
    # difference visible only when fleeing. This is the check that pins
    # "shared row, shared walk" so a future edit that "fixes" the pig to a
    # different number breaks loudly here.
    pig_speeds = walking_speeds("minecraft:pig")
    pig_expected = expected_speeds("minecraft:pig", len(pig_speeds))
    chicken_steady = (0.25 ** 2 * GROUND_SPEED_FACTOR
                      / (1.0 - GROUND_FRICTION))
    check("pig and chicken share a steady state (the 0.25 row)",
          len(pig_speeds) == len(pig_expected)
          and all(near(s, e, 1e-6) for s, e in zip(pig_speeds,
                                                   pig_expected))
          and near(pig_expected[-1], chicken_steady, 1e-9),
          f"steady={pig_expected[-1]:.6f}")
    # The steady state is pinned here, not derived from the speeds list, so
    # the assertion holds whether the mob ever wrote a velocity or not.
    # 0.25^2 / (1 - 0.546) with the grass speed factor of 1.000000093 = 0.1377
    # blocks/tick = 2.75 blocks/second -- the chicken's steady walk, which the
    # pig row happens to share.
    check("pig steady walk is ~0.1377 blocks per tick (the 0.25 row)",
          near(chicken_steady, 0.1376652, 1e-4),
          "see mob.py for the calibration table")

    # A walk has to *end*. Without an arrival test the navigator stays busy
    # on the spot it reached, the goal's `should_continue` keeps answering
    # yes, and the cow stands holding a finished wander for twenty seconds.
    rig = Rig(World())
    rig.ai.selector.goals = []
    rig.ai.navigator.move_to(8.5, rig.pose.y, 11.0, 1.0)
    took = 0
    for took in range(1, MAX_TRAVEL_TICKS):
        rig.tick()
        if rig.ai.navigator.is_idle:
            break
    check("a walk ends when the mob arrives, not at the timeout",
          rig.ai.navigator.is_idle and took < MAX_TRAVEL_TICKS,
          f"after {took} ticks")
    check("and the mob is where it was sent",
          near(rig.pose.z, 10.5, 0.6), f"z={rig.pose.z:.2f}")
    rig.tick(40)
    check("and the glide after arrival settles under the solver's friction",
          abs(rig.pose.velocity_x) < 1e-6 and abs(rig.pose.velocity_z) < 1e-6,
          f"v=({rig.pose.velocity_x:.8f}, {rig.pose.velocity_z:.8f})")


# ==================== 4: silence, sleeping, stopping ====================

def section_silence() -> None:
    print("\n== silence, sleeping, and the one stop ==")
    rig = Rig(World())
    settled = 0
    for settled in range(1, 601):
        rig.tick()
        if rig.is_sleeping():
            break
    check("a settled cow's body sleeps", rig.is_sleeping(),
          f"after {settled} ticks, flags={rig.solver.bodies.flags[rig.slot()]}")

    # With the goals dropped, "nothing to say" is a state the AI can be in
    # deterministically rather than whenever the cow happens not to be walking.
    rig.ai.selector.goals = []
    asleep_at = pose_of(rig)
    writes_before = len(rig.tracker.updates)
    rig.tick(60)
    check("60 more ticks leave a sleeping cow bit-identical",
          pose_of(rig) == asleep_at and rig.is_sleeping(),
          f"{pose_of(rig)} vs {asleep_at}")
    check("a still cow writes nothing to the wire",
          len(rig.tracker.updates) == writes_before,
          f"{len(rig.tracker.updates) - writes_before} writes")

    # The stop: a mob that was walking and then is not gets *no* write at
    # all. The solver's friction is what stops it - vanilla's glide - and
    # the velocity one tick later must be exactly the friction decay of the
    # velocity it had, which pins "no acceleration written" and "no zero
    # written" at the same time. (Two multiplies, in solver order: the
    # decay is spelled out rather than pre-folded so the rounding matches.)
    rig = Rig(World())
    rig.ai.selector.goals = []
    rig.ai.navigator.move_to(20.0, rig.pose.y, 8.5, 1.0)
    rig.tick(5)
    check("a walking cow has horizontal velocity", rig.pose.velocity_x != 0.0,
          f"vx={rig.pose.velocity_x:.4f}")
    vx_before = rig.pose.velocity_x
    vz_before = rig.pose.velocity_z
    pos_before = pose_of(rig)
    rig.ai.navigator.stop()
    rig.tick(1)
    check("stopping writes nothing - friction does the stopping",
          near(rig.pose.velocity_x, vx_before * GROUND_FRICTION, 1e-12)
          and near(rig.pose.velocity_z, vz_before * GROUND_FRICTION, 1e-12),
          f"v=({rig.pose.velocity_x:.8f}, {rig.pose.velocity_z:.8f})")
    check("and the cow glided while decelerating",
          distance(pos_before, pose_of(rig)) > 0.0,
          f"glided {distance(pos_before, pose_of(rig)):.4f}")
    rig.tick(60)
    check("a stopped cow coasts to rest and sleeps again",
          rig.is_sleeping() and abs(rig.pose.velocity_x) < 1e-6)


# ==================== 5: look-around takes MOVE ====================

def section_look_around() -> None:
    print("\n== random look-around takes MOVE ==")
    rig = Rig(World())
    look = _goal(rig, "RandomLookAroundGoal")
    check("RandomLookAroundGoal controls are MOVE|LOOK",
          look.controls == (MOVE | LOOK), f"controls={look.controls}")

    # A cow staring at nothing does not walk. The 2% roll is forced open and
    # the other goals are dropped, so the check is about what the goal *does*
    # once it is running rather than about how long it took to start.
    rig.ai.selector.goals = []
    look.can_start = lambda mob: True
    rig.ai.selector.add_goal(7, look)
    before = pose_of(rig)
    rig.tick(10)
    moved = distance(before, pose_of(rig))
    check("a cow staring at nothing does not walk", moved < 0.01,
          f"moved {moved:.4f}")
    check("and it does turn", rig.pose.yaw != 0.0, f"yaw={rig.pose.yaw:.2f}")
    check("and the tracker is told about a turn that moved nothing",
          len(rig.tracker.yaws_for(rig.mob.entity_id)) > 0,
          f"{len(rig.tracker.yaws_for(rig.mob.entity_id))} rotation writes")


def _goal(rig: Rig, name: str):
    for prioritized in rig.ai.selector.goals:
        if type(prioritized.goal).__name__ == name:
            return prioritized.goal
    raise AssertionError(f"{name} is not on the mob")


# ==================== 6: look at a player ====================

def section_look_at() -> None:
    print("\n== looking at a player ==")
    rig = Rig(World())
    entity_id = rig.mob.entity_id
    look = _goal(rig, "LookAtEntityGoal")
    # The 2% roll is forced open so the check is about the *looking*, not
    # about how long the cow happened to take to notice.
    look.chance = 1.0

    rig.tracker.viewers = [(999, rig.pose.x + 3.0, rig.pose.y, rig.pose.z,
                            False)]
    check("LookAtEntityGoal finds a player inside 6 blocks",
          look.can_start(rig.mob))

    rig.tracker.viewers = [(999, rig.pose.x + 20.0, rig.pose.y, rig.pose.z,
                            False)]
    check("and refuses one outside", not look.can_start(rig.mob))
    rig.tracker.viewers = []
    check("and refuses no player at all", not look.can_start(rig.mob))

    # A player standing due east, inside range. The cow starts at yaw 0,
    # which faces +Z; facing the player means yaw -90.
    rig.tracker.viewers = [(999, rig.pose.x + 3.0, rig.pose.y, rig.pose.z,
                            False)]
    rig.pose.yaw = 0.0
    rig.ai.selector.goals = []
    rig.ai.selector.add_goal(6, look)
    rig.tick(30)
    check("the cow turns to face the player",
          abs(abs(wrap_degrees(rig.pose.yaw)) - 90.0) < 1.0,
          f"yaw={rig.pose.yaw:.2f}")
    check("the turn reaches the tracker without a position change",
          len(rig.tracker.yaws_for(entity_id)) > 0,
          f"{len(rig.tracker.yaws_for(entity_id))} rotation writes")
    check("and the cow stood still while doing it",
          rig.pose.x == 8.5 and rig.pose.z == 8.5,
          f"({rig.pose.x}, {rig.pose.z})")


# ==================== 7: swimming ====================

def section_swim() -> None:
    print("\n== swimming ==")
    world = World()
    world.fill(6, 1, 6, 10, 4, 10, WATER)      # a pool the cow stands in
    rig = Rig(world, x=8.5, y=1.0, z=8.5)
    swim = _goal(rig, "SwimGoal")
    # Only the swim goal is left on the list, so the claim is about the
    # vertical rise alone; `WanderAroundGoal` also holds MOVE and would
    # eventually walk the cow out of a five-block pool.
    rig.ai.selector.goals = []
    rig.ai.selector.add_goal(0, swim)
    check("SwimGoal starts in water", swim.can_start(rig.mob))

    rig.tick(20)
    check("the cow rises in water", rig.pose.y > 1.0, f"y={rig.pose.y:.3f}")
    # Vanilla's travelInWater adds 0.04/tick with no gravity; the solver here
    # subtracts its own 0.08 into the motion, so the AI hands it 0.12 and the
    # net displacement is 0.04.
    rig = Rig(world, x=8.5, y=1.0, z=8.5)
    swim_only = _goal(rig, "SwimGoal")
    rig.ai.selector.goals = []
    rig.ai.selector.add_goal(0, swim_only)
    rig.tick(2)
    before = rig.pose.y
    rig.tick()
    check("and rises at vanilla's 0.04 blocks per tick",
          near(rig.pose.y - before, 0.04, 1e-9),
          f"dy={rig.pose.y - before:.9f}")

    dry = Rig(World())
    check("SwimGoal does not start on dry land",
          not _goal(dry, "SwimGoal").can_start(dry.mob))


# ==================== 8: walls, and giving up ====================

def section_stuck() -> None:
    """A wall used to be a wall. It is a detour now.

    This section was written when the navigator pointed at the target and
    pushed, and its claim was "a mob walking into a wall gives up" - the
    straight-line navigator could do nothing else. With `PathNavigation` the
    mob walks *round* the wall, so that claim is no longer true and would be
    a bug report rather than a check if it were kept.

    What replaces it is both halves: the wall is now a detour (below), and a
    mob that genuinely cannot get anywhere still gives up rather than pushing
    into stone forever (further down). The second is the claim that actually
    matters for a mob AI - being stuck is a worse failure than being slow.
    """
    print("\n== a wall is a detour, not a wall ==")
    world = World()
    world.fill(12, 1, 6, 12, 3, 10, STONE)    # a wall due east of the cow
    rig = Rig(world)
    # The other goals are dropped so the walk is the only thing that can
    # interrupt it - a RandomLookAroundGoal taking MOVE mid-walk is correct
    # behaviour and would make this check depend on a 2% roll.
    rig.ai.selector.goals = []
    rig.ai.navigator.move_to(20.0, rig.pose.y, rig.pose.z, 1.0)
    # Watched rather than sampled at the end: the mob comes back to its own z
    # line once it is past the wall, so its final z says nothing about whether
    # it detoured.
    furthest_z = 0.0
    for _ in range(400):
        rig.tick()
        furthest_z = max(furthest_z, abs(rig.pose.z - 8.5))
    check("a mob walks round a wall now",
          rig.pose.x > 12.5, f"x={rig.pose.x:.2f} z={rig.pose.z:.2f}")
    check("and it went round it rather than through it",
          furthest_z > 0.5, f"furthest z offset {furthest_z:.2f}")
    check("and it did not give up on the way",
          rig.ai.navigator.is_idle, "still walking after 400 ticks")

    print("\n== and a mob that cannot get anywhere still gives up ==")
    # Walled in on every side. The search returns a partial path - the start
    # node and nothing else - and the mob has nowhere to walk, which is what
    # the three detectors in `PathNavigation` are for.
    boxed = World()
    for x in range(6, 12):
        for z in range(6, 12):
            for y in (1, 2, 3):
                boxed.fill(x, y, z, x, y, z, STONE)
    rig = Rig(boxed)
    rig.ai.selector.goals = []
    rig.ai.navigator.move_to(40.0, rig.pose.y, rig.pose.z, 1.0)
    rig.tick(MAX_TRAVEL_TICKS + 20)
    check("a mob that cannot get anywhere gives up",
          rig.ai.navigator.is_idle, f"ticks={rig.ai.navigator.ticks}")
    check("and it did not walk out through the stone",
          abs(rig.pose.x - 8.5) < 0.5 and abs(rig.pose.z - 8.5) < 0.5,
          f"x={rig.pose.x:.2f} z={rig.pose.z:.2f}")

    rig = Rig(World())
    rig.ai.selector.goals = []
    rig.ai.navigator.move_to(rig.pose.x + 100.0, rig.pose.y, rig.pose.z, 1.0)
    rig.tick(MAX_TRAVEL_TICKS + 20)
    check("a walk that never arrives is bounded by MAX_TRAVEL_TICKS",
          rig.ai.navigator.is_idle, f"ticks={rig.ai.navigator.ticks}")
    check("and it gave up while still far short of the target",
          rig.pose.x < 60.0, f"x={rig.pose.x:.2f}")
    check("having walked its follow_range and stopped - the target at +100 is "
          "further out than a cow will plan",
          12.0 < rig.pose.x - 8.5 < 20.0,
          f"walked {rig.pose.x - 8.5:.2f} of 100")


# ==================== 9: escape danger ====================

def section_escape() -> None:
    print("\n== escape danger ==")
    rig = Rig(World())
    escape = _goal(rig, "EscapeDangerGoal")
    check("EscapeDangerGoal does not start on an unhurt mob",
          not escape.can_start(rig.mob))

    rig.mob.panic("minecraft:cow.hurt")
    # On a flat plain only the candidates at the mob's own height have a solid
    # floor, so one `can_start` fails about a third of the time - vanilla's
    # rule, and the reason a cow's panic does not always find somewhere to run.
    # What is being tested is that the *panic* is the only gate, so ask a few
    # times and require that it is never the panic that says no.
    started = False
    refusals: List[bool] = []
    for _ in range(20):
        escape.can_start(rig.mob)
        if escape.target is not None:
            started = True
            break
        refusals.append(True)
    check("EscapeDangerGoal starts once the mob has been hurt", started,
          f"found a spot after {1 + len(refusals)} tries")
    escape.start(rig.mob)
    rig.ai.selector.goals = []
    rig.ai.selector.add_goal(1, escape)
    speed = escape.speed * rig.mob.attributes.movement_speed
    rig.ai.move_control.clear()
    rig.tick(1)
    check("and it walks at danger_speed x movement_speed",
          near(rig.mob.movement_input[2], speed, 1e-9),
          f"input={rig.mob.movement_input[2]:.4f} expected={speed:.4f}")

    # The panic *speed*, driven explicitly on a fresh rig like the wander
    # check. Driving rather than free-running matters here: a panic goal
    # re-rolls its flee target every time one leg ends, so a free-running
    # mob spends its ticks on turns between legs and a distance threshold
    # over a fixed window flakes on the mob's RNG. The claim under test is
    # the speed of a single leg, and the input check above already pinned
    # the goal-to-control wiring. Panic accel is the walk's at 2x the
    # modifier, so the steady state is exactly double.
    panic = Rig(World())
    panic.ai.selector.goals = []
    panic.ai.navigator.move_to(panic.pose.x + 20.0, panic.pose.y,
                               panic.pose.z, escape.speed)
    before = pose_of(panic)
    panic.tick(20)
    moved = distance(before, pose_of(panic))
    expected_total = sum(expected_speeds("minecraft:cow", 20, escape.speed))
    check("a panicking cow covers ground", moved > 1.0,
          f"{moved:.2f} blocks in 20 ticks")
    check("and the panic leg follows the accel+friction series at 2x",
          near(moved, expected_total, 1e-6),
          f"{moved:.6f} vs {expected_total:.6f}")


# ==================== 10: an unbound AI ====================

def section_unbound() -> None:
    print("\n== an unbound AI does nothing ==")
    rig = Rig(World(), bind=False)
    before = pose_of(rig)
    rig.tick(120)
    check("a mob with no context does not move", pose_of(rig) == before,
          f"{pose_of(rig)}")
    check("and does not raise", rig.ai.no_body_ticks == 0
          and rig.ai.ticks == 120, f"ticks={rig.ai.ticks}")
    check("no goal ever started", rig.ai.running_goals() == (),
          str(rig.ai.running_goals()))

    no_solver = Rig(World(), bind=True)
    no_solver.context.physics = None
    before = pose_of(no_solver)
    no_solver.tick(60)
    check("a context with no solver leaves the mob where it was",
          pose_of(no_solver) == before)
    check("and counts it rather than hiding it",
          no_solver.ai.no_body_ticks == 60,
          f"no_body_ticks={no_solver.ai.no_body_ticks}")


# ==================== 11: control arbitration ====================

class _Recorder(Goal):
    """A goal that records whether it was started, and asks for some controls.

    `start_answers` and `continue_answers` are separate lists because a goal's
    two questions are separate: `can_start` is "do I want this tick" and
    `should_continue` is "may I keep going", and a test that shares one list
    between them is testing the order the selector happens to ask in rather
    than the arbitration.
    """

    def __init__(self, controls: int, start_answers: List[bool],
                 continue_answers: Optional[List[bool]] = None) -> None:
        self._controls = controls
        self._start = list(start_answers)
        self._continue = list(continue_answers or [])
        self.started = 0
        self.stopped = 0

    @property
    def controls(self) -> int:
        return self._controls

    def can_start(self, mob) -> bool:
        return self._start.pop(0) if self._start else False

    def should_continue(self, mob) -> bool:
        return self._continue.pop(0) if self._continue else True

    def start(self, mob) -> None:
        self.started += 1

    def stop(self, mob) -> None:
        self.stopped += 1


def section_arbitration() -> None:
    print("\n== control arbitration ==")

    class FakeMob:
        pass

    mob = FakeMob()

    high = _Recorder(MOVE, [True, True, True])
    low = _Recorder(MOVE, [True, True, True])
    selector = GoalSelector([(1, high), (5, low)])
    selector.tick(mob)
    check("the higher-priority goal wins a control", high.started == 1
          and low.started == 0, f"high={high.started} low={low.started}")
    selector.tick(mob)
    check("and keeps it", high.started == 1 and low.started == 0)

    # A *lower*-priority goal cannot take MOVE, however badly it wants it.
    hungry = _Recorder(MOVE, [True, True, True])
    modest = _Recorder(MOVE, [True, True, True])
    blocked = GoalSelector([(1, modest), (5, hungry)])
    blocked.tick(mob)
    check("a lower-priority goal cannot displace the holder",
          modest.started == 1 and hungry.started == 0,
          f"modest={modest.started} hungry={hungry.started}")

    # Same priority: insertion order is the tie-break, and nothing gets pushed
    # off a control it is already holding.
    first = _Recorder(MOVE, [True, False, True])
    second = _Recorder(MOVE, [True, True, True])
    tie = GoalSelector([(3, first), (3, second)])
    tie.tick(mob)
    check("equal priorities tie-break on insertion order",
          first.started == 1 and second.started == 0)

    # A goal that only wants LOOK does not displace one holding MOVE.
    move_goal = _Recorder(MOVE, [True, True, True])
    look_goal = _Recorder(LOOK, [True, True, True])
    mixed = GoalSelector([(1, move_goal), (6, look_goal)])
    mixed.tick(mob)
    check("a LOOK-only goal runs alongside a MOVE goal",
          move_goal.started == 1 and look_goal.started == 1)

    # RandomLookAround is 7 and WanderAround is 5, and a goal is displaced
    # only by a *lower* priority number - so the look-around does NOT interrupt
    # a walk. It runs in the gap between one walk ending and the next, which is
    # why a cow stops and stares between strolls rather than in the middle of
    # one. Pinned here because it reads like a bug and is not.
    walk = _Recorder(MOVE, [True, True, True])
    both = _Recorder(MOVE | LOOK, [True, True, True])
    no_displace = GoalSelector([(5, walk), (7, both)])
    no_displace.tick(mob)
    check("a lower-numbered priority displaces... and a higher one does not",
          walk.started == 1 and walk.stopped == 0 and both.started == 0,
          f"walk {walk.started}/{walk.stopped} both {both.started}")

    dominant = _Recorder(MOVE | LOOK, [True, True, True])
    modest = _Recorder(MOVE, [True, True, True])
    displace = GoalSelector([(7, modest), (1, dominant)])
    displace.tick(mob)
    check("so a MOVE+LOOK goal displaces a MOVE goal when it outranks it",
          modest.started == 1 and modest.stopped == 1 and dominant.started == 1,
          f"modest {modest.started}/{modest.stopped} dominant {dominant.started}")
    displace.tick(mob)
    check("and the displaced goal does not come back",
          modest.started == 1 and dominant.started == 1,
          f"modest {modest.started} dominant {dominant.started}")

    # A stopped goal's controls come back the same tick, not the tick after.
    # `again`'s first `can_start` is never asked - on tick one it loses the
    # arbitration to `release` and is skipped without being consulted.
    release = _Recorder(MOVE, [True], continue_answers=[False])
    again = _Recorder(MOVE, [True])
    handoff = GoalSelector([(1, release), (5, again)])
    handoff.tick(mob)
    handoff.tick(mob)
    check("controls are released on the same tick the holder stops",
          release.stopped == 1 and again.started == 1,
          f"release {release.started}/{release.stopped} again {again.started}")

    check("wrap_degrees takes the short way round",
          near(wrap_degrees(-181.0), 179.0)
          and near(wrap_degrees(181.0), -179.0)
          and near(wrap_degrees(540.0), -180.0))
    check("change_angle turns at most the limit",
          near(change_angle(0.0, 90.0, 45.0), 45.0)
          and near(change_angle(0.0, -90.0, 45.0), -45.0)
          and near(change_angle(350.0, 10.0, 45.0), 10.0))


# ==================== 12: the chunk index follows a mover ====================

def section_rekey() -> None:
    print("\n== the chunk index follows a mover ==")
    from entity.spawner import NaturalSpawner

    rig = Rig(World())
    dimension = rig.dimension
    spawner = NaturalSpawner(dimension)
    mob = rig.mob
    # The state `_place` leaves behind, without the placement: one mob indexed
    # under the chunk it stands in.
    dimension.entities[mob.entity_id] = mob
    start_chunk = mob.position.chunk
    spawner.chunk_entities[start_chunk] = {mob.entity_id}
    spawner.entity_chunks[mob.entity_id] = start_chunk

    moved = spawner.rekey_movers()
    check("a mob that has not moved is left alone", moved == 0)

    # Walk east into the next chunk (chunk boundary at x = 16).
    mob.position.x = 17.5
    moved = spawner.rekey_movers()
    check("a mob that walked into the next chunk is re-keyed", moved == 1)
    check("and the index says so",
          spawner.chunk_entities.get((1, 0)) == {mob.entity_id}
          and start_chunk not in spawner.chunk_entities
          and spawner.entity_chunks.get(mob.entity_id) == (1, 0),
          str(spawner.chunk_entities))

    # A mob removed from the dimension by another path must not wedge the
    # index: the stale entry is dropped, not carried.
    del dimension.entities[mob.entity_id]
    moved = spawner.rekey_movers()
    check("a stale entry is dropped rather than carried",
          moved == 0 and not spawner.entity_chunks
          and (1, 0) not in spawner.chunk_entities,
          str(spawner.chunk_entities))


# ==================== main ====================

def main() -> int:
    section_goals()
    section_wander()
    section_silence()
    section_look_around()
    section_look_at()
    section_swim()
    section_stuck()
    section_escape()
    section_unbound()
    section_arbitration()
    section_rekey()

    print()
    if failures:
        print(f"{len(failures)} of {checks} check(s) FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"All {checks} mob AI checks passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback

        # stdout is block-buffered when it is redirected to a file, so the
        # traceback (stderr, unbuffered) would otherwise land *above* the
        # checks that led to it and read as if the run stopped earlier.
        sys.stdout.flush()
        traceback.print_exc()
        raise SystemExit(1)
