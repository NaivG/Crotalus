"""Acceptance check for the physics solver: gravity, collision, sleeping, gaps.

The whole engine, on a synthetic world this file builds itself -- no server, no
client, no network, no worldgen. Every case is "given this world and this body,
tick it and assert where it ends up":

1. a fall lands **exactly** on the block surface, not a millimetre into it and
   not a millimetre above it;
2. a landed body sleeps, and 100 further ticks leave its position bit-identical;
3. a fall from 100 blocks never overlaps a solid block and never rises;
4. a full-block wall stops a walker, and the wall face is where it stops;
5. a half slab is stepped onto (step_height 0.6), a two-block wall is not;
6. a diagonal walk into a wall slides along it rather than stopping dead;
7. a fence post -- whose collision box leaves its own block cell and reaches
   y = 1.5 -- is landed on at 1.5, not at 1.0;
8. **the invariant**: after every tick of every scenario, the body's AABB
   overlaps no solid block by more than an epsilon. This is the generic version
   of every "does not clip into the world" claim above;
9. a body whose swept span reaches an unloaded chunk is skipped, counted, and
   keeps its position *and* its velocity;
10. a zero motion does not read the world at all (proved with an empty view:
    if the fast path did not short-circuit, the body would be deferred);
11. a body that falls past the world floor is removed through the caller's path;
12. the tracker is told about a body that moved and not about one that did not;
13. many bodies at once stay inside the world;
14. **the load level gates the whole thing**: a body in a chunk that is not
    entity-ticking (level 32 "block ticking" / 33 "border", `Chunk.mdx`) is
    paused -- no gravity, no drag, no sweep, no probe, no void removal -- while
    staying in the store and staying movable from outside physics, and it
    resumes where it was left when its chunk ticks again.

Run: python tools/check_physics_solver.py

Exit code 0 when everything passes, 1 otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from entity.position import EntityPosition  # noqa: E402
from physics.block_shapes import block_shapes  # noqa: E402
from physics.solver import GRAVITY, VOID_MARGIN, PhysicsSolver  # noqa: E402
from physics.world_view import WorldView  # noqa: E402
from world import block_states  # noqa: E402
from world.chunk import Chunk, ChunkPos  # noqa: E402

#: The real overworld window, so the section arithmetic is the live one.
MIN_Y, HEIGHT = -64, 384
FLOOR_SECTION = (0 - MIN_Y) // 16      # the section holding world Y 0..15
FLOOR_TOP = 1.0                        # a stone layer at y = 0 ends at y = 1

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


class FakeEntity:
    """The smallest thing `add_entity` needs: a position and a collision box."""

    def __init__(self, entity_id: int, width: float, height: float,
                 x: float, y: float, z: float, on_ground: bool = False) -> None:
        self.entity_id = entity_id
        self.entity_type = SimpleNamespace(width=width, height=height)
        self.position = EntityPosition(x=x, y=y, z=z, on_ground=on_ground)


class RecordingTracker:
    def __init__(self) -> None:
        self.calls: List[Tuple[int, float, float, float, bool]] = []

    def update_entity(self, entity_id, x, y, z, yaw, pitch, on_ground) -> None:
        self.calls.append((entity_id, x, y, z, bool(on_ground)))


class World:
    """Chunks plus the view over them, so a test can unload one."""

    def __init__(self, radius: int = 1) -> None:
        self.chunks: Dict[Tuple[int, int], Chunk] = {}
        self.view = WorldView(MIN_Y, HEIGHT)
        self.radius = radius
        for cx in range(-radius, radius + 1):
            for cz in range(-radius, radius + 1):
                self.chunks[(cx, cz)] = self._flat(cx, cz)
        self.sync()

    def _flat(self, cx: int, cz: int) -> Chunk:
        chunk = Chunk(position=ChunkPos(cx, cz), min_y=MIN_Y, height=HEIGHT)
        section = chunk.sections[FLOOR_SECTION]
        section.blocks.reshape(16, 16, 16)[0, :, :] = STONE
        section.recompute_non_air_count()
        return chunk

    def sync(self) -> None:
        self.view.rebuild(self.chunks.values())

    def put(self, x: int, y: int, z: int, state: int) -> None:
        cx, cz = x >> 4, z >> 4
        chunk = self.chunks.get((cx, cz))
        if chunk is None:
            chunk = Chunk(position=ChunkPos(cx, cz), min_y=MIN_Y, height=HEIGHT)
            self.chunks[(cx, cz)] = chunk
        chunk.set_block(x & 15, y, z & 15, state)
        self.view.add(chunk)

    def unload(self, cx: int, cz: int) -> None:
        self.chunks.pop((cx, cz), None)
        self.view.remove(cx, cz)

    def clear_column(self, x: int, z: int, y0: int, y1: int) -> None:
        for y in range(y0, y1 + 1):
            self.put(x, y, z, 0)


def solid_overlap(world: World, entity: FakeEntity) -> float:
    """The largest volume any solid block overlaps this body's box by.

    The generic invariant: whatever the engine did this tick, the body must not
    be inside the world. Sampling the block span rather than re-running the
    collision code means this cannot agree with a bug by construction.

    Reads the chunks directly rather than through the physics `WorldView`, so it
    cannot inherit the view's own idea of what is loaded -- an unloaded chunk
    here is air, and a body that has drifted into one shows up as a body that
    fell out of the world rather than as a passed check.
    """
    pose = entity.position
    half = entity.entity_type.width / 2.0
    bx0, by0, bz0 = pose.x - half, pose.y, pose.z - half
    bx1, by1, bz1 = pose.x + half, pose.y + entity.entity_type.height, pose.z + half
    worst = 0.0
    shapes = block_shapes()
    for block_x in range(int(np.floor(bx0)) - 1, int(np.floor(bx1)) + 2):
        for block_y in range(int(np.floor(by0)) - 1, int(np.floor(by1)) + 2):
            for block_z in range(int(np.floor(bz0)) - 1, int(np.floor(bz1)) + 2):
                chunk = world.chunks.get((block_x >> 4, block_z >> 4))
                state = 0 if chunk is None else int(
                    chunk.get_block(block_x & 15, block_y, block_z & 15))
                for box in shapes.boxes(state):
                    ox = min(bx1, block_x + box[3]) - max(bx0, block_x + box[0])
                    oy = min(by1, block_y + box[4]) - max(by0, block_y + box[1])
                    oz = min(bz1, block_z + box[5]) - max(bz0, block_z + box[2])
                    if ox > 0 and oy > 0 and oz > 0:
                        worst = max(worst, ox * oy * oz)
    return worst


# ==================== harness ====================


class Rig:
    """A world, a solver and one body, with the bookkeeping the tests share."""

    def __init__(self, world: World, width: float = 0.6, height: float = 1.8,
                 x: float = 0.5, y: float = 20.0, z: float = 0.5,
                 entity_id: int = 1, gate=None) -> None:
        self.world = world
        self.entity = FakeEntity(entity_id, width, height, x, y, z)
        self.entities = {entity_id: self.entity}
        self.tracker = RecordingTracker()
        self.removed: List[List[int]] = []
        self.solver = PhysicsSolver(
            world.view, block_shapes(), MIN_Y, HEIGHT,
            entities=self.entities, tracker=self.tracker,
            on_removed=self.removed.append,
            ticks_entities=gate)
        self.solver.warm_up()
        self.solver.add_entity(self.entity)
        self.worst_overlap = 0.0
        self.peak_y = y

    @property
    def pose(self) -> EntityPosition:
        return self.entity.position

    def set_motion(self, vx: float, vy: float, vz: float) -> None:
        """Set the body's velocity through the solver, as an AI or knockback would.

        Writing `EntityPosition.velocity_*` instead would look right and do
        nothing: that field is the solver's *output* for the wire, and the state
        it integrates is `BodyStore`. `set_velocity` also wakes the body, which
        is the whole reason it exists.
        """
        self.solver.set_velocity(self.entity.entity_id, vx, vy, vz)

    def step(self, ticks: int = 1, motion: Optional[Tuple[float, float, float]] = None,
             track_overlap: bool = True) -> None:
        """Tick the solver, optionally re-asserting a constant horizontal motion.

        Gravity is the solver's job -- tests never pass a vertical velocity --
        so `motion` is the horizontal velocity a walker would keep, reapplied
        every tick the way an AI goal would.
        """
        for _ in range(ticks):
            if motion is not None:
                self.set_motion(motion[0], 0.0, motion[2])
            self.solver.tick()
            self.peak_y = max(self.peak_y, self.pose.y)
            if track_overlap:
                self.worst_overlap = max(self.worst_overlap, solid_overlap(self.world, self.entity))


def main() -> int:
    from registry import init_registries

    init_registries(log_summary=False)

    global STONE, SLAB, STAIRS, FENCE
    STONE = block_states.default_state_id("minecraft:stone")
    SLAB = block_states.state_id("minecraft:oak_slab", {"type": "bottom"})
    STAIRS = block_states.state_id("minecraft:oak_stairs",
                                   {"facing": "east", "half": "bottom", "shape": "straight"})
    FENCE = block_states.state_id("minecraft:oak_fence",
                                  {"north": "false", "south": "false",
                                   "east": "false", "west": "false"})

    print("1. A fall lands exactly on the surface")
    world = World()
    rig = Rig(world, y=100.0)
    check("nothing overlaps anything at the start", solid_overlap(world, rig.entity) == 0.0)
    rig.step(200)
    check("it came to rest", rig.pose.on_ground, f"on_ground={rig.pose.on_ground}")
    check("at exactly the block surface", rig.pose.y == FLOOR_TOP, repr(rig.pose.y))
    check("and it never clipped into the world",
          rig.worst_overlap == 0.0, f"worst overlap {rig.worst_overlap}")
    check("the solver says it slept", bool(rig.solver.bodies.sleeping_mask()[0]))

    print("\n2. A sleeping body does not move, ever")
    before = (rig.pose.x, rig.pose.y, rig.pose.z)
    velocities = (rig.pose.velocity_x, rig.pose.velocity_y, rig.pose.velocity_z)
    rig.step(100)
    after = (rig.pose.x, rig.pose.y, rig.pose.z)
    check("100 ticks leave the position bit-identical", before == after,
          f"{before} -> {after}")
    check("and the velocity at exactly zero",
          (rig.pose.velocity_x, rig.pose.velocity_y, rig.pose.velocity_z) == (0.0, 0.0, 0.0))
    check("the start velocity was already zero", velocities == (0.0, 0.0, 0.0))
    check("the probe kept it asleep", bool(rig.solver.bodies.sleeping_mask()[0]))
    stats = rig.solver.stats()
    check("sleeping bodies are counted, not run as awake",
          stats["awake"] == 0 and stats["sleeping"] == 1,
          f"awake={stats['awake']} sleeping={stats['sleeping']}")
    check("nothing was deferred in a fully loaded world",
          stats["deferred_unloaded"] == 0)

    print("\n3. Removing the floor wakes it, and it starts falling")
    world.clear_column(0, 0, 0, 0)
    world.sync()
    rig.solver.tick()
    check("the probe woke it", bool(rig.solver.stats()["woken_by_probe"] >= 1),
          str(rig.solver.stats()["woken_by_probe"]))
    rig.step(20)
    check("it fell through the hole", rig.pose.y < FLOOR_TOP - 1.0, repr(rig.pose.y))
    check("and it never clipped into the world", rig.worst_overlap == 0.0)

    print("\n4. A terminal-velocity fall never passes through the ground")
    world = World()
    # A column with a floor at y = 0 and nothing between: the body reaches
    # terminal velocity (-3.92/tick) long before it arrives.
    deep = Rig(world, x=8.5, z=8.5, y=200.0)
    previous = deep.pose.y
    rose = False
    for _ in range(300):
        deep.solver.tick()
        if deep.pose.y > previous + 1e-12:
            rose = True
        previous = deep.pose.y
        if solid_overlap(world, deep.entity) > 0.0:
            break
    check("it never rose", not rose)
    check("it stopped exactly on the surface", deep.pose.y == FLOOR_TOP, repr(deep.pose.y))
    check("it is on the ground", deep.pose.on_ground)
    check("it never clipped into the world",
          solid_overlap(world, deep.entity) == 0.0)
    fastest = deep.solver.stats()
    check("the fall was long enough to reach terminal velocity", fastest["ticks"] >= 100,
          f"{fastest['ticks']} ticks")

    print("\n5. A full-block wall stops a walker at its face")
    world = World()
    for z in range(16):
        world.put(2, 1, z, STONE)
        world.put(2, 2, z, STONE)
    world.sync()
    walker = Rig(world, x=0.5, y=FLOOR_TOP, z=8.5)
    walker.step(60, motion=(0.3, 0.0, 0.0))
    check("it stopped", walker.pose.x < 2.0, repr(walker.pose.x))
    check("at the wall face (x = 1.7 with a 0.6 wide body)",
          near(walker.pose.x, 2.0 - 0.3, 1e-9), repr(walker.pose.x))
    check("it is still on the ground", walker.pose.on_ground)
    check("and it never clipped into the world", walker.worst_overlap == 0.0)

    print("\n6. A half slab is stepped onto, a two-block wall is not")
    world = World()
    for z in range(16):
        world.put(2, 1, z, SLAB)
    world.sync()
    climber = Rig(world, x=0.5, y=FLOOR_TOP, z=8.5)
    climber.step(60, motion=(0.3, 0.0, 0.0))
    # The slab is one block wide, so a walker crosses it and drops back to the
    # floor on the far side: what proves the step is the peak, not the end state.
    check("it stepped up onto the slab", near(climber.peak_y, 1.5, 1e-9), repr(climber.peak_y))
    check("it got past the slab", climber.pose.x > 2.0, repr(climber.pose.x))
    check("and it never clipped into the world", climber.worst_overlap == 0.0)

    world2 = World()
    for z in range(16):
        for y in (1, 2, 3):
            world2.put(2, y, z, STONE)
    world2.sync()
    blocked = Rig(world2, x=0.5, y=FLOOR_TOP, z=8.5)
    blocked.step(60, motion=(0.3, 0.0, 0.0))
    check("a three-block wall is not climbed", near(blocked.pose.y, FLOOR_TOP, 1e-9),
          repr(blocked.pose.y))
    check("and it is stopped at the face", near(blocked.pose.x, 1.7, 1e-9), repr(blocked.pose.x))
    check("it never clipped into the world", blocked.worst_overlap == 0.0)

    print("\n7. A diagonal walk into a wall slides along it")
    world = World()
    # The wall spans the whole loaded world, so the walker cannot slide past its
    # end and start moving in z again -- that would be correct behaviour and a
    # useless assertion.
    for x in range(-16, 32):
        world.put(x, 1, 2, STONE)
        world.put(x, 2, 2, STONE)
    world.sync()
    slider = Rig(world, x=8.5, y=FLOOR_TOP, z=0.5)
    slider.step(60, motion=(0.3, 0.0, 0.3))
    check("the wall stopped it in z", near(slider.pose.z, 2.0 - 0.3, 1e-9), repr(slider.pose.z))
    check("it kept sliding in x", slider.pose.x > 12.0, repr(slider.pose.x))
    check("and it never clipped into the world", slider.worst_overlap == 0.0)

    print("\n8. A fence post is landed on at its own height (1.5, above its cell)")
    world = World()
    world.put(8, 1, 8, FENCE)
    world.sync()
    # The fence's collision box is 0.375..0.625 in x and z, so the body has to
    # be centred on the post to land on it.
    faller = Rig(world, width=0.4, height=1.8, x=8.5, y=20.0, z=8.5)
    faller.step(200)
    check("it rests on top of the post at y = 2.5", near(faller.pose.y, 2.5, 1e-9),
          repr(faller.pose.y))
    check("and it never clipped into the world", faller.worst_overlap == 0.0)

    print("\n9. An unloaded chunk in the swept span defers the body")
    world = World(radius=0)
    world.unload(0, 0)
    world.sync()
    stranded = Rig(world, x=0.5, y=30.0, z=0.5)
    stranded.set_motion(0.0, -1.25, 0.0)
    before_pos = (stranded.pose.x, stranded.pose.y, stranded.pose.z)
    stranded.solver.tick()
    check("the position is unchanged",
          (stranded.pose.x, stranded.pose.y, stranded.pose.z) == before_pos)
    # The state the solver integrates is `BodyStore`; `EntityPosition` is the
    # wire copy and is deliberately not written for a body that did not move.
    check("the velocity is unchanged too",
          float(stranded.solver.bodies.vy[0]) == -1.25,
          repr(float(stranded.solver.bodies.vy[0])))
    check("and it was counted", stranded.solver.stats()["deferred_unloaded"] == 1,
          str(stranded.solver.stats()["deferred_unloaded"]))
    check("it is not asleep", not bool(stranded.solver.bodies.sleeping_mask()[0]))

    print("\n10. A zero motion does not read the world at all")
    # An awake body always has gravity in its motion, so the only way to reach
    # the kernel's zero-motion fast path is a velocity that cancels it exactly.
    # This is a unit test of that branch, not a scenario from the game.
    empty = World(radius=0)
    empty.chunks.clear()
    empty.sync()
    still = Rig(empty, x=0.5, y=30.0, z=0.5)
    still.set_motion(0.0, GRAVITY, 0.0)
    before = (still.pose.x, still.pose.y, still.pose.z)
    still.solver.tick()
    check("no deferral: the fast path never asked for a chunk",
          still.solver.stats()["deferred_unloaded"] == 0,
          str(still.solver.stats()["deferred_unloaded"]))
    check("nothing moved", still.pose.y == before[1], repr(still.pose.y))
    check("and it was not reported as moved", still.solver.stats()["moved"] == 0,
          str(still.solver.stats()["moved"]))

    print("\n11. Falling past the world floor removes the body")
    world = World(radius=0)
    world.clear_column(8, 8, 0, 0)
    world.sync()
    doomed = Rig(world, x=8.5, y=MIN_Y - 50, z=8.5)
    doomed.set_motion(0.0, -3.92, 0.0)
    doomed.step(40, track_overlap=False)
    check("the body was removed", not doomed.solver.has_body(1))
    check("the callback saw it", doomed.removed == [[1]], str(doomed.removed))
    check("and it is counted", doomed.solver.stats()["removed_void"] == 1,
          str(doomed.solver.stats()["removed_void"]))

    print("\n12. Only bodies that moved reach the tracker")
    world = World()
    talker = Rig(world, x=0.5, y=100.0, z=0.5)
    talker.solver.tick()
    check("a falling body produced a packet", len(talker.tracker.calls) == 1,
          str(len(talker.tracker.calls)))
    talker.step(200)
    landed = len(talker.tracker.calls)
    talker.step(50)
    check("a landed, sleeping body produces none", len(talker.tracker.calls) == landed,
          f"{landed} -> {len(talker.tracker.calls)}")
    check("the last packet said on_ground", talker.tracker.calls[-1][4])
    check("the packet carried the solver's own position",
          talker.tracker.calls[-1][2] == FLOOR_TOP, str(talker.tracker.calls[-1][2]))

    print("\n13. Many bodies at once stay inside the world")
    world = World(radius=2)
    entities = {}
    solver = PhysicsSolver(world.view, block_shapes(), MIN_Y, HEIGHT,
                           entities=entities)
    solver.warm_up()
    rng = np.random.default_rng(7)
    for index in range(200):
        x = float(rng.uniform(-16, 32))
        z = float(rng.uniform(-16, 32))
        entity = FakeEntity(1000 + index, 0.6, 1.8, x, 60.0 + rng.uniform(0, 20), z)
        entities[entity.entity_id] = entity
        solver.add_entity(entity)
    worst = 0.0
    for _ in range(200):
        solver.tick()
        for entity in entities.values():
            worst = max(worst, solid_overlap(world, entity))
    check("200 bodies land and stay out of the world", worst == 0.0, f"worst {worst}")
    landed = sum(1 for e in entities.values() if e.position.on_ground)
    check("all of them landed", landed == 200, f"{landed}/200")
    stats = solver.stats()
    check("most of them are asleep by the end", stats["sleeping"] > 150,
          f"sleeping={stats['sleeping']} awake={stats['awake']}")
    check("nothing overflowed a buffer", stats["overflow"] == 0, str(stats["overflow"]))
    check("nothing was deferred", stats["deferred_unloaded"] == 0)

    print("\n14. The load level decides whether a body is stepped at all")
    # A real ticket model, not a stub predicate: two player tickets on chunk
    # (0, 0), loading at radius 2 and simulating at radius 1. That puts chunk
    # (1, 0) at level 31 (entity ticking), (2, 0) at 32 (block ticking) and
    # (3, 0) at 33 (border) -- the three load types `Chunk.mdx` gives entities
    # different answers for, on one row, so the test walks west to east and the
    # only thing that changes is the level.
    from world.chunk_ticket import (FORCED, PLAYER_LOADING, PLAYER_SIMULATION,
                                    TicketManager)

    tickets = TicketManager("gate")
    tickets.add(PLAYER_LOADING, ChunkPos(0, 0), radius=2, key="p1")
    tickets.add(PLAYER_SIMULATION, ChunkPos(0, 0), radius=1, key="p1")

    def gate(chunk_x: int, chunk_z: int) -> bool:
        return tickets.ticks_entities(ChunkPos(chunk_x, chunk_z))

    # The residency model (level()) and the ticking model (simulation_level())
    # are deliberately different questions: chunks (2, 0) and (3, 0) are
    # *loaded* at levels 31/32 -- which is why the bodies in them are accessible
    # at all -- while their ticking level is 32/33, which is what pauses them.
    sim = tuple(tickets.simulation_level(ChunkPos(x, 0)) for x in (1, 2, 3, 4))
    loaded = tuple(tickets.level(ChunkPos(x, 0)) for x in (1, 2, 3, 4))
    check("the fixture lands on ticking levels 31/32/33/34",
          sim == (31, 32, 33, 34), f"{sim}")
    check("and those chunks stay loaded through the border level",
          loaded == (31, 31, 32, 33), f"{loaded}")
    check("only level 31 ticks entities (and only <= 32 ticks blocks)",
          gate(1, 0) and tickets.ticks_blocks(ChunkPos(2, 0))
          and not gate(2, 0) and not gate(3, 0) and not gate(4, 0)
          and not tickets.ticks_blocks(ChunkPos(3, 0)))

    world = World(radius=4)
    # One body per load type, all dropped from the same height on the same floor.
    ticking = Rig(world, x=24.5, y=40.0, z=8.5, gate=gate)      # chunk (1, 0)
    block = Rig(world, x=40.5, y=40.0, z=8.5, gate=gate)        # chunk (2, 0)
    border = Rig(world, x=56.5, y=40.0, z=8.5, gate=gate)       # chunk (3, 0)

    ticking.step(200)
    check("the entity-ticking body fell and landed", ticking.pose.y == FLOOR_TOP,
          repr(ticking.pose.y))

    before = [(rig.pose.x, rig.pose.y, rig.pose.z, rig.solver.bodies.vy[0])
              for rig in (block, border)]
    block.step(100)
    border.step(100)
    after = [(rig.pose.x, rig.pose.y, rig.pose.z, rig.solver.bodies.vy[0])
             for rig in (block, border)]
    check("the level-32 body did not move for 100 ticks", before[0] == after[0],
          f"{before[0]} -> {after[0]}")
    check("the level-33 body did not move for 100 ticks", before[1] == after[1],
          f"{before[1]} -> {after[1]}")
    check("no gravity reached them: velocity is still exactly zero",
          after[0][3] == 0.0 and after[1][3] == 0.0, str([a[3] for a in after]))
    for label, rig in (("level 32", block), ("level 33", border)):
        stats = rig.solver.stats()
        check(f"the {label} body is counted as paused, not awake or asleep",
              stats["paused_not_ticking"] == 1 and stats["awake"] == 0
              and stats["sleeping"] == 0 and stats["moved"] == 0,
              f"paused={stats['paused_not_ticking']} awake={stats['awake']} "
              f"sleeping={stats['sleeping']}")
        check(f"and the {label} body sent no packet",
              not rig.tracker.calls, str(rig.tracker.calls))
        check(f"and the {label} body is still in the store, readable",
              rig.solver.has_body(1)
              and rig.solver.bodies.position_of(1) == (rig.pose.x, rig.pose.y,
                                                       rig.pose.z))

    # "Accessible" is the other half of the rule: a paused body can be read and
    # moved from outside physics, and the move reaches the entity and the wire
    # in the same call -- there is no tick coming that would carry it there.
    check("an external teleport of a paused body is accepted",
          block.solver.set_position(1, 41.5, 60.0, 8.5, on_ground=False))
    check("it published the new pose onto the entity",
          (block.pose.x, block.pose.y, block.pose.z)
          == (41.5, 60.0, 8.5) and block.pose.on_ground is False,
          str(block.pose))
    check("and pushed it to the tracker",
          block.tracker.calls == [(1, 41.5, 60.0, 8.5, False)],
          str(block.tracker.calls))
    block.step(50)
    check("and the pause does not undo it", block.pose.y == 60.0
          and block.tracker.calls[-1][2] == 60.0, repr(block.pose.y))

    # Resume: the chunk ticks again, and the body continues from where the
    # external write left it.
    tickets.add(FORCED, ChunkPos(2, 0))
    check("the forced ticket makes level 32 an entity-ticking chunk again",
          gate(2, 0) and tickets.simulation_level(ChunkPos(2, 0)) == 31)
    block.step(3)
    check("the resumed body started falling", block.pose.y < 60.0,
          repr(block.pose.y))
    mid = (block.pose.y, float(block.solver.bodies.vy[0]))
    check("a resumed body carries momentum", mid[1] < 0.0, repr(mid[1]))

    # Suspend mid-fall, which is the case `Chunk.mdx` writes down in words: the
    # body keeps the velocity it had and picks it up again when the chunk ticks.
    tickets.remove(FORCED, ChunkPos(2, 0))
    block.step(20)
    check("suspended mid-fall it keeps its velocity bit for bit",
          (block.pose.y, float(block.solver.bodies.vy[0])) == mid,
          f"{mid} -> {(block.pose.y, float(block.solver.bodies.vy[0]))}")
    tickets.add(FORCED, ChunkPos(2, 0))
    block.step(400)
    check("and lands exactly on the floor once its chunk ticks again",
          block.pose.y == FLOOR_TOP and block.pose.on_ground,
          repr(block.pose.y))
    check("no body is paused on the final tick",
          block.solver.stats()["paused_not_ticking"] == 0,
          str(block.solver.stats()["paused_not_ticking"]))
    tickets.remove(FORCED, ChunkPos(2, 0))
    landed = (block.pose.x, block.pose.y, block.pose.z)
    block.step(10)
    check("a landed body pauses again when its chunk goes lazy",
          (block.pose.x, block.pose.y, block.pose.z) == landed
          and block.solver.stats()["paused_not_ticking"] == 1,
          f"{landed} -> {(block.pose.x, block.pose.y, block.pose.z)}")

    # The sleeping probe is part of "processed", so a sleeping body in a lazy
    # chunk is not asked whether its floor is still there -- until the chunk
    # ticks, at which point it is asked before anything else can move it.
    sleeper = Rig(world, x=24.5, y=40.0, z=8.5, gate=gate)
    sleeper.step(200)
    check("the sleeper landed and slept",
          sleeper.pose.y == FLOOR_TOP and bool(sleeper.solver.bodies.sleeping_mask()[0]))
    tickets.remove(PLAYER_SIMULATION, ChunkPos(0, 0), key="p1")
    check("dropping the simulation ticket makes its chunk lazy",
          not gate(1, 0) and tickets.level(ChunkPos(1, 0)) == 31)
    world.clear_column(24, 8, 0, 0)
    world.sync()
    sleeper.step(10)
    check("the floor is gone but a paused sleeper is not woken",
          sleeper.pose.y == FLOOR_TOP
          and bool(sleeper.solver.bodies.sleeping_mask()[0])
          and sleeper.solver.stats()["woken_by_probe"] == 0,
          f"y={sleeper.pose.y} woken={sleeper.solver.stats()['woken_by_probe']}")
    tickets.add(PLAYER_SIMULATION, ChunkPos(0, 0), radius=1, key="p1")
    sleeper.solver.tick()
    check("the first ticking tick probes it and wakes it",
          sleeper.solver.stats()["woken_by_probe"] == 1
          and not bool(sleeper.solver.bodies.sleeping_mask()[0]),
          str(sleeper.solver.stats()["woken_by_probe"]))
    sleeper.step(20)
    check("and then it falls through the hole",
          sleeper.pose.y < FLOOR_TOP - 1.0, repr(sleeper.pose.y))

    # Void removal deletes the entity, so it is not something a paused chunk may
    # do either: a body put below the world from outside stays until its chunk
    # runs again.
    lost = Rig(world, x=40.5, y=MIN_Y - VOID_MARGIN - 6, z=8.5, gate=gate)
    lost.set_motion(0.0, -3.92, 0.0)
    lost.step(5, track_overlap=False)
    check("a paused body below the world is not removed",
          lost.solver.has_body(1) and lost.removed == []
          and lost.solver.stats()["removed_void"] == 0,
          f"removed={lost.removed}")
    tickets.add(FORCED, ChunkPos(2, 0))
    lost.step(5, track_overlap=False)
    check("and is removed as soon as its chunk ticks",
          not lost.solver.has_body(1) and lost.removed == [[1]]
          and lost.solver.stats()["removed_void"] == 1,
          f"removed={lost.removed}")
    # The tick that dropped it is the case the gauges used to get wrong: `awake`
    # was read before the removal and `bodies` after it, so the three state
    # counters summed to one more than the store held. Removals now run first.
    gone_stats = lost.solver.stats()
    check("the counters still add up on the tick that removed it",
          gone_stats["bodies"] == gone_stats["awake"] + gone_stats["sleeping"]
          + gone_stats["paused_not_ticking"],
          f"bodies={gone_stats['bodies']} awake={gone_stats['awake']} "
          f"sleeping={gone_stats['sleeping']} "
          f"paused={gone_stats['paused_not_ticking']}")
    # Sharper, because a one-body solver summing to all zeros cannot tell the
    # old accounting apart from the new: two bodies, one of which dies while the
    # other is being stepped. The doomed body must appear in neither `awake` nor
    # `bodies`.
    doomed_entities: Dict[int, FakeEntity] = {}
    doomed = PhysicsSolver(world.view, block_shapes(), MIN_Y, HEIGHT,
                           entities=doomed_entities, ticks_entities=gate)
    doomed.warm_up()
    survivor = FakeEntity(1, 0.6, 1.8, 24.5, 40.0, 8.5)          # chunk (1, 0)
    falling = FakeEntity(2, 0.6, 1.8, 40.5, MIN_Y - VOID_MARGIN - 6, 8.5)
    doomed_entities[1] = survivor
    doomed_entities[2] = falling
    doomed.add_entity(survivor)
    doomed.add_entity(falling)
    doomed.tick()
    mixed_stats = doomed.stats()
    check("a removal tick counts the survivor and not the corpse",
          mixed_stats["bodies"] == 1 and mixed_stats["awake"] == 1
          and mixed_stats["removed_void"] == 1
          and mixed_stats["bodies"] == mixed_stats["awake"]
          + mixed_stats["sleeping"] + mixed_stats["paused_not_ticking"],
          f"bodies={mixed_stats['bodies']} awake={mixed_stats['awake']} "
          f"removed_void={mixed_stats['removed_void']}")

    # The gate is asked once per tick, from the position at the top of it: a
    # body that crosses the boundary during its own move keeps that move and is
    # paused from the next tick on. This is vanilla's `EntityTickList` lag, and
    # it is pinned here so nobody "fixes" it into a mid-tick clip.
    tickets.remove(FORCED, ChunkPos(2, 0))
    check("chunk (2, 0) is lazy again for the boundary case",
          gate(1, 0) and not gate(2, 0))
    crosser = Rig(world, x=31.5, y=FLOOR_TOP, z=8.5, gate=gate)
    crosser.set_motion(1.0, 0.0, 0.0)
    crosser.solver.tick()
    crossed = (crosser.pose.x, crosser.pose.y, crosser.pose.z)
    check("it walked out of the ticking chunk on its own tick",
          crossed[0] > 32.0 and int(crossed[0]) >> 4 == 2, repr(crossed))
    crosser.step(5)
    check("and is frozen from the next tick, at that position",
          (crosser.pose.x, crosser.pose.y, crosser.pose.z) == crossed
          and crosser.solver.stats()["paused_not_ticking"] == 1,
          f"{crossed} -> {(crosser.pose.x, crosser.pose.y, crosser.pose.z)}")

    # The counters have to add up, or a body can be in the store and in none of
    # the three states.
    mixed_entities: Dict[int, FakeEntity] = {}
    mixed = PhysicsSolver(world.view, block_shapes(), MIN_Y, HEIGHT,
                          entities=mixed_entities, ticks_entities=gate)
    mixed.warm_up()
    for index, x in enumerate((24.5, 40.5, 56.5), start=1):
        entity = FakeEntity(index, 0.6, 1.8, x, 40.0, 8.5)
        mixed_entities[index] = entity
        mixed.add_entity(entity)
    mixed.tick()
    stats = mixed.stats()
    check("one tick runs only the entity-ticking body",
          stats["awake"] == 1 and stats["paused_not_ticking"] == 2,
          f"awake={stats['awake']} paused={stats['paused_not_ticking']}")
    check("bodies == awake + sleeping + paused",
          stats["bodies"] == stats["awake"] + stats["sleeping"]
          + stats["paused_not_ticking"], str(stats))
    untouched = [(mixed_entities[index].position.y,
                  float(mixed.bodies.y[mixed.bodies.slot_of(index)]))
                 for index in (2, 3)]
    check("the other two are untouched bit for bit",
          untouched == [(40.0, 40.0), (40.0, 40.0)], str(untouched))

    # The chunk a body is in comes from `floor`, and the gate is keyed by the
    # packed negative-safe chunk key. Both are one-character mistakes away from
    # reading the neighbouring chunk's level, and the only place that shows up
    # is the boundary -- so the boundary is what this case stands on: chunk (0, 0)
    # ticks (simulation radius 0), chunk (-1, 0) is loaded and lazy, and the two
    # bodies are 0.5 blocks apart across x = 0.
    negative = TicketManager("negative")
    negative.add(PLAYER_LOADING, ChunkPos(0, 0), radius=2, key="p1")
    negative.add(PLAYER_SIMULATION, ChunkPos(0, 0), radius=0, key="p1")

    def negative_gate(chunk_x: int, chunk_z: int) -> bool:
        return negative.ticks_entities(ChunkPos(chunk_x, chunk_z))

    check("the fixture has the ticking chunk east of x = 0 and a lazy one west",
          negative_gate(0, 0) and not negative_gate(-1, 0))
    east = Rig(world, x=0.5, y=40.0, z=8.5, gate=negative_gate)
    west = Rig(world, x=-0.5, y=40.0, z=8.5, gate=negative_gate)
    east.step(60)
    west.step(60)
    check("x = +0.5 is in chunk 0 and fell", east.pose.y < 40.0, repr(east.pose.y))
    check("x = -0.5 is in chunk -1 and is paused",
          west.pose.y == 40.0
          and west.solver.stats()["paused_not_ticking"] == 1,
          f"y={west.pose.y} paused={west.solver.stats()['paused_not_ticking']}")

    print()
    if failures:
        print(f"{len(failures)} of {checks} check(s) FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"All {checks} physics solver checks passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        raise SystemExit(1)
