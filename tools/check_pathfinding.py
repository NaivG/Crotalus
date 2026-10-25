"""Pathfinding checks. `py -3.14t tools/check_pathfinding.py`

Five groups, in the order the code does its work: the cost model and block
classification, the two corner judgements (`is_diagonal_valid` and
`find_accepted_node`), the search itself, water, and then a real mob driven
through the real physics solver.

**Two worlds, two jobs.** The node-level questions - "is this diagonal
allowed", "does a four-block drop get a node" - run against `Ground`, a dict
of blocks with a stub `BlockFacts`. That is deliberate: the claim under test
is the rule, and a stub makes a wrong answer a failure of the rule rather
than of the 14 MB block payload. The navigation checks run against the same
`World`/`Rig` harness `check_mob_ai.py` uses, because a check that only
asserted on path nodes would pass with a search no mob could execute.

**The searches go through `PathFinder.create_path`, not through a copy of the
search.** An earlier draft re-implemented the A* loop here so it could drive a
stub mob; it drifted, and the drift was invisible because both copies were
individually correct. `search` is now a thin wrapper over the real entry
point, and its only extra knob is `max_nodes` - vanilla's own budget valve.

Every line of output is a claim. A failure prints what it saw.
"""

from __future__ import annotations

import sys
from pathlib import Path as FsPath
from typing import Dict, Optional, Tuple

sys.path.insert(0, str(FsPath(__file__).resolve().parents[1]))

from entity.ai.pathfinding.context import PathfindingContext  # noqa: E402
from entity.ai.pathfinding.evaluator import MobData  # noqa: E402
from entity.ai.pathfinding.finder import PathFinder  # noqa: E402
from entity.ai.pathfinding.node import MALUS, PathType  # noqa: E402
from entity.ai.pathfinding.path import Path  # noqa: E402
from world import block_states  # noqa: E402

MIN_Y = -64
HEIGHT = 384

PASSED = 0
FAILED: list = []


def check(claim: str, ok: bool, detail: str = "") -> None:
    global PASSED
    if ok:
        PASSED += 1
        print(f"  ok   {claim}")
    else:
        FAILED.append(claim)
        print(f"  FAIL {claim}   {detail}")


def section(title: str) -> None:
    print(f"\n== {title} ==")


# ==================== the world ====================

STONE = block_states.state_id("minecraft:stone")
DIRT = block_states.state_id("minecraft:dirt")
GRASS = block_states.state_id("minecraft:grass_block")
OAK_LOG = block_states.state_id("minecraft:oak_log")
OAK_FENCE = block_states.state_id("minecraft:oak_fence")
OAK_DOOR = block_states.state_id("minecraft:oak_door")
WATER = block_states.state_id("minecraft:water")
LAVA = block_states.state_id("minecraft:lava")
AIR = block_states.state_id("minecraft:air")


_NAMES = {
    AIR: "minecraft:air",
    STONE: "minecraft:stone",
    DIRT: "minecraft:dirt",
    GRASS: "minecraft:grass_block",
    OAK_LOG: "minecraft:oak_log",
    OAK_FENCE: "minecraft:oak_fence",
    OAK_DOOR: "minecraft:oak_door",
    WATER: "minecraft:water",
    LAVA: "minecraft:lava",
}

_TAGS = {
    OAK_DOOR: frozenset(("minecraft:doors",)),
    OAK_FENCE: frozenset(("minecraft:fences",)),
}

_FLUID = {WATER: "minecraft:water", LAVA: "minecraft:lava"}

_SOLID = frozenset((STONE, DIRT, GRASS, OAK_LOG, OAK_FENCE, OAK_DOOR))
_FULL_CUBE = frozenset((STONE, DIRT, GRASS, OAK_LOG))

#: Block-local collision boxes, `(min, max)` pairs. A full cube is `(0,0,0)`
#: to `(1,1,1)`; a fence is a thin post.
_BOXES = {
    STONE: ((0.0, 0.0, 0.0, 1.0, 1.0, 1.0),),
    DIRT: ((0.0, 0.0, 0.0, 1.0, 1.0, 1.0),),
    GRASS: ((0.0, 0.0, 0.0, 1.0, 1.0, 1.0),),
    OAK_LOG: ((0.0, 0.0, 0.0, 1.0, 1.0, 1.0),),
    OAK_FENCE: ((0.375, 0.0, 0.375, 0.625, 1.0, 0.625),),
    OAK_DOOR: ((0.0, 0.0, 0.0, 1.0, 1.0, 1.0),),
    WATER: (),
}


class StubFacts:
    """A `BlockFacts`-shaped object over a dict of positions."""

    def __init__(self, states: Dict[Tuple[int, int, int], int]) -> None:
        self.states = states
        #: `BlockFacts.names` is a list indexed by state id; a dict keyed by
        #: state id answers the same question for the handful of blocks these
        #: checks use.
        self.names = _NAMES

    def is_air(self, state_id: int) -> bool:
        return state_id == AIR

    def is_solid(self, state_id: int) -> bool:
        return state_id in _SOLID

    def is_full_cube(self, state_id: int) -> bool:
        return state_id in _FULL_CUBE

    def fluid_of(self, state_id: int) -> str:
        return _FLUID.get(state_id, "minecraft:empty")

    def in_tag(self, state_id: int, tag: str) -> bool:
        return tag in _TAGS.get(state_id, frozenset())

    def collision_shape_ids(self, state_id: int):
        return tuple(range(len(_BOXES.get(state_id, ()))))

    def collision_shapes(self, state_id: int):
        return _BOXES.get(state_id, ())

    @property
    def name_table(self):
        return _NAMES


class StubContext:
    """Enough of `AIContext` for `PathfindingContext`: `state` and `facts`.

    It carries the real context's per-tick search budget interface too, so
    `_search` can talk to either one. The default (`path_nodes_per_tick=0`)
    answers "budget off", which is what every check in this file wants --
    they are claims about the search, not about the valve -- except the
    per-tick budget section, which passes a number and exercises the valve
    exactly as the dimension actor's mob herd would.
    """

    def __init__(self, states: Dict[Tuple[int, int, int], int],
                 min_y: int = MIN_Y, path_nodes_per_tick: int = 0) -> None:
        self.states = states
        self.dimension = object()
        self.min_y = min_y
        self.max_y = min_y + HEIGHT
        self._facts = StubFacts(states)
        self.path_nodes_per_tick = max(0, int(path_nodes_per_tick))
        self._path_nodes_left = self.path_nodes_per_tick

    def state(self, x: int, y: int, z: int) -> Optional[int]:
        return self.states.get((x, y, z))

    @property
    def facts(self):
        return self._facts

    # ----- the budget interface, mirroring `AIContext` -----

    def begin_tick(self) -> None:
        self._path_nodes_left = self.path_nodes_per_tick

    def path_nodes_left(self) -> int:
        if self.path_nodes_per_tick <= 0:
            return -1
        return self._path_nodes_left

    def spend_path_nodes(self, used: int) -> None:
        if self.path_nodes_per_tick > 0:
            self._path_nodes_left -= used


class Ground:
    """Flat stone at y=0, so the mob walks on top at y=1.

    Air fills `-6..5` and stone sits at y=0, so a pit is made by *removing*
    the stone rather than by stacking a negative-height world. Anything
    outside the filled box reads as unloaded, which is what the pathfinder
    calls `BLOCKED` - deliberately the same as the edge of a loaded chunk.
    """

    def __init__(self, radius: int = 16, floor: int = STONE,
                 y_lo: int = -6, y_hi: int = 5) -> None:
        self.states: Dict[Tuple[int, int, int], int] = {}
        for x in range(-radius, radius + 1):
            for z in range(-radius, radius + 1):
                for y in range(y_lo, y_hi + 1):
                    self.states[(x, y, z)] = AIR
                self.states[(x, 0, z)] = floor

    def put(self, x: int, y: int, z: int, state_id: int) -> None:
        self.states[(x, y, z)] = state_id

    def carve(self, x: int, y: int, z: int) -> None:
        self.states[(x, y, z)] = AIR

    def pit(self, x: int, z0: int, z1: int, floor_y: int) -> None:
        """A shaft down column `x`, from the surface to a floor at `floor_y`.

        The floor is the block *at* `floor_y`, so a mob standing at y=1 drops
        onto the node at `floor_y + 1` - that is, `1 - floor_y` blocks.
        """
        for z in range(z0, z1 + 1):
            for y in range(floor_y + 1, 1):
                self.carve(x, y, z)
            self.put(x, floor_y, z, STONE)

    def wall(self, x: int, z: int, y0: int = 1, y1: int = 3,
             state_id: int = STONE) -> None:
        for y in range(y0, y1 + 1):
            self.put(x, y, z, state_id)

    def wall_x(self, x: int, z0: int, z1: int, y0: int = 1, y1: int = 3) -> None:
        """A wall across the whole world at one x, closing it off entirely."""
        for z in range(z0, z1 + 1):
            self.wall(x, z, y0, y1)

    def wall_z(self, z: int, x0: int, x1: int, y0: int = 1, y1: int = 3) -> None:
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                self.put(x, y, z, STONE)

    def ai(self, path_nodes_per_tick: int = 0) -> StubContext:
        return StubContext(self.states, path_nodes_per_tick=path_nodes_per_tick)


def make_finder(ground: Ground, position=(0.5, 1.0, 0.5), width=0.6,
                height=1.95, on_ground=True):
    """A prepared evaluator over a synthetic world.

    Used by the checks that ask the evaluator a *specific* question - "is this
    diagonal allowed", "does this drop get a node" - where going through a
    whole search would only obscure the answer. The search itself is driven
    through `search` below, against the real entry point.
    """
    ai = ground.ai()
    finder = PathFinder()
    mob_data = MobData(position, width, height, 1.0)
    mob_data.on_ground = on_ground
    mob_data.min_y = ai.min_y
    context = PathfindingContext(ai, mob_data.block_position(), min_y=ai.min_y)
    finder.evaluator.prepare(context, mob_data)
    return finder, mob_data, context


class FakeMob:
    """Just enough mob for `PathFinder.create_path`.

    The point of going through the real entry point rather than poking at the
    evaluator directly is that the checks below are then claims about the
    search that ships, not about a copy of it in this file. A re-implementation
    here would drift from `finder.py` and the drift would be invisible: both
    would be right about the thing they were each written to say.
    """

    class _Position:
        __slots__ = ("x", "y", "z", "on_ground")

        def __init__(self, position, on_ground):
            self.x, self.y, self.z = position
            self.on_ground = on_ground

    class _Attributes:
        __slots__ = ("follow_range",)

        def __init__(self, follow_range):
            self.follow_range = follow_range

    class _EntityType:
        __slots__ = ("width", "height")

        def __init__(self, width, height):
            self.width = width
            self.height = height

    class _AI:
        __slots__ = ("context",)

        def __init__(self, context):
            self.context = context

    def __init__(self, ground: Ground, position=(0.5, 1.0, 0.5), width=0.6,
                 height=1.95, on_ground=True, follow_range=16.0):
        self.position = self._Position(position, on_ground)
        self.attributes = self._Attributes(follow_range)
        self.entity_type = self._EntityType(width, height)
        self.ai = self._AI(ground.ai())


def search(mob: FakeMob, dest: Tuple[float, float, float],
           reach_range: int = 1,
           max_nodes: float = 1.0) -> Optional[Path]:
    """Run the real search, once, on a mob in a synthetic world.

    `follow_range` is a property of the mob (16 for a cow) and it is a *hard*
    radius: a path whose walked length exceeds it is cut off mid-search and
    comes back partial. Test worlds here are sized to fit inside it, which is
    not a convenience - a world scaled past it produces a partial path and no
    bug. Pass a mob with a bigger `follow_range` when the check is about the
    cost model rather than about what a cow will actually plan.
    """
    return _search_with(PathFinder(), mob, dest, reach_range, max_nodes)


def _search_with(finder: PathFinder, mob: FakeMob,
                 dest: Tuple[float, float, float],
                 reach_range: int = 1,
                 max_nodes: float = 1.0) -> Optional[Path]:
    """The same, on a finder that has already been configured.

    `max_nodes` is vanilla's `max_visited_nodes_multiplier`, the valve
    `WanderAroundGoal` turns down to 0.5. It buys node budget, not range: a
    check that has to search a wide area buys it here rather than shrinking
    the world, because the width is part of what is under test.
    """
    finder.max_visited_nodes_multiplier = max_nodes
    return finder.create_path(mob, dest, reach_range)


def path_positions(path: Path):
    return [node.pos for node in path.nodes]


# ==================== 1: the malus table ====================

def section_malus() -> None:
    section("the cost model is one number per block")
    check("27 types, and the table is that long",
          len(MALUS) == 27, f"{len(MALUS)}")
    check("lava is forbidden, not merely expensive",
          MALUS[PathType.LAVA] == -1.0, f"{MALUS[PathType.LAVA]}")
    check("leaves are forbidden (vanilla behaviour, not a gap)",
          MALUS[PathType.LEAVES] == -1.0, f"{MALUS[PathType.LEAVES]}")
    check("water is expensive, so a mob may wade but prefers to detour",
          MALUS[PathType.WATER] == 8.0, f"{MALUS[PathType.WATER]}")
    check("open ground is free", MALUS[PathType.OPEN] == 0.0)
    check("a closed wooden door is forbidden with no mob able to open it",
          MALUS[PathType.DOOR_WOOD_CLOSED] == -1.0)


# ==================== 2: classification ====================

def section_classification() -> None:
    section("block -> PathType")
    ground = Ground()
    ground.put(1, 1, 1, STONE)
    ground.put(2, 1, 1, WATER)
    ground.put(3, 1, 1, OAK_FENCE)
    ground.put(4, 1, 1, OAK_DOOR)
    _, _, ctx = make_finder(ground)

    check("air is open", ctx.compute_path_type_from_state((1, 3, 1)) == PathType.OPEN)
    check("stone is blocked - it is what you stand on, not walk in",
          ctx.compute_path_type_from_state((1, 1, 1)) == PathType.BLOCKED,
          str(ctx.compute_path_type_from_state((1, 1, 1))))
    check("water is water", ctx.compute_path_type_from_state((2, 1, 1)) == PathType.WATER)
    check("a fence is a fence", ctx.compute_path_type_from_state((3, 1, 1)) == PathType.FENCE)
    check("a closed oak door is a closed wooden door",
          ctx.compute_path_type_from_state((4, 1, 1)) == PathType.DOOR_WOOD_CLOSED,
          str(ctx.compute_path_type_from_state((4, 1, 1))))

    # An unloaded coordinate. This is the check the whole context file exists
    # for: it must not read as air, or a cow paths into space it cannot see.
    partial = StubContext({(0, 0, 0): STONE})
    partial_ctx = PathfindingContext(partial, (0, 1, 0), min_y=MIN_Y)
    check("an unloaded block is BLOCKED, not air",
          partial_ctx.compute_path_type_from_state((50, 1, 50)) == PathType.BLOCKED,
          str(partial_ctx.compute_path_type_from_state((50, 1, 50))))
    check("and it is not pathfindable either",
          not partial_ctx.is_pathfindable((50, 1, 50)))


def section_land_type() -> None:
    section("a column, not a block")
    ground = Ground()
    ground.put(5, 0, 5, LAVA)
    _, _, ctx = make_finder(ground)
    # Vanilla's rule (`pathfinding_context.rs:191`) puts Open, Water, Lava and
    # Walkable in one arm together, so open air over lava is `OPEN` - not
    # "standing in lava". A mob walking over a lava pool does not fall in; it
    # walks on the air above it, which is only *dangerous* because the
    # neighbour scan then finds the lava and marks the cell DANGER_FIRE. The
    # distinction matters: the first answer decides whether there is a floor,
    # the second decides what it costs.
    check("open air over lava is OPEN - there is no floor to fall onto",
          ctx.land_node_type((5, 1, 5)) == PathType.OPEN,
          str(ctx.land_node_type((5, 1, 5))))
    check("open air over stone, away from the lava, is WALKABLE - this is "
          "the ground",
          ctx.land_node_type((10, 1, 10)) == PathType.WALKABLE,
          str(ctx.land_node_type((10, 1, 10))))
    check("open air over more open air is still open (a hole, not a floor)",
          ctx.land_node_type((6, 3, 6)) == PathType.OPEN,
          str(ctx.land_node_type((6, 3, 6))))
    check("the block itself is still LAVA whatever is above it",
          ctx.path_type((5, 0, 5)) == PathType.LAVA)


def section_danger_propagation() -> None:
    section("danger spreads")
    ground = Ground()
    ground.put(10, 0, 10, LAVA)
    _, _, ctx = make_finder(ground)
    # (9, 1, 10) is walkable ground with lava at (10, 0, 10) - below and to
    # the right. Vanilla's scan is 26-deep, so it finds it.
    resolved = ctx.land_node_type((9, 1, 10))
    check("ground beside lava is DANGER_FIRE, not plain WALKABLE",
          resolved == PathType.DANGER_FIRE, str(resolved))
    check("and DANGER_FIRE costs 8, so a path can still choose it",
          MALUS[resolved] == 8.0, f"{MALUS[resolved]}")
    check("ground far from lava stays WALKABLE",
          ctx.land_node_type((2, 1, 2)) == PathType.WALKABLE,
          str(ctx.land_node_type((2, 1, 2))))
    # The cell directly above the lava is *not* blocked - vanilla treats it as
    # open, and it is the danger ring around it that makes crossing cost
    # something. A mob walks over a one-block lava ravine; it does not fall in.
    check("and the cell over the lava is open, priced by its neighbours",
          ctx.land_node_type((10, 1, 10)) == PathType.OPEN,
          str(ctx.land_node_type((10, 1, 10))))
    check("while the lava cell itself is forbidden outright",
          ctx.path_type((10, 0, 10)) == PathType.LAVA
          and MALUS[PathType.LAVA] == -1.0)


# ==================== 3: the two corner judgements ====================

def section_diagonal() -> None:
    section("diagonals and corners")
    ground = Ground()
    ground.put(0, 1, -1, STONE)   # north neighbour of (0, 1, 0) - a wall
    _, _, ctx = make_finder(ground)
    from entity.ai.pathfinding.node import Node
    finder, mob_data, _ = make_finder(ground)
    evaluator = finder.evaluator

    current = evaluator.get_start_node((0, 1, 0))
    north = evaluator._get_node((0, 1, -1))
    north.path_type = PathType.BLOCKED
    north.cost_malus = -1.0
    east = evaluator.get_start_node((1, 1, 0))

    check("a diagonal with one blocked corner is refused",
          not evaluator.is_diagonal_valid(current, north, east))
    check("a diagonal with two clear corners is allowed",
          evaluator.is_diagonal_valid(current, east, east))

    # The fence-pair exception: two fences, mob narrower than half a block.
    chicken, _, _ = make_finder(ground, width=0.4)
    c_eval = chicken.evaluator
    c_current = c_eval.get_start_node((0, 1, 0))
    fence_a = c_eval._get_node((1, 1, 0))
    fence_a.path_type = PathType.FENCE
    fence_a.cost_malus = MALUS[PathType.FENCE]
    fence_b = c_eval._get_node((0, 1, -1))
    fence_b.path_type = PathType.FENCE
    fence_b.cost_malus = MALUS[PathType.FENCE]
    check("two fences pass a mob narrower than half a block",
          c_eval.is_diagonal_valid(c_current, fence_a, fence_b))

    cow, _, _ = make_finder(ground, width=0.6)
    w_eval = cow.evaluator
    w_current = w_eval.get_start_node((0, 1, 0))
    check("and do not pass a cow",
          not w_eval.is_diagonal_valid(w_current, fence_a, fence_b))

    # A door in either corner kills the diagonal outright, even for a mob that
    # could open it - the villager case.
    villager, _, _ = make_finder(ground, width=0.6)
    v_eval = villager.evaluator
    v_current = v_eval.get_start_node((0, 1, 0))
    door = v_eval._get_node((1, 1, 0))
    door.path_type = PathType.WALKABLE_DOOR
    door.cost_malus = 0.0
    check("a door in a corner refuses the diagonal, openable or not",
          not v_eval.is_diagonal_valid(v_current, door, door))


def section_falls_and_steps() -> None:
    section("falls, steps and jumps")
    # The mob stands at y=1 on the y=0 floor. A pit whose floor is the block at
    # y=F presents a drop of (1 - F) blocks, and the landing node is at y=F.
    ground = Ground(radius=12)
    ground.pit(4, -4, 4, floor_y=-4)   # floor block at y=-4 -> a 4-block drop
    ground.pit(8, -4, 4, floor_y=-3)   # floor block at y=-3 -> a 3-block drop

    finder, mob_data, _ = make_finder(ground, position=(3.5, 1.0, 0.5))
    evaluator = finder.evaluator
    current = evaluator.get_start_node((3, 1, 0))
    pos_height = evaluator.get_floor_level(current.pos)

    # Four is one too many. The answer is a *blocked* node rather than None,
    # so the search records "there is a hole here" once instead of
    # re-discovering it from every neighbour.
    four = evaluator.find_accepted_node(4, 1, 0, 1, pos_height, (0, 0),
                                        PathType.WALKABLE)
    check("a four-block drop is refused", four is not None
          and four.cost_malus < 0.0,
          f"{None if four is None else (four.pos, four.cost_malus)}")

    three = evaluator.find_accepted_node(8, 1, 0, 1, pos_height, (0, 0),
                                         PathType.WALKABLE)
    check("a three-block drop is survivable and sinks to the floor",
          three is not None and three.pos == (8, -2, 0),
          f"{None if three is None else three.pos}")

    # The step up. The node asked about is the *raised block*, so the answer
    # comes back one above it - asking about the block in front of the step
    # would only ever test flat ground.
    step = Ground(radius=6)
    for z in range(-3, 4):
        step.put(2, 1, z, STONE)
    finder2, _, _ = make_finder(step, position=(0.5, 1.0, 0.5))
    ev2 = finder2.evaluator
    cur2 = ev2.get_start_node((0, 1, 0))
    up = ev2.find_accepted_node(2, 1, 0, 1, ev2.get_floor_level(cur2.pos), (0, 0),
                                PathType.WALKABLE)
    check("a one-block step up is climbed",
          up is not None and up.pos == (2, 2, 0),
          f"{None if up is None else up.pos}")

    # The same step with the jump budget already spent: the search may not
    # plan a climb it cannot pay for.
    no_jump = ev2.find_accepted_node(2, 1, 0, 0, ev2.get_floor_level(cur2.pos),
                                     (0, 0), PathType.WALKABLE)
    check("and not with the jump budget already spent",
          no_jump is None or no_jump.cost_malus < 0.0,
          f"{None if no_jump is None else (no_jump.pos, no_jump.cost_malus)}")

    # A two-block wall is not climbed even with the budget.
    wall = Ground(radius=6)
    for y in (1, 2, 3):
        for z in range(-3, 4):
            wall.put(2, y, z, STONE)
    finder3, _, _ = make_finder(wall, position=(0.5, 1.0, 0.5))
    ev3 = finder3.evaluator
    cur3 = ev3.get_start_node((0, 1, 0))
    two_up = ev3.find_accepted_node(2, 1, 0, 1, ev3.get_floor_level(cur3.pos),
                                    (0, 0), PathType.WALKABLE)
    check("a two-block wall is not climbed",
          two_up is None or two_up.cost_malus < 0.0,
          f"{None if two_up is None else (two_up.pos, two_up.cost_malus)}")


def section_search() -> None:
    section("flat ground")
    ground = Ground(radius=20)
    mob = FakeMob(ground)
    path = search(mob, (12.5, 1.0, 0.5))
    check("a straight-line target gets a path", path is not None)
    if path is not None:
        positions = path_positions(path)
        check("and it is reached", path.can_reach(), str(path))
        check("every step is one block in x, on the same y",
              all(p[1] == 1 for p in positions)
              and all(abs(positions[i + 1][0] - positions[i][0]) == 1
                      for i in range(len(positions) - 1)),
              str(positions[:6]))
        # `reach_range` is 1 and Manhattan, so the search stops a block short:
        # 0..11 is twelve nodes and the mob arrives *beside* x=12.
        check("and it stops one block short, at reach_range",
              len(positions) == 12 and positions[-1] == (11, 1, 0),
              f"{len(positions)} {positions[-1]}")

    section("around an obstacle")
    walled = Ground(radius=20)
    # Short enough to walk round, long enough that walking round it costs more
    # than the straight line - which is the whole claim being tested.
    walled.wall_x(4, -3, 3)
    mob2 = FakeMob(walled)
    path2 = search(mob2, (8.5, 1.0, 0.5))
    check("a wall with a way round is still reached", path2 is not None
          and path2.can_reach(), "" if path2 is None else str(path2))
    if path2 is not None:
        positions = path_positions(path2)
        check("and the path does not contain the wall",
              not any(p[0] == 4 and -3 <= p[2] <= 3 for p in positions),
              str([p for p in positions if p[0] == 4]))
        check("it leaves the wall by going round, in z",
              max(abs(p[2]) for p in positions) > 3, str(positions))
        check("and every node it walks on is standable",
              all(p[1] == 1 for p in positions))

    section("a wall with no way round")
    boxed = Ground(radius=16)
    boxed.wall_x(3, -16, 16)
    boxed.wall_z(3, -16, 16)
    mob3 = FakeMob(boxed)
    path3 = search(mob3, (10.5, 1.0, 0.5))
    check("an unreachable target still returns a partial path",
          path3 is not None, "None")
    if path3 is not None:
        check("marked not reached", not path3.can_reach(), str(path3))
        check("and non-empty, so the mob walks as far as it can",
              len(path3.nodes) > 0, f"{len(path3.nodes)}")
        check("with a distance-to-target saying how far short it fell",
              path3.get_dist_to_target() > 0)
        check("and every node it did reach is inside the box",
              all(abs(p[0]) < 3 and abs(p[2]) < 3
                  for p in path_positions(path3)))

    section("budget")
    maze = Ground(radius=20)
    for z in range(-20, 21):
        if z % 2 == 0:
            maze.wall(4, z, 1, 3)
    # The multiplier is vanilla's pressure valve and 0.0 pins the budget at
    # the floor (560), which a comb this long cannot be searched inside.
    finder4 = PathFinder()
    finder4.max_visited_nodes_multiplier = 0.0
    ai = maze.ai()
    fake = FakeMob(maze, follow_range=48.0)
    fake.ai.context = ai
    mob4 = finder4.build_mob_data(fake)
    ctx4 = PathfindingContext(ai, mob4.block_position(), min_y=ai.min_y)
    finder4.evaluator.prepare(ctx4, mob4)
    path4 = _search_with(finder4, fake, (18.5, 1.0, 0.5))
    check("a search that runs out of budget returns a partial path rather "
          "than None", path4 is not None, "None")
    if path4 is not None:
        check("and the partial path is short, not the whole way",
              len(path4.nodes) < 20, f"{len(path4.nodes)}")

    section("per-tick budget")
    # The dimension-level valve (`advanced.path-nodes-per-tick` on the
    # AIContext): every search of the tick spends against one allowance the
    # spawner refills. The case that made it necessary is a flood of a
    # hostile radius -- one such search costs more than a tick -- so the
    # hostile world tests the cap and a nearby target tests that the charge
    # is what was actually spent rather than the whole allowance.
    pillar = Ground(radius=20)
    pillar.wall(12, 6, 1, 4)
    budget_ai = pillar.ai(path_nodes_per_tick=64)
    budget_mob = FakeMob(pillar, follow_range=16.0)
    budget_mob.ai.context = budget_ai
    flood = search(budget_mob, (12.5, 5.0, 6.5))
    check("a search bigger than the allowance returns the partial path it "
          "could afford", flood is not None,
          "None" if flood is None else str(flood))
    check("and the allowance is exactly drained by it",
          budget_ai.path_nodes_left() == 0,
          f"left={budget_ai.path_nodes_left()}")
    after = search(budget_mob, (12.5, 5.0, 6.5))
    check("a search with an empty allowance answers None, not a guess",
          after is None and budget_ai.path_nodes_left() == 0,
          f"left={budget_ai.path_nodes_left()}")
    budget_ai.begin_tick()
    check("the refill is the spawner's one line per tick, and it refills",
          budget_ai.path_nodes_left() == 64,
          f"left={budget_ai.path_nodes_left()}")
    near = Ground(radius=20)
    near_ai = near.ai(path_nodes_per_tick=64)
    near_mob = FakeMob(near)
    near_mob.ai.context = near_ai
    short = search(near_mob, (3.5, 1.0, 0.5))
    check("a small search is charged what it spent, not the allowance",
          short is not None and 0 < near_ai.path_nodes_left() < 64,
          f"left={near_ai.path_nodes_left()}")
    unlimited = pillar.ai()
    check("the default context answers unlimited, so tools are unaffected",
          unlimited.path_nodes_left() == -1)


def section_water() -> None:
    section("water")
    # A watercourse running the whole length of the world. There is nothing to
    # go round, which is what makes this the right shape for the question:
    # is water a price or a prohibition?
    # A small world, so "no way round" is true by construction rather than by
    # hoping the search never noticed a gap: outside the filled box every
    # column is unloaded, which is BLOCKED.
    river = Ground(radius=8)
    for x in range(3, 6):
        for z in range(-8, 9):
            for y in (1, 2):
                river.put(x, y, z, WATER)
    mob = FakeMob(river, follow_range=16.0)
    path = search(mob, (7.5, 1.0, 0.5), max_nodes=4.0)
    check("a river with no way round is crossed", path is not None
          and path.can_reach(), "" if path is None else str(path))
    waded = [] if path is None else [n.pos for n in path.nodes
                                     if n.path_type == PathType.WATER]
    check("by wading - malus 8 is a price, not a prohibition",
          bool(waded), "no water node in the path")
    check("with the nodes sunk to the bottom of the water, not the surface",
          bool(waded) and all(p[1] == 1 for p in waded), str(waded))
    if path is not None:
        check("and priced above the ground it is standing on",
              all(n.cost_malus >= 0.0 for n in path.nodes)
              and any(n.cost_malus > 0.0 for n in path.nodes))

    # A pond wide enough that 8 malus per water block outweighs going round.
    # Both routes have to fit inside the search radius or the check measures
    # the radius instead of the cost, so this mob gets a wide one - a cow at
    # 16 would not even be offered the detour.
    pond = Ground(radius=20)
    for x in range(5, 17):
        for z in range(-1, 2):
            for y in (1, 2):
                pond.put(x, y, z, WATER)
    mob2 = FakeMob(pond, follow_range=48.0)
    path2 = search(mob2, (18.5, 1.0, 0.5), max_nodes=4.0)
    check("a pond that can be walked round is walked round",
          path2 is not None and path2.can_reach(),
          "" if path2 is None else str(path2))
    if path2 is not None:
        waded2 = [n.pos for n in path2.nodes if n.path_type == PathType.WATER]
        check("and that path has no water node at all", not waded2, str(waded2))
        check("it went round in z instead",
              max(abs(p[2]) for p in path_positions(path2)) > 1,
              str(path_positions(path2)))

    # And the radius itself. The same pond, at a cow's real 16-block radius:
    # the detour needs 22 walked blocks and the wade needs 18, so *neither* is
    # inside the budget and the search comes back partial. That is vanilla -
    # follow_range is a promise about what a mob will commit to, not a
    # suggestion - and it reads as a bug from outside, so it gets a check.
    narrow = FakeMob(pond)
    path3 = search(narrow, (18.5, 1.0, 0.5), max_nodes=4.0)
    check("the same pond at follow_range 16 is out of range either way",
          path3 is not None and not path3.can_reach(),
          "None" if path3 is None else str(path3))
    check("and comes back partial rather than as nothing",
          path3 is not None and len(path3.nodes) > 0,
          "empty" if path3 is None else f"{len(path3.nodes)} nodes")


def section_mob() -> None:
    section("a real mob, a real solver")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_mob_ai_check", FsPath(__file__).parent / "check_mob_ai.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - environment guard
        print(f"  skip the mob checks: could not load check_mob_ai ({exc})")
        return

    Rig = module.Rig
    World = module.World

    # Ten blocks, because that is what a wander goal actually asks for and it
    # fits inside a cow's 16-block follow range. A target further than that is
    # a different claim, made below.
    rig = Rig(World())
    rig.ai.selector.goals = []
    start_x = rig.pose.x
    rig.ai.navigator.move_to(rig.pose.x + 10.0, rig.pose.y, rig.pose.z, 1.0)
    rig.tick(300)
    check("a mob walks its path and arrives", rig.ai.navigator.is_idle,
          f"x={rig.pose.x:.2f} target={start_x + 10.0:.2f}")
    # It arrives at the *end of its path*, and the path stops one block short
    # of the target (`reach_range` 1). Landing inside half a block of the
    # target is not on offer, and vanilla does not offer it either.
    check("and it got there, to within a block of the target",
          rig.pose.x > start_x + 8.0,
          f"x={rig.pose.x:.2f} from {start_x:.2f}")

    section("and the follow range is a hard limit on the path")
    rig = Rig(World())
    rig.ai.selector.goals = []
    start_x = rig.pose.x
    rig.ai.navigator.move_to(rig.pose.x + 40.0, rig.pose.y, rig.pose.z, 1.0)
    rig.tick(600)
    check("a goal 40 blocks away comes back as a partial path, and the mob "
          "walks its 16 and stops",
          rig.ai.navigator.is_idle and 12.0 < rig.pose.x - start_x < 20.0,
          f"moved {rig.pose.x - start_x:.2f}")

    section("and now it goes round a wall")
    world = World()
    world.fill(12, 1, -8, 12, 3, 8, module.STONE)   # a wall due east
    rig = Rig(world)
    rig.ai.selector.goals = []
    rig.ai.navigator.move_to(rig.pose.x + 8.0, rig.pose.y, rig.pose.z, 1.0)
    # The detour has to be watched for rather than read off the end position:
    # the mob comes back to its own z line once it is past the wall, so its
    # final z says nothing about whether it detoured.
    furthest_z = 0.0
    for _ in range(400):
        rig.tick(1)
        furthest_z = max(furthest_z, abs(rig.pose.z - 8.5))
    check("the mob got past the wall rather than stopping against it",
          rig.pose.x > 12.5,
          f"x={rig.pose.x:.2f} z={rig.pose.z:.2f} idle={rig.ai.navigator.is_idle}")
    check("and it got there by going round it in z, not through it",
          furthest_z > 0.5, f"furthest z offset {furthest_z:.2f}")

    section("an unbound mob")
    unbound = Rig(World())
    unbound.ai.context = None
    check("move_to on an unbound AI returns True (it does not know better)",
          unbound.ai.navigator.move_to(50.0, 64.0, 0.0, 1.0))
    before = unbound.pose.x
    unbound.tick(50)
    check("and it does not move", abs(unbound.pose.x - before) < 1e-9)
    # `MobAI.tick` returns before the navigator when there is no context, so
    # the navigator is not idle - it was never asked. The target it is still
    # holding is the whole story: nothing reached it to stop it.
    check("and the navigator was never reached at all, so its target is "
          "untouched", not unbound.ai.navigator.is_idle)

    section("is_stuck distinguishes wedged from nowhere-to-go")
    # Case 1: no route at all. The mob stops, and reports *not* stuck - it did
    # not try and fail, it had nothing to try. That distinction is the whole
    # reason `is_stuck` exists alongside `is_idle`.
    boxed = World()
    for x in range(6, 12):
        for z in range(6, 12):
            for y in (1, 2, 3):
                boxed.fill(x, y, z, x, y, z, module.STONE)
    rig = Rig(boxed)
    rig.ai.selector.goals = []
    rig.ai.navigator.move_to(40.0, rig.pose.y, rig.pose.z, 1.0)
    rig.tick(600)
    check("a mob with no route at all stops",
          rig.ai.navigator.is_idle, "still walking after 600 ticks")
    check("and reports not-stuck, because it never tried and failed",
          not rig.ai.navigator.is_stuck,
          f"stuck={rig.ai.navigator.is_stuck}")

    # Case 2: genuinely wedged. The mob has a path and is walking it, and then
    # a wall appears across it. This is the case `is_stuck` is for, and it is
    # why the flag has to latch: the question cannot be answered while the walk
    # is still running, because a wedged mob is wedged exactly when the walk
    # has already ended.
    world = World()
    rig = Rig(world)
    rig.ai.selector.goals = []
    rig.ai.navigator.move_to(rig.pose.x + 10.0, rig.pose.y, rig.pose.z, 1.0)
    rig.tick(30)                      # it is walking, with a real path
    check("and it was walking before the world changed",
          not rig.ai.navigator.is_idle and not rig.ai.navigator.is_stuck,
          f"idle={rig.ai.navigator.is_idle} "
          f"stuck={rig.ai.navigator.is_stuck}")
    # Built *ahead* of the mob, not on top of it: filling stone into the block
    # a body is standing in is a different test (the solver ejects it) and
    # would measure that instead of the thing under test.
    wall_x = int(rig.pose.x) + 3
    for x in range(wall_x, wall_x + 10):
        for y in range(1, 4):
            world.fill(x, y, 8, x, y, 9, module.STONE)
    rig.tick(200)
    check("a mob that walks into a wall it did not plan for reports stuck",
          rig.ai.navigator.is_stuck and rig.ai.navigator.is_idle,
          f"stuck={rig.ai.navigator.is_stuck} "
          f"idle={rig.ai.navigator.is_idle}")
    check("and did not walk through it", rig.pose.x < wall_x,
          f"x={rig.pose.x:.2f} wall at x={wall_x}")
    rig.ai.navigator.move_to(rig.pose.x + 10.0, rig.pose.y, rig.pose.z, 1.0,
                             mob=rig.mob)
    check("and starting a new walk clears the flag",
          not rig.ai.navigator.is_stuck, "still latched after move_to")

    section("move_to answers reachability")
    rig = Rig(World())
    ok = rig.ai.navigator.move_to(rig.pose.x + 10.0, rig.pose.y, rig.pose.z,
                                   1.0, mob=rig.mob)
    check("a nearby target is reachable", ok is True)
    check("and it has a real path, not None",
          rig.ai.navigator.path is not None
          and rig.ai.navigator.path.can_reach())

    # **The return value means "the search found something", not "you can get
    # there".** A target in an unloaded region gives a *partial* path - which
    # is not None, so `move_to` says yes - and the mob walks its sixteen
    # blocks and stops. That is the vanilla contract and it is the right one:
    # "I found no route" and "the route stops short" call for the same
    # response from the goal, which is to try again somewhere else.
    far = rig.ai.navigator.move_to(100000.0, rig.pose.y, rig.pose.z, 1.0,
                                   mob=rig.mob)
    check("a target in another country still returns True - it is a partial "
          "path, not a failure", far is True, f"move_to={far}")
    check("but the path is marked not reached",
          rig.ai.navigator.path is not None
          and not rig.ai.navigator.path.can_reach())

    # The one case that really is False: a mob with no world has no blocks to
    # search, so there is no path at all.
    unbound2 = Rig(World())
    unbound2.ai.context = None
    check("and a mob with no world at all returns False",
          unbound2.ai.navigator.move_to(20.0, 64.0, 0.0, 1.0,
                                        mob=unbound2.mob) is False)


# ==================== run ====================

def main() -> int:
    print("pathfinding checks")
    section_malus()
    section_classification()
    section_land_type()
    section_danger_propagation()
    section_diagonal()
    section_falls_and_steps()
    section_search()
    section_water()
    section_mob()

    print(f"\n{PASSED} passed, {len(FAILED)} failed")
    for claim in FAILED:
        print(f"  - {claim}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())