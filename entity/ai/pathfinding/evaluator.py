"""
The walk evaluator: which of the eight neighbouring blocks can a mob step to.

This is the layer where a 1x1x1 grid stops being a nice abstraction and
becomes a set of judgement calls about bodies. Everything above it - the
search in `finder.py`, the waypoint following in `navigation.py` - assumes
that if this file returns a node, a mob could actually stand there. Two
judgements dominate, and both are about corners:

* **`is_diagonal_valid`** - can a mob step diagonally without clipping the two
  blocks it squeezes between? A cow walking north-east passes the corner
  between the north block and the east block; if either of those is a full
  block at the mob's own height, the cow passes through it. That is the rule
  vanilla encodes, and the fence exception in it is narrow and deliberate.
* **`find_accepted_node`** - an open neighbour block is not a place to walk,
  it is a place to *fall from*. This walks down the column until it finds
  floor, refuses to fall further than `max_fall_distance`, and is where a
  one-block step up, a three-block drop and a five-block drop diverge into
  three different answers.

**Only the walk evaluator is here.** Pumpkin also ports fly, swim and
amphibious evaluators, and they differ from this one in almost every branch -
`FlyNodeEvaluator` is longer than this file and has its own concept of what a
neighbour is. Every animal that spawns in Crotalus is a ground walker, and an
abstract `EvaluatorKind` with one implementation is an indirection with no
second implementation to justify it.

**`prepare` and `done` bracket one search.** The path-type cache is valid
between them and invalid outside, and that is not a convention - it is why
`finder.py` always calls `done()` even on the early-return paths. A cache
that outlives its world is a mob walking into a door another mob has since
opened, or a path that routes through a block the search loaded a tick ago.
"""

from __future__ import annotations

from math import ceil, floor, sqrt
from typing import List, Optional, Tuple

from entity.ai.pathfinding.context import PathfindingContext
from entity.ai.pathfinding.node import (
    Node,
    Pos,
    PathType,
    Target,
    get_malus,
)

__all__ = ["MobData", "WalkNodeEvaluator"]


class MobData:
    """The body the search is for.

    Built fresh for every `compute_path` (`node_evaluator.rs:34`). Not
    shared, not cached on the mob, because two of its fields are read from
    live body state - `on_ground` and the position - and a search that ran
    while the mob was falling has a different question than one that ran while
    it was standing. The other two, `width` and `height`, come off the entity
    type and are the only reason this is not three floats hard-coded.

    **`path_type_malus` is the per-mob override, and it is the whole reason
    this is a class and not a table.** The class default says lava costs 8
    extra blocks of walking; a cow overrides that to -1, meaning lava is a
    wall, and no amount of money talks it across. Water defaults to 8 and the
    override is conditional on `can_float`. Those are per-mob policy
    statements and they cannot live in a per-block table, so they live here,
    and `get_pathfinding_malus` is the one place the two meet.
    """

    __slots__ = ("position", "width", "height", "max_step_height",
                 "max_fall_distance", "can_swim", "can_walk_on_water",
                 "avoids_fire", "avoids_water", "on_ground", "is_in_water",
                 "sea_level", "min_y", "path_type_malus")

    def __init__(self, position: Tuple[float, float, float],
                 width: float = 0.6, height: float = 1.95,
                 max_step_height: float = 1.0) -> None:
        #: Where the mob's feet are, in world floats - not floored. The
        #: evaluator's start-node search reads the fractional part to decide
        #: whether the mob is standing on this block or straddling two.
        self.position = position
        self.width = width
        self.height = height
        #: How high a step the mob takes without jumping. Vanilla's step
        #: height attribute; 1.0 for every mob in vanilla, not 0.6, which is
        #: the *player's* number and the one the physics solver here carries.
        self.max_step_height = max_step_height
        #: How far it may fall without the node being refused. Vanilla's
        #: `maxFallDistance`; 3.0 for every mob.
        self.max_fall_distance = 3.0
        self.can_swim = False
        self.can_walk_on_water = False
        self.avoids_fire = True
        self.avoids_water = False
        #: Read from the body, not assumed. A mob in mid-air starts its search
        #: from where it will land, not from where it is - `get_start`.
        self.on_ground = True
        self.is_in_water = False
        self.sea_level = 63
        self.min_y = -64
        #: 27 slots, one per `PathType`. `None` means "no opinion, use the
        #: class default", which is what keeps the table sparse.
        self.path_type_malus: List[Optional[float]] = [None] * 27

    def set_pathfinding_malus(self, path_type: PathType, malus: float) -> None:
        self.path_type_malus[int(path_type)] = malus

    def get_pathfinding_malus(self, path_type: PathType) -> float:
        override = self.path_type_malus[int(path_type)]
        return get_malus(path_type) if override is None else override

    def block_position(self) -> Pos:
        x, y, z = self.position
        return (int(floor(x)), int(floor(y)), int(floor(z)))

    def get_bb_width(self) -> int:
        """Cells across the mob's width, rounded up.

        A 0.6-wide cow gets `floor(1.6) == 1`: one cell. A 2-block-wide mob
        gets 3, which is over-counted by a rounding, and that over-count is
        what makes the bounding-box rule conservative for large mobs rather
        than letting a two-wide mob decide a gap is passable.
        """
        return int(floor(self.width + 1.0))

    def get_bb_height(self) -> int:
        return int(floor(self.height + 1.0))

    @property
    def half_width(self) -> float:
        return self.width * 0.5


class WalkNodeEvaluator:
    """Vanilla's `WalkNodeEvaluator` (`walk_node_evaluator.rs`).

    One per `PathFinder`, reused across searches, with its caches cleared in
    `prepare` and `done`. Reused because allocating a dict per mob per tick is
    the kind of cost that shows up as a server-wide frame time before anyone
    notices which mob caused it.
    """

    __slots__ = ("context", "mob_data", "nodes", "entity_width",
                 "entity_height", "entity_depth", "can_pass_doors",
                 "can_open_doors", "can_float", "can_walk_over_fences",
                 "path_types_cache", "reusable_neighbors", "is_amphibious")

    def __init__(self) -> None:
        self.context: Optional[PathfindingContext] = None
        self.mob_data: Optional[MobData] = None
        #: Nodes interned by position for the current computation. Shared with
        #: the search, which reads `came_from` out of it to rebuild a path.
        self.nodes: dict = {}
        self.entity_width = 1
        self.entity_height = 2
        self.entity_depth = 1
        self.can_pass_doors = True
        self.can_open_doors = False
        self.can_float = False
        self.can_walk_over_fences = False
        #: Per-computation `PathType` cache. Separate from the one in
        #: `PathfindingContext` because the two answer different questions:
        #: the context caches one block's type, this caches one *position's*
        #: type once the mob's body has been folded in. See
        #: `get_path_type_of_mob` for what folding in means.
        self.path_types_cache: dict = {}
        #: Four slots reused across the neighbour expansion, one per
        #: horizontal direction. They have to outlive the loop because the
        #: diagonal test reads the two orthogonal ones.
        self.reusable_neighbors: List[Optional[Node]] = [None, None, None, None]
        self.is_amphibious = False

    # ----- lifecycle -----

    def prepare(self, context: PathfindingContext, mob_data: MobData) -> None:
        """Bind the world and the body for one search.

        The bounding-box dimensions come from the body rather than from the
        context because they change who counts as large: `entity_width > 1`
        is the switch that turns on the big-mob danger rule below, and a cow
        at 1 cell is not a big mob while a 2-block-wide one is.
        """
        self.entity_width = mob_data.get_bb_width()
        self.entity_height = mob_data.get_bb_height()
        self.entity_depth = mob_data.get_bb_width()
        self.context = context
        self.mob_data = mob_data
        self.nodes = {}
        self.path_types_cache = {}
        self.reusable_neighbors = [None, None, None, None]

    def done(self) -> None:
        """Release the world and drop every cache. Always called, including
        on `compute_path`'s early returns."""
        self.context = None
        self.mob_data = None
        self.nodes = {}
        self.path_types_cache = {}
        self.reusable_neighbors = [None, None, None, None]

    # ----- node interning -----

    def _get_node(self, pos: Pos) -> Node:
        """The interned node for a position, created on first ask."""
        node = self.nodes.get(pos)
        if node is None:
            node = Node(pos)
            self.nodes[pos] = node
        return node

    def _node_with_cost(self, x: int, y: int, z: int, path_type: PathType,
                        cost: float) -> Node:
        """The node at a position, priced at `cost` - and priced at the *worst*
        thing seen there, not the cheapest.

        `cost_malus = max(existing, cost)` is the accumulation of every visit
        to this position. A cell whose bounding box touches water is `WATER`
        at 8 for one body and `OPEN` at 0 for another; taking the max means
        the expensive reading sticks, so a path never becomes cheaper by being
        recomputed from a slightly different angle.
        """
        node = self._get_node((x, y, z))
        node.path_type = path_type
        node.cost_malus = max(node.cost_malus, cost)
        return node

    def _blocked_node(self, x: int, y: int, z: int) -> Node:
        node = self._get_node((x, y, z))
        node.path_type = PathType.BLOCKED
        node.cost_malus = -1.0
        return node

    def _closed_node(self, x: int, y: int, z: int,
                     path_type: PathType) -> Node:
        """A node marked closed before expansion.

        Only ever built for the partial-collision branches - a fence or a
        closed door that the mob's body cannot pass. It is marked closed so no
        other neighbour proposes it either, which is how "two fences in a row
        with a gap too small" stops being explored four times over.
        """
        node = self._get_node((x, y, z))
        node.closed = True
        node.path_type = path_type
        node.cost_malus = get_malus(path_type)
        return node

    # ----- costs and geometry -----

    def get_mob_penalty(self, path_type: PathType) -> float:
        if self.mob_data is None:
            return get_malus(path_type)
        return self.mob_data.get_pathfinding_malus(path_type)

    def get_mob_jump_height(self) -> float:
        """How high a single step up the mob can take, in blocks.

        `max(1.0, 1.125)`: vanilla's step height is 1.0, and the search is
        allowed to plan a step of 1.125 because the *jump* clears it - a mob
        that finds a one-and-an-eighth step will jump it, and a planner that
        refused to plan it would route around a staircase the mob could have
        climbed in one hop. Two levels of `try_jump_on` recursion are allowed
        before the search gives up on climbing.
        """
        if self.mob_data is None:
            return 1.125
        return max(float(self.mob_data.max_step_height), 1.125)

    def get_floor_level(self, pos: Pos) -> float:
        if self.context is None:
            return float(pos[1])
        if (self.can_float or self.is_amphibious) and self.context.is_water(pos):
            # A mob that floats rests half a block up, not on the floor.
            return float(pos[1]) + 0.5
        return self.context.floor_level(pos)

    def _has_collision(self, box: Tuple[float, float, float,
                                        float, float, float]) -> bool:
        if self.context is None:
            return False
        return self.context.has_collision(box)

    def _cached_path_type(self, pos: Pos) -> PathType:
        """The `PathType` of a position *as this mob would see it*.

        This is the difference between "this block is water" and "this cell is
        water for a 0.6-wide cow", and it is `get_path_type_of_mob`'s job -
        see there. Blocked is the answer when there is no context, which is
        the same answer a missing block gets: refuse rather than guess.
        """
        cached = self.path_types_cache.get(pos)
        if cached is not None:
            return cached
        if self.context is None or self.mob_data is None:
            path_type = PathType.BLOCKED
        else:
            path_type = self._path_type_of_mob(pos, self.mob_data)
        self.path_types_cache[pos] = path_type
        return path_type

    def _get_path_type(self, pos: Pos) -> PathType:
        """The land-node type of a position, before the mob's body is folded in."""
        if self.context is None:
            return PathType.BLOCKED
        return self.context.land_node_type(pos)

    # ----- the classification that needs a body -----

    def _path_type_within_mob_bb(self, x: int, y: int, z: int) -> List[PathType]:
        """Every `PathType` inside this mob's footprint, de-duplicated.

        A node is a cell, but a cow is not - it occupies parts of up to two
        cells in each axis. Vanilla samples the whole `entity_width x height x
        depth` box starting at this position and collects the types it finds
        (`walk_node_evaluator.rs:370`). The answer is not a set union: the
        types are sorted and de-duplicated, because the caller below walks
        them in that order and the order decides which one wins.

        Three of the adjustments are policy rather than classification, and
        they are the flags this evaluator carries:
        * a closed wooden door becomes `WALKABLE_DOOR` when the mob can open
          doors - the villager. No mob in Crotalus has that flag yet, so the
          branch is dead but present, and the *door* classifications are what
          make it a one-flag change rather than a rewrite.
        * an open door becomes `BLOCKED` when the mob cannot pass doors.
        * a rail becomes `UNPASSABLE_RAIL` unless the mob is standing on, or
          directly under, another rail - otherwise a cow would route along the
          rails of a minecart track, which is what vanilla does not do.
        """
        block_types: List[PathType] = []
        mob_data = self.mob_data
        assert mob_data is not None
        for dx in range(self.entity_width):
            for dy in range(self.entity_height):
                for dz in range(self.entity_depth):
                    block_type = self._get_path_type((x + dx, y + dy, z + dz))

                    if (block_type == PathType.DOOR_WOOD_CLOSED
                            and self.can_open_doors and self.can_pass_doors):
                        block_type = PathType.WALKABLE_DOOR
                    if block_type == PathType.DOOR_OPEN and not self.can_pass_doors:
                        block_type = PathType.BLOCKED
                    if block_type == PathType.RAIL:
                        mob_block = mob_data.block_position()
                        mob_type = self._get_path_type(mob_block)
                        mob_below = (mob_block[0], mob_block[1] - 1,
                                     mob_block[2])
                        mob_below_type = self._get_path_type(mob_below)
                        if mob_type != PathType.RAIL and mob_below_type != PathType.RAIL:
                            block_type = PathType.UNPASSABLE_RAIL

                    block_types.append(block_type)

        block_types = sorted(set(block_types))
        return block_types

    def _path_type_of_mob(self, pos: Pos, mob_data: MobData) -> PathType:
        """Fold the mob's footprint into a cell's type (`get_path_type_of_mob`).

        This is where "the block" becomes "a place this mob can be". Seven
        rules, in order, and the order is the algorithm:

        1. One type covering the whole footprint - no decision to make.
        2. A fence anywhere in the footprint wins outright. Not because fences
           are the worst thing there is, but because a fence is *thin*: the
           other blocks in the box are open air above and solid ground below,
           and a cell that is mostly walkable with a fence through it is a
           cell a cow has to squeeze, and vanilla wants it priced as a fence.
        3. Unpassable rail, same reasoning: the cell is mostly air and the one
           thing in it is a rail the mob would have to walk on.
        4. Any type with negative malus anywhere in the footprint is
           disqualifying - that is the whole "no mobs here" contract.
        5. Otherwise take the most expensive type in the footprint. A cell that
           is half water is water.
        6. A big mob (two cells wide or more) gets a special case: standing
           next to danger is *worse* for it than the block it is standing on
           suggests, because it cannot turn around in the space. So a big mob
           whose own cell is cheap but whose neighbourhood is dangerous is
           marked `BIG_MOBS_CLOSE_TO_DANGER` at a cost of 4.
        7. A small mob standing in an open cell keeps `OPEN`, even when the
           cell next to it is expensive. This is the asymmetry with rule 6 and
           it is deliberate: a cow beside a lava pool is fine, an ocelot-sized
           mob beside one is not, and vanilla draws that line by cell count
           rather than by species.
        """
        block_types = self._path_type_within_mob_bb(pos[0], pos[1], pos[2])
        if len(block_types) == 1:
            return block_types[0]

        if PathType.FENCE in block_types:
            return PathType.FENCE
        if PathType.UNPASSABLE_RAIL in block_types:
            return PathType.UNPASSABLE_RAIL

        highest_type = PathType.BLOCKED
        highest_malus = mob_data.get_pathfinding_malus(highest_type)
        for path_type in block_types:
            malus = mob_data.get_pathfinding_malus(path_type)
            if malus < 0.0:
                return path_type
            if malus >= highest_malus:
                highest_malus = malus
                highest_type = path_type

        current_node_type = self._get_path_type(pos)
        if self.entity_width > 1:
            is_current_node_cheaper = (mob_data.get_pathfinding_malus(current_node_type)
                                       < highest_malus)
            cap_malus_due_to_cheap_node = (
                is_current_node_cheaper
                and mob_data.get_pathfinding_malus(PathType.BIG_MOBS_CLOSE_TO_DANGER)
                < highest_malus)
            return (PathType.BIG_MOBS_CLOSE_TO_DANGER if cap_malus_due_to_cheap_node
                    else highest_type)

        if (current_node_type == PathType.OPEN
                and highest_type != PathType.OPEN and highest_malus == 0.0):
            return PathType.OPEN
        return highest_type

    # ----- the two corner judgements -----

    @staticmethod
    def is_neighbor_valid(neighbor: Optional[Node],
                          current: Node) -> bool:
        """Whether a proposed neighbour may be returned to the search.

        The `current.cost_malus < 0.0` clause is not a typo and not
        defensive: it lets a search *leave* a cell it could not legally be
        in. A cow that has been knocked into a half-block of snow is standing
        in a negative-malus cell, and refusing to plan any way out of it is
        how a mob ends up stuck inside the one obstacle it is standing in.
        """
        if neighbor is None or neighbor.closed:
            return False
        return neighbor.cost_malus >= 0.0 or current.cost_malus < 0.0

    def is_diagonal_valid(self, current: Node, first: Optional[Node],
                          second: Optional[Node]) -> bool:
        """Can a mob step diagonally past the corner between two neighbours?

        Walking north-east from `(x, y, z)` means passing the corner where
        the north cell `(x, y, z-1)` and the east cell `(x+1, y, z)` meet. The
        mob's body sweeps through both, so if either of them is solid at the
        mob's height the mob clips it. Four rules, and every one of them has
        been got wrong by a plausible simplification:

        * **Neither neighbour may be higher than the current node.** A step up
          and sideways at once is a jump, and the jump is not a diagonal - it
          is an ordinary step up handled by `try_jump_on`. Allowing it here
          would produce paths with diagonal nodes one block above their
          neighbours that the mob cannot execute.
        * **A door in either corner kills the diagonal outright**, even if the
          other corner is wide open. `WALKABLE_DOOR` means the cell is a door
          this mob could open, and a mob opening a door while turning a corner
          is a mob that gets stuck in the frame. Returning `false` in the
          `else` branch - rather than falling through to the width test - is
          what makes this unconditional.
        * **The mob-width branch.** A mob one cell or narrower takes the
          corner if both neighbours are no higher and passable. A mob two cells
          wide needs both neighbours to be *free* (`cost_malus <= 0`) as well,
          because it is not squeezing through a gap, it is turning inside one.
        * **`can_pass_between_posts`**, the narrow exception, and it is worth
          reading closely: both neighbours are fences *and* the mob is under
          half a block wide. That is a chicken, and it is the only mob in
          vanilla that fits between two fence posts. Everything else treats the
          pair as impassable no matter how much space is nominally there.

        The two neighbours are named `first`/`second` rather than after
        compass directions because the rule is symmetric in them and the two
        call-site orderings differ per diagonal; naming them would assert an
        east-west / north-south distinction the code does not make.
        """
        if first is None or second is None:
            return False
        if second.pos[1] > current.pos[1] or first.pos[1] > current.pos[1]:
            return False
        if (first.path_type == PathType.WALKABLE_DOOR
                or second.path_type == PathType.WALKABLE_DOOR):
            return False

        mob_width = 0.6 if self.mob_data is None else self.mob_data.width
        if mob_width <= 1.0 or (first.cost_malus <= 0.0
                                and second.cost_malus <= 0.0):
            can_pass_between_posts = (second.path_type == PathType.FENCE
                                      and first.path_type == PathType.FENCE
                                      and mob_width < 0.5)
            return ((second.pos[1] < current.pos[1]
                     or second.cost_malus >= 0.0 or can_pass_between_posts)
                    and (first.pos[1] < current.pos[1]
                         or first.cost_malus >= 0.0 or can_pass_between_posts))
        return False

    @staticmethod
    def is_diagonal_node_valid(diagonal: Optional[Node]) -> bool:
        """Whether the diagonal's own cell may be stepped into.

        Stricter than `is_neighbor_valid`: no "escape from a negative cell"
        exception, because a diagonal is a shortcut and a shortcut out of a
        cell the mob should not be in is not a shortcut.
        """
        return (diagonal is not None and not diagonal.closed
                and diagonal.path_type != PathType.WALKABLE_DOOR
                and diagonal.cost_malus >= 0.0)

    @staticmethod
    def does_block_have_partial_collision(path_type: PathType) -> bool:
        """Fences and *closed* doors - the blocks thin enough to squeeze past.

        Note this is a narrower set than `has_partial_collision` in `node.py`,
        which also includes `WALKABLE_DOOR` and `DOOR_OPEN`. The two are not
        the same question: that one asks "can this block have a gap", this one
        asks "is this block a gap-shaped obstacle", and an open door is not an
        obstacle at all.
        """
        return path_type in (PathType.FENCE, PathType.DOOR_WOOD_CLOSED,
                             PathType.DOOR_IRON_CLOSED)

    def can_reach_without_collision(self, x: int, y: int, z: int) -> bool:
        """Whether the mob's body can slide from here to there unobstructed.

        Not "is the destination cell free" - the cell is already known free,
        or this would not be called. This walks the mob's *bounding box* along
        the straight line in `ceil(distance / average_side)` steps and asks
        whether the box collides at each one. It is the difference between a
        cow fitting through a one-block gap in a fence line and clipping the
        fence post at the midpoint, which no per-cell test can see.

        A `None` body answers True, matching vanilla: with no body there is no
        box, and refusing to move on that basis would make a body-less search
        find nothing at all rather than finding the cells it was asked about.
        """
        mob_data = self.mob_data
        if mob_data is None:
            return True

        mx, my, mz = mob_data.position
        width = float(mob_data.width)
        height = float(mob_data.height)
        min_x = mx - width * 0.5
        min_y = my
        min_z = mz - width * 0.5
        box = (min_x, min_y, min_z,
               min_x + width, min_y + height, min_z + width)

        x_size = box[3] - box[0]
        y_size = box[4] - box[1]
        z_size = box[5] - box[2]
        avg_size = (x_size + y_size + z_size) / 3.0

        dx = float(x) - mx + x_size * 0.5
        dy = float(y) - my + y_size * 0.5
        dz = float(z) - mz + z_size * 0.5
        steps = int(ceil(sqrt(dx * dx + dy * dy + dz * dz) / avg_size)) if avg_size > 0 else 0
        if steps <= 0:
            return True
        dx /= steps
        dy /= steps
        dz /= steps

        for _ in range(steps):
            box = (box[0] + dx, box[1] + dy, box[2] + dz,
                   box[3] + dx, box[4] + dy, box[5] + dz)
            if self._has_collision(box):
                return False
        return True

    # ----- the core acceptance test -----

    def find_accepted_node(self, x: int, y: int, z: int, jump_size: int,
                           node_height: float, travel_direction: Tuple[int, int],
                           block_path_type_current: PathType) -> Optional[Node]:
        """The node at `(x, y, z)` if a mob can stand there having come from
        the current one, else None.

        This is `find_accepted_node` (`walk_node_evaluator.rs:134`) and it is
        the branch tree the whole evaluator is built around. Read top to
        bottom:

        1. **Too high to step onto.** Compare the destination's *floor level*
           against the current node's floor level. Floor level, not block Y,
           because a slab is half a block of floor and treating it as a full
           step is what makes mobs refuse to walk over slabs.
        2. **The obvious answer** - the cell is passable at its own price.
        3. **Unless the mob's body cannot fit**, for partial-collision
           obstacles: a fence the cell is otherwise fine gets rejected if the
           bounding box cannot slide through. This is only consulted when the
           current node *is* partial-collision, because that is the case where
           a free cell next to the obstacle is really a squeeze.
        4. **Otherwise, four alternatives, tried in order:**
           * **Climb.** `jump_size > 0` means the search still has jump budget
             (it starts at `floor(jump_height)` and is spent one per level).
             Fences are excluded unless the mob can walk over fences, and rails
             and trapdoors and powder snow are excluded outright - a mob does
             not jump *onto* a trapdoor.
           * **Sink to the bottom of the water**, when the cell is water and
             the mob cannot float. Not a swim path: the node is moved to the
             lowest water cell above the floor so the cow wades at the bottom
             rather than trying to stand on the surface.
           * **Fall to the ground**, when the cell is open air. An open
             neighbour is a hole, not a step, and this is where it stops being
             one.
           * **Close it**, when the cell is partial-collision and nothing else
             worked. Marking the node closed is how the search stops
             re-proposing it from every other neighbour.
        """
        max_y_target = self.get_floor_level((x, y, z))
        if max_y_target - node_height > self.get_mob_jump_height():
            return None

        path_type = self._cached_path_type((x, y, z))
        path_cost = self.get_mob_penalty(path_type)
        best = (self._node_with_cost(x, y, z, path_type, path_cost)
                if path_cost >= 0.0 else None)

        if (self.does_block_have_partial_collision(block_path_type_current)
                and best is not None and best.cost_malus >= 0.0
                and not self.can_reach_without_collision(x, y, z)):
            best = None

        if path_type != PathType.WALKABLE and not (
                self.is_amphibious and path_type == PathType.WATER):
            if ((best is None or best.cost_malus < 0.0)
                    and jump_size > 0
                    and (path_type != PathType.FENCE or self.can_walk_over_fences)
                    and path_type != PathType.UNPASSABLE_RAIL
                    and path_type != PathType.TRAPDOOR
                    and path_type != PathType.POWDER_SNOW):
                best = self._try_jump_on(x, y, z, jump_size, node_height,
                                         travel_direction,
                                         block_path_type_current)
            elif (not self.is_amphibious and path_type == PathType.WATER
                    and not self.can_float):
                best = self._try_find_first_non_water_below(x, y, z, best)
            elif path_type == PathType.OPEN:
                best = self._try_find_first_ground_node_below(x, y, z)
            elif (self.does_block_have_partial_collision(path_type)
                    and best is None):
                best = self._closed_node(x, y, z, path_type)

        return best

    def _try_jump_on(self, x: int, y: int, z: int, jump_size: int,
                     node_height: float, travel_direction: Tuple[int, int],
                     block_path_type_current: PathType) -> Optional[Node]:
        """Recurse one block up with one less jump.

        The recursion *is* the climb: `find_accepted_node` at `y + 1` will,
        if that is also too high, recurse to `y + 2` with the budget spent
        again. Two levels of budget is one block of step height plus the
        1.125 jump allowance, which is why a mob can plan a one-block step but
        not a two-block wall.

        **The head-clearance check below is the second half, and it only runs
        for narrow mobs.** A cow jumping onto a step has to get its *head*
        over the edge too - the part of it hanging behind, under the lip of the
        block it is landing on. The `grow` box is that body: from the centre of
        the cell it came from, spanning the height between the two floors, and
        if anything solid is in it the jump is refused. Wide mobs skip it
        because they are already too wide for the corner to be their problem.
        """
        node_above = self.find_accepted_node(x, y + 1, z, jump_size - 1,
                                             node_height, travel_direction,
                                             block_path_type_current)
        if node_above is None:
            return None
        mob_data = self.mob_data
        if mob_data is None:
            return node_above
        if mob_data.width >= 1.0:
            return node_above
        if node_above.path_type not in (PathType.OPEN, PathType.WALKABLE):
            return node_above

        centre_x = float(x) - float(travel_direction[0]) + 0.5
        centre_z = float(z) - float(travel_direction[1]) + 0.5
        half_width = mob_data.half_width
        floor_below = self.get_floor_level((int(floor(centre_x)), y + 1,
                                            int(floor(centre_z))))
        floor_above = self.get_floor_level(node_above.pos)
        grow = (centre_x - half_width, floor_below + 0.001,
                centre_z - half_width,
                centre_x + half_width,
                float(mob_data.height) + floor_above - 0.002,
                centre_z + half_width)
        if self._has_collision(grow):
            return None
        return node_above

    def _try_find_first_non_water_below(self, x: int, y: int, z: int,
                                        best: Optional[Node]) -> Optional[Node]:
        """Sink a water node to the bottom of its water column.

        Keeps walking down while the cell is still water, remembering the
        deepest one, and returns that when it hits something that is not. The
        reason a cow's node ends up at the pond's bottom rather than its
        surface is that a cow's node is *where its feet are*; putting it at
        the surface would ask the mob to stand on water. If the column is
        water all the way down to the world floor, the deepest node found is
        returned anyway - a bad node beats no path, and the mob's `SwimGoal`
        handles the case of actually being in deep water.
        """
        mob_data = self.mob_data
        min_y = -64 if mob_data is None else mob_data.min_y
        y -= 1
        while y > min_y:
            path_type = self._cached_path_type((x, y, z))
            if path_type != PathType.WATER:
                return best
            penalty = self.get_mob_penalty(path_type)
            best = self._node_with_cost(x, y, z, path_type, penalty)
            y -= 1
        return best

    def _try_find_first_ground_node_below(self, x: int, y: int,
                                          z: int) -> Node:
        """Fall from an open cell until there is floor.

        Walks down from the cell, and returns the first cell that is not open
        air - either a standable one (returned with its price) or a blocked
        one (returned marked blocked, so the search treats the fall as a dead
        end rather than as an infinite hole).

        **`max_fall_distance` is checked *before* the cell at each depth, and
        the check is `>`, not `>=`.** A three-block drop is allowed; four is
        not, and the node returned at the limit is a *blocked* node, so a mob
        never plans a fall that would hurt it. This is the single most
        consequential number in the evaluator - it is why a cow walks to the
        edge of a three-block ledge and stops, and why it will happily step off
        a two-block one.
        """
        mob_data = self.mob_data
        max_fall_distance = 3 if mob_data is None else int(floor(mob_data.max_fall_distance))
        min_y = -64 if mob_data is None else mob_data.min_y

        current_y = y - 1
        while current_y >= min_y:
            if y - current_y > max_fall_distance:
                return self._blocked_node(x, current_y, z)
            path_type = self._cached_path_type((x, current_y, z))
            if path_type != PathType.OPEN:
                path_cost = self.get_mob_penalty(path_type)
                if path_cost >= 0.0:
                    return self._node_with_cost(x, current_y, z, path_type,
                                               path_cost)
                return self._blocked_node(x, current_y, z)
            current_y -= 1
        return self._blocked_node(x, y, z)

    # ----- start and target -----

    def can_start_at(self, pos: Pos) -> bool:
        """Whether the search may begin in this cell.

        `path_type != OPEN` is the interesting half: the mob's own cell is not
        classified as open air, which is true of any cell it is standing in,
        because the block is *under* the cell. Requiring it means the search
        does not start inside a hole it would immediately fall out of.
        """
        path_type = self._cached_path_type(pos)
        return path_type != PathType.OPEN and self.get_mob_penalty(path_type) >= 0.0

    def get_start_node(self, pos: Pos) -> Node:
        node = self._get_node(pos)
        node.path_type = self._cached_path_type(pos)
        node.cost_malus = self.get_mob_penalty(node.path_type)
        return node

    def get_start(self) -> Optional[Node]:
        """The node the search starts from (`walk_node_evaluator.rs:440`).

        Four cases, and the mob's own body decides which:

        * **Walking on water** - start at the mob's own cell. Dead this round
          (`can_walk_on_water` is never set); the case is here because removing
          it later would mean re-deriving the start-Y rule from scratch.
        * **Floating in water** - scan upward until the cell above stops being
          water, and start there. That is the surface, which is where a mob
          that floats actually is.
        * **On the ground** - `floor(y + 0.5)`. The half is the mob's feet
          tolerance: a mob standing with its feet a third of the way up a
          block belongs to the lower cell, and rounding at `y` instead would
          start the search half a block inside the ground.
        * **Airborne** - look *down* from above the mob for the first cell that
          is not air and not pathfindable, and start there. A falling cow
          starts its search from where it is going to land, which is what stops
          a path from being planned from a position the mob is about to leave.

        If the resulting cell is not startable - a cow standing inside a fence,
        say - the four corners of its bounding box are tried, and the first
        startable one wins. That is the "I am stuck in a hedge" escape hatch,
        and it is checked in a fixed corner order because which corner wins
        is observable and should not depend on dict ordering.
        """
        mob_data = self.mob_data
        if mob_data is None:
            return None
        mob_x, mob_y, mob_z = mob_data.position
        block_x = int(floor(mob_x))
        block_y = int(floor(mob_y))
        block_z = int(floor(mob_z))
        context = self.context

        start_y = block_y
        if not mob_data.can_walk_on_water:
            if self.can_float and mob_data.is_in_water:
                check_y = block_y
                if context is not None:
                    while context.is_water((block_x, check_y + 1, block_z)):
                        check_y += 1
                start_y = check_y
            elif mob_data.on_ground:
                start_y = int(floor(mob_y + 0.5))
            else:
                start_y = int(floor(mob_y + 1.0))
                check_y = start_y
                while check_y > mob_data.min_y:
                    start_y = check_y
                    check_y -= 1
                    below = (block_x, check_y, block_z)
                    if (context is not None and not context.is_air(below)
                            and not context.is_pathfindable(below)):
                        break

        start_pos = (block_x, start_y, block_z)
        if not self.can_start_at(start_pos):
            half_width = mob_data.half_width
            min_x = int(floor(mob_x - half_width))
            max_x = int(floor(mob_x + half_width))
            min_z = int(floor(mob_z - half_width))
            max_z = int(floor(mob_z + half_width))
            for candidate in ((min_x, start_y, min_z), (min_x, start_y, max_z),
                              (max_x, start_y, min_z), (max_x, start_y, max_z)):
                if self.can_start_at(candidate):
                    return self.get_start_node(candidate)

        return self.get_start_node(start_pos)

    def get_target(self, pos: Pos) -> Target:
        return Target(self._get_node(pos))

    # ----- expansion -----

    def get_neighbors(self, current: Node, out: List[Node]) -> None:
        """Every cell the search may consider next from `current`.

        Four orthogonal and four diagonal, no up and no straight down: up is
        `try_jump_on`'s business and down is `_try_find_first_ground_node_below`'s,
        and neither is a separate neighbour. Vanilla's order is orthogonal
        first, all four, then diagonal - because the diagonal test needs two
        orthogonal answers and computing them on demand would mean computing
        them twice.

        The four orthogonal results are kept in `reusable_neighbors` for the
        diagonal pass to read. That is the whole reason the field exists, and
        it is why `prepare` resets it: a stale entry from a previous search
        would be a cell in a different world being used as a corner check.
        """
        block_path_type_above = self._cached_path_type(
            (current.pos[0], current.pos[1] + 1, current.pos[2]))
        block_path_type_current = self._cached_path_type(current.pos)

        # Jump budget for this expansion: one step if there is headroom above
        # and the mob is not in honey, zero otherwise. A mob under a ceiling
        # cannot jump, and planning a jump it cannot make is how a path sends
        # a cow into a tunnel roof.
        if (self.get_mob_penalty(block_path_type_above) >= 0.0
                and block_path_type_current != PathType.STICKY_HONEY):
            jump_size = int(floor(self.get_mob_jump_height()))
        else:
            jump_size = 0

        pos_height = self.get_floor_level(current.pos)

        neighbours = self.reusable_neighbors
        neighbours[0] = neighbours[1] = neighbours[2] = neighbours[3] = None

        # Slot order is south(0), west(1), north(2), east(3) - vanilla's
        # indexing, which the diagonal table below is written against.
        for slot, dx, dz, direction in (
                (2, 0, -1, (0, 0)), (0, 0, 1, (0, 0)),
                (1, -1, 0, (-1, 0)), (3, 1, 0, (1, 0))):
            node = self.find_accepted_node(
                current.pos[0] + dx, current.pos[1], current.pos[2] + dz,
                jump_size, pos_height, direction, block_path_type_current)
            neighbours[slot] = node
            if self.is_neighbor_valid(node, current):
                out.append(node)

        for slot, other_slot, dx, dz, direction in (
                (2, 3, 1, -1, (1, 0)), (0, 1, -1, 1, (-1, 0)),
                (1, 2, -1, -1, (0, 0)), (3, 0, 1, 1, (0, 0))):
            if self.is_diagonal_valid(current, neighbours[slot],
                                      neighbours[other_slot]):
                diagonal_node = self.find_accepted_node(
                    current.pos[0] + dx, current.pos[1], current.pos[2] + dz,
                    jump_size, pos_height, direction, block_path_type_current)
                if self.is_diagonal_node_valid(diagonal_node):
                    out.append(diagonal_node)