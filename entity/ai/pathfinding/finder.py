"""
The search.

A* over the 1x1x1 node graph the evaluator produces, with `heapq` and lazy
deletion in place of Pumpkin's `binary_heap.rs`. That substitution is the one
piece of this round's structure that is not a port, and it is deliberate: the
binary heap there exists to support `decrease-key` in place, keeping exactly
one entry per open node. `heapq` has no `decrease-key`, and adding one means
maintaining an index map inside a wrapper - which is the same code, written
again, to end up in the same place. Lazy deletion pushes a second entry when
a node improves and discards the first when it is popped, which costs a few
extra tuples per improvement and removes the index map entirely.

**Two numbers control how much work one search may do, and they are
deliberately different.** `max_iters` bounds the pops; `follow_range` bounds
the radius. The radius is the mob's own `follow_range` attribute (16 blocks,
so a cow will not plan a walk to something 40 blocks away), and `max_iters`
is `follow_range * 16 * multiplier` clamped to 100..2048 with a floor of 560.
The clamp is a latency bound, not a correctness one: a search that runs out of
iterations returns a partial path rather than failing, and the multiplier is
vanilla's pressure valve - `WanderAroundGoal` sets it to 0.5, which is the
first thing to turn if a mob is measurably costing too much per tick.

**The heuristic is Euclidean distance times 1.5, and the 1.5 is not tuning.**
It is the same inflation factor vanilla uses to make an admissible heuristic
admissible-in-practice: the true cost is walked distance *plus* malus, and a
mob that pays even one `WATER` at 8 is walking much slower than its straight
line suggests. At 1.0 the heuristic under-estimates so hard that the search
degenerates toward Dijkstra and opens a huge front; at 1.5 it still never
over-estimates (a malus of 0 is the common case) but it expands far fewer
nodes for the same answer.

**`Target.best_node` is how a failed search still returns something.** Every
expansion updates it; when the budget runs out or the target is walled off,
the path unwinds from it. A partial path is a shorter version of the real
answer, not a different answer, and the navigation layer walks it and then
gives up cleanly - which is the behaviour a mob wants when the player is
behind a wall it cannot get round.
"""

from __future__ import annotations

from heapq import heappop, heappush
from math import sqrt
from typing import List, Optional, Tuple

from entity.ai.pathfinding.context import PathfindingContext
from entity.ai.pathfinding.evaluator import MobData, WalkNodeEvaluator
from entity.ai.pathfinding.node import Node, PathType
from entity.ai.pathfinding.path import Path

__all__ = ["MAX_ITERS", "PathFinder", "TARGET_DISTANCE_MULTIPLIER"]

#: Vanilla's `MAX_ITERS` (`mod.rs:34`) - a floor on the budget so that a mob
#: with a tiny follow range can still find the four nodes to its goal.
MAX_ITERS = 560

#: The heuristic inflation (`mod.rs:35`). See the module docstring: it is
#: about how much of the true cost the straight line misses, not about being
#: in a hurry.
TARGET_DISTANCE_MULTIPLIER = 1.5

#: Malus overrides applied to every mob before its search starts
#: (`mod.rs:583-587`). Lava and fire are not merely expensive, they are
#: forbidden: no mob in vanilla walks into fire on purpose, and the "danger"
#: types - the ones you get by standing *next* to it - are what the search
#: actually routes around.
_DEFAULT_MALUS = (
    (PathType.DANGER_FIRE, 16.0),
    (PathType.DAMAGE_FIRE, -1.0),
    (PathType.WATER, 8.0),
    (PathType.LAVA, -1.0),
    (PathType.DANGER_OTHER, 8.0),
)


class PathFinder:
    """Owns the evaluator and runs searches. One per mob.

    The evaluator is held rather than rebuilt because its caches are the
    expensive part and rebuilding them per search would mean allocating four
    dicts on every pathfinding tick. Its contents are worthless between
    searches - `prepare` clears them - so holding it is not caching stale
    answers, it is holding a scratch buffer.
    """

    __slots__ = ("evaluator", "max_visited_nodes_multiplier",
                 "_open_heap", "_open_g", "_closed", "_neighbours",
                 "_required_path_length", "_malus_overrides")

    def __init__(self) -> None:
        self.evaluator = WalkNodeEvaluator()
        #: Vanilla's per-goal valve. 1.0 by default; `WanderAroundGoal` uses
        #: 0.5, and it is the first thing to lower if searches are too costly.
        self.max_visited_nodes_multiplier = 1.0
        #: Vanilla's `requiredPathLength` - the shortest path the mob will
        #: bother planning, independent of how far away the goal is. 16.0.
        self._required_path_length = 16.0
        #: Extra per-mob malus, applied on top of `_DEFAULT_MALUS`. No mob
        #: sets one yet; a villager's "willing to cross a doorway" and a
        #: chicken's "will not go near water" are both this field.
        self._malus_overrides: List[Tuple[PathType, float]] = []
        self._open_heap: List[Tuple[float, int, Tuple[int, int, int]]] = []
        self._open_g: dict = {}
        self._closed: set = set()
        self._neighbours: List[Node] = []

    def set_max_visited_nodes_multiplier(self, multiplier: float) -> None:
        """Vanilla's budget valve, named in the plan as the first thing to turn
        if a search is measurably costing too much. `WanderAroundGoal` uses
        0.5 upstream; nothing here sets it yet, and `bench_pathfinding.py`
        drives the attribute directly."""
        self.max_visited_nodes_multiplier = multiplier

    def set_pathfinding_malus(self, path_type: PathType, malus: float) -> None:
        self._malus_overrides.append((path_type, malus))

    def mob_max_follow_range(self, mob) -> float:
        """How far the search will look, in blocks.

        `max(follow_range, required_path_length)`. The attribute alone is
        wrong for a mob whose follow range was tuned for *seeing* something -
        a mob that can see 32 blocks but is only willing to walk 16 of them
        would plan paths it will not finish.
        """
        return max(float(mob.attributes.follow_range),
                   self._required_path_length)

    # ----- building the body -----

    def build_mob_data(self, mob) -> MobData:
        """A `MobData` for this mob, for this one search.

        Width and height come off the entity type rather than off the mob
        class. A chicken is 0.4 x 0.7 and a cow is 0.9 x 1.4, and the
        difference is not cosmetic: it is what decides whether the mob can
        pass a gap in a fence line and whether its head clears a step's lip.
        Both are read from `assets/entities.json`, which is the same source
        the physics solver reads them from, so a cow's collision box and its
        pathfinding box cannot disagree.

        `on_ground` is read from the body rather than assumed, and it is the
        one field that changes what the search is even asking: a standing cow
        starts at the block under its feet, an airborne one starts where it
        is going to land.
        """
        pos = mob.position
        entity_type = getattr(mob, "entity_type", None)
        width = float(getattr(entity_type, "width", 0.6) or 0.6)
        height = float(getattr(entity_type, "height", 1.95) or 1.95)

        mob_data = MobData((pos.x, pos.y, pos.z), width, height, 1.0)
        mob_data.on_ground = bool(pos.on_ground)
        mob_data.is_in_water = False
        context = mob.ai.context
        if context is not None:
            mob_data.min_y = context.min_y
        for path_type, malus in _DEFAULT_MALUS:
            mob_data.set_pathfinding_malus(path_type, malus)
        for path_type, malus in self._malus_overrides:
            mob_data.set_pathfinding_malus(path_type, malus)
        return mob_data

    # ----- the search -----

    def create_path(self, mob, destination: Tuple[float, float, float],
                    reach_range: int = 1) -> Optional[Path]:
        """Find a path from where the mob is to where it wants to be.

        Returns None only when the mob is not in a world at all. Every other
        failure - unreachable target, budget exhausted, surrounded by walls -
        returns a partial path, because "go as far as you can toward it" is
        what a mob should do in all three cases and "stand still" is what it
        would do in none of them.

        `reach_range` is a Manhattan radius, not a Euclidean one: a node
        within 1 block of the target counts as having arrived, which is how a
        cow stops *at* the block next to a player rather than trying to stand
        in them.
        """
        context_handle = mob.ai.context
        if context_handle is None or context_handle.dimension is None:
            # No world, so no blocks, so every node is BLOCKED. Returning
            # None rather than an empty path is what makes an unbound mob
            # stand still instead of trying to walk through the floor of an
            # empty search space.
            return None

        pos = mob.position
        mob_data = self.build_mob_data(mob)
        context = PathfindingContext(
            context_handle, mob_data.block_position(), min_y=mob_data.min_y,
            sea_level=mob_data.sea_level)

        self.evaluator.prepare(context, mob_data)
        try:
            return self._search(mob, mob_data, destination, reach_range)
        finally:
            # Unconditional. The caches in here are only valid for the world
            # they were built from, and a search that raised or returned early
            # still filled them.
            self.evaluator.done()

    def _search(self, mob, mob_data: MobData,
                destination: Tuple[float, float, float],
                reach_range: int) -> Optional[Path]:
        start_node = self.evaluator.get_start()
        if start_node is None:
            return None
        target = self.evaluator.get_target(
            (int(destination[0]), int(destination[1]), int(destination[2])))

        start_node.g = 0.0
        start_dist = start_node.distance_to(target.node)
        target.update_best(start_dist, start_node)
        start_node.h = start_dist
        start_node.f = start_node.h
        start_node.walked_dist = 0.0
        start_node.came_from = None

        start_pos = start_node.pos
        follow_range = self.mob_max_follow_range(mob)
        max_iters = max(560, min(2048, max(100, int(
            follow_range * 16.0 * self.max_visited_nodes_multiplier))))

        # The dimension's per-tick search allowance (`advanced.path-nodes-per-tick`,
        # on the AIContext every mob of this dimension shares). Two exits, and
        # the difference matters: exhausted *before* the search returns None,
        # because "the tick has already paid for pathfinding" is an honest
        # "not searched" and the goal re-rolls next tick; exhausted *during*
        # the search is vanilla's budget-exhausted answer, a partial path from
        # the best node seen, because the pops are already spent. Either way a
        # herd on hostile terrain spends its way down to crumbs across ticks
        # instead of stacking 48 ms searches into one discarded tick.
        context_handle = mob.ai.context
        budget = context_handle.path_nodes_left()
        if budget == 0:
            return None
        if budget > 0:
            max_iters = min(max_iters, budget)

        heap = self._open_heap
        open_g = self._open_g
        closed = self._closed
        nodes = self.evaluator.nodes
        heap.clear()
        open_g.clear()
        closed.clear()

        # A monotonic tie-break counter, so two nodes with equal `f` are
        # ordered by insertion rather than by comparing positions. Using the
        # heap's length would not do: it shrinks on every pop, so two entries
        # could share a value and the tie would fall through to the position
        # comparison, making the order of equally-priced nodes a function of
        # their coordinates.
        tiebreak = 0
        push = heappush
        push(heap, (start_node.f, tiebreak, start_pos))
        open_g[start_pos] = 0.0

        reached = False
        iterations = 0
        neighbours = self._neighbours
        target_pos = target.node.pos

        while heap:
            iterations += 1
            if iterations >= max_iters:
                break

            _, _, pos = heappop(heap)
            g = open_g.pop(pos, None)
            if g is None:
                continue  # a stale entry from before the node improved
            if pos in closed:
                continue
            current = nodes[pos]
            closed.add(pos)

            if current.manhattan_to_pos(target_pos) <= reach_range:
                target.set_reached()
                reached = True
                target.update_best(0.0, current)
                break

            dx = pos[0] - start_pos[0]
            dy = pos[1] - start_pos[1]
            dz = pos[2] - start_pos[2]
            if sqrt(dx * dx + dy * dy + dz * dz) >= follow_range:
                # Too far from home to be walking to. Close it without
                # expanding, so it is not re-proposed by its neighbours.
                continue

            neighbours.clear()
            self.evaluator.get_neighbors(current, neighbours)

            for neighbor in neighbours:
                if neighbor.pos in closed:
                    continue
                step_cost = current.distance_to(neighbor)
                # `walked_dist` is computed into a local and only committed
                # when the visit is accepted. Pumpkin gets this for free
                # because `get_neighbors` hands out Node *copies* and a
                # rejected copy is dropped on the floor; here the neighbours
                # are the interned nodes themselves, so writing before the
                # accept test would let a long approach overwrite a node's
                # walked distance and then cut that node off from every
                # cheaper route - which silently truncates detours that a cow
                # would otherwise walk.
                walked_dist = current.walked_dist + step_cost
                if walked_dist >= follow_range:
                    continue
                tentative_g = g + step_cost + neighbor.cost_malus
                existing = open_g.get(neighbor.pos)
                if existing is not None and tentative_g >= existing:
                    continue

                neighbor.walked_dist = walked_dist
                neighbor.came_from = pos
                neighbor.g = tentative_g
                dist_to_target = neighbor.distance_to_pos(target_pos)
                target.update_best(dist_to_target, neighbor)
                neighbor.h = dist_to_target * TARGET_DISTANCE_MULTIPLIER
                neighbor.f = neighbor.g + neighbor.h
                open_g[neighbor.pos] = tentative_g
                tiebreak += 1
                push(heap, (neighbor.f, tiebreak, neighbor.pos))

        context_handle.spend_path_nodes(iterations)
        return self._build_path(target, target_pos, reached)

    def _build_path(self, target, target_pos, reached: bool) -> Optional[Path]:
        """Unwind `came_from` from the best node found into a `Path`.

        The walk has two exits beyond the obvious one - a parent that is not
        in the node table, and a parent that repeats a node already on the
        path - and both are reachable, because a node can be re-improved after
        the path through it was recorded. The repeat check is what keeps that
        from producing a path that visits the same cell twice; the
        not-in-table check is what keeps it from indexing a node that was
        never built.
        """
        best_node = target.best_node
        if best_node is None:
            return None

        nodes = self.evaluator.nodes
        path_nodes: List[Node] = [best_node]
        visited = {best_node.pos}
        current_pos = best_node.pos
        while True:
            node = nodes.get(current_pos)
            if node is None:
                break
            previous = node.came_from
            if previous is None or previous == current_pos:
                break
            if previous in visited:
                break
            previous_node = nodes.get(previous)
            if previous_node is None:
                break
            visited.add(previous)
            path_nodes.append(previous_node)
            current_pos = previous

        path_nodes.reverse()
        return Path(path_nodes, target_pos, reached)

    def can_reach_within(self, mob, destination: Tuple[float, float, float],
                         distance: float) -> bool:
        """Whether the mob can get close enough to act on `destination`.

        Vanilla's `canReachWithin` (`mod.rs:552`): either the search actually
        arrived, or it got within `distance` of where it wanted to be. The
        second clause is what makes "can I attack that mob" work when the
        target is on the other side of a fence - the path gets to the fence,
        which is close enough, and the goal proceeds.
        """
        path = self.create_path(mob, destination, 1)
        if path is None:
            return False
        return path.can_reach() or path.get_dist_to_target() <= distance