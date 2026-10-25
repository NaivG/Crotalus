"""
Vanilla's mob pathfinder, in the order the pieces call each other.

    PathNavigation  - "get there": owns the walk, picks waypoints, gives up
        |
        +-> PathFinder  - the A* search: budget, heuristic, partial paths
                |
                +-> WalkNodeEvaluator - "may I stand here": corners, falls, jumps
                        |
                        +-> PathfindingContext - the only thing that reads blocks
                                |
                                +-> Node / PathType / Target - the graph and its costs

**The dependency arrows all point down.** Nothing below knows anything is
above it: the context does not know a search is running, the evaluator does not
know a mob is walking, and `Node` is a dataclass that has never heard of a
cow. That is what makes the pieces individually testable - a corner rule can
be exercised against a hand-built world with no mob, a malus table against no
world at all - and it is why `PathfindingContext` being the single reader of
`AIContext.state` is a structural property rather than a convention.

**One context per search, one evaluator per mob.** See the two modules'
docstrings: the caches in both are only valid against the world they were
built from, so their lifetimes are tied to a single `compute_path` call.

`node.py` and `path.py` are the two leaves and are the ones to read first if
you are new to this - `PathType` and its malus table are the entire cost
model, and `Path` is a list with a cursor.
"""

from entity.ai.pathfinding.context import PathfindingContext
from entity.ai.pathfinding.evaluator import MobData, WalkNodeEvaluator
from entity.ai.pathfinding.finder import PathFinder
from entity.ai.pathfinding.navigation import PathNavigation
from entity.ai.pathfinding.node import MALUS, Node, PathType, Target
from entity.ai.pathfinding.path import Path

__all__ = [
    "MALUS",
    "MobData",
    "Node",
    "Path",
    "PathFinder",
    "PathNavigation",
    "PathType",
    "PathfindingContext",
    "Target",
    "WalkNodeEvaluator",
]