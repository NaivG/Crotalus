"""
The node type, the malus table, and the search bookkeeping.

A path is a graph search over 1x1x1 blocks. This file holds everything about
that graph that is not "how to look at a block": what a node *is*, what kinds
of block there are, and what each kind costs. `context.py` answers the block
question, `evaluator.py` answers the "which nodes may I step to" question, and
`finder.py` runs the search over what is here.

**`PathType` is vanilla's `PathType` and the malus table is its
`getMalus`.** The idea - and it is the whole cost model - is that every block
gets one float. Negative means "no mob may be here", zero means "free", and
positive is a price in "extra blocks of walking". A cow pays 0 to cross grass
and 8 to cross shallow water, so a path through the grass detouring 8 blocks
around the pond wins over the one wading through it. That is the whole
mechanism: there is no separate "avoid water" rule, only a number, and the A*
in `finder.py` balances it against distance by itself.

**The per-mob override exists because malus is not a property of a block.**
`MobData` carries a 27-slot table of its own that replaces the class default
for chosen types - a zombie sets `DamageFire` to -1 (never walk through fire,
even though the block class only calls it expensive) and `DangerFire` to 16
(walk around the fire rather than close to it). That distinction between
"expensive" and "forbidden" cannot live in a per-block table, which is why
`MobData` exists at all.

**Two fields of Pumpkin's `Node` are deliberately gone**: `hash` (an i32 with
negative-shift overflow guards, `node.rs:77` - it bought Pumpkin an
integer-keyed map and buys Python nothing) and `heap_idx` (the in-place heap
index behind `binary_heap.rs`'s decrease-key). `finder.py` uses `heapq` with
lazy deletion instead, which is a dictionary of positions and a heap of
`(f, tiebreak, pos)` triples. The `closed` flag survives because it means
"this exact node was expanded", which is not the same question as "a node is
known at this position".

**Nodes are mutable and interned by position.** `BaseNodeEvaluator.get_node`
returns the same object for the same block every time a search touches it, so
`g`/`f`/`came_from` are written in place. The alternative - copying a node per
improvement, as vanilla's `PathNode` does - is what makes `came_from` a pointer
into a pool rather than a coordinate; the coordinate is used here because it
serialises through `Path` with no other structure attached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
# from math import sqrt
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "MALUS",
    "Node",
    "PATH_TYPE_COUNT",
    "PathType",
    "Target",
]

#: A block position. Nodes are 1x1x1 blocks, so this is an integer triple and
#: nothing else - see the module docstring on why this is the key and not a
#: packed integer.
Pos = Tuple[int, int, int]


class PathType(IntEnum):
    """Vanilla's `PathType`, all 27 of them.

    `IntEnum` rather than a bare `Enum` for two reasons that are both load
    bearing. The malus table below is indexed by member value, so the numbers
    have to be integers. And `WalkNodeEvaluator.get_path_type_within_mob_bb`
    sorts and de-duplicates the types it collected from a mob's bounding box
    before deciding which one wins (`walk_node_evaluator.rs:415`) - that
    ordering is part of the algorithm, and it needs the declaration order to
    be *a* total order, which it does not have to be correct but must be
    stable. Declaration order is what vanilla's declaration order gives.
    """

    BLOCKED = 0
    OPEN = 1
    WALKABLE = 2
    WALKABLE_DOOR = 3
    TRAPDOOR = 4
    POWDER_SNOW = 5
    DANGER_POWDER_SNOW = 6
    FENCE = 7
    LAVA = 8
    WATER = 9
    WATER_BORDER = 10
    RAIL = 11
    UNPASSABLE_RAIL = 12
    DANGER_FIRE = 13
    DAMAGE_FIRE = 14
    DANGER_OTHER = 15
    DAMAGE_OTHER = 16
    DOOR_OPEN = 17
    DOOR_WOOD_CLOSED = 18
    DOOR_IRON_CLOSED = 19
    BREACH = 20
    LEAVES = 21
    STICKY_HONEY = 22
    COCOA = 23
    DAMAGE_CAUTIOUS = 24
    DANGER_TRAPDOOR = 25
    BIG_MOBS_CLOSE_TO_DANGER = 26


#: How many types there are, which is the length of both tables below and the
#: width of `MobData.path_type_malus`.
PATH_TYPE_COUNT = 27

#: Vanilla's `PathType.getMalus`, by member (`node.rs:407`).
#:
#: Negative is a wall, zero is free, positive is a price in blocks of walking.
#: The two entries worth reading twice are `WATER` at 8 and `LEAVES` at -1: a
#: cow *may* wade into shallow water and is charged for it, but may not walk
#: into leaves at any price. The second is not a bug and not a gap - vanilla's
#: walker refuses leaves for the same reason a person would not choose to climb
#: a hedge. `Lava` at -1 and `DAMAGE_FIRE` at 16 are the two classes a mob
#: overrides per-mob rather than arguing with here.
MALUS: Tuple[float, ...] = (
    -1.0,   # BLOCKED
    0.0,    # OPEN
    0.0,    # WALKABLE
    0.0,    # WALKABLE_DOOR
    0.0,    # TRAPDOOR
    -1.0,   # POWDER_SNOW
    0.0,    # DANGER_POWDER_SNOW
    -1.0,   # FENCE
    -1.0,   # LAVA
    8.0,    # WATER
    8.0,    # WATER_BORDER
    0.0,    # RAIL
    -1.0,   # UNPASSABLE_RAIL
    8.0,    # DANGER_FIRE
    16.0,   # DAMAGE_FIRE
    8.0,    # DANGER_OTHER
    -1.0,   # DAMAGE_OTHER
    0.0,    # DOOR_OPEN
    -1.0,   # DOOR_WOOD_CLOSED
    -1.0,   # DOOR_IRON_CLOSED
    4.0,    # BREACH
    -1.0,   # LEAVES
    8.0,    # STICKY_HONEY
    0.0,    # COCOA
    0.0,    # DAMAGE_CAUTIOUS
    0.0,    # DANGER_TRAPDOOR
    4.0,    # BIG_MOBS_CLOSE_TO_DANGER
)

def get_malus(path_type: PathType) -> float:
    """The class-default price of a path type.

    The only classifier this module exports. Vanilla's `PathType` has six
    predicates on top of it (`isBlocked`, `isWater`, `isDoor`, `isDangerous`,
    `hasPartialCollision`, `isPassable`) and none of them are carried here,
    because every question they answer is answered by the evaluator on the
    *mob's* reading of a cell rather than on the block's - see
    `WalkNodeEvaluator.does_block_have_partial_collision` for the one that
    looks most like a direct translation and is deliberately not one.
    """
    return MALUS[int(path_type)]


@dataclass
class Node:
    """One 1x1x1 block the search may stand in.

    Mutable and shared: `WalkNodeEvaluator` interns one per position per
    computation, so `g`, `f`, `came_from` and `closed` are written in place
    and `finder.py` reads them back. Equality is over every field, matching
    Pumpkin's derived `PartialEq` (`node.rs:16`) - `Path.same_as_path` compares
    node lists and would otherwise call two different paths equal because they
    happened to pass through the same blocks.
    """

    pos: Pos
    g: float = 0.0
    """Cost from the start: walked distance plus every malus paid so far."""
    h: float = 0.0
    """Remaining estimate, already multiplied by `TARGET_DISTANCE_MULTIPLIER`."""
    f: float = 0.0
    """`g + h`. The only field the open set sorts on."""
    came_from: Optional[Pos] = None
    """The node to walk back to. A position, not a reference - see the module
    docstring on why."""
    closed: bool = False
    walked_dist: float = 0.0
    """Blocks actually walked, without malus. Bounds the search radius."""
    cost_malus: float = 0.0
    """The most expensive thing inside this node's bounding box."""
    path_type: PathType = PathType.BLOCKED

    def distance_to(self, other: "Node") -> float:
        """Euclidean distance between two nodes' block positions."""
        dx = other.pos[0] - self.pos[0]
        dy = other.pos[1] - self.pos[1]
        dz = other.pos[2] - self.pos[2]
        return np.sqrt(dx * dx + dy * dy + dz * dz)

    def distance_to_pos(self, pos: Pos) -> float:
        dx = pos[0] - self.pos[0]
        dy = pos[1] - self.pos[1]
        dz = pos[2] - self.pos[2]
        return np.sqrt(dx * dx + dy * dy + dz * dz)

    def manhattan_to_pos(self, pos: Pos) -> int:
        return (abs(self.pos[0] - pos[0]) + abs(self.pos[1] - pos[1])
                + abs(self.pos[2] - pos[2]))

    @property
    def center(self) -> Tuple[float, float, float]:
        """Where a mob standing in this node is - the block centre, `(x+.5, y, z+.5)`."""
        return (self.pos[0] + 0.5, float(self.pos[1]), self.pos[2] + 0.5)


@dataclass
class Target:
    """What the search is looking for, and the best it has found so far.

    `best_node` is the load-bearing field and it is the whole partial-path
    mechanism. Vanilla does not return "failed" when the budget runs out or
    the target is walled off; it returns the node that got *closest* and
    unwinds the parent pointers from there. A mob walks as far as it can and
    stops, which is the right answer for a goal that only asked to "get over
    there" - it is not the same as failing, and it is not a fallback bolted on
    afterwards. The search is allowed to end at any node at any time and this
    is where it says which one.
    """

    node: Node
    best_heuristic: float = float("inf")
    best_node: Optional[Node] = None
    reached: bool = False

    def update_best(self, heuristic: float, node: Node) -> None:
        if heuristic < self.best_heuristic:
            self.best_heuristic = heuristic
            self.best_node = node

    def set_reached(self) -> None:
        self.reached = True

    def is_reached(self) -> bool:
        return self.reached