"""
A finished path, and the cursor walking along it.

Once `finder.py` has a node list, everything else is bookkeeping: a mob is at
index `i`, it wants index `i + 1`, and when it gets close enough the index
moves. `Path` owns that cursor and the few questions about it that the
navigation layer asks.

**`next_node_index` is the cursor, and it starts at 0 pointing at the mob's
own current node.** That is not an off-by-one that has been left in - it is
how the mob gets a "where am I going" answer on the first tick of a walk, when
the only node it has reached is the one it started on. The first node of a
path is always the node the mob is standing in, and it is consumed immediately.

**Two distances, and they answer different questions.** `dist_to_target` is
Manhattan distance from the path's *last* node to where the search was asked
to go, and it exists to answer "how close did we get" when the target turned
out to be unreachable. `remaining_distance` is the true walking length of the
nodes not yet consumed, and it is what `needs_new_path` scales its drift
tolerance by: a mob with four blocks left should abandon its path when the
goal moves, and a mob with sixteen blocks left should not. Using the straight
line to the target for that second question would make every long walk
recompute constantly.
"""

from __future__ import annotations

from typing import List, Optional

from entity.ai.pathfinding.node import Node, Pos

__all__ = ["Path"]


class Path:
    """A node list, a cursor into it, and whether it arrived.

    Vanilla's `Path` (`path.rs`). Of the things that are there for a caller
    which does not exist yet, exactly two survive: `truncate` and
    `replace_node`. Both are for the avoid-sun path trimming (§三-4 of the
    plan - zombie-only, no consumer this round), and they are kept because
    trimming a path you cannot truncate means changing this data structure
    when the mob arrives, rather than adding a goal. Everything else vanilla
    hangs off a `Path` and this server does not - no `sameAs` (nothing
    compares two paths), no `getNextEntityPos` (the navigation layer reads
    the node and asks `Node.center`), no `getRemainingNodes` (the only
    question asked of the tail is its length, which `get_remaining_distance`
    answers).
    """

    __slots__ = ("nodes", "next_node_index", "target", "dist_to_target",
                 "reached")

    def __init__(self, nodes: List[Node], target: Pos, reached: bool) -> None:
        self.nodes = nodes
        self.next_node_index = 0
        self.target = target
        self.reached = reached
        self.dist_to_target = (
            float("inf") if not nodes else self._manhattan(nodes[-1].pos, target))

    @staticmethod
    def _manhattan(a: Pos, b: Pos) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

    @classmethod
    def empty(cls, target: Pos) -> "Path":
        return cls([], target, False)

    # ----- the cursor -----

    def advance(self) -> None:
        self.next_node_index += 1

    def is_done(self) -> bool:
        return self.next_node_index >= len(self.nodes)

    def node_count(self) -> int:
        return len(self.nodes)

    def get_node(self, index: int) -> Optional[Node]:
        if 0 <= index < len(self.nodes):
            return self.nodes[index]
        return None

    def get_next_node(self) -> Optional[Node]:
        return self.get_node(self.next_node_index)

    def get_next_node_pos(self) -> Optional[Pos]:
        node = self.get_next_node()
        return None if node is None else node.pos

    # ----- distances -----

    def get_remaining_distance(self) -> float:
        """Walking length of every node not yet reached.

        Zero when the path is finished, which is what `needs_new_path` wants:
        a mob at the end of its path should recompute the moment the goal
        moves at all.
        """
        if self.next_node_index >= len(self.nodes):
            return 0.0
        total = 0.0
        for i in range(self.next_node_index + 1, len(self.nodes)):
            total += self.nodes[i - 1].distance_to(self.nodes[i])
        return total

    def get_dist_to_target(self) -> float:
        """Manhattan distance from the path's last node to the requested target."""
        return self.dist_to_target

    def can_reach(self) -> bool:
        return self.reached

    def get_target(self) -> Pos:
        return self.target

    def is_valid(self) -> bool:
        return bool(self.nodes)

    # ----- editing -----

    def truncate(self, index: int) -> None:
        """Cut the path short after `index` nodes.

        Used by the avoid-sun trimming, which stops a zombie as soon as it is
        back in daylight. With no sun-avoiding mob this round there is no
        caller, and it is ported rather than left out so that adding one is
        adding a goal rather than adding a data structure.
        """
        if index < len(self.nodes):
            del self.nodes[index:]
            if self.next_node_index > len(self.nodes):
                self.next_node_index = len(self.nodes)

    def replace_node(self, index: int, replacement: Node) -> None:
        if 0 <= index < len(self.nodes):
            self.nodes[index] = replacement

    def __len__(self) -> int:
        return len(self.nodes)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (f"<Path {len(self.nodes)} nodes at {self.next_node_index} "
                f"reached={self.reached}>")