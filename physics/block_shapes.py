"""Block geometry for the physics engine: state id -> collision AABB.

    from physics import block_shapes

    shapes = block_shapes()
    shapes.ids[shapes.offsets[state]:shapes.offsets[state + 1]]   # kernel view
    shapes.boxes(state)                                           # test view

The engine needs three things the feature stage never asked for, all of them
already parsed in `world.vanilla.features.BlockFacts`:

* **where** a state's collision boxes are, not just whether it has any.
  `blocks_motion` collapses 683 AABBs into one bool; a slab, a fence and a
  full cube are only distinguishable by their boxes.
* the same thing as **flat arrays**, because the hot path is an `njit` kernel
  that must not touch a Python object per block look-up.
* the three per-block material values (`slipperiness`, `velocity_multiplier`,
  `jump_velocity_multiplier`) expanded to per-state, so the round that adds
  friction does not have to reopen `assets/blocks.json`.

Nothing here re-reads the 14 MB payload: every array is an alias of the one
`BlockFacts` built, so there is exactly one parse and one truth per process.

Deliberate limits of this round (the behaviour is absent, not faked):

* friction, block-material effects and jumping are **not implemented**; the
  material columns are carried but unread.
* a state id outside the vanilla table -- a plugin block registered after the
  table was built -- has no boxes and reads as air, matching
  `BlockFacts.blocks_motion` and `is_air`. It is not "solid by default".
* liquids have no collision boxes in this table, which is correct (`water` and
  `lava` collide with nothing); swimming and buoyancy are separate work.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from world.vanilla.features import BlockFacts, block_facts

__all__ = ["BlockShapes", "block_shapes"]


class BlockShapes:
    """A state id's collision boxes, as arrays an `njit` kernel can index.

    Every attribute **aliases** `BlockFacts`' own arrays rather than copying
    them (52511 shape references and 29671 states would otherwise be stored
    twice, and the second copy is the one that goes stale).

    The kernel-facing layout is `offsets`/`ids`:

        shapes.ids[shapes.offsets[state] : shapes.offsets[state + 1]]

    which is the row numbers of `min_xyz`/`max_xyz` for that state, in payload
    order. `offsets` has a sentinel entry at the end, so "how many boxes" is a
    subtraction and never a branch.
    """

    __slots__ = ("offsets", "counts", "ids", "min_xyz", "max_xyz",
                 "slipperiness", "velocity_multiplier",
                 "jump_velocity_multiplier", "state_count", "shape_count",
                 "_facts")

    def __init__(self, facts: BlockFacts) -> None:
        self._facts = facts
        #: `int64[state_count + 1]`, aliased from `BlockFacts`.
        self.offsets = facts.shape_offsets
        #: `int64[state_count]`, derived from `offsets` (`counts == diff`).
        self.counts = np.diff(facts.shape_offsets)
        #: `int32[total shape references]` -> row of `min_xyz`/`max_xyz`.
        self.ids = facts.shape_ids
        #: `float64[n_shapes, 3]`, block-local, `0.0 .. 1.0`.
        self.min_xyz = facts.shape_min
        self.max_xyz = facts.shape_max
        #: `float64[state_count]`, per state; unread this round (no friction).
        self.slipperiness = facts.block_slipperiness
        self.velocity_multiplier = facts.block_velocity_multiplier
        self.jump_velocity_multiplier = facts.block_jump_velocity_multiplier
        self.state_count = int(facts.shape_offsets.shape[0] - 1)
        self.shape_count = int(facts.shape_min.shape[0])

    # ----- lookups -----

    def has_collision(self, state_id: int) -> bool:
        """Whether a state blocks movement at all (`blocks_motion`)."""
        return self.count(state_id) > 0

    def count(self, state_id: int) -> int:
        if not (0 <= state_id < self.state_count):
            return 0
        return int(self.offsets[state_id + 1] - self.offsets[state_id])

    def shape_ids(self, state_id: int) -> Tuple[int, ...]:
        """Row numbers into `min_xyz`/`max_xyz`; `()` when the state is air."""
        if not (0 <= state_id < self.state_count):
            return ()
        start = int(self.offsets[state_id])
        end = int(self.offsets[state_id + 1])
        return tuple(int(shape) for shape in self.ids[start:end])

    def boxes(self, state_id: int) -> Tuple[Tuple[float, ...], ...]:
        """A state's boxes as `(min_x, min_y, min_z, max_x, max_y, max_z)`.

        Block-local: a full cube is `(0, 0, 0, 1, 1, 1)`. The block position is
        added by the caller. This is the readable form -- the kernel reads
        `min_xyz`/`max_xyz` directly.
        """
        return tuple(
            (float(self.min_xyz[shape, 0]), float(self.min_xyz[shape, 1]),
             float(self.min_xyz[shape, 2]), float(self.max_xyz[shape, 0]),
             float(self.max_xyz[shape, 1]), float(self.max_xyz[shape, 2]))
            for shape in self.shape_ids(state_id)
        )

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (f"<BlockShapes states={self.state_count} "
                f"shapes={self.shape_count} refs={int(self.ids.shape[0])}>")


_SHAPES: Optional[BlockShapes] = None


def block_shapes(refresh: bool = False) -> BlockShapes:
    """The shared `BlockShapes`. Building it reuses `BlockFacts`, which is already cached."""
    global _SHAPES
    if _SHAPES is None or refresh:
        _SHAPES = BlockShapes(block_facts(refresh=refresh))
    return _SHAPES
