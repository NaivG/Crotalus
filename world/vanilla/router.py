"""
The noise router: the fifteen named functions that decide where blocks go.

`assets/density_function.json` stores, per generator preset, the fully resolved
router -- every `minecraft:` reference already inlined, so each slot is a
self-contained tree. This module turns one preset into something you can ask
questions of.

    from world.vanilla.router import NoiseRouter

    router = NoiseRouter.from_assets("overworld", seed=1234)
    router.begin_chunk(chunk_x=0, chunk_z=0)
    density = router.final_density(x, y, z)          # > 0 means solid
    surface = router.preliminary_surface_level(x, z)

Reading a slot
--------------
The slot names in the dump are Pumpkin's parsed-struct names (`finalDensity`,
`barrierNoise`), not vanilla's (`final_density`, `barrier`). `NOISE_ROUTER_SLOTS`
in `registry/worldgen.py` is the project's mapping between them; this module
exposes the dump's names and adds the vanilla ones as aliases so either
spelling works.

Which slots matter, and to whom
--------------------------------
`final_density` is the one that decides terrain: positive is stone, negative is
air, and the surface is wherever it crosses zero. `barrier` floors the world
(you cannot fall out), `fluid_level_*` and `lava` drive aquifers, the six
climate slots pick biomes, and `vein_*` place ore veins. `depth` and `factor`
have no geometric effect on their own -- they are inputs the `spline`s inside
`final_density` shape the terrain with.

Chunk lifecycle
---------------
`FlatCache` nodes are valid for one chunk's generation pass and no longer.
`begin_chunk` invalidates them. Forgetting to call it is a subtle and *quiet*
bug: the same column returns its first sampled value for the rest of the
session, so terrain gets smeared along Z in a way that looks like a noise bug.
`evaluate_column` and `evaluate` both call it; direct slot access does not, so
if you are driving the slots by hand, call `begin_chunk` yourself.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from world.vanilla.density import (
    Context,
    Node,
    cell_size_for,
    collect_flat_caches,
    load_parameters,
    parse,
)

F = np.float32

#: Dump slot name -> vanilla router slot name. Same mapping, same order, as
#: `NOISE_ROUTER_SLOTS` in `registry/worldgen.py` (which reads the other way).
VANILLA_SLOT_NAMES: Dict[str, str] = {
    "barrierNoise": "barrier",
    "fluidLevelFloodednessNoise": "fluid_level_floodedness",
    "fluidLevelSpreadNoise": "fluid_level_spread",
    "lavaNoise": "lava",
    "temperature": "temperature",
    "vegetation": "vegetation",
    "continents": "continents",
    "erosion": "erosion",
    "depth": "depth",
    "ridges": "ridges",
    "initialDensityWithoutJaggedness": "preliminary_surface_level",
    "finalDensity": "final_density",
    "veinToggle": "vein_toggle",
    "veinRidged": "vein_ridged",
    "veinGap": "vein_gap",
}


def _assets_path(name: str) -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, "assets", name)


class NoiseRouter:
    """One generator preset's router, parsed and ready to sample.

    Parsing is not cheap -- `final_density` in the overworld is a few thousand
    nodes -- and it is entirely seed-dependent, so build one per preset per
    world and keep it. Sampling, by contrast, is cheap.
    """

    __slots__ = ("preset", "seed", "context", "_slots", "_aliases", "_caches", "_chunk")

    def __init__(self, preset: str, seed: int, trees: Dict[str, dict],
                 parameters: Optional[Dict[str, dict]] = None) -> None:
        self.preset = preset
        self.seed = seed
        self.context = Context(seed, parameters if parameters is not None else load_parameters())
        self._chunk: Optional[tuple] = None

        self._slots: Dict[str, Node] = {}
        self._caches: List[Node] = []
        for slot, tree in trees.items():
            node = parse(tree, self.context, cell_size_for(preset, slot))
            self._slots[slot] = node
            self._caches.extend(collect_flat_caches(node))

        # Both spellings resolve to the same node, so a caller can use whichever
        # the surrounding code already uses.
        self._aliases = dict(self._slots)
        for dump_name, vanilla_name in VANILLA_SLOT_NAMES.items():
            if dump_name in self._slots:
                self._aliases.setdefault(vanilla_name, self._slots[dump_name])

    @classmethod
    def from_assets(cls, preset: str = "overworld", seed: int = 0,
                    density_function_path: Optional[str] = None,
                    parameters: Optional[Dict[str, dict]] = None) -> "NoiseRouter":
        """Build a router from `assets/density_function.json`."""
        path = density_function_path or _assets_path("density_function.json")
        with open(path, encoding="utf-8") as handle:
            routers = json.load(handle)
        if preset not in routers:
            known = ", ".join(sorted(routers))
            raise KeyError(f"unknown generator preset {preset!r}; have: {known}")
        return cls(preset, seed, routers[preset], parameters)

    # -- chunk lifecycle ---------------------------------------------------

    def begin_chunk(self, chunk_x: int, chunk_z: int) -> None:
        """Invalidate every `FlatCache`. Call once per chunk."""
        token = (chunk_x, chunk_z)
        if token == self._chunk:
            return
        self._chunk = token
        for cache in self._caches:
            cache.begin_context(token)

    # -- sampling ----------------------------------------------------------

    def slot(self, name: str) -> Node:
        """The parsed tree for a slot, under either spelling."""
        try:
            return self._aliases[name]
        except KeyError:
            known = ", ".join(sorted(self._slots))
            raise KeyError(f"unknown router slot {name!r}; have: {known}") from None

    def has(self, name: str) -> bool:
        return name in self._aliases

    def __getitem__(self, name: str) -> Node:
        return self.slot(name)

    def evaluate(self, name: str, x: int, y: int, z: int) -> np.float32:
        """Sample one slot at one position. Does *not* manage chunk state."""
        return self.slot(name).eval(self.context, x, y, z)

    def evaluate_array(self, name: str, xs, ys, zs) -> np.ndarray:
        """Sample one slot at many positions at once. No chunk-state management.

        `xs`, `ys`, `zs` are equal-length sequences of block coordinates; the
        result is a float32 array. This is the same computation as `evaluate`,
        point for point (see `tools/check_density_vectorized.py`), batched so
        the shared subtrees parse-time hash-consing collapsed are evaluated
        once per batch instead of once per visit.
        """
        node = self.slot(name)
        xs = np.asarray(xs, dtype=np.int64)
        ys = np.asarray(ys, dtype=np.int64)
        zs = np.asarray(zs, dtype=np.int64)
        return node.eval_array(self.context, xs, ys, zs)

    def evaluate_column(self, name: str, x: int, z: int, y_min: int, y_max: int) -> List[np.float32]:
        """Sample one slot down a column -- the shape terrain generation wants.

        Scalar loop on purpose: at one column's few hundred points the batch
        path's per-node numpy overhead is not amortised (see `surface_height`).
        Use `fill_chunk` for volumes.
        """
        node = self.slot(name)
        return [node.eval(self.context, x, y, z) for y in range(y_min, y_max)]

    def __iter__(self) -> Iterator[str]:
        return iter(self._slots)

    def __contains__(self, name: str) -> bool:
        return name in self._aliases

    # -- convenience -------------------------------------------------------

    def final_density(self, x: int, y: int, z: int) -> np.float32:
        """Positive means stone. This is the terrain function."""
        return self.evaluate("finalDensity", x, y, z)

    def is_solid(self, x: int, y: int, z: int) -> bool:
        return self.final_density(x, y, z) > 0

    def surface_height(self, x: int, z: int, min_y: int, max_y: int) -> Optional[int]:
        """Highest Y where `final_density` is positive, scanning downward.

        Returns None if the whole column is empty. Scans from the top because
        the density is not monotonic in Y -- overhangs and cave roofs have air
        *below* them, and a bottom-up scan reports those instead of the ground.

        Scalar scan: one column is too few points for `evaluate_array` to pay
        for itself. For a whole chunk use `surface_heights`.
        """
        node = self.slot("finalDensity")
        for y in range(max_y, min_y, -1):
            if node.eval(self.context, x, y, z) > 0:
                return y
        return None

    def fill_chunk(self, chunk_x: int, chunk_z: int,
                   y_min: int = -64, y_max: int = 320,
                   name: str = "finalDensity") -> np.ndarray:
        """The density volume for one chunk, shape (y_max - y_min, 16, 16).

        Layout is [y, z, x] -- the same axis order `world/generator.py` fills
        block arrays in. Y runs `y_min` (bottom) to `y_max` (exclusive).
        Manages chunk state: calls `begin_chunk` first, so `FlatCache` nodes
        are invalidated exactly once per volume.
        """
        self.begin_chunk(chunk_x, chunk_z)
        y = np.arange(y_min, y_max, dtype=np.int64)
        z = np.arange(chunk_z * 16, chunk_z * 16 + 16, dtype=np.int64)
        x = np.arange(chunk_x * 16, chunk_x * 16 + 16, dtype=np.int64)
        grid_y, grid_z, grid_x = np.meshgrid(y, z, x, indexing="ij")
        values = self.evaluate_array(name, grid_x, grid_y, grid_z)
        return values.reshape(y.size, 16, 16)

    def surface_heights(self, chunk_x: int, chunk_z: int,
                        min_y: int = -64, max_y: int = 320) -> np.ndarray:
        """Per-column surface height for one chunk: (16, 16) int32, [z, x].

        The batched form of `surface_height`: the highest Y in
        (min_y, max_y] where `final_density` is positive, or `min_y` where the
        column is empty (`min_y` is never inside the scanned range, so it
        cannot collide with a real height). Manages chunk state.
        """
        self.begin_chunk(chunk_x, chunk_z)
        y = np.arange(min_y + 1, max_y + 1, dtype=np.int64)
        z = np.arange(chunk_z * 16, chunk_z * 16 + 16, dtype=np.int64)
        x = np.arange(chunk_x * 16, chunk_x * 16 + 16, dtype=np.int64)
        grid_y, grid_z, grid_x = np.meshgrid(y, z, x, indexing="ij")
        column = self.evaluate_array("finalDensity", grid_x, grid_y, grid_z)
        column = column.reshape(y.size, 16, 16)
        # Last positive along Y per column: where a row is positive keep its
        # index, else -1, then take the max -- -1 survives as "empty column".
        positive = column > 0
        row_idx = np.where(positive, np.arange(y.size)[:, None, None], -1).max(axis=0)
        heights = np.where(row_idx >= 0, y[np.maximum(row_idx, 0)], min_y)
        return heights.astype(np.int32)

    def bounds(self) -> Dict[str, Tuple[np.float32, np.float32]]:
        """Each slot's declared [min, max] -- useful for sanity-checking scale."""
        return {name: (node.min_value, node.max_value) for name, node in self._slots.items()}

    def __repr__(self) -> str:
        return f"<NoiseRouter {self.preset!r} seed={self.seed} slots={len(self._slots)}>"


__all__ = ["NoiseRouter", "VANILLA_SLOT_NAMES"]
