"""
The multi-noise biome source: six climate numbers, then a nearest-neighbour tree.

Terrain decides *where* blocks are; this decides *what* they are. Vanilla picks a
biome by sampling six climate values at a quart position, quantising each to an
integer, and walking an R-tree of parameter boxes looking for the closest leaf.
Nothing here is approximate: the tree, the boxes and the quantisation are all
read out of `assets/`, because a biome one step off moves the surface rule that
dresses the terrain and the feature list that decorates it.

    from world.vanilla import biome_source

    source = biome_source("overworld", seed=0)
    source.biome_at_quart(21, 2, 21)            # -> "minecraft:forest"
    source.fill_volume(0, 0, -64, 384)          # -> (96, 4, 4) ids for chunk (0,0)

Four things that are easy to get wrong
--------------------------------------

**The sampler takes quart coordinates, not block coordinates.** Vanilla's
`MultiNoiseSampler.sample(quartX, quartY, quartZ)` shifts each argument left by
two before sampling the router. Passing block coordinates works, raising no
error, and puts every biome four times as far apart as it should be.

**The climate values are floats that get quantised to ints.** `quantize_coord`
multiplies by 10000 and truncates *toward zero* in f32. The tree's boxes are
integers in that same space, so the truncation is part of the answer, not
rounding noise: `-0.00005` and `0.00005` are 0 and 0, but `-0.00015` is -1 and
`0.00015` is 1, and a box boundary sits at exactly those integers.

**The climate slots have no `Interpolated` wrapper, so both density phases agree
for them.** The dump wraps only `finalDensity`, `veinToggle` and `veinRidged`
(see `tools/check_biomes.py`, which asserts it), so the proto-phase precept that
governs the terrain router simply does not bite here. The sampler is built in the
proto phase anyway - it is what vanilla's `MultiNoiseSampler` does, and skipping
the interpolation machinery is free - but the value would be identical either
way, and claiming otherwise would send the next reader hunting a difference that
is not there.

**The tree search is stateful.** Vanilla caches the last leaf on the tree and
seeds the next search's distance bound with it, so a tie goes to whichever leaf
was found first *in that history*. The cache is kept here for the same reason it
exists there - the golden data was captured with it - but `BiomeTree.get` can be
called with no candidate for a history-independent answer, which is what the
golden-data check needs. Ties are vanishingly rare in real climate data and
invisible to it, so `tools/check_biomes.py` layer 5 pins the tie-break rules on
trees built to tie.

Which biomes, and what they carry
---------------------------------
`assets/multi_noise_biome_tree.json` names the biome at every leaf but says
nothing about it. `assets/biome.json` holds the rest: the numeric id the wire
uses, the carvers that run in it and the feature lists for each decoration step.
`BiomeTable` joins the two so the carver and feature layers have one place to
ask.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from world.vanilla.density import parse as parse_density
from world.vanilla.density import Context as DensityContext
from world.vanilla.density import load_parameters as load_noise_parameters
from world.vanilla.router import NoiseRouter

try:  # pragma: no cover - numba is a declared dependency
    from numba import njit

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def wrap(func):
            return func

        return wrap


#: Vanilla's `Climate.QUANTIZATION_FACTOR`. The tree's boxes are in this space.
QUANTIZATION_FACTOR = np.float32(10000.0)

#: Router slots the six climate values come from, in the order vanilla stores
#: them (`TargetPoint.to_parameter_array`): temperature, humidity,
#: continentalness, erosion, depth, weirdness. The dump spells them
#: `vegetation` for humidity and `ridges` for weirdness.
CLIMATE_SLOTS: Tuple[str, ...] = (
    "temperature", "vegetation", "continents", "erosion", "depth", "ridges",
)

#: The same six under vanilla's names, which is what a `Climate` tuple is indexed by.
CLIMATE_NAMES: Tuple[str, ...] = (
    "temperature", "humidity", "continentalness", "erosion", "depth", "weirdness",
)

#: The parameter-list boxes use seven axes; the last is the entry's `offset`,
#: which is 0 for every biome in the overworld and nether lists. A sampled point
#: is six numbers, so it is padded with a zero on the way into the tree - the
#: search compares like for like and a point is never shorter than a box.
PARAMETER_AXES = 7

#: Biomes per chunk along one axis (16 blocks / 4), and per section along Y.
BIOMES_PER_AXIS = 4


def _assets_path(name: str) -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, "assets", name)


def _point7(point) -> np.ndarray:
    """A climate point as the tree's seven axes, padding the offset with 0.

    Silently padding is the point: the offset axis is a constant, and a caller
    holding the six sampled values should not have to remember to append it. A
    seven-element input passes through untouched.
    """
    values = np.asarray(point, dtype=np.int64)
    if values.size == 6:
        padded = np.zeros(PARAMETER_AXES, dtype=np.int64)
        padded[:6] = values
        return padded
    if values.size != PARAMETER_AXES:
        raise ValueError(f"a climate point has 6 axes, got {values.size}")
    return values


# ==================== quantisation ====================

def quantize(coord) -> int:
    """`f32(coord) * 10000`, truncated toward zero - vanilla's `quantizeCoord`.

    The multiplication is done in f32 and *then* truncated, which is not the same
    as truncating in double: `0.00005 * 10000.0` is 0.5 exactly in double but
    0.49999997 in f32, and those truncate to 0 and 0 - but `-0.0001` is -1 either
    way while `0.00005 * 10000` at a hair above compiles to 1 in double.
    """
    return int(np.float32(coord) * QUANTIZATION_FACTOR)


def unquantize(coord: int) -> np.float32:
    """The inverse, for reading a box's bounds back as a climate value."""
    return np.float32(np.float32(coord) / QUANTIZATION_FACTOR)


# ==================== parameter boxes ====================

class Parameter:
    """One `ParameterRange`: an inclusive `[min, max]` box on one climate axis."""

    __slots__ = ("min", "max")

    def __init__(self, minimum: int, maximum: int) -> None:
        self.min = int(minimum)
        self.max = int(maximum)

    @classmethod
    def point(cls, value) -> "Parameter":
        quantized = quantize(value)
        return cls(quantized, quantized)

    def distance(self, target: int) -> int:
        """Blocks outside the box, 0 inside - vanilla's `Parameter.distance`."""
        above = target - self.max
        below = self.min - target
        if above > 0:
            return above
        if below > 0:
            return below
        return 0

    def __repr__(self) -> str:
        if self.min == self.max:
            return f"Parameter({self.min})"
        return f"Parameter({self.min}..{self.max})"


class ParameterPoint:
    """One entry of a multi-noise parameter list: six boxes plus an offset."""

    __slots__ = ("boxes", "offset")

    def __init__(self, boxes: Sequence[Parameter], offset: int = 0) -> None:
        self.boxes: Tuple[Parameter, ...] = tuple(boxes)
        self.offset = int(offset)

    def fitness(self, point: Sequence[int]) -> int:
        """Squared distance from `point`, offset included - vanilla's `fitness`."""
        total = self.offset * self.offset
        for box, value in zip(self.boxes, point):
            d = box.distance(value)
            total += d * d
        return total

    def __repr__(self) -> str:
        return f"ParameterPoint({', '.join(map(repr, self.boxes))}, offset={self.offset})"


# ==================== the tree ====================

class BiomeTree:
    """Vanilla's `Climate.RTree`, flattened for a numba kernel.

    The tree is a binary-ish hierarchy over parameter boxes: a branch tests its
    own box for pruning and then visits children in order, a leaf is a biome. The
    search returns the leaf with the smallest squared distance, where "distance"
    is how far the sampled point sits outside the box, summed over the axes.

    Parsing keeps the tree as arrays rather than objects because the search runs
    once per quart cell - 1536 times per chunk - and a Python object walk costs
    about 40x the numba kernel that replaces it. Both paths share these arrays,
    so they cannot disagree about the traversal; only about how long it takes.
    """

    __slots__ = ("params", "is_leaf", "child_off", "child_count", "children",
                 "biome_ids", "root", "names", "node_count")

    def __init__(self, payload: dict, biome_ids: Dict[str, int]) -> None:
        params: List[List[Tuple[int, int]]] = []
        is_leaf: List[int] = []
        child_off: List[int] = []
        child_count: List[int] = []
        children: List[int] = []
        biome: List[int] = []
        names: List[str] = []

        # Recursive build: the overworld tree is ~9000 nodes at depth 5, and a
        # deep tree from a plugin's parameter list would need an explicit stack
        # if Python's recursion limit ever bites. `children` holds only *direct*
        # children - the slots are reserved before recursing, because `add`
        # appends the child's own descendants to `children` before it returns,
        # and a range that swallowed them would make the walk re-expand whole
        # subtrees once per ancestor, multiplying node tests per point.
        def add(node: dict) -> int:
            index = len(params)
            boxes = [
                (int(box["min"]), int(box["max"]))
                for box in node["parameters"]
            ]
            params.append(boxes)
            child_off.append(len(children))
            kind = node.get("_type")
            if kind == "leaf":
                is_leaf.append(1)
                child_count.append(0)
                name = node.get("biome") or node.get("value") or ""
                names.append(name)
                resolved = biome_ids.get(name)
                if resolved is None:
                    raise KeyError(
                        f"biome tree leaf names {name!r}, which is not in biome.json"
                    )
                biome.append(resolved)
                return index
            if kind != "branch":
                raise ValueError(f"unknown biome tree node type {kind!r}")
            is_leaf.append(0)
            biome.append(-1)
            names.append("")
            kids = node.get("subTree") or ()
            child_count.append(len(kids))
            start = len(children)
            for _ in kids:
                children.append(-1)
            child_off[index] = start
            for i, child in enumerate(kids):
                children[start + i] = add(child)
            return index

        self.root = add(payload)
        self.node_count = len(params)
        self.params = np.asarray(params, dtype=np.int64).reshape(self.node_count, PARAMETER_AXES, 2)
        self.is_leaf = np.asarray(is_leaf, dtype=np.uint8)
        self.child_off = np.asarray(child_off, dtype=np.int64)
        self.child_count = np.asarray(child_count, dtype=np.int64)
        self.children = np.asarray(children, dtype=np.int64)
        self.biome_ids = np.asarray(biome, dtype=np.int32)
        self.names = tuple(names)

    @classmethod
    def from_assets(cls, source: str = "overworld",
                    biome_ids: Optional[Dict[str, int]] = None,
                    path: Optional[str] = None) -> "BiomeTree":
        """Build from `assets/multi_noise_biome_tree.json`."""
        with open(path or _assets_path("multi_noise_biome_tree.json"), encoding="utf-8") as handle:
            trees = json.load(handle)
        if source not in trees:
            known = ", ".join(sorted(trees))
            raise KeyError(f"unknown multi-noise source {source!r}; have: {known}")
        return cls(trees[source], biome_ids if biome_ids is not None else biome_ids_from_assets())

    def get_index(self, point: Sequence[int], candidate: int = -1) -> int:
        """Node index of the closest leaf, seeding the search with `candidate`.

        `candidate` is vanilla's `lastResult`: a node whose distance becomes the
        initial bound, so the search may return it unchanged. Pass -1 for a
        history-independent answer.
        """
        return _search(self.params, self.is_leaf, self.child_off, self.child_count,
                       self.children, _point7(point), self.root, int(candidate))

    def get(self, point: Sequence[int], candidate: Optional[int] = None) -> str:
        """Biome name at a quantized climate point."""
        index = self.get_index(point, -1 if candidate is None else candidate)
        return self.names[index]

    def get_many(self, points: np.ndarray, candidate: int = -1) -> np.ndarray:
        """Leaf indices for a batch of points, shape (n, 6) or (n, 7).

        Keeps the cache warm the way vanilla does: `lastResult` carries from one
        sample to the next, so row order is part of the answer at ties.
        """
        points = np.ascontiguousarray(points, dtype=np.int64)
        if points.ndim != 2 or points.shape[1] not in (6, PARAMETER_AXES):
            raise ValueError(f"points must be (n, 6) or (n, 7), got {points.shape}")
        if points.shape[1] == 6:
            padded = np.zeros((points.shape[0], PARAMETER_AXES), dtype=np.int64)
            padded[:, :6] = points
            points = padded
        out = np.empty(points.shape[0], dtype=np.int32)
        _search_many(self.params, self.is_leaf, self.child_off, self.child_count,
                     self.children, self.biome_ids, points, self.root, int(candidate), out)
        return out

    def __repr__(self) -> str:
        leaves = int(self.is_leaf.sum())
        return f"<BiomeTree nodes={self.node_count} leaves={leaves}>"


def _search_walk(params, is_leaf, child_off, child_count, children, point, root, candidate):
    """Closest leaf index, or `candidate` when nothing beats it. Vanilla's search.

    This is the *definition* of the answer: a depth-first walk in child order
    that prunes `dist >= best`, so a tie goes to the earlier-visited leaf and a
    `candidate` tie goes to the candidate. `_search_kernel` is this walk
    compiled; `tools/check_biomes.py` layer 5 pits the two against each other
    and against an independent spec on crafted ties.
    """
    best_dist = _node_distance(params, candidate, point) if candidate >= 0 else (1 << 62)
    best_node = candidate if candidate >= 0 else root
    stack = [root]
    while stack:
        node = stack.pop()
        dist = _node_distance(params, node, point)
        if dist >= best_dist:
            continue
        if is_leaf[node]:
            best_dist = dist
            best_node = node
            continue
        start = child_off[node]
        for i in range(child_count[node] - 1, -1, -1):
            stack.append(children[start + i])
    return best_node


def _node_distance(params, node, point) -> int:
    total = 0
    boxes = params[node]
    for axis in range(PARAMETER_AXES):
        low = boxes[axis, 0]
        high = boxes[axis, 1]
        above = point[axis] - high
        below = low - point[axis]
        if above > 0:
            total += above * above
        elif below > 0:
            total += below * below
    return total


@njit(cache=True, nogil=True)
def _search_kernel(params, is_leaf, child_off, child_count, children, point, root, candidate,
                   stack):
    """The same traversal as `_search_walk`, compiled.

    An explicit stack, pushed in reverse so children come off in order - the
    order decides ties, and ties decide biomes at borders, so it has to match the
    interpreted version exactly.

    `stack` is the caller's scratch buffer. Allocating it here instead costs one
    9112-entry `np.empty` per sampled position, which is 1536 allocations of
    73 KB for a single chunk and dominated the search outright.
    """
    big = np.int64(1) << np.int64(62)
    best_dist = big
    best_node = root
    if candidate >= 0:
        best_dist = _node_distance_kernel(params, candidate, point)
        best_node = candidate
    top = 0
    stack[0] = root
    top = 1
    while top > 0:
        top -= 1
        node = stack[top]
        dist = _node_distance_kernel(params, node, point)
        if dist >= best_dist:
            continue
        if is_leaf[node] == 1:
            best_dist = dist
            best_node = node
            continue
        start = child_off[node]
        count = child_count[node]
        for i in range(count - 1, -1, -1):
            stack[top] = children[start + i]
            top += 1
    return best_node


@njit(cache=True, nogil=True)
def _node_distance_kernel(params, node, point):
    total = np.int64(0)
    for axis in range(params.shape[1]):
        low = params[node, axis, 0]
        high = params[node, axis, 1]
        above = point[axis] - high
        below = low - point[axis]
        if above > 0:
            total += above * above
        elif below > 0:
            total += below * below
    return total


@njit(cache=True, nogil=True)
def _search_many_kernel(params, is_leaf, child_off, child_count, children, biome_ids,
                        points, root, candidate, out):
    """One batch, carrying `lastResult` from row to row exactly as vanilla does."""
    big = np.int64(1) << np.int64(62)
    stack = np.empty(params.shape[0], dtype=np.int64)
    last = candidate
    for row in range(points.shape[0]):
        point = points[row]
        best_dist = big
        best_node = root
        if last >= 0:
            best_dist = _node_distance_kernel(params, last, point)
            best_node = last
        top = 0
        stack[0] = root
        top = 1
        while top > 0:
            top -= 1
            node = stack[top]
            dist = _node_distance_kernel(params, node, point)
            if dist >= best_dist:
                continue
            if is_leaf[node] == 1:
                best_dist = dist
                best_node = node
                continue
            start = child_off[node]
            count = child_count[node]
            for i in range(count - 1, -1, -1):
                stack[top] = children[start + i]
                top += 1
        out[row] = best_node
        last = best_node
    return last


#: Dispatch: the numba kernel when it is installed, the interpreted walk
#: otherwise. `tools/check_biomes.py` layer 5 asserts the two agree, including
#: at crafted ties where any order-dependent shortcut would show up.


def _search(params, is_leaf, child_off, child_count, children, point, root, candidate):
    """Closest leaf index, or `candidate` when nothing beats it."""
    if _HAS_NUMBA:
        return _search_kernel(params, is_leaf, child_off, child_count, children,
                              point, root, candidate,
                              np.empty(params.shape[0], dtype=np.int64))
    return _search_walk(params, is_leaf, child_off, child_count, children,
                        point, root, candidate)


def _search_many(params, is_leaf, child_off, child_count, children, biome_ids,
                 points, root, candidate, out):
    if _HAS_NUMBA:
        return _search_many_kernel(params, is_leaf, child_off, child_count, children,
                                   biome_ids, points, root, candidate, out)
    last = candidate
    for row in range(points.shape[0]):
        last = _search_walk(params, is_leaf, child_off, child_count, children,
                            points[row], root, last)
        out[row] = last
    return last


# ==================== the climate sampler ====================

class ClimateSampler:
    """The six router slots, evaluated at quart positions and quantized.

    Built in the proto phase for parity with vanilla's `MultiNoiseSampler`. The
    six slots in `assets/density_function.json` contain no `Interpolated` node,
    so this is a statement of intent rather than a value difference - see the
    module docstring.
    """

    __slots__ = ("preset", "seed", "context", "nodes", "_flat_caches", "_chunk")

    def __init__(self, preset: str, seed: int, trees: Dict[str, dict],
                 parameters: Optional[Dict[str, dict]] = None) -> None:
        self.preset = preset
        self.seed = seed
        self.context = DensityContext(seed, parameters if parameters is not None else load_noise_parameters())
        self.nodes = tuple(
            parse_density(trees[slot], self.context, pass_through_wrappers=True)
            for slot in CLIMATE_SLOTS
        )
        self._flat_caches = []
        self._chunk: Optional[Tuple[int, int]] = None

    @classmethod
    def from_assets(cls, preset: str = "overworld", seed: int = 0,
                    density_function_path: Optional[str] = None) -> "ClimateSampler":
        path = density_function_path or _assets_path("density_function.json")
        with open(path, encoding="utf-8") as handle:
            routers = json.load(handle)
        if preset not in routers:
            known = ", ".join(sorted(routers))
            raise KeyError(f"unknown generator preset {preset!r}; have: {known}")
        return cls(preset, seed, routers[preset])

    @classmethod
    def from_router(cls, router: NoiseRouter) -> "ClimateSampler":
        """Reuse a router that is already parsed (it must be in the proto phase)."""
        sampler = cls.__new__(cls)
        sampler.preset = router.preset
        sampler.seed = router.seed
        sampler.context = router.context
        sampler.nodes = tuple(router.slot(slot) for slot in CLIMATE_SLOTS)
        sampler._flat_caches = []
        sampler._chunk = None
        return sampler

    def sample(self, quart_x: int, quart_y: int, quart_z: int) -> Tuple[int, ...]:
        """Six quantized climate values at one quart position."""
        x, y, z = quart_x << 2, quart_y << 2, quart_z << 2
        return tuple(
            quantize(node.eval(self.context, x, y, z)) for node in self.nodes
        )

    def sample_many(self, quart_xs, quart_ys, quart_zs) -> np.ndarray:
        """Batched `sample`: (n, 6) int64, one row per position.

        The shift to block coordinates happens once, on the arrays, so the batch
        path and the scalar path address the same points.
        """
        xs = np.asarray(quart_xs, dtype=np.int64) << 2
        ys = np.asarray(quart_ys, dtype=np.int64) << 2
        zs = np.asarray(quart_zs, dtype=np.int64) << 2
        out = np.empty((xs.size, len(CLIMATE_SLOTS)), dtype=np.int64)
        for column, node in enumerate(self.nodes):
            values = node.eval_array(self.context, xs, ys, zs)
            # `np.float32 * float32` keeps the product in f32, which is where
            # vanilla truncates. Doing it in double moves boundary samples.
            out[:, column] = (values * QUANTIZATION_FACTOR).astype(np.int64)
        return out

    def __repr__(self) -> str:
        return f"<ClimateSampler {self.preset!r} seed={self.seed}>"


# ==================== biome sources ====================

class MultiNoiseBiomeSource:
    """Vanilla's `MultiNoiseBiomeSource` for one preset."""

    __slots__ = ("preset", "seed", "tree", "sampler", "last_node", "_names", "_ids")

    def __init__(self, preset: str, seed: int, tree: BiomeTree,
                 sampler: Optional[ClimateSampler] = None) -> None:
        self.preset = preset
        self.seed = seed
        self.tree = tree
        self.sampler = sampler if sampler is not None else ClimateSampler.from_assets(preset, seed)
        self.last_node = -1
        self._names = tree.names
        self._ids = tree.biome_ids

    @classmethod
    def from_assets(cls, preset: str = "overworld", seed: int = 0,
                    biome_ids: Optional[Dict[str, int]] = None) -> "MultiNoiseBiomeSource":
        ids = biome_ids if biome_ids is not None else biome_ids_from_assets()
        # `large_biomes` and `amplified` share the overworld biome tree; only the
        # shape and the terrain splines differ (see `chunk_gen_settings.json`).
        tree_source = "nether" if preset == "nether" else "overworld"
        return cls(preset, seed, BiomeTree.from_assets(tree_source, ids))

    # -- scalar ------------------------------------------------------------

    def biome_index_at_quart(self, quart_x: int, quart_y: int, quart_z: int) -> int:
        """Leaf index. Advances the `lastResult` cache, like vanilla."""
        point = self.sampler.sample(quart_x, quart_y, quart_z)
        self.last_node = self.tree.get_index(point, self.last_node)
        return self.last_node

    def biome_at_quart(self, quart_x: int, quart_y: int, quart_z: int) -> str:
        return self._names[self.biome_index_at_quart(quart_x, quart_y, quart_z)]

    def biome_id_at_quart(self, quart_x: int, quart_y: int, quart_z: int) -> int:
        return int(self._ids[self.biome_index_at_quart(quart_x, quart_y, quart_z)])

    # -- batched -----------------------------------------------------------

    def fill_volume(self, chunk_x: int, chunk_z: int, min_y: int, height: int) -> np.ndarray:
        """Biome ids for one chunk: (height // 4, 4, 4), index `y*16 + z*4 + x`.

        Layout matches vanilla's biome container: quart cells, X fastest, then Z,
        then Y upward from `min_y`. The four columns are the chunk's own 16
        blocks, so `fill_volume` is self-contained - it does not read neighbours.

        The cache carried between rows is why this is one call and not 1536: the
        result would otherwise depend on where the previous chunk's generation
        finished.
        """
        qy = height // 4
        qx = chunk_x << 2
        qz = chunk_z << 2
        ys = np.arange(qy, dtype=np.int64)
        zs = np.arange(BIOMES_PER_AXIS, dtype=np.int64)
        xs = np.arange(BIOMES_PER_AXIS, dtype=np.int64)
        # Index order of the *output* is y*16 + z*4 + x, so the point order has to
        # be built the same way or the cache walk is the only thing that differs.
        gy, gz, gx = np.meshgrid(ys, zs, xs, indexing="ij")
        points = self.sampler.sample_many(
            gx.reshape(-1) + qx,
            gy.reshape(-1) + (min_y >> 2),
            gz.reshape(-1) + qz,
        )
        leaves = self.tree.get_many(points, self.last_node)
        self.last_node = int(leaves[-1]) if leaves.size else -1
        return self._ids[leaves].reshape(qy, BIOMES_PER_AXIS, BIOMES_PER_AXIS).astype(np.int32)

    def __repr__(self) -> str:
        return f"<MultiNoiseBiomeSource {self.preset!r} seed={self.seed}>"


class FixedBiomeSource:
    """One biome everywhere - vanilla's fixed source, and the fallback."""

    __slots__ = ("name", "numeric_id")

    def __init__(self, name: str, numeric_id: int) -> None:
        self.name = name
        self.numeric_id = int(numeric_id)

    def biome_at_quart(self, quart_x: int, quart_y: int, quart_z: int) -> str:
        return self.name

    def fill_volume(self, chunk_x: int, chunk_z: int, min_y: int, height: int) -> np.ndarray:
        return np.full((height // 4, BIOMES_PER_AXIS, BIOMES_PER_AXIS),
                       self.numeric_id, dtype=np.int32)

    def __repr__(self) -> str:
        return f"<FixedBiomeSource {self.name!r}>"


class EndBiomeSource:
    """Vanilla's `TheEndBiomeSource`: a centre disc, then four erosion bands."""

    __slots__ = ("seed", "sampler", "_ids")

    CENTER = "minecraft:the_end"
    HIGHLANDS = "minecraft:end_highlands"
    MIDLANDS = "minecraft:end_midlands"
    SMALL_ISLANDS = "minecraft:small_end_islands"
    BARRENS = "minecraft:end_barrens"

    #: Section radius squared of the central island, and the erosion thresholds.
    CENTER_RADIUS_SQUARED = 4096
    HIGHLANDS_THRESHOLD = np.float32(0.25)
    MIDLANDS_THRESHOLD = np.float32(-0.0625)
    SMALL_ISLANDS_THRESHOLD = np.float32(-0.21875)

    def __init__(self, seed: int, sampler: ClimateSampler, biome_ids: Dict[str, int]) -> None:
        self.seed = seed
        self.sampler = sampler
        self._ids = biome_ids

    @classmethod
    def from_assets(cls, seed: int = 0, biome_ids: Optional[Dict[str, int]] = None) -> "EndBiomeSource":
        ids = biome_ids if biome_ids is not None else biome_ids_from_assets()
        return cls(seed, ClimateSampler.from_assets("end", seed), ids)

    def biome_at_quart(self, quart_x: int, quart_y: int, quart_z: int) -> str:
        x = quart_x << 2
        y = quart_y << 2
        z = quart_z << 2
        section_x = x >> 4
        section_z = z >> 4
        if section_x * section_x + section_z * section_z <= self.CENTER_RADIUS_SQUARED:
            return self.CENTER
        # Vanilla samples erosion at the *centre* of the 2x2 section block the
        # quart belongs to, which is what the (2n+1)*8 spelling means.
        sample_x = (section_x * 2 + 1) * 8
        sample_z = (section_z * 2 + 1) * 8
        erosion = np.float32(self.sampler.nodes[3].eval(self.sampler.context, sample_x, y, sample_z))
        if erosion > self.HIGHLANDS_THRESHOLD:
            return self.HIGHLANDS
        if erosion >= self.MIDLANDS_THRESHOLD:
            return self.MIDLANDS
        if erosion < self.SMALL_ISLANDS_THRESHOLD:
            return self.SMALL_ISLANDS
        return self.BARRENS

    def fill_volume(self, chunk_x: int, chunk_z: int, min_y: int, height: int) -> np.ndarray:
        qy = height // 4
        out = np.empty((qy, BIOMES_PER_AXIS, BIOMES_PER_AXIS), dtype=np.int32)
        for qx in range(BIOMES_PER_AXIS):
            for qz in range(BIOMES_PER_AXIS):
                out[:, qz, qx] = self._ids[self.biome_at_quart(
                    (chunk_x << 2) + qx, min_y >> 2, (chunk_z << 2) + qz)]
        return out

    def __repr__(self) -> str:
        return f"<EndBiomeSource seed={self.seed}>"


def biome_source(preset: str, seed: int = 0,
                 biome_ids: Optional[Dict[str, int]] = None):
    """The biome source a generator preset uses.

    `overworld`/`large_biomes`/`amplified` share the overworld tree; `nether` has
    its own; `end` is not multi-noise at all.
    """
    if preset == "end":
        return EndBiomeSource.from_assets(seed, biome_ids)
    return MultiNoiseBiomeSource.from_assets(preset, seed, biome_ids)


# ==================== biome metadata ====================

def _canonical(name: str) -> str:
    """A resource location with its `minecraft:` namespace filled in.

    The file mixes conventions: biome keys are bare (`plains`) but every
    `type` inside `spawners`/`spawn_costs` is already prefixed. Normalising
    here means the spawner can compare against `EntityType` names as they
    come out of the registry without a second prefix check at every use.
    """
    return name if name.startswith("minecraft:") else f"minecraft:{name}"


#: The eight vanilla spawn categories, in the order `assets/biome.json` keys
#: them. This list is the only place a category name is spelled; the spawner
#: looks them up by string, so a typo here would silently empty that
#: category's draw rather than raise.
SPAWN_CATEGORIES = (
    "monster",
    "creature",
    "ambient",
    "axolotls",
    "underground_water_creature",
    "water_creature",
    "water_ambient",
    "misc",
)


class SpawnerEntry(NamedTuple):
    """One row of a biome's `spawners` list: "this type, this often".

    `min_count`/`max_count` are the size of the *herd* the spawner commits
    to before it starts walking positions; `weight` is that row's share of
    the weighted draw against its siblings in the same category.
    """
    type_name: str
    min_count: int
    max_count: int
    weight: int


class SpawnCost(NamedTuple):
    """One row of a biome's `spawn_costs`: the spawner's per-block budget.

    `energy_budget` is how much of a category's per-chunk energy a single
    mob of this type costs, and `charge` is how much of that energy a placed
    mob returns to the local pool. Both are 0 for every type not named in a
    biome's `spawn_costs` map.
    """
    energy_budget: float
    charge: float


class BiomeTable:
    """`assets/biome.json`, indexed by name and by the numeric id the wire uses.

    The id here is the one recorded in the file, which is the position in the
    synchronized `worldgen/biome` registry - the same number a chunk section's
    biome palette holds. It is *not* an index into this table's own ordering.
    """

    __slots__ = ("ids", "names", "carvers", "features", "temperature", "downfall",
                 "modifier", "spawners", "spawn_costs")

    def __init__(self, payload: Dict[str, dict]) -> None:
        self.ids: Dict[str, int] = {}
        self.names: Dict[int, str] = {}
        self.carvers: Dict[str, Tuple[str, ...]] = {}
        self.features: Dict[str, Tuple[Tuple[str, ...], ...]] = {}
        self.temperature: Dict[str, float] = {}
        self.downfall: Dict[str, float] = {}
        #: `temperature_modifier` - `"frozen"` for the biomes whose temperature is
        #: rewritten by noise. The surface rules and `freeze_top_layer` both read
        #: it; without it a frozen ocean is warm and grows no ice.
        self.modifier: Dict[str, str] = {}
        #: biome -> category -> the types that biome spawns, in file order.
        #: Every biome defines all eight categories; most are empty lists.
        self.spawners: Dict[str, Dict[str, Tuple[SpawnerEntry, ...]]] = {}
        #: biome -> entity type -> its cost/charge. Sparse on purpose: only two
        #: biomes in the 774 report record one, and a type absent from here
        #: costs nothing and charges nothing rather than being forbidden.
        self.spawn_costs: Dict[str, Dict[str, SpawnCost]] = {}
        for name, entry in payload.items():
            canonical = name if name.startswith("minecraft:") else f"minecraft:{name}"
            recorded = entry.get("id")
            if recorded is None:
                continue
            self.ids[canonical] = int(recorded)
            self.names[int(recorded)] = canonical
            self.carvers[canonical] = tuple(
                value if value.startswith("minecraft:") else f"minecraft:{value}"
                for value in (entry.get("carvers") or ())
            )
            steps: List[Tuple[str, ...]] = []
            for step in entry.get("features") or ():
                if isinstance(step, str):
                    steps.append((step,))
                    continue
                steps.append(tuple(
                    value if value.startswith("minecraft:") else f"minecraft:{value}"
                    for value in (step or ())
                ))
            self.features[canonical] = tuple(steps)
            self.temperature[canonical] = float(entry.get("temperature", 0.0))
            self.downfall[canonical] = float(entry.get("downfall", 0.0))
            # Absent means "none"; the file only records the interesting case.
            self.modifier[canonical] = str(entry.get("temperature_modifier") or "")

            categories = entry.get("spawners") or {}
            self.spawners[canonical] = {
                category: tuple(
                    SpawnerEntry(
                        _canonical(row["type"]),
                        int(row.get("minCount", 0)),
                        int(row.get("maxCount", 0)),
                        int(row.get("weight", 0)),
                    )
                    for row in categories.get(category) or ()
                )
                for category in SPAWN_CATEGORIES
            }
            self.spawn_costs[canonical] = {
                _canonical(mob_type): SpawnCost(
                    float(cost.get("energy_budget", 0.0)),
                    float(cost.get("charge", 0.0)),
                )
                for mob_type, cost in (entry.get("spawn_costs") or {}).items()
            }

    @classmethod
    def from_assets(cls, path: Optional[str] = None) -> "BiomeTable":
        with open(path or _assets_path("biome.json"), encoding="utf-8") as handle:
            return cls(json.load(handle))

    def id_of(self, name: str) -> int:
        return self.ids[name if name.startswith("minecraft:") else f"minecraft:{name}"]

    def name_of(self, numeric_id: int) -> str:
        return self.names[numeric_id]

    def spawners_for(self, name: str, category: str) -> Tuple[SpawnerEntry, ...]:
        """The `category` spawn rows of one biome, by name.

        An unknown biome or category gives an empty tuple rather than a
        KeyError: a half-implemented category is a *result*, not a crash,
        and the spawner treats "nothing to draw" as "skip this round".
        """
        return self.spawners.get(_canonical(name), {}).get(category, ())

    def spawn_cost_of(self, name: str, type_name: str) -> SpawnCost:
        """One type's cost/charge in one biome, defaulting to free and empty.

        Vanilla's default for a type with no recorded cost is exactly this -
        it costs no energy and places no charge - which is why 63 of the 65
        biomes here have an empty map and still spawn monsters fine.
        """
        return self.spawn_costs.get(_canonical(name), {}).get(
            _canonical(type_name), SpawnCost(0.0, 0.0)
        )

    def steps(self, name: str) -> int:
        """How many decoration steps this biome's feature lists define."""
        return len(self.features.get(name, ()))

    def iter_features(self) -> Iterator[Tuple[str, int, str]]:
        """`(biome, step, placed feature)` over everything the table lists."""
        for biome, steps in self.features.items():
            for step, entries in enumerate(steps):
                for feature in entries:
                    yield biome, step, feature

    def __repr__(self) -> str:
        return f"<BiomeTable biomes={len(self.ids)}>"


_BIOME_TABLE: Optional[BiomeTable] = None


def biome_table(refresh: bool = False) -> BiomeTable:
    """The shared `BiomeTable`. Cached: 65 entries, but every chunk asks."""
    global _BIOME_TABLE
    if _BIOME_TABLE is None or refresh:
        _BIOME_TABLE = BiomeTable.from_assets()
    return _BIOME_TABLE


def biome_ids_from_assets() -> Dict[str, int]:
    """Name -> numeric id, from `assets/biome.json`."""
    return dict(biome_table().ids)


__all__ = [
    "BIOMES_PER_AXIS",
    "CLIMATE_NAMES",
    "CLIMATE_SLOTS",
    "PARAMETER_AXES",
    "QUANTIZATION_FACTOR",
    "SPAWN_CATEGORIES",
    "BiomeTable",
    "BiomeTree",
    "ClimateSampler",
    "EndBiomeSource",
    "FixedBiomeSource",
    "MultiNoiseBiomeSource",
    "Parameter",
    "ParameterPoint",
    "SpawnCost",
    "SpawnerEntry",
    "biome_ids_from_assets",
    "biome_source",
    "biome_table",
    "quantize",
    "unquantize",
]
