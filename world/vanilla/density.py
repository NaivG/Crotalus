"""
A density-function interpreter that reproduces vanilla's values exactly.

A density function is a little expression tree: sample some noise, shift it,
clamp it, blend it, and hand back a number whose sign says whether a block is
solid or air. Vanilla's trees come from `assets/density_function.json` -- the
noise router, already resolved, with every `minecraft:` reference inlined. This
module walks those trees.

    from world.vanilla.density import parse, DensityContext

    tree = json.load(open("assets/density_function.json"))["overworld"]["finalDensity"]
    fn = parse(tree, context)
    fn.eval(context, 0, 64, 0)

Three things make this more than a recursive walk
-------------------------------------------------
**Everything is f32.** Vanilla's `DensityFunction` interface returns `float`,
not `double`, even though noise coordinates are `double`. Every node here
rounds to f32 on the way out. A Python float is a double, so without the
rounding the low bits drift and surface levels move by a block.

**The wrappers come in two flavours.** `CacheOnce`, `Cache2D` and `FlatCache`
are pure memoisation -- they cannot change a value, only the cost of computing
it. `Interpolated` is the opposite: it samples only the *corners* of a cell and
blends, which genuinely changes the number. Vanilla uses it to trade exactness
for a large speedup on expensive subtrees. Treating the two groups the same is
the single easiest way to get subtly wrong terrain.

**There are two evaluation paths, one set of numbers.** `eval` takes one
position, `eval_array` takes a batch of them. The batch path exists because a
chunk is ~100k positions and per-point Python overhead cannot be amortised any
other way; it is not allowed to change a single rounding. Nodes round to f32 at
exactly the places the scalar path does, and
`tools/check_density_vectorized.py` asserts the two agree point for point.

A note on `Interpolated`'s cell size
------------------------------------
The vanilla datapack carries it per node (`cell_size_xz`, `cell_size_y`), but
the resolved dump in `assets/density_function.json` drops those fields. The
values are recovered in `INTERPOLATED_CELL_SIZES` below, straight from the
datapack, and the reasoning is recorded there rather than being magic numbers.

A note on the parse-time hash-consing
------------------------------------
The resolved dump inlines every `minecraft:` reference, so the same shift-field
cluster appears ~240 times in `finalDensity` -- 2955 nodes, but only ~134
distinct subtrees. `parse` merges structurally identical subtrees into one
shared instance (keyed by a structural fingerprint of the JSON plus the parse
options). Sharing cannot change a value: every node is a pure function of its
position, and the memoisation wrappers hold only memo tables. What it changes
is cost -- the shared `Cache2D` inside a duplicated cluster computes its shift
field once instead of once per copy -- and it is what makes the batch path's
common-subexpression memo (one evaluation per unique node per batch) effective.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

from world.vanilla.perlin import (
    DoublePerlinNoiseSampler,
    NormalNoise,
    create_fbm,
    fbm_range_max,
    lerp3,
)
from world.vanilla.random_source import LegacyRandom, Xoroshiro

F = np.float32
_INF = np.float32(np.inf)
_F32_0 = np.float32(0.0)
_F32_1 = np.float32(1.0)

#: `Interpolated` cell sizes, recovered from the vanilla datapack by scanning
#: every `minecraft:interpolated` node:
#:
#:   overworld/chunk_surface_level.json   (16, 1)   <- the surface estimator
#:   overworld/final_density.json          (4, 8)
#:   overworld/caves/noodle.json           (4, 8)   x4
#:   overworld/ore_vein/{mask,toggle}      (4, 8)
#:   nether                                (4, 8)
#:   end                                   (8, 4)
#:
#: So the overworld family is (4, 8) throughout, the surface-level function is
#: the lone (16, 1), and the End -- a flat, wide dimension -- is (8, 4). The
#: resolved dump carries no cell sizes, so they are selected by which slot the
#: tree came from.
DEFAULT_CELL_SIZE = (4, 8)
SURFACE_CELL_SIZE = (16, 1)
END_CELL_SIZE = (8, 4)

#: `RarityValueMapper`, transcribed from the control flow quoted in
#: `Density_function.mdx` (the wiki source for this project) and cross-checked
#: against its own prose, which says `type_1` spans "minimum scale 0.75,
#: maximum 2.0" and `type_2` spans "minimum 0.5, maximum 3.0" -- the two agree
#: on the ranges, so the labels below are the wiki's.
#:
#: Both are piecewise-constant in the *input* `d`; the result picks a rarity,
#: evaluates the named noise at `pos / rarity`, and scales back by that rarity:
#:
#:   type_1:  d < -0.5 -> 0.75;  d < 0.0 -> 1.0;  d < 0.5 -> 1.5;  else -> 2.0
#:   type_2:  d < -0.75 -> 0.5;  d < -0.5 -> 0.75;  d < 0.5 -> 1.0;
#:            d < 0.75 -> 2.0;  else -> 3.0
#:
#: A caveat worth recording: Mojang's 26.2 snapshot 5 notes, which describe the
#: `interval_select` that replaced this function, label these two the other way
#: round. The wiki is preferred here because its prose and its quoted control
# flow corroborate each other; if a future world looks like its caves are
#: systematically the wrong thickness, this swap is the first thing to try.
_RARITY_TYPE_1 = ((-0.5, 0.75), (0.0, 1.0), (0.5, 1.5), (math.inf, 2.0))
_RARITY_TYPE_2 = ((-0.75, 0.5), (-0.5, 0.75), (0.5, 1.0), (0.75, 2.0), (math.inf, 3.0))


def _rarity(table, value: float) -> float:
    for threshold, rarity in table:
        if value < threshold:
            return rarity
    return table[-1][1]


def _apply_unary(op: str, d: np.float32) -> np.float32:
    """One of vanilla's eleven unary density operations, at f32."""
    if op == "ABS":
        return F(abs(d))
    if op == "SQUARE":
        return F(d * d)
    if op == "CUBE":
        return F(d * d * d)
    if op == "HALF_NEGATIVE":
        return F(d if d > 0 else d * 0.5)
    if op == "QUARTER_NEGATIVE":
        return F(d if d > 0 else d * 0.25)
    if op == "SQUEEZE":
        # Quintic approximation to sign(); smooth where vanilla's "Squeeze" is.
        c = F(min(max(d, F(-1.0)), F(1.0)))
        return F(c / 2.0 - c * c * c / 24.0)
    if op == "INVERT":
        return _INF if d == 0 else F(F(1.0) / d)
    if op == "NEGATE":
        return F(-d)
    if op == "SQRT":
        return F(np.sqrt(d, dtype=np.float32)) if d >= 0 else F(np.nan)
    if op == "LOG":
        return F(np.float32(math.log(float(d)))) if d > 0 else F(-_INF if d == 0 else np.float32(np.nan))
    if op == "SIGN":
        return F(1.0 if d > 0 else (-1.0 if d < 0 else 0.0))
    raise ValueError(f"unknown unary operation: {op}")


def _apply_binary(op: str, a: np.float32, b: np.float32) -> np.float32:
    if op == "ADD":
        return F(a + b)
    if op == "MUL":
        return F(a * b)
    if op == "MIN":
        return F(min(a, b))
    if op == "MAX":
        return F(max(a, b))
    if op == "SUB":
        return F(a - b)
    if op == "DIV":
        # Vanilla short-circuits on a zero *numerator*, not a zero divisor.
        return F(0.0) if a == 0 else F(a / b)
    if op == "POW":
        # Computed in f64 then narrowed, exactly as Java's Math.pow + cast.
        return F(math.pow(float(a), float(b)))
    raise ValueError(f"unknown binary operation: {op}")


# ==========================================================================
# Array forms of the two op tables.
#
# numpy arrays follow the same NEP-50 weak-promotion rules the scalar np.float32
# values do -- a bare Python float is absorbed into the f32 computation rather
# than widening it -- so writing the *same expression text* over float32 arrays
# reproduces the scalar path's every rounding. Two places need care, both about
# NaN, where numpy and CPython disagree:
#
#   * `MIN`/`MAX` must reproduce CPython `min`/`max`, which return the *first*
#     argument when the comparison is unordered: `min(1.0, nan)` is `1.0`.
#     `np.where(b < a, b, a)` does exactly that, including the asymmetry.
#   * `Clamp` and `SQUEEZE` write `min(max(d, lo), hi)` with the NaN candidate
#     in first position, where CPython's max *propagates* the NaN -- the same
#     behaviour as `np.maximum`/`np.minimum`. Keep the argument order.
# ==========================================================================


def _py_min_arr(a, b):
    return np.where(b < a, b, a)


def _py_max_arr(a, b):
    return np.where(a < b, b, a)


def _apply_unary_arr(op: str, d: np.ndarray) -> np.ndarray:
    if op == "ABS":
        return np.abs(d)
    if op == "SQUARE":
        return d * d
    if op == "CUBE":
        return d * d * d
    if op == "HALF_NEGATIVE":
        return np.where(d > 0, d, d * 0.5)
    if op == "QUARTER_NEGATIVE":
        return np.where(d > 0, d, d * 0.25)
    if op == "SQUEEZE":
        c = np.minimum(np.maximum(d, -1.0), 1.0)
        return c / 2.0 - c * c * c / 24.0
    if op == "INVERT":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(d == 0, _INF, _F32_1 / d)
    if op == "NEGATE":
        return -d
    if op == "SQRT":
        with np.errstate(invalid="ignore"):
            return np.where(d >= 0, np.sqrt(d, dtype=np.float32), np.float32(np.nan))
    if op == "LOG":
        with np.errstate(divide="ignore", invalid="ignore"):
            logged = np.log(d.astype(np.float64)).astype(np.float32)
        return np.where(d > 0, logged, np.where(d == 0, -_INF, np.float32(np.nan)))
    if op == "SIGN":
        return np.where(d > 0, _F32_1, np.where(d < 0, -_F32_1, _F32_0))
    raise ValueError(f"unknown unary operation: {op}")


def _apply_binary_arr(op: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if op == "ADD":
        return a + b
    if op == "MUL":
        return a * b
    if op == "MIN":
        return _py_min_arr(a, b)
    if op == "MAX":
        return _py_max_arr(a, b)
    if op == "SUB":
        return a - b
    if op == "DIV":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(a == 0, _F32_0, a / b)
    if op == "POW":
        with np.errstate(invalid="ignore"):
            return np.power(a.astype(np.float64), b.astype(np.float64)).astype(np.float32)
    raise ValueError(f"unknown binary operation: {op}")


class NoiseLibrary:
    """Builds and memoises noise samplers, keyed by parameter name.

    The memo matters more than it looks: a `Noise` node is cheap to *describe*
    but constructing its sampler means a Stafford mix, a splitter derivation and
    an MD5 per octave. The overworld router asks for the same handful of
    parameters hundreds of times.
    """

    def __init__(self, splitter, parameters: Dict[str, dict]) -> None:
        self._splitter = splitter
        self._parameters = parameters
        self._cache: Dict[str, DoublePerlinNoiseSampler] = {}

    def sampler(self, name: str) -> DoublePerlinNoiseSampler:
        cached = self._cache.get(name)
        if cached is not None:
            return cached
        param = self._parameters.get(name)
        if param is None:
            raise KeyError(f"unknown noise parameter: {name}")
        first_octave = int(param.get("firstOctave", param.get("baseOctave", -7)))
        amplitudes = param.get("amplitudes") or param.get("amplitude_modifiers")
        if amplitudes is None:
            count = int(param.get("octave_count", 1))
            amplitudes = [1.0] * max(count, 1)
        # MD5("minecraft:<name>") -- the id is fully qualified before hashing.
        random = self._splitter.split_string(f"minecraft:{name}")
        sampler = DoublePerlinNoiseSampler(random, first_octave, [float(a) for a in amplitudes])
        self._cache[name] = sampler
        return sampler


class InterpolatedNoiseLibrary:
    """Memoises `InterpolatedNoiseSampler` instances, keyed by their parameters.

    Each instance costs 23 smeared Perlin constructions, so building them lazily
    and once matters.
    """

    _BLENDED_BASE_SCALE = 684.412
    _BLENDED_LIMIT_FACTOR = 0.99998474
    _BLENDED_MAIN_FACTOR = 12.75
    _BLENDED_LIMIT_FIRST_OCTAVE = -15
    _BLENDED_MAIN_FIRST_OCTAVE = -7

    def __init__(self, splitter) -> None:
        self._splitter = splitter
        self._cache: Dict[tuple, "_InterpolatedNoise"] = {}

    def get(self, data: dict) -> "_InterpolatedNoise":
        key = (
            data["xzScale"],
            data["yScale"],
            data["xzFactor"],
            data["yFactor"],
            data["smearScaleMultiplier"],
        )
        cached = self._cache.get(key)
        if cached is None:
            cached = _InterpolatedNoise(self._splitter, data, self)
            self._cache[key] = cached
        return cached


class _InterpolatedNoise:
    """`minecraft:interpolated`-style sampler: three FBMs blended by a fourth.

    The main noise picks a blend factor in [0, 1]; the two limit noises supply
    the endpoints. Because the blend factor comes from a *third* independent
    field, the result has far fewer visible grid features than either endpoint
    alone -- that is the whole point of the construction.
    """

    def __init__(self, splitter, data: dict, library: InterpolatedNoiseLibrary) -> None:
        k = InterpolatedNoiseLibrary
        self.xz_multiplier = k._BLENDED_BASE_SCALE * data["xzScale"]
        self.y_multiplier = k._BLENDED_BASE_SCALE * data["yScale"]
        limit_smear_scale_y = self.y_multiplier * data["smearScaleMultiplier"]
        main_smear_scale_y = limit_smear_scale_y / data["yFactor"]

        # Each FBM is built from its own deriver, so they do not share state.
        def build(first_octave: int, smear: float, factor: float):
            random = splitter.split_u64(first_octave * 1000003 + int(smear * 1000))
            return create_fbm(random, first_octave, smear, factor)

        self.min_limit = build(k._BLENDED_LIMIT_FIRST_OCTAVE, limit_smear_scale_y, k._BLENDED_LIMIT_FACTOR)
        self.max_limit = build(k._BLENDED_LIMIT_FIRST_OCTAVE, limit_smear_scale_y, k._BLENDED_LIMIT_FACTOR)
        self.main = build(k._BLENDED_MAIN_FIRST_OCTAVE, main_smear_scale_y, k._BLENDED_MAIN_FACTOR)

        self.main_xz_multiplier = self.xz_multiplier / data["xzFactor"]
        self.main_y_multiplier = self.y_multiplier / data["yFactor"]
        self.max_value = fbm_range_max(
            k._BLENDED_LIMIT_FIRST_OCTAVE, limit_smear_scale_y, k._BLENDED_LIMIT_FACTOR
        )

    def sample(self, x: int, y: int, z: int) -> np.float32:
        alpha = F(min(max(self.main.get(
            x * self.main_xz_multiplier, y * self.main_y_multiplier, z * self.main_xz_multiplier
        ) + 0.5, F(0.0)), F(1.0)))
        if alpha == 0:
            return F(self.min_limit.get(x * self.xz_multiplier, y * self.y_multiplier, z * self.xz_multiplier))
        if alpha == 1:
            return F(self.max_limit.get(x * self.xz_multiplier, y * self.y_multiplier, z * self.xz_multiplier))
        lo = F(self.min_limit.get(x * self.xz_multiplier, y * self.y_multiplier, z * self.xz_multiplier))
        hi = F(self.max_limit.get(x * self.xz_multiplier, y * self.y_multiplier, z * self.xz_multiplier))
        return F(lo + alpha * (hi - lo))

    def sample_array(self, xs, ys, zs) -> np.ndarray:
        alpha = np.minimum(np.maximum(
            self.main.get_array(
                xs * self.main_xz_multiplier, ys * self.main_y_multiplier, zs * self.main_xz_multiplier
            ) + 0.5,
            _F32_0,
        ), _F32_1)
        xm = xs * self.xz_multiplier
        ym = ys * self.y_multiplier
        zm = zs * self.xz_multiplier
        out = np.empty(alpha.shape, dtype=np.float32)
        lo_mask = alpha == 0
        hi_mask = alpha == 1
        mid = ~(lo_mask | hi_mask)
        # Branch by alpha exactly as `sample` does: only the stacks a branch
        # needs are sampled, so the array path cannot drift into evaluating a
        # limit noise at positions the scalar path skipped (which matters only
        # for cost -- the values would agree either way).
        if lo_mask.any():
            out[lo_mask] = self.min_limit.get_array(xm[lo_mask], ym[lo_mask], zm[lo_mask])
        if hi_mask.any():
            out[hi_mask] = self.max_limit.get_array(xm[hi_mask], ym[hi_mask], zm[hi_mask])
        if mid.any():
            lo = self.min_limit.get_array(xm[mid], ym[mid], zm[mid])
            hi = self.max_limit.get_array(xm[mid], ym[mid], zm[mid])
            out[mid] = lo + alpha[mid] * (hi - lo)
        return out


# ==========================================================================
# Nodes
# ==========================================================================


class _BatchMemo:
    """Per-batch common-subexpression memo.

    After parse-time hash-consing the tree is a DAG: the ~240 copies of a shift
    cluster are one node reached ~240 ways. Keying on node identity collapses
    each unique node to one evaluation per batch. The point arrays are kept
    alive in the entries because node/array `id()`s are the keys, and a freed
    array's id could otherwise be reused mid-batch by a later subset.

    `decomps` caches point-set decompositions (unique columns, unique cells)
    the same way: the split of a batch into unique keys depends only on the
    point set, not on which node asks, and `np.unique` on a 100k-row key is
    the kind of sort worth doing once per batch instead of once per wrapper.
    """

    __slots__ = ("entries", "decomps")

    def __init__(self) -> None:
        self.entries: Dict[tuple, tuple] = {}
        self.decomps: Dict[tuple, tuple] = {}

    def decomp(self, key: tuple, keepalive, build):
        hit = self.decomps.get(key)
        if hit is not None:
            return hit[1]
        data = build()
        self.decomps[key] = (keepalive, data)
        return data


class Node:
    """Base class: `eval` returns f32; `min_value`/`max_value` bound it.

    The bounds are real slots rather than class attributes: with `__slots__` in
    play, a class-level default shadows the slot descriptor and assigning to it
    on an instance raises `AttributeError: read-only`. So every node sets its
    own, explicitly, in `__init__`.

    `eval` is one position; `eval_array` is a batch of equal-shape int64
    coordinate arrays and returns a float32 array. The public `eval_array`
    wraps the per-class `_eval_array` with the batch memo -- subclasses override
    `_eval_array` only.
    """

    __slots__ = ("min_value", "max_value")

    def __init__(self, min_value: np.float32, max_value: np.float32) -> None:
        self.min_value = min_value
        self.max_value = max_value

    def eval(self, ctx, x: int, y: int, z: int) -> np.float32:
        raise NotImplementedError

    def _eval_array(self, ctx, x, y, z) -> np.ndarray:
        raise NotImplementedError

    def eval_array(self, ctx, x, y, z) -> np.ndarray:
        batch = ctx._arr_batch
        started = batch is None
        if started:
            batch = _BatchMemo()
            ctx._arr_batch = batch
        try:
            key = (id(self), id(x), id(y), id(z))
            entry = batch.entries.get(key)
            if entry is not None:
                return entry[3]
            if x.size == 0:
                return np.empty(x.shape, dtype=np.float32)
            value = self._eval_array(ctx, x, y, z)
            batch.entries[key] = (x, y, z, value)
            return value
        finally:
            if started:
                ctx._arr_batch = None


class Constant(Node):
    __slots__ = ("value",)

    def __init__(self, value: float) -> None:
        self.value = F(value)
        super().__init__(self.value, self.value)

    def eval(self, ctx, x, y, z):
        return self.value

    def _eval_array(self, ctx, x, y, z):
        return np.full(x.shape, self.value, dtype=np.float32)


class UnaryOperation(Node):
    __slots__ = ("op", "input")

    def __init__(self, op: str, input_node: Node) -> None:
        self.op = op
        self.input = input_node
        super().__init__(input_node.min_value, input_node.max_value)

    def eval(self, ctx, x, y, z):
        return _apply_unary(self.op, self.input.eval(ctx, x, y, z))

    def _eval_array(self, ctx, x, y, z):
        return _apply_unary_arr(self.op, self.input.eval_array(ctx, x, y, z))


class BinaryOperation(Node):
    __slots__ = ("op", "a", "b")

    def __init__(self, op: str, a: Node, b: Node) -> None:
        self.op = op
        self.a = a
        self.b = b
        lo = min(a.min_value, b.min_value)
        hi = max(a.max_value, b.max_value)
        super().__init__(lo, hi)

    def eval(self, ctx, x, y, z):
        a = self.a.eval(ctx, x, y, z)
        # Vanilla short-circuits these, skipping the second subtree entirely --
        # which matters for side effects as well as speed.
        if self.op == "MUL" and a == 0:
            return F(0.0)
        if self.op == "DIV" and a == 0:
            return F(0.0)
        return _apply_binary(self.op, a, self.b.eval(ctx, x, y, z))

    def _eval_array(self, ctx, x, y, z):
        a = self.a.eval_array(ctx, x, y, z)
        b = self.b.eval_array(ctx, x, y, z)
        out = _apply_binary_arr(self.op, a, b)
        if self.op == "MUL" or self.op == "DIV":
            # Same short-circuit as `eval`, applied per element. _apply_binary_arr
            # already zeroes MUL/DIV by a zero numerator where that is the
            # arithmetic result; this also pins the +0.0 the scalar path returns.
            out = np.where(a == 0, _F32_0, out)
        return out


class LinearOperation(Node):
    __slots__ = ("op", "input", "argument")

    def __init__(self, op: str, input_node: Node, argument: float) -> None:
        self.op = op
        self.input = input_node
        self.argument = F(argument)
        lo, hi = input_node.min_value, input_node.max_value
        if op == "ADD":
            super().__init__(F(lo + self.argument), F(hi + self.argument))
        else:
            a = abs(self.argument)
            super().__init__(F(lo * a), F(hi * a))

    def eval(self, ctx, x, y, z):
        d = self.input.eval(ctx, x, y, z)
        return F(d + self.argument) if self.op == "ADD" else F(d * self.argument)

    def _eval_array(self, ctx, x, y, z):
        d = self.input.eval_array(ctx, x, y, z)
        return d + self.argument if self.op == "ADD" else d * self.argument


class ClampNode(Node):
    __slots__ = ("input", "lo", "hi")

    def __init__(self, input_node: Node, lo: float, hi: float) -> None:
        self.input = input_node
        self.lo = F(lo)
        self.hi = F(hi)
        super().__init__(self.lo, self.hi)

    def eval(self, ctx, x, y, z):
        return F(min(max(self.input.eval(ctx, x, y, z), self.lo), self.hi))

    def _eval_array(self, ctx, x, y, z):
        d = self.input.eval_array(ctx, x, y, z)
        # Argument order matters: the NaN candidate sits in first position,
        # where CPython's max propagates it -- matching np.maximum. See the
        # op-table notes above.
        return np.minimum(np.maximum(d, self.lo), self.hi)


class YClampedGradient(Node):
    """A linear ramp in Y, clamped at both ends.

    `clamped_map` rather than a plain lerp: outside [fromY, toY] the value holds
    at the endpoint instead of extrapolating, which is what keeps the world's
    floor and ceiling flat instead of ramping off forever.
    """

    __slots__ = ("from_y", "to_y", "from_value", "to_value")

    def __init__(self, from_y: int, to_y: int, from_value: float, to_value: float) -> None:
        self.from_y = F(from_y)
        self.to_y = F(to_y)
        self.from_value = F(from_value)
        self.to_value = F(to_value)
        super().__init__(F(min(from_value, to_value)), F(max(from_value, to_value)))

    def eval(self, ctx, x, y, z):
        span = self.to_y - self.from_y
        t = F((F(y) - self.from_y) / span) if span != 0 else F(0.0)
        t = F(min(max(t, F(0.0)), F(1.0)))
        return F(self.from_value + t * (self.to_value - self.from_value))

    def _eval_array(self, ctx, x, y, z):
        span = np.float32(self.to_y - self.from_y)
        if span == 0:
            t = np.zeros(x.shape, dtype=np.float32)
        else:
            t = (y.astype(np.float32) - self.from_y) / span
            t = np.minimum(np.maximum(t, _F32_0), _F32_1)
        return self.from_value + t * (self.to_value - self.from_value)


class NoiseNode(Node):
    """`minecraft:noise` -- a NormalNoise sampled at scaled coordinates."""

    __slots__ = ("_name", "_sampler", "xz_scale", "y_scale")

    def __init__(self, name: str, xz_scale: float, y_scale: float, library: NoiseLibrary) -> None:
        self._name = name
        self._sampler = library.sampler(name)
        self.xz_scale = xz_scale
        self.y_scale = y_scale
        bound = F(self._sampler.max_value)
        super().__init__(F(-bound), bound)

    def eval(self, ctx, x, y, z):
        return F(self._sampler.get(x * self.xz_scale, y * self.y_scale, z * self.xz_scale))

    def _eval_array(self, ctx, x, y, z):
        return self._sampler.get_array(x * self.xz_scale, y * self.y_scale, z * self.xz_scale)


class ShiftA(Node):
    """A 2D shift field, 4x the amplitude, no Y dependence."""

    __slots__ = ("_sampler",)

    def __init__(self, name: str, library: NoiseLibrary) -> None:
        self._sampler = library.sampler(name)
        bound = F(self._sampler.max_value * 4.0)
        super().__init__(F(-bound), bound)

    def eval(self, ctx, x, y, z):
        return F(self._sampler.get(x * 0.25, 0.0, z * 0.25) * 4.0)

    def _eval_array(self, ctx, x, y, z):
        zero = np.zeros(x.shape, dtype=np.float64)
        return self._sampler.get_array(x * 0.25, zero, z * 0.25) * 4.0


class ShiftB(Node):
    """A 2D shift field with X and Z swapped relative to `ShiftA`."""

    __slots__ = ("_sampler",)

    def __init__(self, name: str, library: NoiseLibrary) -> None:
        self._sampler = library.sampler(name)
        bound = F(self._sampler.max_value * 4.0)
        super().__init__(F(-bound), bound)

    def eval(self, ctx, x, y, z):
        return F(self._sampler.get(z * 0.25, x * 0.25, 0.0) * 4.0)

    def _eval_array(self, ctx, x, y, z):
        zero = np.zeros(x.shape, dtype=np.float64)
        return self._sampler.get_array(z * 0.25, x * 0.25, zero) * 4.0


class ShiftedNoise(Node):
    """`minecraft:shifted_noise` -- noise displaced by three shift functions.

    The shifts are what break the grid alignment of the base noise. They are
    almost always wrapped in `Cache2D`, which is safe precisely because they
    have no Y dependence.
    """

    __slots__ = ("shift_x", "shift_y", "shift_z", "_sampler", "xz_scale", "y_scale")

    def __init__(self, sx: Node, sy: Node, sz: Node, xz_scale: float, y_scale: float,
                 name: str, library: NoiseLibrary) -> None:
        self.shift_x, self.shift_y, self.shift_z = sx, sy, sz
        self._sampler = library.sampler(name)
        self.xz_scale = xz_scale
        self.y_scale = y_scale
        bound = F(self._sampler.max_value)
        super().__init__(F(-bound), bound)

    def eval(self, ctx, x, y, z):
        return F(self._sampler.get(
            x * self.xz_scale + float(self.shift_x.eval(ctx, x, y, z)),
            y * self.y_scale + float(self.shift_y.eval(ctx, x, y, z)),
            z * self.xz_scale + float(self.shift_z.eval(ctx, x, y, z)),
        ))

    def _eval_array(self, ctx, x, y, z):
        return self._sampler.get_array(
            x * self.xz_scale + self.shift_x.eval_array(ctx, x, y, z).astype(np.float64),
            y * self.y_scale + self.shift_y.eval_array(ctx, x, y, z).astype(np.float64),
            z * self.xz_scale + self.shift_z.eval_array(ctx, x, y, z).astype(np.float64),
        )


class InterpolatedNoiseSamplerNode(Node):
    __slots__ = ("_impl",)

    def __init__(self, data: dict, library: InterpolatedNoiseLibrary) -> None:
        self._impl = library.get(data)
        bound = F(self._impl.max_value)
        super().__init__(F(-bound), bound)

    def eval(self, ctx, x, y, z):
        return F(self._impl.sample(x, y, z))

    def _eval_array(self, ctx, x, y, z):
        return self._impl.sample_array(x, y, z)


class RangeChoice(Node):
    """Picks one of two branches on whether the input is in [lo, hi)."""

    __slots__ = ("input", "lo", "hi", "when_in", "when_out")

    def __init__(self, inp: Node, lo: float, hi: float, when_in: Node, when_out: Node) -> None:
        self.input, self.lo, self.hi = inp, F(lo), F(hi)
        self.when_in, self.when_out = when_in, when_out
        super().__init__(
            F(min(when_in.min_value, when_out.min_value)),
            F(max(when_in.max_value, when_out.max_value)),
        )

    def eval(self, ctx, x, y, z):
        v = self.input.eval(ctx, x, y, z)
        return self.when_in.eval(ctx, x, y, z) if self.lo <= v < self.hi else self.when_out.eval(ctx, x, y, z)

    def _eval_array(self, ctx, x, y, z):
        v = self.input.eval_array(ctx, x, y, z)
        # Chained `lo <= v < hi` is False for NaN v, so NaN goes to the
        # out-of-range branch -- same as the scalar comparison chain.
        inside = (self.lo <= v) & (v < self.hi)
        out = np.empty(v.shape, dtype=np.float32)
        if inside.any():
            out[inside] = self.when_in.eval_array(ctx, x[inside], y[inside], z[inside])
        outside = ~inside
        if outside.any():
            out[outside] = self.when_out.eval_array(ctx, x[outside], y[outside], z[outside])
        return out


class SplineNode(Node):
    """A cubic Hermite spline over a piecewise-sorted set of points.

    Outside the outer points the curve is extended *linearly* along the end
    derivative rather than clamped -- so a spline's tails do run off. Getting
    this wrong turns the ends of every cave tunnel into a hard shelf.

    Only the two points bracketing the location are evaluated. Vanilla's
    `CubicSpline` does the same; a point's value node is a pure function of the
    position, so skipping the others cannot change the returned number, only
    its cost -- and with nested value-splines that cost is the whole curve.
    """

    __slots__ = ("location", "points", "_locs_f32", "_span_f32", "_startx_f32",
                 "_dstart_f32", "_dend_f32", "_tail0", "_tailn")

    def __init__(self, location: Node, points) -> None:
        self.location = location
        # (location, value_node, derivative) triples; `value_node` is a Node
        # because a spline point's value may itself be a spline.
        self.points = points
        n = len(points)
        locs = [p[0] for p in points]
        derivs = [p[2] for p in points]
        self._locs_f32 = np.asarray(locs, dtype=np.float32)
        # The scalar path computes `end_x - start_x` in f64 (two Python floats)
        # and only rounds to f32 when it divides -- so the per-segment spans are
        # pre-rounded here from the f64 difference, not from f32 endpoints.
        spans = [locs[i + 1] - locs[i] for i in range(n - 1)]
        self._span_f32 = np.asarray(spans, dtype=np.float32)
        self._startx_f32 = np.asarray(locs[:-1], dtype=np.float32)
        self._dstart_f32 = np.asarray(
            [derivs[i] * spans[i] for i in range(n - 1)], dtype=np.float32
        )
        self._dend_f32 = np.asarray(
            [derivs[i + 1] * spans[i] for i in range(n - 1)], dtype=np.float32
        )
        self._tail0 = (np.float32(locs[0]), np.float32(derivs[0]))
        self._tailn = (np.float32(locs[-1]), np.float32(derivs[-1]))
        super().__init__(F(-math.inf), F(math.inf))

    def eval(self, ctx, x, y, z):
        loc = self.location.eval(ctx, x, y, z)

        n = len(self.points)
        idx = 0
        while idx < n and loc >= self.points[idx][0]:
            idx += 1

        if idx == 0:
            p = self.points[0]
            return F(p[1].eval(ctx, x, y, z) + p[2] * (loc - p[0]))
        if idx == n:
            p = self.points[n - 1]
            return F(p[1].eval(ctx, x, y, z) + p[2] * (loc - p[0]))

        start_x, start_val, start_d = self.points[idx - 1]
        end_x, end_val, end_d = self.points[idx]
        start_v = start_val.eval(ctx, x, y, z)
        end_v = end_val.eval(ctx, x, y, z)

        t = F((loc - start_x) / (end_x - start_x))
        h00 = F((1 + 2 * t) * (1 - t) * (1 - t))
        h10 = F(t * (1 - t) * (1 - t))
        h01 = F(t * t * (3 - 2 * t))
        h11 = F(t * t * (t - 1))
        return F(
            h00 * start_v
            + h10 * (start_d * (end_x - start_x))
            + h01 * end_v
            + h11 * (end_d * (end_x - start_x))
        )

    def _eval_array(self, ctx, x, y, z):
        loc = self.location.eval_array(ctx, x, y, z)
        n = len(self.points)

        # Segment index per point: #{points with location <= loc}, in the f32
        # comparison domain the scalar scan uses (numpy absorbs the Python-float
        # endpoint into f32). A NaN location sorts as "not >= anything", so the
        # scalar scan stops at 0; searchsorted alone would say n -- mask it.
        idx = np.searchsorted(self._locs_f32, loc, side="right")
        idx = np.where(np.isnan(loc), 0, idx)

        out = np.empty(loc.shape, dtype=np.float32)

        low = idx == 0
        if low.any():
            loc0, d0 = self._tail0
            out[low] = self.points[0][1].eval_array(ctx, x[low], y[low], z[low]) + d0 * (loc[low] - loc0)

        high = idx == n
        if high.any():
            locn, dn = self._tailn
            out[high] = self.points[n - 1][1].eval_array(ctx, x[high], y[high], z[high]) + dn * (loc[high] - locn)

        mid = (~low) & (~high)
        if mid.any():
            seg = idx[mid] - 1
            start_v = np.empty(mid.sum(), dtype=np.float32)
            end_v = np.empty(mid.sum(), dtype=np.float32)
            # Evaluate each bracketing value node only where it is needed.
            for s in np.unique(seg):
                where = seg == s
                px = x[mid][where]
                py = y[mid][where]
                pz = z[mid][where]
                start_v[where] = self.points[s][1].eval_array(ctx, px, py, pz)
                end_v[where] = self.points[s + 1][1].eval_array(ctx, px, py, pz)

            t = (loc[mid] - self._startx_f32[seg]) / self._span_f32[seg]
            h00 = (1 + 2 * t) * (1 - t) * (1 - t)
            h10 = t * (1 - t) * (1 - t)
            h01 = t * t * (3 - 2 * t)
            h11 = t * t * (t - 1)
            out[mid] = (
                h00 * start_v
                + h10 * self._dstart_f32[seg]
                + h01 * end_v
                + h11 * self._dend_f32[seg]
            )
        return out


class WeirdScaledSampler(Node):
    """`minecraft:weird_scaled_sampler` -- a noise rescaled by a rarity band.

    The input picks a *rarity*; the named noise is then sampled at
    `pos / rarity` and the result multiplied by that rarity. Where the rarity
    is large the noise varies slowly, which is how vanilla carves the long,
    wide "spaghetti" tunnels instead of uniform-diameter pipes.
    """

    __slots__ = ("input", "_sampler", "_table")

    def __init__(self, inp: Node, name: str, mapper: str, library: NoiseLibrary) -> None:
        self.input = inp
        self._sampler = library.sampler(name)
        self._table = _RARITY_TYPE_1 if mapper.upper().replace("_", "") == "TYPE1" else _RARITY_TYPE_2
        # `abs(...)` means the result is never negative, and the largest rarity
        # band (3.0 for type_1) is the widest bound.
        super().__init__(F(0.0), F(self._sampler.max_value * 3.0))

    def eval(self, ctx, x, y, z):
        rarity = _rarity(self._table, float(self.input.eval(ctx, x, y, z)))
        return F(abs(rarity * self._sampler.get(x / rarity, y / rarity, z / rarity)))

    def _eval_array(self, ctx, x, y, z):
        v = self.input.eval_array(ctx, x, y, z)
        out = np.empty(v.shape, dtype=np.float32)
        # Bands in table order; a NaN input falls through every `v < threshold`
        # and lands in the last band, which is what `_rarity`'s fallthrough does.
        covered = np.zeros(v.shape, dtype=bool)
        last = len(self._table) - 1
        for i, (threshold, rarity) in enumerate(self._table):
            band = (~covered) if i == last else (~covered) & (v < threshold)
            if band.any():
                out[band] = np.abs(
                    rarity * self._sampler.get_array(x[band] / rarity, y[band] / rarity, z[band] / rarity)
                )
            covered |= band
        return out


class BlendOffset(Node):
    """No beardifier structures means a constant 0."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(F(0.0), F(0.0))

    def eval(self, ctx, x, y, z):
        return F(0.0)

    def _eval_array(self, ctx, x, y, z):
        return np.zeros(x.shape, dtype=np.float32)


class BlendAlpha(Node):
    """No beardifier structures means a constant 1 -- the blend is a pass-through."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(F(1.0), F(1.0))

    def eval(self, ctx, x, y, z):
        return F(1.0)

    def _eval_array(self, ctx, x, y, z):
        return np.ones(x.shape, dtype=np.float32)


class BlendDensity(Node):
    __slots__ = ("input",)

    def __init__(self, inp: Node) -> None:
        self.input = inp
        super().__init__(inp.min_value, inp.max_value)

    def eval(self, ctx, x, y, z):
        return F(self.input.eval(ctx, x, y, z))

    def _eval_array(self, ctx, x, y, z):
        return self.input.eval_array(ctx, x, y, z)


class EndIslands(Node):
    """`minecraft:end_islands` -- a sparse 2D field driving the End's islands.

    Scans a 25x25 neighbourhood of "candidate" cells whose 2D simplex noise
    falls below -0.9, and keeps the highest falloff. It is 625 simplex samples
    per call, which is why the End is slow to generate and why this is the only
    node that still uses the legacy LCG.
    """

    __slots__ = ("_sampler",)

    def __init__(self, seed: int) -> None:
        random = LegacyRandom(seed)
        random.skip(17292)
        from world.vanilla.perlin import _GRAD_F32  # noqa: F401  (keeps import local)
        self._sampler = _SimplexNoise(random)
        self.min_value = F(-0.84375)
        self.max_value = F(0.5625)

    def _sample_2d(self, x: int, z: int) -> float:
        i, j = x // 2, z // 2
        k, l = x % 2, z % 2
        best = -100.0
        for m in range(-12, 13):
            for n in range(-12, 13):
                o, p = i + m, j + n
                if (o * o + p * p) > 4096 and self._sampler.sample_2d(float(o), float(p)) < -0.9:
                    g = (math.fmod(abs(float(o)) * 3439.0 + abs(float(p)) * 147.0, 13.0) + 9.0)
                    r = 100.0 - g * math.hypot(float(k - m * 2), float(l - n * 2))
                    best = max(best, min(max(r, -100.0), 80.0))
        return best

    def eval(self, ctx, x, y, z):
        return F((self._sample_2d(x // 8, z // 8) - 8.0) / 128.0)

    def _eval_array(self, ctx, x, y, z):
        # The End is generated rarely enough that a point loop is honest here;
        # vectorising 625 simplex samples per point is not where the time goes.
        flat_x = x.ravel()
        flat_z = z.ravel()
        out = np.empty(flat_x.shape, dtype=np.float32)
        for i in range(flat_x.shape[0]):
            out[i] = self.eval(ctx, int(flat_x[i]), 0, int(flat_z[i]))
        return out.reshape(x.shape)


class _SimplexNoise:
    """The 2D simplex sampler the End's islands use, from the legacy LCG."""

    __slots__ = ("_perm", "_grads")

    _GRAD_2 = ((1, 1), (-1, 1), (1, -1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1))

    _F2 = 0.5 * (math.sqrt(3.0) - 1.0)
    _G2 = (3.0 - math.sqrt(3.0)) / 6.0

    def __init__(self, random: LegacyRandom) -> None:
        p = list(range(256))
        for i in range(255, 0, -1):
            j = random.next_bounded_i32(i + 1)
            p[i], p[j] = p[j], p[i]
        table = [p[(i + p[(i + 1) % 256]) % 256] for i in range(256)]
        # The doubled table is what `sample_2d` indexes: `ii + perm[jj]` and
        # `ii + 1 + perm[jj + 1]` reach 510/511, so a 256-entry table raises
        # IndexError wherever a corner index overflows. Duplicating is the
        # classic Gustavson layout and is what vanilla's 512-entry `perms` is;
        # it is equivalent to masking each lookup with & 255 instead.
        self._perm = table + table

    def sample_2d(self, xin: float, yin: float) -> float:
        perm = self._perm
        s = (xin + yin) * self._F2
        i = math.floor(xin + s)
        j = math.floor(yin + s)
        t = (i + j) * self._G2
        x0 = xin - (i - t)
        y0 = yin - (j - t)
        i1, j1 = (1, 0) if x0 > y0 else (0, 1)
        x1, y1 = x0 - i1 + self._G2, y0 - j1 + self._G2
        x2, y2 = x0 - 1.0 + 2 * self._G2, y0 - 1.0 + 2 * self._G2
        ii, jj = int(i) & 255, int(j) & 255
        n = 0.0
        for dx, dy, gi in ((x0, y0, perm[ii + perm[jj]]), (x1, y1, perm[ii + i1 + perm[jj + j1]]),
                           (x2, y2, perm[ii + 1 + perm[jj + 1]])):
            t = 0.5 - dx * dx - dy * dy
            if t > 0:
                g = self._GRAD_2[gi & 7]
                t *= t
                n += t * t * (g[0] * dx + g[1] * dy)
        return 70.0 * n


# ==========================================================================
# Wrappers: memoisation, and the one that changes values
# ==========================================================================


class CacheOnce(Node):
    """Single-slot memo on the full position. Vanilla's cheapest wrapper."""

    __slots__ = ("input", "_key", "_value")

    def __init__(self, inner: Node) -> None:
        self.input = inner
        self._key = None
        self._value = F(0.0)
        self.min_value = inner.min_value
        self.max_value = inner.max_value

    def eval(self, ctx, x, y, z):
        key = (x, y, z)
        if self._key == key:
            return self._value
        value = self.input.eval(ctx, x, y, z)
        self._key = key
        self._value = value
        return value

    def _eval_array(self, ctx, x, y, z):
        return _eval_array_unique3(self.input, ctx, x, y, z)


class Cache2D(Node):
    """Memo keyed on (x, z) only -- sound because shift fields ignore Y.

    The Y-blindness is not an approximation, it is why vanilla is allowed to do
    it. A `Cache2D` over something that *did* depend on Y would return the wrong
    answer, and would do so silently.
    """

    __slots__ = ("input", "_entries")

    def __init__(self, inner: Node) -> None:
        self.input = inner
        self._entries: Dict[Tuple[int, int], np.float32] = {}
        self.min_value = inner.min_value
        self.max_value = inner.max_value

    def eval(self, ctx, x, y, z):
        key = (x, z)
        hit = self._entries.get(key)
        if hit is not None:
            return hit
        value = self.input.eval(ctx, x, y, z)
        self._entries[key] = value
        return value

    def _eval_array(self, ctx, x, y, z):
        return _eval_array_unique2(self.input, ctx, x, y, z, x, z)


class FlatCache(Node):
    """Memo for one chunk's columns, dropped when the chunk context changes.

    Distinct from `Cache2D` only in lifetime: `FlatCache` lives for a single
    chunk generation pass, `Cache2D` for the life of the router. The values are
    identical; only the invalidation differs.
    """

    __slots__ = ("input", "_entries", "_context")

    def __init__(self, inner: Node) -> None:
        self.input = inner
        self._entries: Dict[Tuple[int, int], np.float32] = {}
        self._context = None
        self.min_value = inner.min_value
        self.max_value = inner.max_value

    def begin_context(self, context) -> None:
        """Invalidate. Call once per chunk before evaluating."""
        if self._context != context:
            self._context = context
            self._entries.clear()

    def eval(self, ctx, x, y, z):
        key = (x, z)
        hit = self._entries.get(key)
        if hit is not None:
            return hit
        value = self.input.eval(ctx, x, y, z)
        self._entries[key] = value
        return value

    def _eval_array(self, ctx, x, y, z):
        return _eval_array_unique2(self.input, ctx, x, y, z, x, z)


def _eval_array_unique3(inner: Node, ctx, x, y, z) -> np.ndarray:
    """Evaluate `inner` once per unique (x, y, z) in the batch, then scatter.

    This is what the memoisation wrappers compute anyway -- the value at each
    position -- just batched: the scalar `CacheOnce` would return the first
    result for a repeated position, and the value is the same either way.
    """
    shape = x.shape
    x, y, z = x.ravel(), y.ravel(), z.ravel()
    first, inv, sx, sy, sz = ctx._arr_batch.decomp(
        ("u3", id(x), id(y), id(z)), (x, y, z),
        lambda: _split_unique3(x, y, z),
    )
    vals = inner.eval_array(ctx, sx, sy, sz)
    return vals[inv].reshape(shape)


#: Coordinate ranges the packed-unique fast path can address: 26 bits of X and
#: Z (the world border sits at ~30M, well inside) and 12 bits of Y (vanilla
#: worlds span -2048..2047 with room to spare). Outside these the helpers fall
#: back to `np.unique(axis=0)`, which is slower but takes any int64.
_PACK_XZ = 1 << 25
_PACK_Y = 1 << 11


def _unique_rows(x, y, z):
    """First-occurrence index and inverse for unique (x, y, z) rows.

    The three coordinates are packed into one uint64 and sorted as a 1-D
    array. `np.unique(axis=0)` would sort 24-byte void rows instead, which is
    several times slower on a 100k-point batch -- and on a chunk fill this
    split runs once per distinct point set, several times per chunk.
    """
    if (x.min(initial=0) >= -_PACK_XZ and x.max(initial=0) < _PACK_XZ
            and z.min(initial=0) >= -_PACK_XZ and z.max(initial=0) < _PACK_XZ
            and y.min(initial=0) >= -_PACK_Y and y.max(initial=0) < _PACK_Y):
        key = ((x + _PACK_XZ).astype(np.uint64) << np.uint64(38)
               | (y + _PACK_Y).astype(np.uint64) << np.uint64(26)
               | (z + _PACK_XZ).astype(np.uint64))
        _, first, inv = np.unique(key, return_index=True, return_inverse=True)
        return first, inv
    key = np.stack([x, y, z], axis=1)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return first, inv


def _unique_rows2(kx, kz):
    """As `_unique_rows`, for (x, z) columns. 26+26 bits fit a plain int64."""
    if (kx.min(initial=0) >= -_PACK_XZ and kx.max(initial=0) < _PACK_XZ
            and kz.min(initial=0) >= -_PACK_XZ and kz.max(initial=0) < _PACK_XZ):
        key = (kx + _PACK_XZ).astype(np.int64) << 26 | (kz + _PACK_XZ)
        _, first, inv = np.unique(key, return_index=True, return_inverse=True)
        return first, inv
    key = np.stack([kx, kz], axis=1)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return first, inv


def _split_unique3(x, y, z):
    first, inv = _unique_rows(x, y, z)
    return first, inv, x[first], y[first], z[first]


def _eval_array_unique2(inner: Node, ctx, x, y, z, kx, kz) -> np.ndarray:
    """Evaluate `inner` once per unique (x, z) column in the batch.

    `y` is taken from the first row of each column -- exactly what the scalar
    memo stores. For a well-formed (Y-blind) input any `y` gives the same
    answer; for a misused one this reproduces the scalar cache's silent
    behaviour rather than inventing a different wrong answer.
    """
    shape = x.shape
    x, y, z, kx, kz = x.ravel(), y.ravel(), z.ravel(), kx.ravel(), kz.ravel()
    first, inv, sx, sy, sz = ctx._arr_batch.decomp(
        ("u2", id(kx), id(kz), id(x), id(y), id(z)), (kx, kz, x, y, z),
        lambda: _split_unique2(x, y, z, kx, kz),
    )
    vals = inner.eval_array(ctx, sx, sy, sz)
    return vals[inv].reshape(shape)


def _split_unique2(x, y, z, kx, kz):
    first, inv = _unique_rows2(kx, kz)
    return first, inv, x[first], y[first], z[first]


class Interpolated(Node):
    """Samples only cell corners and trilinearly blends between them.

    The one wrapper that changes values. Inside a cell the field is *not* the
    wrapped function -- it is a straight-line blend between the eight corners,
    so a 4x8x4 cell replaces 128 evaluations with 8. On a noisy subtree that is
    a visible change, and it is the reason vanilla's terrain is smoother inside
    cells than the raw noise would be.

    Vanilla computes the eight corners once per *cell*, not once per block, so
    the scalar path caches the current cell's corners (`_corners`) and the
    batch path evaluates the batch's unique cells in one shot. Both are pure
    memoisation of values that were already defined per cell.
    """

    __slots__ = ("input", "cell_xz", "cell_y", "inv_xz", "inv_y", "_cell", "_corners")

    def __init__(self, inner: Node, cell_size: Tuple[int, int]) -> None:
        self.input = inner
        self.cell_xz, self.cell_y = cell_size
        self.inv_xz = F(1.0 / cell_size[0])
        self.inv_y = F(1.0 / cell_size[1])
        self._cell = None
        self._corners = None
        self.min_value = inner.min_value
        self.max_value = inner.max_value

    def eval(self, ctx, x, y, z):
        cxz, cy = self.cell_xz, self.cell_y
        x_in = x % cxz
        y_in = y % cy
        z_in = z % cxz
        if x_in == 0 and y_in == 0 and z_in == 0:
            return self.input.eval(ctx, x, y, z)

        ox, oy, oz = x - x_in, y - y_in, z - z_in
        corners = self._corners
        if corners is None or self._cell != (ox, oy, oz):
            v = self.input
            sx, sy, sz = cxz, cy, cxz
            corners = (
                v.eval(ctx, ox, oy, oz),
                v.eval(ctx, ox + sx, oy, oz),
                v.eval(ctx, ox, oy + sy, oz),
                v.eval(ctx, ox + sx, oy + sy, oz),
                v.eval(ctx, ox, oy, oz + sz),
                v.eval(ctx, ox + sx, oy, oz + sz),
                v.eval(ctx, ox, oy + sy, oz + sz),
                v.eval(ctx, ox + sx, oy + sy, oz + sz),
            )
            self._cell = (ox, oy, oz)
            self._corners = corners
        return lerp3(
            F(x_in * self.inv_xz), F(y_in * self.inv_y), F(z_in * self.inv_xz),
            corners[0], corners[1], corners[2], corners[3],
            corners[4], corners[5], corners[6], corners[7],
        )

    def _eval_array(self, ctx, x, y, z):
        shape = x.shape
        x, y, z = x.ravel(), y.ravel(), z.ravel()
        cxz, cy = self.cell_xz, self.cell_y
        x_in, y_in, z_in, uniq, inv = ctx._arr_batch.decomp(
            ("cell", id(x), id(y), id(z), cxz, cy), (x, y, z),
            lambda: _split_cells(x, y, z, cxz, cy),
        )
        c = uniq.shape[0]

        sx, sy, sz = cxz, cy, cxz
        order = (
            (0, 0, 0), (sx, 0, 0), (0, sy, 0), (sx, sy, 0),
            (0, 0, sz), (sx, 0, sz), (0, sy, sz), (sx, sy, sz),
        )
        cx = np.concatenate([uniq[:, 0] + d[0] for d in order])
        cy = np.concatenate([uniq[:, 1] + d[1] for d in order])
        cz = np.concatenate([uniq[:, 2] + d[2] for d in order])
        vals = self.input.eval_array(ctx, cx, cy, cz).reshape(8, c)

        dx = (x_in * self.inv_xz).astype(np.float32)
        dy = (y_in * self.inv_y).astype(np.float32)
        dz = (z_in * self.inv_xz).astype(np.float32)
        out = _lerp3_arr_batch(
            dx, dy, dz,
            vals[0][inv], vals[1][inv], vals[2][inv], vals[3][inv],
            vals[4][inv], vals[5][inv], vals[6][inv], vals[7][inv],
        )

        # Points on a cell origin take the scalar path's shortcut -- the direct
        # corner value, not the blend. The two agree whenever the corners are
        # finite, but the blend's `0 * (b - a)` turns into NaN the moment some
        # other corner is +-inf (an INVERTed zero, say), so pin the shortcut.
        corner = (x_in == 0) & (y_in == 0) & (z_in == 0)
        if corner.any():
            out[corner] = vals[0][inv[corner]]
        return out.reshape(shape)


def _split_cells(x, y, z, cxz, cy):
    """Split a batch into unique interpolation cells: in-cell offsets plus the
    unique cell origins and each point's cell index."""
    x_in = x % cxz
    y_in = y % cy
    z_in = z % cxz
    first, inv = _unique_rows(x - x_in, y - y_in, z - z_in)
    uniq = np.stack([x[first] - x_in[first], y[first] - y_in[first], z[first] - z_in[first]], axis=1)
    return x_in, y_in, z_in, uniq, inv


def _lerp3_arr_batch(dx, dy, dz, v000, v100, v010, v110, v001, v101, v011, v111):
    """`perlin.lerp3`'s X-then-Y-then-Z nesting, elementwise.

    Written out rather than shared with `perlin` because the array version must
    not introduce a temporary copy of the scalar signature's argument order --
    the nesting order *is* the rounding order.
    """
    def lerp(delta, start, end):
        return start + delta * (end - start)

    x00 = lerp(dx, v000, v100)
    x10 = lerp(dx, v010, v110)
    x01 = lerp(dx, v001, v101)
    x11 = lerp(dx, v011, v111)
    return lerp(dz, lerp(dy, x00, x10), lerp(dy, x01, x11))


# ==========================================================================
# Parsing
# ==========================================================================


class Context:
    """Everything a tree needs beyond the position: samplers and the seed.

    `_cons` is the parse-time hash-cons table; `_arr_batch` is the batch memo
    of the `eval_array` call currently in flight (see `_BatchMemo`).
    """

    __slots__ = ("seed", "splitter", "noise", "interpolated_noise", "_cons", "_arr_batch")

    def __init__(self, seed: int, parameters: Dict[str, dict]) -> None:
        self.seed = seed
        splitter = Xoroshiro.from_seed(seed).next_splitter()
        self.splitter = splitter
        self.noise = NoiseLibrary(splitter, parameters)
        self.interpolated_noise = InterpolatedNoiseLibrary(splitter)
        self._cons: Dict[tuple, Node] = {}
        self._arr_batch = None

    def begin_chunk(self, chunk_x: int, chunk_z: int) -> object:
        """A fresh token per chunk; `FlatCache` nodes compare against it."""
        return (chunk_x, chunk_z)


def _freeze(obj, memo: dict):
    """A hashable structural fingerprint of a JSON subtree.

    Memoised by `id` of the JSON object while the parse is running (the caller
    holds the whole tree alive for the duration), which keeps the walk linear.
    """
    if isinstance(obj, dict):
        hit = memo.get(id(obj))
        if hit is not None:
            return hit
        out = ("d", tuple((k, _freeze(v, memo)) for k, v in sorted(obj.items())))
        memo[id(obj)] = out
        return out
    if isinstance(obj, list):
        hit = memo.get(id(obj))
        if hit is not None:
            return hit
        out = ("l", tuple(_freeze(v, memo) for v in obj))
        memo[id(obj)] = out
        return out
    return ("p", obj)


def _parse_spline(
    spline: dict, ctx: Context, cell_size: Tuple[int, int], f_memo: dict,
    pass_through: bool = False,
) -> Node:
    """Build a `SplineNode` from the dump's `_type`-tagged spline payload.

    The resolved dump stores a spline as three parallel arrays rather than a
    list of point objects, and a point's *value* is itself a spline -- so the
    structure is genuinely recursive, not just a flat curve with constant ends.
    Vanilla's terrain-shaper splines nest to two levels.
    """
    key = (_freeze(spline, f_memo), cell_size, pass_through)
    hit = ctx._cons.get(key)
    if hit is not None:
        return hit

    if spline.get("_type") == "fixed":
        built = Constant(spline["value"]["value"])
    else:
        body = spline["value"]
        location = _parse(body["locationFunction"], ctx, cell_size, pass_through, f_memo)
        points = [
            (loc, _parse_spline(val, ctx, cell_size, f_memo, pass_through), deriv)
            for loc, val, deriv in zip(body["locations"], body["values"], body["derivatives"])
        ]
        built = SplineNode(location, points)
    ctx._cons[key] = built
    return built


class PassThrough(Node):
    """A wrapper resolved away: the inner function, sampled directly.

    Vanilla builds its router in two phases, and the distinction is not
    cosmetic. The *proto* router resolves every wrapper into an indexed
    component list and evaluates without any cell machinery; the *chunk* router
    adds caching and `Interpolated` on top. Values from the first phase are a
    strict, testable subset of the second.

    Getting this wrong in either direction is a trap: compare a proto-phase
    golden value against a chunk-phase evaluation and every `Interpolated`
    interior point disagrees, which looks like a broken interpolator when the
    interpolator is fine.
    """

    __slots__ = ("input",)

    def __init__(self, inner: Node) -> None:
        self.input = inner
        super().__init__(inner.min_value, inner.max_value)

    def eval(self, ctx, x, y, z):
        return self.input.eval(ctx, x, y, z)

    def _eval_array(self, ctx, x, y, z):
        return self.input.eval_array(ctx, x, y, z)


def _parse(
    node: Any,
    ctx: Context,
    cell_size: Tuple[int, int],
    pass_through_wrappers: bool,
    f_memo: dict,
) -> Node:
    key = (_freeze(node, f_memo), cell_size, pass_through_wrappers)
    hit = ctx._cons.get(key)
    if hit is not None:
        return hit

    cls = node["_class"]
    value = node.get("value")

    if cls == "Constant":
        built = Constant(value["value"])
    elif cls == "BlendOffset":
        built = BlendOffset()
    elif cls == "BlendAlpha":
        built = BlendAlpha()
    elif cls == "BlendDensity":
        built = BlendDensity(_parse(value["input"], ctx, cell_size, pass_through_wrappers, f_memo))
    elif cls == "UnaryOperation":
        built = UnaryOperation(value["type"], _parse(value["input"], ctx, cell_size, pass_through_wrappers, f_memo))
    elif cls == "BinaryOperation":
        built = BinaryOperation(
            value["type"],
            _parse(value["argument1"], ctx, cell_size, pass_through_wrappers, f_memo),
            _parse(value["argument2"], ctx, cell_size, pass_through_wrappers, f_memo),
        )
    elif cls == "LinearOperation":
        built = LinearOperation(
            value["specificType"],
            _parse(value["input"], ctx, cell_size, pass_through_wrappers, f_memo),
            value["argument"],
        )
    elif cls == "Clamp":
        built = ClampNode(
            _parse(value["input"], ctx, cell_size, pass_through_wrappers, f_memo),
            value["minValue"], value["maxValue"],
        )
    elif cls == "YClampedGradient":
        built = YClampedGradient(
            value["fromY"], value["toY"], value["fromValue"], value["toValue"]
        )
    elif cls == "Noise":
        built = NoiseNode(value["noise"], value["xzScale"], value["yScale"], ctx.noise)
    elif cls == "ShiftA":
        built = ShiftA(value["offsetNoise"], ctx.noise)
    elif cls == "ShiftB":
        built = ShiftB(value["offsetNoise"], ctx.noise)
    elif cls == "ShiftedNoise":
        built = ShiftedNoise(
            _parse(value["shiftX"], ctx, cell_size, pass_through_wrappers, f_memo),
            _parse(value["shiftY"], ctx, cell_size, pass_through_wrappers, f_memo),
            _parse(value["shiftZ"], ctx, cell_size, pass_through_wrappers, f_memo),
            value["xzScale"], value["yScale"], value["noise"], ctx.noise,
        )
    elif cls == "InterpolatedNoiseSampler":
        built = InterpolatedNoiseSamplerNode(value, ctx.interpolated_noise)
    elif cls == "RangeChoice":
        built = RangeChoice(
            _parse(value["input"], ctx, cell_size, pass_through_wrappers, f_memo),
            value["minInclusive"], value["maxExclusive"],
            _parse(value["whenInRange"], ctx, cell_size, pass_through_wrappers, f_memo),
            _parse(value["whenOutOfRange"], ctx, cell_size, pass_through_wrappers, f_memo),
        )
    elif cls == "Spline":
        return _parse_spline(value["spline"], ctx, cell_size, f_memo, pass_through_wrappers)
    elif cls == "WeirdScaledSampler":
        built = WeirdScaledSampler(
            _parse(value["input"], ctx, cell_size, pass_through_wrappers, f_memo),
            value["noise"], value["rarityValueMapper"], ctx.noise,
        )
    elif cls == "EndIslands":
        built = EndIslands(ctx.seed)
    elif cls == "Wrapping":
        inner = _parse(value["wrapped"], ctx, cell_size, pass_through_wrappers, f_memo)
        if pass_through_wrappers:
            built = PassThrough(inner)
        else:
            kind = value["type"]
            if kind == "CacheOnce":
                built = CacheOnce(inner)
            elif kind == "Cache2D":
                built = Cache2D(inner)
            elif kind == "FlatCache":
                built = FlatCache(inner)
            elif kind == "Interpolated":
                built = Interpolated(inner, cell_size)
            else:
                raise ValueError(f"unknown wrapper type: {kind}")
    else:
        raise ValueError(f"unknown density function class: {cls}")

    ctx._cons[key] = built
    return built


def parse(
    node: Any,
    ctx: Context,
    cell_size: Tuple[int, int] = DEFAULT_CELL_SIZE,
    pass_through_wrappers: bool = False,
) -> Node:
    """Turn one resolved JSON node into an evaluable `Node`.

    `cell_size` applies to any `Interpolated` wrapper in the subtree; the dump
    does not record it per node, so it comes from the slot being parsed.

    `pass_through_wrappers` selects the proto phase, where every wrapper is
    erased. Leave it false for anything that generates real terrain.

    Structurally identical subtrees are shared (see the module docstring);
    the fingerprint memo is local to this call so it cannot outlive the JSON
    tree it describes.
    """
    f_memo: dict = {}
    return _parse(node, ctx, cell_size, pass_through_wrappers, f_memo)


#: The router slots, in the order `assets/density_function.json` names them.
#: `NOISE_ROUTER_SLOTS` in `registry/worldgen.py` maps these to the vanilla
#: slot names; keep the two in step.
SLOT_NAMES = (
    "barrierNoise", "fluidLevelFloodednessNoise", "fluidLevelSpreadNoise", "lavaNoise",
    "temperature", "vegetation", "continents", "erosion", "depth", "ridges",
    "initialDensityWithoutJaggedness", "finalDensity", "veinToggle", "veinRidged", "veinGap",
)

#: Slots whose `Interpolated` wrappers use a different cell size than the rest.
#: See `DEFAULT_CELL_SIZE` for where these come from.
_SLOT_CELL_SIZES = {"initialDensityWithoutJaggedness": SURFACE_CELL_SIZE}
_PRESET_CELL_SIZES = {"end": END_CELL_SIZE}


def cell_size_for(preset: str, slot: Optional[str] = None) -> Tuple[int, int]:
    """The `Interpolated` cell size a given slot should use."""
    if slot in _SLOT_CELL_SIZES:
        return _SLOT_CELL_SIZES[slot]
    return _PRESET_CELL_SIZES.get(preset, DEFAULT_CELL_SIZE)


def load_parameters(path: Optional[str] = None) -> Dict[str, dict]:
    """Read `assets/noise_parameters.json`."""
    if path is None:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(root, "assets", "noise_parameters.json")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_corpus(path: Optional[str] = None) -> Dict[str, Any]:
    """Read `assets/density_function_tests.json`, the acceptance corpus."""
    if path is None:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(root, "assets", "density_function_tests.json")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def collect_flat_caches(node: Node, out: Optional[list] = None, _seen: Optional[set] = None) -> list:
    """Every `FlatCache` under `node`, for per-chunk invalidation.

    Walks with a visited set because the tree is a DAG after hash-consing --
    without it a shared cluster is re-walked once per parent. Spline point
    values sit in `(loc, node, deriv)` tuples, so tuples are searched
    recursively rather than one level deep.
    """
    if out is None:
        out = []
        _seen = set()
    if id(node) in _seen:
        return out
    _seen.add(id(node))
    if isinstance(node, FlatCache):
        out.append(node)
    for slot in getattr(node, "__slots__", ()):
        child = getattr(node, slot, None)
        if isinstance(child, Node):
            collect_flat_caches(child, out, _seen)
        elif isinstance(child, (list, tuple)):
            stack = list(child)
            while stack:
                item = stack.pop()
                if isinstance(item, Node):
                    collect_flat_caches(item, out, _seen)
                elif isinstance(item, (list, tuple)):
                    stack.extend(item)
    return out


__all__ = [
    "Context",
    "DEFAULT_CELL_SIZE",
    "END_CELL_SIZE",
    "Interpolated",
    "Node",
    "SLOT_NAMES",
    "SURFACE_CELL_SIZE",
    "cell_size_for",
    "collect_flat_caches",
    "load_corpus",
    "load_parameters",
    "parse",
]
