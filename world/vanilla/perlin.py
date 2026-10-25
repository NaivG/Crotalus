"""
Vanilla's Perlin noise, rebuilt so its output matches the Java original bit for
bit.

This is not a Perlin implementation in the general sense -- it is a
transliteration of `net.minecraft.util.math.noise.PerlinNoise`,
`SmearedPerlinNoise` and `NormalNoise`, because "looks like Perlin noise" is not
the requirement. The requirement is that a player who types the same seed into
Crotalus and into 1.21.11 sees the same coastline.

    from world.vanilla.perlin import PerlinNoise, NormalNoise
    from world.vanilla.random_source import Xoroshiro

    noise = PerlinNoise(Xoroshiro.from_seed(42))
    noise.get(10.0, 64.0, -3.0)

Three things make this hard to get right, and each is commented where it appears
below:

* **Coordinates are f64, everything downstream is f32.** Noise is sampled at
  `double` precision, the corner gradients are computed at `float`, and the
  interpolation between them happens at `float` too. Mixing the two up shifts
  results by more than a block.
* **The permutation table is seeded, not fixed.** It comes from a
  Fisher-Yates shuffle driven by the noise's own RNG, and it draws exactly 262
  values (3 offsets + 255 swaps) no matter what. A skip that is off by one
  desynchronises every noise created after it.
* **The octave amplitudes are normalised by a formula nobody remembers.** See
  `parity_base_amplitude` -- it is three layers of correction factors that exist
  purely to keep the noise's standard deviation near 1/3 across octave counts.

Two execution paths, one set of numbers
---------------------------------------
Every sampler exposes `get` (one point) and `get_array` (a numpy batch). The
scalar path runs its hot chain -- wrap, floor, eight gradient dots, the lerp
tree -- inside a numba kernel when numba is installed, so a single sample costs
one Python-to-JIT transition instead of a dozen interpreted function calls. The
array path is pure numpy on purpose: batching already removes the per-point
Python overhead, and the simpler build keeps the two paths trivially comparable.
Both paths round to f32 at exactly the places Java's `float` casts do, which is
what `tools/check_density_vectorized.py` verifies point by point.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover - numba is a declared dependency
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator if not args else decorator(args[0])


#: The 16 gradient directions. Vanilla indexes this table with
#: `permutation[hash] & 15`, so the *order* is load-bearing, not just the set.
GRADIENTS: Tuple[Tuple[float, float, float], ...] = (
    (1.0, 1.0, 0.0),
    (-1.0, 1.0, 0.0),
    (1.0, -1.0, 0.0),
    (-1.0, -1.0, 0.0),
    (1.0, 0.0, 1.0),
    (-1.0, 0.0, 1.0),
    (1.0, 0.0, -1.0),
    (-1.0, 0.0, -1.0),
    (0.0, 1.0, 1.0),
    (0.0, -1.0, 1.0),
    (0.0, 1.0, -1.0),
    (0.0, -1.0, -1.0),
    (1.0, 1.0, 0.0),
    (0.0, -1.0, 1.0),
    (-1.0, 1.0, 0.0),
    (0.0, -1.0, -1.0),
)

_GRAD_F32 = np.asarray(GRADIENTS, dtype=np.float32)

#: Perlin noise has a period of 2^16; vanilla wraps inputs into this range.
_ROUND_OFF = 33554432.0  # 2^25
_HALF_ROUND_OFF = float.fromhex("0x1.6A09E667F3BCDP+26")  # 1.5 * 2^25
_OFFSET_SCALE = 256.0

#: The smeared variant shifts each unit of Y by this much, quantised to whole
#: units, so that the noise is constant across a slab of Y and only steps at
#: boundaries. That blockiness is intentional -- it is what makes the
#: cave-carving noise look layered rather than smooth.
_SMEARED_RANGE_BASE = 2.0
_SHIFT_UP_EPSILON = float(np.float32(1.0e-7))

#: Normalisation constants for the octave stack. `STANDARD_DEVIATION` is the
#: measured deviation of a single octave; dividing by 3 keeps a well-formed
#: stack inside [-1, 1] the way the density functions expect.
_STANDARD_DEVIATION = 0.2702247831245211
_TARGET_DEVIATION = 1.0 / 3.0
_INPUT_FACTOR = 1.0181268882175227
_PERSISTENCE = 0.5
_LACUNARITY = 2.0

#: f32 constants for the kernels. Written explicitly rather than as bare
#: literals because a bare `6.0` is a *double* in numba and would silently
#: promote the whole fade curve to f64 -- one rounding wider than Java's
#: `float` math, and enough to move the surface by a block over a few octaves.
_F32_0 = np.float32(0.0)
_F32_1 = np.float32(1.0)
_F32_6 = np.float32(6.0)
_F32_10 = np.float32(10.0)
_F32_15 = np.float32(15.0)


def _f32(value) -> np.float32:
    """Round to f32 -- Java's `float`. Applied at every vanilla `float` cast."""
    return np.float32(value)


# ==========================================================================
# Kernels
#
# One code body each, runnable interpreted and compilable by numba. Everything
# below the wrap/floor stage is f32 arithmetic with one rounding per operator,
# matching both numpy scalar ops and Java's strict float math. The wrappers at
# the bottom of the file box the results back to np.float32 so the Python-side
# type of every sample stays what it was before the kernels existed.
# ==========================================================================


@njit(cache=True, nogil=True)
def _wrap_k(x: float) -> float:
    """Fold a coordinate into the noise's period.

    Coordinates far outside +/-2^25 are folded; near ones pass through
    untouched. The `x / ROUND_OFF + 0.5` floor is the "round half toward zero"
    trick that makes the fold symmetric about the origin.
    """
    if -_HALF_ROUND_OFF < x < _HALF_ROUND_OFF:
        return x
    return x - math.floor(x / _ROUND_OFF + 0.5) * _ROUND_OFF


@njit(cache=True, nogil=True)
def _smoothstep_k(x):
    """The 6t^5 - 15t^4 + 10t^3 fade curve, at f32 precision throughout."""
    return x * x * x * (x * (x * _F32_6 - _F32_15) + _F32_10)


@njit(cache=True, nogil=True)
def _lerp_k(delta, start, end):
    return start + delta * (end - start)


@njit(cache=True, nogil=True)
def _lerp3_k(delta_x, delta_y, delta_z,
             x0y0z0, x1y0z0, x0y1z0, x1y1z0,
             x0y0z1, x1y0z1, x0y1z1, x1y1z1):
    """Trilinear blend, interpolating X first, then Y, then Z.

    The nesting order is not arbitrary: it is the order the f32 rounding
    happens in, and f32 addition is not associative. Reordering these lerps
    changes the low bits of every sample.
    """
    x00 = _lerp_k(delta_x, x0y0z0, x1y0z0)
    x10 = _lerp_k(delta_x, x0y1z0, x1y1z0)
    x01 = _lerp_k(delta_x, x0y0z1, x1y0z1)
    x11 = _lerp_k(delta_x, x0y1z1, x1y1z1)
    return _lerp_k(delta_z, _lerp_k(delta_y, x00, x10), _lerp_k(delta_y, x01, x11))


@njit(cache=True, nogil=True)
def _grad_dot_k(hash_value, x, y, z):
    g = _GRAD_F32[hash_value & 15]
    return g[0] * x + g[1] * y + g[2] * z


@njit(cache=True, nogil=True)
def _sample_and_lerp_k(perm, x, y, z, relative_x, relative_y, relative_z, original_relative_y):
    """Evaluate the eight corners and blend.

    `original_relative_y` exists only for the smeared variant: it is the
    *unfudged* fractional Y, used to fade the Y axis, while `relative_y` is
    the fudged one used for the gradient dot products. Passing them the same
    value is what plain `PerlinNoise` does.
    """
    x0 = perm[x & 0xFF]
    x1 = perm[(x + 1) & 0xFF]
    xy00 = perm[(x0 + y) & 0xFF]
    xy01 = perm[(x0 + y + 1) & 0xFF]
    xy10 = perm[(x1 + y) & 0xFF]
    xy11 = perm[(x1 + y + 1) & 0xFF]

    one = _F32_1
    d000 = _grad_dot_k(perm[(xy00 + z) & 0xFF], relative_x, relative_y, relative_z)
    d100 = _grad_dot_k(perm[(xy10 + z) & 0xFF], relative_x - one, relative_y, relative_z)
    d010 = _grad_dot_k(perm[(xy01 + z) & 0xFF], relative_x, relative_y - one, relative_z)
    d110 = _grad_dot_k(perm[(xy11 + z) & 0xFF], relative_x - one, relative_y - one, relative_z)
    d001 = _grad_dot_k(perm[(xy00 + z + 1) & 0xFF], relative_x, relative_y, relative_z - one)
    d101 = _grad_dot_k(perm[(xy10 + z + 1) & 0xFF], relative_x - one, relative_y, relative_z - one)
    d011 = _grad_dot_k(perm[(xy01 + z + 1) & 0xFF], relative_x, relative_y - one, relative_z - one)
    d111 = _grad_dot_k(perm[(xy11 + z + 1) & 0xFF], relative_x - one, relative_y - one, relative_z - one)

    return _lerp3_k(
        _smoothstep_k(relative_x),
        _smoothstep_k(original_relative_y),
        _smoothstep_k(relative_z),
        d000, d100, d010, d110, d001, d101, d011, d111,
    )


@njit(cache=True, nogil=True)
def _perlin_get_k(perm, offset_x, offset_y, offset_z, x, y, z):
    fx = _wrap_k(x) + offset_x
    fy = _wrap_k(y) + offset_y
    fz = _wrap_k(z) + offset_z

    floor_x = math.floor(fx)
    floor_y = math.floor(fy)
    floor_z = math.floor(fz)
    rx = np.float32(fx - floor_x)
    ry = np.float32(fy - floor_y)
    rz = np.float32(fz - floor_z)

    return _sample_and_lerp_k(perm, int(floor_x), int(floor_y), int(floor_z), rx, ry, rz, ry)


@njit(cache=True, nogil=True)
def _smeared_get_k(perm, offset_x, offset_y, offset_z, fudge_y_scale, x, original_y, z):
    fx = _wrap_k(x) + offset_x
    fy = _wrap_k(original_y) + offset_y
    fz = _wrap_k(z) + offset_z

    floor_x = math.floor(fx)
    floor_y = math.floor(fy)
    floor_z = math.floor(fz)
    rx = np.float32(fx - floor_x)
    relative_y = fy - floor_y  # f64 on purpose
    rz = np.float32(fz - floor_z)

    # The fudge: quantise the fractional Y down to a whole slab. `limit` picks
    # the raw input Y when it sits inside [0, relative_y), which is vanilla's
    # exact fudge rule -- odd, but the goldens were verified against it.
    limit = original_y if (original_y >= 0.0 and original_y < relative_y) else relative_y
    fudged = np.float32(relative_y - math.floor(limit / fudge_y_scale + _SHIFT_UP_EPSILON) * fudge_y_scale)
    return _sample_and_lerp_k(
        perm, int(floor_x), int(floor_y), int(floor_z), rx, fudged, rz, np.float32(relative_y)
    )


@njit(cache=True, nogil=True)
def _stack_get_k(perm, offsets, freqs, amps, x, y, z):
    """A sum of plain octaves, each sampled at its own frequency.

    `perm` is (L, 256) int64, `offsets` (L, 3) f64, `freqs` (L,) f64 and
    `amps` (L,) f32. The accumulation is `value + amplitude * sample`, one f32
    rounding per octave, exactly as `NoiseStack.get` did it in Python.
    """
    value = _F32_0
    for i in range(freqs.shape[0]):
        f = freqs[i]
        sample = _perlin_get_k(perm[i], offsets[i, 0], offsets[i, 1], offsets[i, 2],
                               x * f, y * f, z * f)
        value = value + amps[i] * sample
    return value


@njit(cache=True, nogil=True)
def _smeared_stack_get_k(perm, offsets, freqs, amps, fudges, x, y, z):
    value = _F32_0
    for i in range(freqs.shape[0]):
        f = freqs[i]
        sample = _smeared_get_k(perm[i], offsets[i, 0], offsets[i, 1], offsets[i, 2], fudges[i],
                                x * f, y * f, z * f)
        value = value + amps[i] * sample
    return value


@njit(cache=True, nogil=True)
def _perlin_get_arr_k(perm, offset_x, offset_y, offset_z, xs, ys, zs):
    """`_perlin_get_k` over a 1-D batch. Serial on purpose: the batch path
    runs inside the project's chunk-worker threads, and a nested numba team
    per worker would oversubscribe. `nogil` still lets workers overlap."""
    n = xs.shape[0]
    out = np.empty(n, np.float32)
    for i in range(n):
        out[i] = _perlin_get_k(perm, offset_x, offset_y, offset_z, xs[i], ys[i], zs[i])
    return out


@njit(cache=True, nogil=True)
def _smeared_get_arr_k(perm, offset_x, offset_y, offset_z, fudge_y_scale, xs, ys, zs):
    n = xs.shape[0]
    out = np.empty(n, np.float32)
    for i in range(n):
        out[i] = _smeared_get_k(perm, offset_x, offset_y, offset_z, fudge_y_scale,
                                xs[i], ys[i], zs[i])
    return out


# ==========================================================================
# Public single-point API (thin boxing wrappers over the kernels)
# ==========================================================================


def smoothstep(x) -> np.float32:
    """The 6t^5 - 15t^4 + 10t^3 fade curve, at f32 precision throughout."""
    return _f32(_smoothstep_k(_f32(x)))


def lerp(delta, start, end) -> np.float32:
    return _f32(_lerp_k(_f32(delta), _f32(start), _f32(end)))


def lerp3(
    delta_x,
    delta_y,
    delta_z,
    x0y0z0,
    x1y0z0,
    x0y1z0,
    x1y1z0,
    x0y0z1,
    x1y0z1,
    x0y1z1,
    x1y1z1,
) -> np.float32:
    """Trilinear blend, interpolating X first, then Y, then Z.

    The nesting order is not arbitrary: it is the order the f32 rounding
    happens in, and f32 addition is not associative. Reordering these lerps
    changes the low bits of every sample.
    """
    return _f32(_lerp3_k(
        _f32(delta_x), _f32(delta_y), _f32(delta_z),
        _f32(x0y0z0), _f32(x1y0z0), _f32(x0y1z0), _f32(x1y1z0),
        _f32(x0y0z1), _f32(x1y0z1), _f32(x0y1z1), _f32(x1y1z1),
    ))


def wrap(x: float) -> float:
    """Fold a coordinate into the noise's period (see `_wrap_k`)."""
    return float(_wrap_k(x))


# ==========================================================================
# Array kernels (pure numpy -- see the module docstring for why)
# ==========================================================================


def _wrap_arr(xs: np.ndarray) -> np.ndarray:
    folded = xs - np.floor(xs / _ROUND_OFF + 0.5) * _ROUND_OFF
    return np.where((xs > -_HALF_ROUND_OFF) & (xs < _HALF_ROUND_OFF), xs, folded)


def _smoothstep_arr(x: np.ndarray) -> np.ndarray:
    return x * x * x * (x * (x * _F32_6 - _F32_15) + _F32_10)


def _lerp_arr(delta, start, end):
    return start + delta * (end - start)


def _lerp3_arr(delta_x, delta_y, delta_z,
               x0y0z0, x1y0z0, x0y1z0, x1y1z0,
               x0y0z1, x1y0z1, x0y1z1, x1y1z1):
    x00 = _lerp_arr(delta_x, x0y0z0, x1y0z0)
    x10 = _lerp_arr(delta_x, x0y1z0, x1y1z0)
    x01 = _lerp_arr(delta_x, x0y0z1, x1y0z1)
    x11 = _lerp_arr(delta_x, x0y1z1, x1y1z1)
    return _lerp_arr(delta_z, _lerp_arr(delta_y, x00, x10), _lerp_arr(delta_y, x01, x11))


def _grad_dot_arr(hash_value: np.ndarray, x, y, z):
    g = _GRAD_F32[hash_value & 15]
    return g[..., 0] * x + g[..., 1] * y + g[..., 2] * z


def _sample_and_lerp_arr(perm, ix, iy, iz, rx, ry, rz, ory):
    x0 = perm[ix & 0xFF]
    x1 = perm[(ix + 1) & 0xFF]
    xy00 = perm[(x0 + iy) & 0xFF]
    xy01 = perm[(x0 + iy + 1) & 0xFF]
    xy10 = perm[(x1 + iy) & 0xFF]
    xy11 = perm[(x1 + iy + 1) & 0xFF]

    one = _F32_1
    d000 = _grad_dot_arr(perm[(xy00 + iz) & 0xFF], rx, ry, rz)
    d100 = _grad_dot_arr(perm[(xy10 + iz) & 0xFF], rx - one, ry, rz)
    d010 = _grad_dot_arr(perm[(xy01 + iz) & 0xFF], rx, ry - one, rz)
    d110 = _grad_dot_arr(perm[(xy11 + iz) & 0xFF], rx - one, ry - one, rz)
    d001 = _grad_dot_arr(perm[(xy00 + iz + 1) & 0xFF], rx, ry, rz - one)
    d101 = _grad_dot_arr(perm[(xy10 + iz + 1) & 0xFF], rx - one, ry, rz - one)
    d011 = _grad_dot_arr(perm[(xy01 + iz + 1) & 0xFF], rx, ry - one, rz - one)
    d111 = _grad_dot_arr(perm[(xy11 + iz + 1) & 0xFF], rx - one, ry - one, rz - one)

    return _lerp3_arr(
        _smoothstep_arr(rx), _smoothstep_arr(ory), _smoothstep_arr(rz),
        d000, d100, d010, d110, d001, d101, d011, d111,
    )


def _perlin_get_arr_np(perm, offset_x, offset_y, offset_z, xs, ys, zs):
    fx = _wrap_arr(np.asarray(xs, dtype=np.float64)) + offset_x
    fy = _wrap_arr(np.asarray(ys, dtype=np.float64)) + offset_y
    fz = _wrap_arr(np.asarray(zs, dtype=np.float64)) + offset_z

    floor_x = np.floor(fx)
    floor_y = np.floor(fy)
    floor_z = np.floor(fz)
    rx = (fx - floor_x).astype(np.float32)
    ry = (fy - floor_y).astype(np.float32)
    rz = (fz - floor_z).astype(np.float32)

    return _sample_and_lerp_arr(
        perm,
        floor_x.astype(np.int64), floor_y.astype(np.int64), floor_z.astype(np.int64),
        rx, ry, rz, ry,
    )


def _smeared_get_arr_np(perm, offset_x, offset_y, offset_z, fudge_y_scale, xs, ys, zs):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    zs = np.asarray(zs, dtype=np.float64)
    fx = _wrap_arr(xs) + offset_x
    fy = _wrap_arr(ys) + offset_y
    fz = _wrap_arr(zs) + offset_z

    floor_x = np.floor(fx)
    floor_y = np.floor(fy)
    floor_z = np.floor(fz)
    rx = (fx - floor_x).astype(np.float32)
    relative_y = fy - floor_y  # f64 on purpose
    rz = (fz - floor_z).astype(np.float32)

    limit = np.where((ys >= 0.0) & (ys < relative_y), ys, relative_y)
    fudged = (relative_y - np.floor(limit / fudge_y_scale + _SHIFT_UP_EPSILON) * fudge_y_scale).astype(np.float32)
    return _sample_and_lerp_arr(
        perm,
        floor_x.astype(np.int64), floor_y.astype(np.int64), floor_z.astype(np.int64),
        rx, fudged, rz, relative_y.astype(np.float32),
    )


def _perlin_get_arr(perm, offset_x, offset_y, offset_z, xs, ys, zs):
    """Batch sampler: numba kernel when available, numpy fallback otherwise.

    Inputs are raveled for the kernel and the output reshaped back, so callers
    may pass any shape. The kernel loops `_perlin_get_k` -- the same function
    the scalar path runs -- so the batch path cannot drift from the scalar one.
    """
    shape = np.broadcast(xs, ys, zs).shape
    fx = np.ascontiguousarray(np.asarray(xs, dtype=np.float64).ravel())
    fy = np.ascontiguousarray(np.asarray(ys, dtype=np.float64).ravel())
    fz = np.ascontiguousarray(np.asarray(zs, dtype=np.float64).ravel())
    if _HAS_NUMBA:
        out = _perlin_get_arr_k(perm, offset_x, offset_y, offset_z, fx, fy, fz)
    else:
        out = _perlin_get_arr_np(perm, offset_x, offset_y, offset_z, fx, fy, fz)
    return out.reshape(shape)


def _smeared_get_arr(perm, offset_x, offset_y, offset_z, fudge_y_scale, xs, ys, zs):
    shape = np.broadcast(xs, ys, zs).shape
    fx = np.ascontiguousarray(np.asarray(xs, dtype=np.float64).ravel())
    fy = np.ascontiguousarray(np.asarray(ys, dtype=np.float64).ravel())
    fz = np.ascontiguousarray(np.asarray(zs, dtype=np.float64).ravel())
    if _HAS_NUMBA:
        out = _smeared_get_arr_k(perm, offset_x, offset_y, offset_z, fudge_y_scale, fx, fy, fz)
    else:
        out = _smeared_get_arr_np(perm, offset_x, offset_y, offset_z, fudge_y_scale, fx, fy, fz)
    return out.reshape(shape)


def _stack_get_arr(perm, offsets, freqs, amps, xs, ys, zs):
    value = np.zeros(np.broadcast(xs, ys, zs).shape, dtype=np.float32)
    for i in range(freqs.shape[0]):
        f = freqs[i]
        sample = _perlin_get_arr(perm[i], offsets[i, 0], offsets[i, 1], offsets[i, 2],
                                 xs * f, ys * f, zs * f)
        value = value + amps[i] * sample
    return value


def _smeared_stack_get_arr(perm, offsets, freqs, amps, fudges, xs, ys, zs):
    value = np.zeros(np.broadcast(xs, ys, zs).shape, dtype=np.float32)
    for i in range(freqs.shape[0]):
        f = freqs[i]
        sample = _smeared_get_arr(perm[i], offsets[i, 0], offsets[i, 1], offsets[i, 2], fudges[i],
                                  xs * f, ys * f, zs * f)
        value = value + amps[i] * sample
    return value


# ==========================================================================
# Samplers
# ==========================================================================


class PerlinNoise:
    """`net.minecraft.util.math.noise.PerlinNoise`.

    Construction consumes a fixed 262 draws from `random` (3 f64 offsets plus a
    255-step shuffle). That count is not incidental -- `NormalNoise` relies on
    it to skip octaves it does not need without disturbing the ones after.
    """

    __slots__ = ("perms", "offset_x", "offset_y", "offset_z")

    def __init__(self, random) -> None:
        self.offset_x = random.next_f64() * _OFFSET_SCALE
        self.offset_y = random.next_f64() * _OFFSET_SCALE
        self.offset_z = random.next_f64() * _OFFSET_SCALE

        perms = list(range(256))
        # Java's `Collections.shuffle` for i in 0..255, inclusive of the last.
        for i in range(256):
            j = random.next_bounded_i32(256 - i) + i
            perms[i], perms[j] = perms[j], perms[i]
        self.perms = np.asarray(perms, dtype=np.int64)

    def get(self, x: float, y: float, z: float) -> np.float32:
        """Sample at a single point. The `x` position is float, `y` is f64."""
        return _f32(_perlin_get_k(self.perms, self.offset_x, self.offset_y, self.offset_z,
                                  x, y, z))

    def get_array(self, xs, ys, zs) -> np.ndarray:
        """Sample a batch; `xs`/`ys`/`zs` are broadcastable coordinate arrays."""
        return _perlin_get_arr(self.perms, self.offset_x, self.offset_y, self.offset_z, xs, ys, zs)

    def _sample_and_lerp(self, x, y, z, relative_x, relative_y, relative_z, original_relative_y):
        """Kept for parity with the old structure; the kernel holds the logic."""
        return _f32(_sample_and_lerp_k(
            self.perms, int(x), int(y), int(z),
            _f32(relative_x), _f32(relative_y), _f32(relative_z), _f32(original_relative_y),
        ))


class SmearedPerlinNoise:
    """Perlin with its Y axis quantised into slabs.

    The fractional Y is shifted down by a whole number of `fudge_y_scale`
    units, so the noise repeats every slab instead of varying smoothly. The
    `+1e-7` epsilon exists to make the floor land on the *previous* slab
    boundary at exact multiples -- without it, integer Y values would flicker
    between two slabs.
    """

    __slots__ = ("noise", "fudge_y_scale")

    def __init__(self, random, fudge_y_scale: float) -> None:
        self.noise = PerlinNoise(random)
        self.fudge_y_scale = fudge_y_scale

    def get(self, x: float, original_y: float, z: float) -> np.float32:
        return _f32(_smeared_get_k(
            self.noise.perms, self.noise.offset_x, self.noise.offset_y, self.noise.offset_z,
            self.fudge_y_scale, x, original_y, z,
        ))

    def get_array(self, xs, ys, zs) -> np.ndarray:
        return _smeared_get_arr(
            self.noise.perms, self.noise.offset_x, self.noise.offset_y, self.noise.offset_z,
            self.fudge_y_scale, xs, ys, zs,
        )


class Layer:
    """One octave: a noise source, a frequency multiplier, an amplitude."""

    __slots__ = ("noise", "frequency", "amplitude")

    def __init__(self, noise, frequency: float, amplitude: np.float32) -> None:
        self.noise = noise
        self.frequency = frequency
        self.amplitude = amplitude


class NoiseStack:
    """A sum of octaves, each sampled at its own frequency.

    The layers are also compiled into parallel arrays so the whole stack
    samples inside one kernel call. Stacks are homogeneous -- `NormalNoise`
    builds all-plain stacks, `create_fbm` all-smeared ones -- so the variant is
    decided once, at construction.
    """

    __slots__ = ("layers", "smeared", "_perm", "_offsets", "_freqs", "_amps", "_fudges")

    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = list(layers)
        self.smeared = bool(layers) and isinstance(layers[0].noise, SmearedPerlinNoise)
        count = len(layers)
        self._perm = np.empty((count, 256), dtype=np.int64)
        self._offsets = np.empty((count, 3), dtype=np.float64)
        self._freqs = np.empty(count, dtype=np.float64)
        self._amps = np.empty(count, dtype=np.float32)
        self._fudges = np.empty(count, dtype=np.float64)
        for i, layer in enumerate(layers):
            noise = layer.noise
            if self.smeared:
                perlin, fudge = noise.noise, noise.fudge_y_scale
            else:
                perlin, fudge = noise, 1.0
            self._perm[i] = perlin.perms
            self._offsets[i] = (perlin.offset_x, perlin.offset_y, perlin.offset_z)
            self._freqs[i] = layer.frequency
            self._amps[i] = layer.amplitude
            self._fudges[i] = fudge

    def get(self, x: float, y: float, z: float) -> np.float32:
        if self.smeared:
            return _f32(_smeared_stack_get_k(self._perm, self._offsets, self._freqs,
                                             self._amps, self._fudges, x, y, z))
        return _f32(_stack_get_k(self._perm, self._offsets, self._freqs, self._amps, x, y, z))

    def get_array(self, xs, ys, zs) -> np.ndarray:
        if self.smeared:
            return _smeared_stack_get_arr(self._perm, self._offsets, self._freqs,
                                          self._amps, self._fudges, xs, ys, zs)
        return _stack_get_arr(self._perm, self._offsets, self._freqs, self._amps, xs, ys, zs)


# --------------------------------------------------------------------------
# Octave amplitude normalisation.
#
# Vanilla stores noise parameters as `firstOctave` + a list of per-octave
# amplitude *modifiers*. Turning that into actual amplitudes takes three nested
# corrections, each undoing part of the previous. Read as a whole they look
# redundant; each exists because the amplitude list is shared between octave
# counts and normalisations.
# --------------------------------------------------------------------------


class _Octave:
    __slots__ = ("index", "frequency", "amplitude")

    def __init__(self, index: int, frequency: float, amplitude: float) -> None:
        self.index = index
        self.frequency = frequency
        self.amplitude = amplitude


def _amplitude_modifier(modifiers: Sequence[float], index: int) -> float:
    return 1.0 if not modifiers else modifiers[index]


def _build_octaves(
    base_octave: int,
    base_amplitude: float,
    octave_count: int,
    normalize: bool,
    modifiers: Sequence[float],
) -> List[_Octave]:
    """Expand modifiers into concrete (frequency, amplitude) octaves.

    Frequencies climb by `LACUNARITY` from `2^base_octave`; amplitudes fall by
    `PERSISTENCE`. A zero modifier drops the octave entirely rather than
    contributing silence -- so it also does not consume an RNG draw later.
    """
    frequency = _LACUNARITY ** base_octave
    if normalize:
        amplitude = base_amplitude * (
            _PERSISTENCE ** -(octave_count - 1) / (_PERSISTENCE**-octave_count - 1.0)
        )
    else:
        amplitude = base_amplitude

    octaves: List[_Octave] = []
    for i in range(octave_count):
        modifier = _amplitude_modifier(modifiers, i)
        if modifier != 0.0:
            octaves.append(_Octave(base_octave + i, frequency, amplitude * modifier))
        frequency *= _LACUNARITY
        amplitude *= _PERSISTENCE
    return octaves


def _estimate_deviation(octaves: Sequence[_Octave]) -> float:
    """Root-sum-square of the per-octave deviations -- noise adds in quadrature."""
    variance = 0.0
    for octave in octaves:
        layer_deviation = _STANDARD_DEVIATION * abs(octave.amplitude)
        variance += layer_deviation * layer_deviation
    return math.sqrt(variance)


def _compute_normalization_factor(target_amplitude: float, octaves: Sequence[_Octave]) -> float:
    """Scale the whole stack so its deviation hits `target_amplitude / 3`."""
    input_deviation = _estimate_deviation(octaves)
    if input_deviation == 0.0:
        return 0.0
    input_sum_deviation = input_deviation * math.sqrt(2.0)
    return (target_amplitude * _TARGET_DEVIATION) / input_sum_deviation


def _parity_expected_deviation(octave_span: int) -> float:
    """The deviation a stack of this many octaves 'should' have."""
    return 0.1 * (1.0 + 1.0 / (octave_span + 1))


def _parity_normalization_factor(
    base_amplitude: float, octave_count: int, modifiers: Sequence[float]
) -> float:
    """Match the legacy stack's deviation, given the same modifier list.

    Named "parity" because its whole job is to make the modern path produce the
    same distribution the legacy LCG path did -- worlds generated by different
    Minecraft versions should not have different noise *character*.
    """
    min_octave = 2**31 - 1
    max_octave = -(2**31)
    for i in range(octave_count):
        if _amplitude_modifier(modifiers, i) != 0.0:
            min_octave = min(min_octave, i)
            max_octave = max(max_octave, i)
    return base_amplitude * 0.5 * _TARGET_DEVIATION / _parity_expected_deviation(max_octave - min_octave)


def _parity_base_amplitude(base_octave: int, amplitudes: Sequence[float]) -> float:
    """The base amplitude that makes the modern stack match legacy parity.

    Works by measuring the stack built at amplitude 1.0, computing what factor
    *would* normalise it, and correcting by the ratio between that and the
    parity factor.
    """
    octaves = _build_octaves(base_octave, 1.0, len(amplitudes), True, amplitudes)
    target_amplitude = sum(abs(o.amplitude) for o in octaves)
    new_normalization_factor = _compute_normalization_factor(target_amplitude, octaves)
    if new_normalization_factor == 0.0:
        return 1.0
    return _parity_normalization_factor(1.0, len(amplitudes), amplitudes) / new_normalization_factor


class NormalNoise:
    """The full octave stack -- what a `Noise` density-function node samples.

    Built the modern (non-legacy) way, which is what the noise router uses.
    Each octave draws *two* independent Perlin sources: one at the octave's
    frequency and one at `INPUT_FACTOR` times it. That pair is the "normal
    noise" trick that makes the field look less grid-aligned than raw Perlin.
    """

    __slots__ = ("stack", "max_value")

    def __init__(self, random, first_octave: int, amplitudes: Sequence[float]) -> None:
        octave_count = len(amplitudes)
        base_amplitude = _parity_base_amplitude(first_octave, amplitudes)
        octaves = _build_octaves(first_octave, base_amplitude, octave_count, True, amplitudes)
        target_amplitude = sum(abs(o.amplitude) for o in octaves)
        normalization_factor = _compute_normalization_factor(target_amplitude, octaves)

        first_random = random.next_splitter()
        second_random = random.next_splitter()

        layers: List[Layer] = []
        for octave in octaves:
            # The seed string is plain "octave_N" -- no "minecraft:" prefix.
            # This is MD5'd by the splitter, so the exact string is load-bearing.
            seed = f"octave_{octave.index}"
            value_factor = _f32(normalization_factor * octave.amplitude)
            layers.append(
                Layer(PerlinNoise(first_random.split_string(seed)), octave.frequency, value_factor)
            )
            layers.append(
                Layer(
                    PerlinNoise(second_random.split_string(seed)),
                    octave.frequency * _INPUT_FACTOR,
                    value_factor,
                )
            )

        self.stack = NoiseStack(layers)
        self.max_value = _f32(target_amplitude * _TARGET_DEVIATION * 6.0)

    def get(self, x: float, y: float, z: float) -> np.float32:
        return self.stack.get(x, y, z)

    def get_array(self, xs, ys, zs) -> np.ndarray:
        return self.stack.get_array(xs, ys, zs)


class DoublePerlinNoiseSampler:
    """`NormalNoise` behind the name the noise router actually uses."""

    __slots__ = ("noise",)

    def __init__(self, random, first_octave: int, amplitudes: Sequence[float]) -> None:
        self.noise = NormalNoise(random, first_octave, amplitudes)

    @property
    def max_value(self) -> np.float32:
        return self.noise.max_value

    def get(self, x: float, y: float, z: float) -> np.float32:
        return self.noise.get(x, y, z)

    def get_array(self, xs, ys, zs) -> np.ndarray:
        return self.noise.get_array(xs, ys, zs)

    # Pumpkin and vanilla both call the sample entry point `sample`; keep the
    # name so call sites read the same as the reference implementation.
    sample = get
    sample_array = get_array


def create_fbm(
    random, first_octave: int, smear_scale_y: float, value_factor: float
) -> NoiseStack:
    """The octave stack behind `InterpolatedNoiseSampler`.

    Note it builds *upward* from the lowest octave rather than downward, and
    uses smeared Perlin throughout -- this noise is meant to be sampled sparsely
    and interpolated, so blockiness is invisible and speed is everything.
    """
    octaves = -first_octave + 1
    factor = 1.0
    value_factor = value_factor / (2.0**octaves - 1.0)
    layers: List[Layer] = []
    for _ in range(octaves - 1, -1, -1):
        layers.append(
            Layer(
                SmearedPerlinNoise(random, smear_scale_y * factor),
                factor,
                _f32(value_factor),
            )
        )
        factor /= 2.0
        value_factor *= 2.0
    return NoiseStack(layers)


def fbm_range_max(first_octave: int, smear_scale_y: float, value_factor: float) -> np.float32:
    """The largest value `create_fbm` can return -- an exact worst case.

    Summed analytically rather than sampled, because the density functions that
    consume this clamp against it and a sampled estimate would be slightly low,
    which shows up as terrain that never quite reaches the top of its range.
    """
    octaves = -first_octave + 1
    factor = 1.0
    value_factor = value_factor / (2.0**octaves - 1.0)
    total = 0.0
    for _ in range(octaves - 1, -1, -1):
        total += (abs(smear_scale_y * factor) + _SMEARED_RANGE_BASE) * value_factor
        factor /= 2.0
        value_factor *= 2.0
    return _f32(total)


__all__ = [
    "DoublePerlinNoiseSampler",
    "Layer",
    "NoiseStack",
    "NormalNoise",
    "PerlinNoise",
    "SmearedPerlinNoise",
    "create_fbm",
    "fbm_range_max",
    "lerp",
    "lerp3",
    "smoothstep",
    "wrap",
]
