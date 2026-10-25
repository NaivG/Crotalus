"""
Simplex noise, as both the biome temperature modifier and several feature types need it.

This is *not* the same sampler `density.py` carries privately for the End's
islands. Vanilla has two simplex implementations in play and they are not
interchangeable:

* `SimplexNoise` (this module) - a 512-entry permutation, a 12-entry 3D gradient
  table indexed with `% 12`, and a random origin per axis. Used by
  `OctaveSimplexNoiseSampler`, which is what `Biome.getTemperature` reads.
* the private one in `density.py`, which is the classic 8-gradient variant.

Mixing them produces a noise field that looks like noise and is not vanilla's,
so the gradient table and the `% 12` are spelled out rather than "cleaned up".

    from world.vanilla.simplex import OctaveSimplexNoiseSampler, SimplexNoise
    from world.vanilla.random_source import LegacyRandom

    noise = OctaveSimplexNoiseSampler(LegacyRandom(1234), [0])
    noise.sample(0.5, 0.25, use_origin=False)

`tools/check_surface_rules.py` checks both classes against vectors Pumpkin took
from a real Minecraft, so a change here that merely looks reasonable fails.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence

import numpy as np

from world.vanilla.random_source import LegacyRandom

#: Vanilla's `SimplexNoise.GRADIENT`: twelve 3D gradients, used as a dot product
#: with (x, y, z). The 2D case feeds z = 0, which is why four of the twelve
#: collapse to a pure x term and four to a pure y term.
_GRADIENTS = (
    (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0),
    (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
    (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1),
)

_SQRT_3 = math.sqrt(3.0)
_SKEW_2D = 0.5 * (_SQRT_3 - 1.0)
_UNSKEW_2D = (3.0 - _SQRT_3) / 6.0
_SKEW_3D = 1.0 / 3.0
_UNSKEW_3D = 1.0 / 6.0

_I32_MAX = 2 ** 31 - 1
_I32_MIN = -(2 ** 31)


def _floor_i32(value: float) -> int:
    """Java's `Mth.floor(double)` as an `int`: floor, truncated into i32.

    Vanilla stores the simplex lattice cell in an `int`, so a coordinate past
    ±2^31 pins to the edge instead of continuing. It matters: the Overworld
    reaches x = 3e7, and the skewed coordinate `x + (x + z) * 0.366` is roughly
    1.4x that before any multiplication, so a feature far from the origin lands
    in the saturated regime.
    """
    floored = math.floor(value)
    if floored > _I32_MAX:
        return _I32_MAX
    if floored < _I32_MIN:
        return _I32_MIN
    return floored


def _wrap_i32(value: int) -> int:
    """Java's `int` overflow: two's complement, wrapping.

    `(i + j)` is an `int` addition in vanilla, and at large coordinates it
    overflows. The wrapped sum feeds the unskew, which then puts the sample
    point far outside its cell and zeroes all three corner contributions - so a
    mismatch here does not look like a small error, it looks like the octave
    going silent.
    """
    return ((value + 0x80000000) & 0xFFFFFFFF) - 0x80000000


class SimplexNoise:
    """Vanilla's `SimplexNoise`: the permutation, three origins, and two samples.

    The origins are drawn from the same generator as the permutation and are
    consumed *before* it, so building this with the wrong RNG call order silently
    produces a different permutation - which is the whole noise field.
    """

    __slots__ = ("permutation", "origin_x", "origin_y", "origin_z")

    def __init__(self, random: Any) -> None:
        self.origin_x = random.next_f64() * 256.0
        self.origin_y = random.next_f64() * 256.0
        self.origin_z = random.next_f64() * 256.0

        table = list(range(256))
        for i in range(256):
            j = random.next_bounded_i32(256 - i)
            table[i], table[i + j] = table[i + j], table[i]
        # Vanilla indexes `p[k + p[l]]`, so the table is read at offsets up to
        # 510: the 256 entries are doubled rather than masked.
        self.permutation = table + table

    @staticmethod
    def _gradient(index: int, x: float, y: float, z: float, distance: float) -> float:
        d = distance - x * x - y * y - z * z
        if d < 0.0:
            return 0.0
        d *= d
        g = _GRADIENTS[index]
        return d * d * (g[0] * x + g[1] * y + g[2] * z)

    def sample_2d(self, x: float, y: float) -> float:
        """2D simplex noise. Vanilla's `SimplexNoise.getValue(x, y)`."""
        d = (x + y) * _SKEW_2D
        i = _floor_i32(x + d)
        j = _floor_i32(y + d)

        unskew = _wrap_i32(i + j) * _UNSKEW_2D
        base_x = i - unskew
        base_y = j - unskew

        rel_x = x - base_x
        rel_y = y - base_y

        step_x, step_y = (1, 0) if rel_x > rel_y else (0, 1)

        mid_x = rel_x - step_x + _UNSKEW_2D
        mid_y = rel_y - step_y + _UNSKEW_2D
        far_x = rel_x - 1.0 + 2.0 * _UNSKEW_2D
        far_y = rel_y - 1.0 + 2.0 * _UNSKEW_2D

        p = self.permutation
        ii = int(i) & 255
        jj = int(j) & 255

        g0 = p[ii + p[jj]] % 12
        g1 = p[ii + step_x + p[jj + step_y]] % 12
        g2 = p[ii + 1 + p[jj + 1]] % 12

        return 70.0 * (
            self._gradient(g0, rel_x, rel_y, 0.0, 0.5)
            + self._gradient(g1, mid_x, mid_y, 0.0, 0.5)
            + self._gradient(g2, far_x, far_y, 0.0, 0.5)
        )

    def sample_3d(self, x: float, y: float, z: float) -> float:
        """3D simplex noise, used only to drop the octave samplers' seeds."""
        e = (x + y + z) * _SKEW_3D
        i = _floor_i32(x + e)
        j = _floor_i32(y + e)
        k = _floor_i32(z + e)

        unskew = _wrap_i32(_wrap_i32(i + j) + k) * _UNSKEW_3D
        rel_x = x - (i - unskew)
        rel_y = y - (j - unskew)
        rel_z = z - (k - unskew)

        if rel_x >= rel_y:
            if rel_y >= rel_z:
                a = (1, 0, 0, 1, 1, 0)
            elif rel_x >= rel_z:
                a = (1, 0, 0, 1, 0, 1)
            else:
                a = (0, 0, 1, 1, 0, 1)
        elif rel_y < rel_z:
            a = (0, 0, 1, 0, 1, 1)
        elif rel_x < rel_z:
            a = (0, 1, 0, 0, 1, 1)
        else:
            a = (0, 1, 0, 1, 1, 0)

        x1 = rel_x - a[0] + _UNSKEW_3D
        y1 = rel_y - a[1] + _UNSKEW_3D
        z1 = rel_z - a[2] + _UNSKEW_3D
        x2 = rel_x - a[3] + 2.0 * _UNSKEW_3D
        y2 = rel_y - a[4] + 2.0 * _UNSKEW_3D
        z2 = rel_z - a[5] + 2.0 * _UNSKEW_3D
        x3 = rel_x - 1.0 + 0.5
        y3 = rel_y - 1.0 + 0.5
        z3 = rel_z - 1.0 + 0.5

        p = self.permutation
        ii = int(i) & 255
        jj = int(j) & 255
        kk = int(k) & 255

        g0 = p[ii + p[jj + p[kk]]] % 12
        g1 = p[ii + a[0] + p[jj + a[1] + p[kk + a[2]]]] % 12
        g2 = p[ii + a[3] + p[jj + a[4] + p[kk + a[5]]]] % 12
        g3 = p[ii + 1 + p[jj + 1 + p[kk + 1]]] % 12

        return 32.0 * (
            self._gradient(g0, rel_x, rel_y, rel_z, 0.6)
            + self._gradient(g1, x1, y1, z1, 0.6)
            + self._gradient(g2, x2, y2, z2, 0.6)
            + self._gradient(g3, x3, y3, z3, 0.6)
        )


class OctaveSimplexNoiseSampler:
    """Vanilla's `OctaveSimplexNoiseSampler`: a sparse set of simplex octaves.

    Two details are easy to get wrong and both change the field everywhere:

    * **The octave list decides how many generators are consumed.** Vanilla walks
      the octave index range backwards and calls `skip(262)` for every octave
      that is *not* wanted, because a `SimplexNoise` constructor is exactly 262
      draws. Skipping in the wrong place leaves every later sampler seeded
      differently.
    * **The loop scales up, not down.** `lacunarity` starts at `2^max_octave` and
      halves; `persistence` starts at `1/(2^n - 1)` and doubles. Multiplying by
      frequency first and amplitude second is what makes the low octaves the
      loud ones.
    """

    __slots__ = ("samplers", "persistence", "lacunarity")

    #: Draws a `SimplexNoise` constructor consumes: 3 origins + 256 bounded ints
    #: + 3 padding, which is `skip(262)` in vanilla's own arithmetic.
    SKIP_PER_OCTAVE = 262

    def __init__(self, random: Any, octaves: Sequence[int]) -> None:
        wanted = sorted(set(int(o) for o in octaves))
        if not wanted:
            self.samplers: List[Optional[SimplexNoise]] = []
            self.persistence = 0.0
            self.lacunarity = 0.0
            return

        first, last = wanted[0], wanted[-1]
        count = -first + last + 1
        samplers: List[Optional[SimplexNoise]] = [None] * count

        base = SimplexNoise(random)
        for m in range(last + 1, count):
            if m >= 0 and (last - m) in wanted:
                samplers[m] = SimplexNoise(random)
            else:
                random.skip(self.SKIP_PER_OCTAVE)

        if last > 0:
            probe = base.sample_3d(base.origin_x, base.origin_y, base.origin_z)
            # Vanilla widens the f32 constant to f64 before multiplying, so the
            # product is a double. Doing it in f32 loses the low bits of a seed
            # that is then used verbatim.
            seed = int(probe * float(np.float32(9.223372e18)))
            derived = LegacyRandom(seed)
            for o in range(last - 1, -1, -1):
                if o < count and (last - o) in wanted:
                    samplers[o] = SimplexNoise(derived)
                else:
                    derived.skip(self.SKIP_PER_OCTAVE)

        if 0 <= last < count and 0 in wanted:
            samplers[last] = base

        self.samplers = samplers
        self.persistence = 1.0 / (2.0 ** count - 1.0)
        self.lacunarity = 2.0 ** last

    def sample(self, x: float, y: float, use_origin: bool = False) -> float:
        """Vanilla's `sample(x, y, useOrigin)`.

        `use_origin` adds each octave's own origin offset. Every call site in
        vanilla's biome temperature code passes false, which is also the faster
        path; it is spelled out rather than assumed because the two disagree.
        """
        total = 0.0
        frequency = self.lacunarity
        amplitude = self.persistence
        for sampler in self.samplers:
            if sampler is not None:
                total += sampler.sample_2d(
                    x * frequency + (sampler.origin_x if use_origin else 0.0),
                    y * frequency + (sampler.origin_y if use_origin else 0.0),
                ) * amplitude
            frequency /= 2.0
            amplitude *= 2.0
        return total

    def __repr__(self) -> str:
        live = sum(1 for s in self.samplers if s is not None)
        return f"<OctaveSimplexNoiseSampler octaves={live}/{len(self.samplers)}>"


__all__ = ["OctaveSimplexNoiseSampler", "SimplexNoise"]
