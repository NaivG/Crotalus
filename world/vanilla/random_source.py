"""
Java-compatible random sources for vanilla world generation.

Vanilla's worldgen seeds its noise from a single 64-bit world seed. Reproducing
that seed's *expansion* into per-noise samplers is the whole game: get one draw
wrong and every block in the world moves. This module provides the four random
sources that expansion touches, each transliterated from the Java original
rather than approximated.

    from world.vanilla.random_source import Xoroshiro, WorldgenRandom

    splitter = Xoroshiro.from_seed(world_seed).next_splitter()
    rand = splitter.split_string("minecraft:offset")   # one noise's own RNG
    rand.next_i32()

The four sources, and why each exists
-------------------------------------
`Xoroshiro`          Xoroshiro128+, the modern generator. Its `next(bits)`
                     takes the *high* bits of a full 64-bit draw.
`XoroshiroSplitter`  Derives independent generators without consuming the
                     parent's stream, which is what lets chunks generate in
                     parallel and in any order.
`WorldgenRandom`     A LCG re-implemented in Java on top of Xoroshiro's bits.
                     It has *different* bounded-draw behaviour from the
                     underlying source, so it is a distinct class, not a
                     subclass.
`LegacyRandom`       The old 48-bit LCG (`java.util.Random`). Still used for
                     the end-island sampler and some feature placement.

`WorldgenRandom` exists at all because vanilla did not delete the old code path
when it switched generators -- it reimplemented `java.util.Random`'s methods on
top of the new bit source. That is why its `next_long` is two 32-bit draws in
*high-then-low* order, and why its `next_bounded_int` uses the LCG's
`value - result` overflow test rather than Xoroshiro's multiply-and-shift. Both
are easy to "clean up" into something more elegant and wrong.
"""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Tuple

MASK64 = 0xFFFFFFFFFFFFFFFF
MASK32 = 0xFFFFFFFF

#: `java.util.Random`'s seed scramble multiplier, reused by the end-island
#: sampler and by any caller that wants a reproducible "vanilla-flavoured"
#: stream that is not Xoroshiro.
LEGACY_MULTIPLIER = 0x5DEECE66D
LEGACY_ADDEND = 0xB
LEGACY_MASK = (1 << 48) - 1

#: 2^24 and 2^53, used to turn a 24/53-bit integer draw into [0, 1).
_F32_SCALE = 1.0 / float(1 << 24)
_F64_SCALE = 1.0 / float(1 << 53)


def _rotl(x: int, k: int) -> int:
    """Rotate a 64-bit word left, staying in unsigned 64-bit range."""
    return ((x << k) | (x >> (64 - k))) & MASK64


def mix_stafford_13(z: int) -> int:
    """Stafford variant 13: the seed-mixing function from SplitMix64.

    Python has no built-in 64-bit type, so every operation here is masked back
    to 64 bits. The masking is the entire point -- without it the intermediates
    become arbitrary-precision integers and the multiplies stop matching Java's
    wrapping `long` arithmetic.
    """
    z &= MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return (z ^ (z >> 31)) & MASK64


def _hash_block_pos(x: int, y: int, z: int) -> int:
    """`Mth.getSeed(x, y, z)` - the position hash `split_pos` derives from.

    Three details, each of which changes every draw downstream:

    * `x * 3129871` is an **int** multiply in Java, so it wraps at 32 bits and is
      then sign-extended into the long. The other two terms are long multiplies
      and do not wrap the same way.
    * `l * l * 42317861 + l * 11` is long arithmetic, wrapping at 64 bits.
    * The final `>> 16` is an **arithmetic** shift on a signed long. Treating it
      as unsigned keeps the high half of the value and produces a completely
      different seed.

    This was wrong until the surface rules needed it: the function was dead
    code, so nothing exercised it, and the hash it had was a different
    (xxHash-shaped) construction. The vectors in `tools/check_surface_rules.py`
    come from a real Minecraft via Pumpkin's `block_position_hash` test.
    """
    mixed_x = x * 3129871
    # i32 wraparound, then sign-extend into the 64-bit word.
    mixed_x = ((mixed_x + 0x80000000) & 0xFFFFFFFF) - 0x80000000

    l = (mixed_x & MASK64) ^ ((z * 116129781) & MASK64) ^ (y & MASK64)
    l = (l * l * 42317861 + l * 11) & MASK64
    if l >= 0x8000000000000000:
        l -= 0x10000000000000000
    return (l >> 16) & MASK64


def name_seed(name: str) -> Tuple[int, int]:
    """Split `md5(name)` into the (lo, hi) pair vanilla seeds anything named from.

    This is how vanilla gets a *stable, order-independent* RNG per named thing -
    a noise parameter, a surface rule's `random_name`, a carver. Java hashes the
    fully-qualified id ("minecraft:offset"); the first 8 bytes of the digest
    become one state word and the next 8 the other, both **big-endian**.

    One recipe, two callers, and it is worth keeping that way: Pumpkin's codegen
    reads the same digest little-endian for its baked surface-rule constants, and
    the vanilla chunk dumps show the resulting bedrock layer is wrong. See
    `tools/check_surface_rules.py`, which pins both this and the derivation.
    """
    digest = hashlib.md5(name.encode("utf-8")).digest()
    lo, hi = struct.unpack(">QQ", digest[:16])
    return lo, hi


def _md5_seed(name: str) -> Tuple[int, int]:
    """Backwards-compatible alias for `name_seed`."""
    return name_seed(name)


class Xoroshiro:
    """Xoroshiro128+, seeded the way vanilla seeds it."""

    __slots__ = ("lo", "hi", "_next_gaussian")

    def __init__(self, lo: int, hi: int, _mixed: bool = False) -> None:
        """
        `lo`/`hi` are the raw state words. The all-zero state is a fixed point
        of the algorithm, so vanilla substitutes a known-good pair instead --
        without this, a seed of zero would emit zeros forever.
        """
        if not _mixed and (lo | hi) == 0:
            lo, hi = 0x9E3779B97F4A7C15, 0x6A09E667F3BCC909
        self.lo = lo & MASK64
        self.hi = hi & MASK64
        self._next_gaussian: float | None = None

    @classmethod
    def from_seed(cls, seed: int) -> "Xoroshiro":
        """Seed from a world seed the way `XoroshiroRandomSource` does.

        The seed is first split by the golden ratio into two words, then each is
        independently Stafford-mixed. Skipping either step yields a valid but
        *wrong* generator -- one that still looks like noise.
        """
        l = (seed ^ 0x6A09E667F3BCC909) & MASK64
        m = (l + 0x9E3779B97F4A7C15) & MASK64
        return cls(mix_stafford_13(l), mix_stafford_13(m))

    def next_raw(self) -> int:
        """One 64-bit draw, advancing the state.

        The three rotates are inlined with their shift counts as constants:
        every noise draw in worldgen passes through here, and three `_rotl`
        calls cost more than the arithmetic around them. Each inlined form is
        bit-for-bit `_rotl`'s - the trailing mask folds the rotate's own mask
        in, since `(x & MASK64) ^ y == (x ^ y) & MASK64` and likewise for `+`.
        """
        l = self.lo
        m = self.hi
        n = (l + m) & MASK64
        n = (((n << 17) | (n >> 47)) + l) & MASK64
        m ^= l
        self.lo = (((l << 49) | (l >> 15)) ^ m ^ (m << 21)) & MASK64
        self.hi = ((m << 28) | (m >> 36)) & MASK64
        return n

    def next(self, bits: int) -> int:
        """`bits` high bits of one draw -- the primitive Java draws from."""
        return self.next_raw() >> (64 - bits)

    def next_i32(self) -> int:
        """A signed 32-bit value from the *low* half of a draw."""
        return self.next_raw() & MASK32

    def next_i64(self) -> int:
        return self.next_raw()

    def next_bool(self) -> bool:
        return (self.next_raw() & 1) != 0

    def next_f32(self) -> float:
        return self.next(24) * _F32_SCALE

    def next_f64(self) -> float:
        return self.next(53) * _F64_SCALE

    def next_bounded_i32(self, bound: int) -> int:
        """A value in [0, bound) by Lemire-style multiply-and-shift.

        The 32x32->64 multiply replaces Java's rejection loop with one
        multiply. Note the intermediate is kept at full 64 bits and only the
        high word is taken -- masking before the shift would change the answer.
        """
        m = (self.next_i32() * bound) & MASK64
        n = m & MASK32
        if n < bound:
            # Rare residue: the low word fell short of `bound`, so the draw is
            # biased upward. Reject and retry until the low word clears it.
            threshold = ((1 << 32) - bound) % bound
            while n < threshold:
                m = (self.next_i32() * bound) & MASK64
                n = m & MASK32
        return (m >> 32) & MASK32

    def next_between(self, low: int, high: int) -> int:
        """Integer in [low, high] -- both ends inclusive, as Java's namesake."""
        if low > high:
            low, high = high, low
        return low + self.next_bounded_i32(high - low + 1)

    def next_gaussian(self) -> float:
        """Box-Muller, with the second value cached like `java.util.Random`.

        Caching the spare is why this generator carries mutable state: the
        sequence of `next_gaussian` calls must consume a *fixed* number of
        underlying draws per pair, or every noise seeded from this source
        shifts.
        """
        if self._next_gaussian is not None:
            value = self._next_gaussian
            self._next_gaussian = None
            return value
        while True:
            v1 = 2 * self.next_f64() - 1
            v2 = 2 * self.next_f64() - 1
            s = v1 * v1 + v2 * v2
            if 0 < s < 1:
                break
        multiplier = math.sqrt(-2 * math.log(s) / s)
        self._next_gaussian = v2 * multiplier
        return v1 * multiplier

    def next_triangular(self, mode: float, deviation: float) -> float:
        """Mean of two uniforms, offset from `mode` -- the difference form.

        Note the argument names: `deviation` is a scale, not a spread, and the
        result is unbounded rather than confined to
        [mode - deviation, mode + deviation].
        """
        return mode + deviation * (self.next_f64() - self.next_f64())

    def next_splitter(self) -> "XoroshiroSplitter":
        """A deriver that does *not* disturb this generator's own stream."""
        return XoroshiroSplitter(self.next_raw(), self.next_raw())

    def skip(self, count: int) -> None:
        """Advance `count` 64-bit draws, so `2 * count` raw steps."""
        for _ in range(count):
            self.next_raw()

    def split(self) -> "Xoroshiro":
        return Xoroshiro(self.next_raw(), self.next_raw())


class XoroshiroSplitter:
    """Derives independent generators from a parent without consuming it.

    This is the reason chunks can be generated in any order, in any thread, and
    still agree. `Xoroshiro.next_splitter` reads two draws to *make* the
    splitter, but every `split_*` call on that splitter is pure: it xors the
    parent's words into the requested seed and touches no shared state.
    """

    __slots__ = ("lo", "hi")

    def __init__(self, lo: int, hi: int) -> None:
        self.lo = lo & MASK64
        self.hi = hi & MASK64

    def split_u64(self, seed: int) -> Xoroshiro:
        return Xoroshiro(seed ^ self.lo, seed ^ self.hi)

    def split_string(self, name: str) -> Xoroshiro:
        """Derive from a name via MD5 -- how vanilla seeds each noise."""
        lo, hi = _md5_seed(name)
        return Xoroshiro(lo ^ self.lo, hi ^ self.hi)

    def from_lo_and_hi(self, lo: int, hi: int) -> Xoroshiro:
        """Derive from a precomputed (lo, hi) pair.

        Identical to `split_string`, but for callers that already hold the
        digest halves -- the noise-parameter table stores them so that
        construction does not re-hash on every lookup.
        """
        return Xoroshiro(lo ^ self.lo, hi ^ self.hi)

    def split_pos(self, x: int, y: int, z: int) -> Xoroshiro:
        l = _hash_block_pos(x, y, z)
        return Xoroshiro(l ^ self.lo, self.hi)


class WorldgenRandom:
    """Java `WorldgenRandom`: Xoroshiro's bits behind `java.util.Random`'s API.

    Every method here deliberately *differs* from the identically-named method
    on `Xoroshiro`. `next_i32` composes two 24-bit draws, `next_i64` is
    high-then-low, and `next_bounded_i32` uses the LCG's overflow test. Porting
    these to the "obvious" Xoroshiro equivalents produces a generator that
    passes a casual smoke test and desynchronises from vanilla.
    """

    __slots__ = ("source", "_next_gaussian")

    def __init__(self, source: Xoroshiro) -> None:
        self.source = source
        self._next_gaussian: float | None = None

    @classmethod
    def from_seed(cls, seed: int) -> "WorldgenRandom":
        return cls(Xoroshiro.from_seed(seed))

    def next(self, bits: int) -> int:
        return self.source.next(bits)

    def next_i32(self) -> int:
        return self.next(32) & MASK32

    def next_i64(self) -> int:
        # Java builds the long as (high << 32) + low, from two 32-bit draws.
        high = self.next_i32()
        low = self.next_i32()
        return ((high << 32) | low) & MASK64

    def next_bool(self) -> bool:
        return self.next(1) != 0

    def next_f32(self) -> float:
        return self.next(24) * _F32_SCALE

    def next_f64(self) -> float:
        # 26 + 27 bits, not 53: this is the LCG's construction, faithfully.
        high = self.next(26)
        low = self.next(27)
        return ((high << 27) | low) * _F64_SCALE

    def next_bounded_i32(self, bound: int) -> int:
        """The LCG's bounded draw: rejection via the overflow test.

        `value - result` wrapping negative is the signal to redraw. This is
        materially slower and materially different from Xoroshiro's
        multiply-shift; both appear in vanilla for different call sites.
        """
        if bound <= 0:
            raise ValueError("bound must be positive")
        if bound & (bound - 1) == 0:  # power of two
            return (bound * self.next(31) >> 31) & MASK32
        while True:
            value = self.next(31)
            result = value % bound
            if (value - result + bound - 1) < (1 << 31):
                return result

    def next_gaussian(self) -> float:
        """Same Box-Muller pair-caching as `java.util.Random`."""
        if self._next_gaussian is not None:
            value = self._next_gaussian
            self._next_gaussian = None
            return value
        while True:
            v1 = 2 * self.next_f64() - 1
            v2 = 2 * self.next_f64() - 1
            s = v1 * v1 + v2 * v2
            if 0 < s < 1:
                break
        import math

        multiplier = math.sqrt(-2 * math.log(s) / s)
        self._next_gaussian = v2 * multiplier
        return v1 * multiplier

    def split(self) -> "WorldgenRandom":
        return WorldgenRandom(self.source.split())


class LegacyRandom:
    """`java.util.Random` -- the 48-bit LCG, kept for the end-island sampler.

    The end-island density function is one of the last places vanilla still
    constructs a legacy generator, and it constructs it from the *world seed*
    after skipping 17292 draws. Reproducing the skip matters as much as
    reproducing the LCG.
    """

    __slots__ = ("seed", "_next_gaussian")

    def __init__(self, seed: int) -> None:
        self.seed = (seed ^ LEGACY_MULTIPLIER) & LEGACY_MASK
        self._next_gaussian: float | None = None

    def next(self, bits: int) -> int:
        self.seed = (self.seed * LEGACY_MULTIPLIER + LEGACY_ADDEND) & LEGACY_MASK
        return self.seed >> (48 - bits)

    def next_i32(self) -> int:
        return self.next(32) & MASK32

    def next_i64(self) -> int:
        hi = self.next(32)
        lo = self.next(32)
        return ((hi << 32) | lo) & MASK64

    def next_bool(self) -> bool:
        return self.next(1) != 0

    def next_f32(self) -> float:
        return self.next(24) * _F32_SCALE

    def next_f64(self) -> float:
        hi = self.next(26)
        lo = self.next(27)
        return ((hi << 27) | lo) * _F64_SCALE

    def next_bounded_i32(self, bound: int) -> int:
        if bound <= 0:
            raise ValueError("bound must be positive")
        if bound & (bound - 1) == 0:
            return (bound * self.next(31) >> 31) & MASK32
        while True:
            value = self.next(31)
            result = value % bound
            if (value - result + bound - 1) < (1 << 31):
                return result

    def skip(self, count: int) -> None:
        """Advance `count` single LCG steps.

        Deliberately a loop of `next_i32` rather than the tempting
        `next(count * 2 + 1)`: the LCG's state is 48 bits, so any request above
        48 is a negative shift. `count` steps is also the correct unit -- the
        generic "skip N longs" default would advance twice as far here, because
        one `next_i64` is two LCG steps, and the end-island sampler skips 17292
        of them.
        """
        for _ in range(count):
            self.next_i32()

    def next_gaussian(self) -> float:
        if self._next_gaussian is not None:
            value = self._next_gaussian
            self._next_gaussian = None
            return value
        while True:
            v1 = 2 * self.next_f64() - 1
            v2 = 2 * self.next_f64() - 1
            s = v1 * v1 + v2 * v2
            if 0 < s < 1:
                break
        multiplier = math.sqrt(-2 * math.log(s) / s)
        self._next_gaussian = v2 * multiplier
        return v1 * multiplier

    def next_triangular(self, mode: float, deviation: float) -> float:
        """Same difference form as `Xoroshiro.next_triangular`."""
        return mode + deviation * (self.next_f64() - self.next_f64())


__all__ = [
    "LegacyRandom",
    "WorldgenRandom",
    "Xoroshiro",
    "XoroshiroSplitter",
    "mix_stafford_13",
]
