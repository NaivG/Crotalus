"""
Kernel-side draws for the compiled worldgen walks.

`world.vanilla.random_source` owns the generators; the compiled kernels in
`carver.py` and `features.py` must not become second, drifting copies of them,
so Python still *seeds* every generator and the compiled code only continues
from a state array it is handed. `state_from` exports that state from the Python
objects; `state_to` writes it back when a kernel is done but the Python stream
carries on -- feature placement reuses one random source across a nested
pipeline, so an ore kernel that forgets to write back loses every draw it made.
`tools/check_carvers.py` and `tools/check_ore_veins.py` replay mixed draw
sequences through both sides and assert every draw is identical, for all three
generators. The three kinds:

  0  WorldgenRandom over Xoroshiro (overworld: `legacy_random_source` false)
  1  LegacyRandom, the 48-bit LCG (nether)
  2  a bare Xoroshiro drawn raw (Pumpkin's behaviour, `random_source="raw"`)

The sign conventions are the ones in random_source.py: 64-bit wrapping,
arithmetic shifts and the low-32-bit read in `next_i32`.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - numba is a declared dependency
    from numba import njit

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator if not args else decorator(args[0])


from world.vanilla.random_source import LegacyRandom, WorldgenRandom

RNG_WORLDGEN = 0
RNG_LEGACY = 1
RNG_RAW = 2

_LCG_MULTIPLIER = np.uint64(0x5DEECE66D)
_LCG_ADDEND = np.uint64(0xB)
_LCG_MASK = np.uint64(0xFFFFFFFFFFFF)
_U32 = np.uint64(0xFFFFFFFF)
_F64_SCALE = 1.0 / float(1 << 53)


def state_from(random) -> np.ndarray:
    """Export a seeded `random_source` generator as a kernel state array.

    `[kind, lo_or_seed, hi, 0]`. Reading `.seed` / `.lo` / `.hi` is reading
    documented state, not re-deriving it, which is the whole point: the seeding
    arithmetic (Stafford mixing, the golden-ratio split, the LCG xor) stays in
    `random_source.py`.
    """
    state = np.zeros(4, dtype=np.uint64)
    if isinstance(random, LegacyRandom):
        state[0] = RNG_LEGACY
        state[1] = np.uint64(random.seed)
        return state
    if isinstance(random, WorldgenRandom):
        state[0] = RNG_WORLDGEN
        source = random.source
    else:
        state[0] = RNG_RAW
        source = random
    state[1] = np.uint64(source.lo)
    state[2] = np.uint64(source.hi)
    return state


def state_to(random, state: np.ndarray) -> None:
    """Write a kernel's state array back into the Python generator.

    The inverse of `state_from`, for streams that continue in Python after the
    kernel returns. The kind is not consulted: the caller passed the generator,
    so the slots to write are determined by what it is.
    """
    if isinstance(random, LegacyRandom):
        random.seed = int(state[1])
        return
    source = random.source if isinstance(random, WorldgenRandom) else random
    source.lo = int(state[1])
    source.hi = int(state[2])


@njit(cache=True, inline="always")
def rotl64(x, k):
    return (x << np.uint64(k)) | (x >> np.uint64(64 - k))


@njit(cache=True, inline="always")
def next_bits(state, bits):
    """One draw, `bits` high bits -- `RandomSource.next(int)`.

    The legacy branch is `LegacyRandomSource.next`; the other is
    `XoroshiroRandomSource.next`, and both generators in `WorldgenRandom` mode
    reach this same method.
    """
    if state[0] == 1:
        seed = (state[1] * _LCG_MULTIPLIER + _LCG_ADDEND) & _LCG_MASK
        state[1] = seed
        return seed >> np.uint64(48 - bits)
    lo = state[1]
    hi = state[2]
    value = lo + hi
    value = rotl64(value, 17) + lo
    hi = hi ^ lo
    state[1] = rotl64(lo, 49) ^ hi ^ (hi << np.uint64(21))
    state[2] = rotl64(hi, 28)
    return value >> np.uint64(64 - bits)


@njit(cache=True, inline="always")
def next_raw(state):
    """One full 64-bit Xoroshiro draw, no bit truncation."""
    lo = state[1]
    hi = state[2]
    value = lo + hi
    value = rotl64(value, 17) + lo
    hi = hi ^ lo
    state[1] = rotl64(lo, 49) ^ hi ^ (hi << np.uint64(21))
    state[2] = rotl64(hi, 28)
    return value


@njit(cache=True, inline="always")
def next_i64(state):
    if state[0] == 2:
        return next_raw(state)
    high = next_bits(state, 32)
    low = next_bits(state, 32)
    return (high << np.uint64(32)) | low


@njit(cache=True, inline="always")
def next_f32(state):
    return np.float32(next_bits(state, 24)) * np.float32(5.9604645e-8)


@njit(cache=True, inline="always")
def next_f64(state):
    """26 + 27 bits for the `java.util.Random` API shapes, 53 for raw.

    `WorldgenRandom`/`LegacyRandom.nextDouble` builds the 53-bit mantissa as
    `(high << 27) | low` from two draws -- high first -- while a bare
    `Xoroshiro.next_f64` truncates one draw to its top 53 bits.
    """
    if state[0] == 2:
        return np.float64(next_raw(state) >> np.uint64(11)) * np.float64(_F64_SCALE)
    high = next_bits(state, 26)
    low = next_bits(state, 27)
    return np.float64((high << np.uint64(27)) | low) * np.float64(_F64_SCALE)


@njit(cache=True, inline="always")
def next_bounded_i32(state, bound):
    """`nextInt(bound)`: Lemire for a bare Xoroshiro, the LCG test otherwise."""
    if state[0] == 2:
        low = np.uint64(next_raw(state)) & _U32
        product = low * np.uint64(bound)
        residue = product & _U32
        if residue < np.uint64(bound):
            threshold = ((np.uint64(1) << np.uint64(32)) - np.uint64(bound)) % np.uint64(bound)
            while residue < threshold:
                low = np.uint64(next_raw(state)) & _U32
                product = low * np.uint64(bound)
                residue = product & _U32
        return product >> np.uint64(32)
    if (bound & (bound - 1)) == 0:
        return (np.uint64(bound) * next_bits(state, 31)) >> np.uint64(31)
    while True:
        value = next_bits(state, 31)
        result = value % np.uint64(bound)
        if (value - result + np.uint64(bound) - np.uint64(1)) < np.uint64(1 << 31):
            return result
