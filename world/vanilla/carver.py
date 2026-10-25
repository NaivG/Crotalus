"""
The carver stage: vanilla's `ChunkStatus.CARVERS`, ported so the tunnels land
where Minecraft puts them.

Terrain and biomes exist by the time this runs. What is added here is the
handful of long winding tunnels and canyons that vanilla *stamps* into that
terrain -- the big cheese and spaghetti caves are not this stage, they are
already inside `final_density`.

    from world.vanilla.carver import carve, load_carvers

    carve(chunk_x, chunk_z, blocks, biomes, preset="overworld", seed=0,
          min_y=-64, height=384, biome_carvers=BIO, carver_data=load_carvers(),
          tags=registry.tags)

`blocks` is `(height, 16, 16)` uint16 `[y, z, x]` holding global block state
ids, and it is edited in place. `biomes` is `(height // 4, 4, 4)` int32 quart
cells. Both layouts are the ones `world/feature/` and
`world/vanilla/biome.py` already produce.

The shape of the stage
----------------------
289 carver chunks -- a 17x17 area centred on the chunk being carved, an 8 chunk
radius -- are visited in `dx` then `dz` order. Each one looks up *its own*
biome, and every carver that biome lists is run once, with its own RNG, seeded
from `(world seed + the carver's index in that biome's list)` and the carver
chunk's coordinates. A carver that passes its `probability` roll stamps
ellipsoids into a mask clipped to this chunk, and when all 289 are done the
mask is applied to the blocks.

That neighbourhood is the reason a cave does not stop at a chunk border: a
tunnel belongs to the carver chunk that started it, and the chunk being
generated only contributes the part of it inside its own 16x16 columns. It is
*not* a reason to read a neighbour's blocks -- see `neighbour_blocks` below,
which this module deliberately never touches.

The traps
---------
**The RNG is per (carver chunk, carver index), and nothing is shared.** Vanilla
seeds it with `WorldgenRandom.setLargeFeatureSeed`: an LCG seeded with
`seed + index`, two `nextLong` draws, then re-seeded with
`chunkX * l ^ chunkZ * m ^ seed`. Reusing one stream across the 289 carver
chunks, or across the carvers of one chunk, still produces plausible caves and
fails only when chunks are generated in a different order.
`tools/check_carvers.py` carves chunk (1,0) first and then (0,0), and compares
against (0,0) alone.

**Which generator.** `chunk_gen_settings.json` carries `legacy_random_source`,
and vanilla turns that into `WorldgenRandom.Algorithm.LEGACY` or `.XOROSHIRO`
(`NoiseGeneratorSettings.getRandomSource()`), then wraps the result in
`WorldgenRandom`. In the overworld that flag is false, so the carvers draw the
*LCG-shaped* methods (`nextInt(bound)`, `nextLong` as two 32-bit draws) over
Xoroshiro bits -- which is exactly what
`world.vanilla.random_source.WorldgenRandom` is. Pumpkin instead constructs a
bare Xoroshiro here, whose `nextInt(bound)` is Lemire multiply-and-shift and
whose `nextLong` is one raw draw; the two streams diverge at the first bounded
draw. `random_source="raw"` selects Pumpkin's behaviour and the default
`"worldgen"` selects vanilla's. This is the one place the port knowingly
disagrees with the read-only reference, and it is a guess: see "Where this is
guessed" below.

**Everything geometric is f32.** A tunnel keeps `x`, `y` and `z` as f64 but
updates them by f32 *increments*, and the two rotation angles are f32
throughout. Doing the increments in double moves caves by blocks over a
hundred-step tunnel.

**Sine is a table, not `math.sin`.** Vanilla's `Mth.sin`/`Mth.cos` index a
65536-entry table with `(int)(v * 10430.378F) & 65535`, built from
`sin(2*pi*i/65536)`. `math.sin` differs in the fourth decimal and changes
ellipsoid radii, which changes which cells are inside. `_sin`/`_cos` reproduce
the table, including the f32 multiply before the truncation.

**The mask has a floor and a ceiling.** It covers `min_y + 1` up to
`min_y + height - 8`: bedrock is never carved, and the world's top seven blocks
are protected from a tunnel that would otherwise breach the surface. That top
bound is also what clamps `carve_ellipsoid`, so it is not merely a filter --
carving without it makes tunnels reach seven blocks higher.

**Two radii, and they are not the same number.** `ChunkGenerator` enumerates
carver chunks in an *8* chunk radius -- 17x17, the 289 invocations the
reference's own javadoc counts -- but `WorldCarver.getRange()`, how far one
carver's tunnel can run from its own chunk, is *4*, so `MAX_DISTANCE` is
`(4 * 2 - 1) * 16 = 112` and not 240. Mixing them up doubles every tunnel and
lets it leave the enumerated area, which would cut caves off at a chunk border;
`MAX_DISTANCE <= RANGE * 16` is the invariant the check tool asserts.

**Carving respects the `replaceable` tag.** Vanilla's `carveBlock` refuses a
cell whose current block is outside the config's `replaceable` tag, so an
already-air cell is left alone (air is not in the tag) and above-surface air
never gets filled with water. Pumpkin's `apply_carving_mask` drops that test
and only skips bedrock; this module restores it, because the tag is data the
config carries and the difference is visible. `tags` is resolved once per chunk
into the union of the tags of the carvers that ran -- with the shipped data
every overworld carver uses `#minecraft:overworld_carver_replaceables` and the
nether carver uses `#minecraft:nether_carver_replaceables`, so the union is
exact.

**Where a carved cell's fluid comes from.** `WorldCarver.getCarveState` tests
`y <= lava_level` *first*, then asks the aquifer, and a `null` answer means "do
not carve this cell at all". With aquifers disabled vanilla falls back to
`default_fluid` below `sea_level` and air above, which is what makes the
nether's caves below y=32 lava. This module does all of that, but it has no
aquifer: see below.

What is *not* here, and why
---------------------------
* **The aquifer.** The overworld's water-filled pockets are placed by
  `CarverAquiferSampler` from the `barrier`, `fluid_level_*` and `lava` router
  slots plus its own positional RNG. That is a separate subsystem and it is not
  in this module's seam, so with `aquifer=None` the overworld carves air. The
  alternative -- treating `y < sea_level` as water the way the aquifers-disabled
  path does -- would flood every cave below y=63, which is much further from
  vanilla than leaving them dry. Pass `aquifer=` to plug one in.
* **The surface material rule.** Vanilla's `carveBlock` re-runs the surface
  rule under a carved cell once the tunnel has passed grass or mycelium in that
  column. `top_material=` is that hook; with `None` the tracking still happens
  and nothing is written, which is the right default while
  `world/vanilla/surface.py` owns the rule. Vanilla also passes a `steep` flag
  derived from the chunk's surface heightmap, which this stage does not have
  here, so `False` is passed.
* **`neighbour_blocks`.** Accepted and never read. The algorithm's only block
  reads and writes are inside the chunk being carved: `carve_ellipsoid` clamps
  its loop to this chunk's columns, and the mask is applied to this chunk. A
  tunnel owned by a neighbour reaches in because the 17x17 loop *runs* that
  neighbour, not because anything is read from it. `tools/check_carvers.py`
  asserts this by passing an object that raises on any attribute access and
  comparing the result with `None`. `None` therefore does **not** mean "the
  neighbours are air"; it means nothing changes.
* **The carver chunk's biome.** Every position in the 17x17 loop needs the
  biome *of that carver chunk*, including the 288 outside this one, because the
  biome's carver list decides which carvers run and in which order. `biomes`
  only covers this chunk, so `biome_at(quart_x, quart_y, quart_z)` is the exact
  path and the fallback clamps the query into this chunk's array. That fallback
  is harmless *for the shipped data*: all 54 overworld biomes list the same
  three carvers and all 5 nether biomes list `minecraft:nether_cave`, so no
  neighbour lookup can change the carver list. A datapack that gave one biome a
  different list would need `biome_at`.
* **`debug_settings`.** Three of the four shipped carvers carry it (with
  crimson-button air and stained-glass lava), but vanilla only honours it when
  `SharedConstants.IS_RUNNING_IN_IDE` is also true, so in production those
  states are inert. The field is recorded on the config and never applied; that
  is the reference behaviour, not a silent skip.

Where this is guessed
---------------------
There is no vanilla dump for this stage, so the port is exact where the
reference is readable and a documented guess where it is not:

* `count`, `thickness`, `weird_thickness_bias` and
  `start_vertical_radius_multiplier` are **not** in this repo's
  `assets/carver.json` -- it holds exactly the fields the Java codec reads, and
  those four are code-side constants in `CaveWorldCarver` (its `getCaveBound`,
  `getThickness` and `getYScale` methods). They are taken from Pumpkin's
  `assets/datapack/.../carver/*.json`, which is a runtime dump and *does* carry
  them, and the two cross-check: Pumpkin names the cave's `yScale` provider
  `room_vertical_radius_multiplier` and gives it the same `uniform 0.1..0.9`
  this repo's `yScale` has, which is what identifies `yScale` as the room's
  aspect ratio rather than the tunnel's. `_CODE_SIDE_DEFAULTS` below feeds them
  in wherever `assets/carver.json` is silent, so a fuller dump supersedes it.
* `TrapezoidFloatProvider` draws **two** floats
  (`min + U*h + U*g`, with `h = g + plateau`); Pumpkin's is a one-draw
  square-root redesign. Two draws is vanilla's shape *and* vanilla's draw
  count, and getting the count wrong desynchronises every tunnel after it.
* `UniformFloatProvider` is `min + nextFloat() * (max - min)` with three f32
  roundings; Pumpkin uses `mul_add`, i.e. one.
* Vanilla's `createTunnel` builds its RNG from the tunnel seed it was handed.
  Whether it re-derives that positionally first (as `setCarverSeed` did in
  1.16) could not be confirmed, and Pumpkin does not. This module follows
  Pumpkin: a positional re-derivation would make one tunnel's shape depend on
  which chunk is being carved, which is the discontinuity the carver-chunk
  parameter exists to prevent.
* `CARVER_RANGE = 4` is inferred, not read: `WorldCarver.getRange()`'s body is
  not in any javadoc, and Pumpkin hardcodes `(4 * 2 - 1)` in *both* the cave and
  the canyon carver, which is where the 4 comes from. It is also the only
  reading that is self-consistent -- `MAX_DISTANCE` has to fit inside the 17x17
  loop's 128 blocks, and 240 would not -- so a `getRange()` of 8 would mean
  vanilla cuts its own tunnels off at chunk borders.
  `tools/check_carvers.py` measures what the other reading would carve.
* `VeryBiasedToBottom` sampling (the cave count) is Pumpkin's formulation. Its
  int and height variants do not agree with each other and neither could be
  checked against Java, so the cave *count* is the least certain number in the
  port. `tools/check_carvers.py` prints the resulting cave density so a wrong
  count is at least visible.
* The mask-first structure is Pumpkin's: vanilla's `carveBlock` writes blocks
  while `carve_ellipsoid` runs, so a cell is read back after it has been
  carved. Here the mask records positions and one apply pass writes them all.
  Since the state a cell becomes is a pure function of its position and the
  cell's own block (not of anything else carved), the outcomes agree; the one
  real difference is that `has_grass` below is tracked per column run here,
  where vanilla's `reachedSurface` flag is shared across one ellipsoid call.
  That only matters through `top_material=`, which is usually `None`.

Unsupported carvers
-------------------
`SUPPORTED_TYPES` is the set of `type` strings this module implements. An
unknown type raises `UnsupportedCarverError`, from `load_carvers` at load time
and again from `carve` if a caller hands in a config built elsewhere. Skipping
it instead would produce terrain that looks entirely plausible and is wrong,
with nothing in any log to say so -- the same reasoning `router.slot` uses when
it raises on an unknown slot name.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

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


from world.vanilla.random_source import LegacyRandom, WorldgenRandom, Xoroshiro

F32 = np.float32

#: Carver `type` strings this module implements. `load_carvers` raises for
#: anything else rather than producing a world with silent holes in it.
SUPPORTED_TYPES = frozenset({"minecraft:cave", "minecraft:nether_cave", "minecraft:canyon"})

#: `WorldCarver.getRange()`: how far a carver reaches from the carver chunk
#: that owns it. Not the same thing as `RANGE` below, and getting the two
#: mixed up doubles every tunnel.
CARVER_RANGE = 4

#: The carver-chunk radius `ChunkGenerator.applyCarvers` iterates: an 8 chunk
#: radius, 17x17 carver chunks (289 of them, which is what the reference's own
#: javadoc counts).
RANGE = 8

#: The world's top 7 blocks are never carved; `min_y + 1` protects bedrock.
PROTECTED_BLOCKS_ON_TOP = 7

#: `(getRange() * 2 - 1) << 4` -- the longest a cave tunnel or a canyon can be.
#:
#: This has to stay below `RANGE * 16`, or a tunnel owned by one of the far
#: carver chunks in the 17x17 loop could leave the enumerated area and be cut
#: off mid-path -- a chunk border with half a cave on each side. `112 <= 128`,
#: which is exactly why the loop radius is 8 and not 7.
MAX_DISTANCE = (CARVER_RANGE * 2 - 1) << 4

#: `Mth.sin`'s table scale: 65536 / (2 * pi).
_SIN_SCALE = F32(10430.378)
_SIN_TABLE_SIZE = 65536

#: Air, fixed at 0 by the registry (`world/block_states.py`).
AIR = 0

_MASK64 = 0xFFFFFFFFFFFFFFFF
_U64_SIGN = 1 << 63
_PREFIX = "minecraft:"

#: Stand-in for "this ellipsoid has no per-height width factors" (caves). It
#: has to be an array, not `None`, because the compiled path types its
#: arguments.
_NO_WIDTH_FACTORS = np.zeros(1, dtype=np.float32)


def _fadd(a, b) -> np.float32:
    return F32(F32(a) + F32(b))


def _fsub(a, b) -> np.float32:
    return F32(F32(a) - F32(b))


def _fmul(a, b) -> np.float32:
    return F32(F32(a) * F32(b))


def _fdiv(a, b) -> np.float32:
    return F32(F32(a) / F32(b))


def _fabs(value) -> np.float32:
    return F32(abs(F32(value)))


# ==========================================================================
# `Mth.sin` / `Mth.cos`
# ==========================================================================

_SIN_TABLE: Optional[np.ndarray] = None


def sin_table() -> np.ndarray:
    """Vanilla's 65536-entry sine table, built once.

    `Mth`'s static block fills it with `(float)Math.sin((double)i * PI * 2.0 /
    65536.0)`. A numpy `sin` over the same doubles agrees to within an f64 ulp,
    which is far below the f32 rounding that follows.
    """
    global _SIN_TABLE
    if _SIN_TABLE is None:
        i = np.arange(_SIN_TABLE_SIZE, dtype=np.float64)
        _SIN_TABLE = np.sin(i * math.pi * 2.0 / _SIN_TABLE_SIZE).astype(np.float32)
    return _SIN_TABLE


def _sin(value) -> np.float32:
    """`Mth.sin`: table lookup after an *f32* multiply and a truncating cast."""
    index = int(F32(value) * _SIN_SCALE) & 0xFFFF
    return sin_table()[index]


def _cos(value) -> np.float32:
    """`Mth.cos`: the same table, a quarter period ahead."""
    index = int(F32(value) * _SIN_SCALE + F32(16384.0)) & 0xFFFF
    return sin_table()[index]


# ==========================================================================
# Value providers
#
# The datapack's `FloatProvider` / `IntProvider` / `HeightProvider` classes.
# Only the variants the carver configs use are implemented; anything else
# raises, because a silently defaulted provider moves every draw after it.
#
# The f32 rounding is not decoration. `min + nextFloat() * (max - min)` is
# three roundings in Java and two if the difference is precomputed in double,
# and the result feeds a radius that decides whether a cell is inside an
# ellipsoid.
# ==========================================================================


class _ConstantFloat:
    __slots__ = ("value",)

    def __init__(self, value: float) -> None:
        self.value = F32(value)

    def get(self, random) -> np.float32:
        return self.value

    def __repr__(self) -> str:
        return f"constant({float(self.value)})"


class _UniformFloat:
    __slots__ = ("min_inclusive", "max_exclusive")

    def __init__(self, low: float, high: float) -> None:
        self.min_inclusive = F32(low)
        self.max_exclusive = F32(high)

    def get(self, random) -> np.float32:
        span = _fsub(self.max_exclusive, self.min_inclusive)
        return _fadd(self.min_inclusive, _fmul(F32(random.next_f32()), span))

    def __repr__(self) -> str:
        return f"uniform({float(self.min_inclusive)}, {float(self.max_exclusive)})"


class _TrapezoidFloat:
    """`min + U*h + U*g`, with `h = g + plateau` -- two draws, not one."""

    __slots__ = ("min", "max", "plateau")

    def __init__(self, low: float, high: float, plateau: float) -> None:
        self.min = F32(low)
        self.max = F32(high)
        self.plateau = F32(plateau)

    def get(self, random) -> np.float32:
        span = _fsub(self.max, self.min)
        ramp = _fdiv(_fsub(span, self.plateau), F32(2.0))
        inner = _fsub(span, ramp)
        return _fadd(
            _fadd(self.min, _fmul(F32(random.next_f32()), inner)),
            _fmul(F32(random.next_f32()), ramp),
        )

    def __repr__(self) -> str:
        return f"trapezoid({float(self.min)}, {float(self.max)}, {float(self.plateau)})"


class _ConstantInt:
    __slots__ = ("value",)

    def __init__(self, value: int) -> None:
        self.value = int(value)

    def get(self, random) -> int:
        return self.value

    def __repr__(self) -> str:
        return f"constant({self.value})"


class _UniformInt:
    __slots__ = ("min_inclusive", "max_inclusive")

    def __init__(self, low: int, high: int) -> None:
        self.min_inclusive = int(low)
        self.max_inclusive = int(high)

    def get(self, random) -> int:
        if self.min_inclusive >= self.max_inclusive:
            return self.min_inclusive
        return self.min_inclusive + random.next_bounded_i32(
            self.max_inclusive - self.min_inclusive + 1
        )

    def __repr__(self) -> str:
        return f"uniform({self.min_inclusive}, {self.max_inclusive})"


class _VeryBiasedToBottomInt:
    """`bound = U(bound) + 1`, `inner + 1` times, then `min + bound - 1`.

    Pumpkin's formulation. It drives the cave count, so it decides how many
    tunnel systems one carver chunk start produces; keeping it identical to the
    reference is what keeps the cave density comparable. See the module
    docstring -- this is the least certain number in the port.
    """

    __slots__ = ("min_inclusive", "max_inclusive", "inner")

    def __init__(self, low: int, high: int, inner: int = 1) -> None:
        self.min_inclusive = int(low)
        self.max_inclusive = int(high)
        self.inner = int(inner)

    def get(self, random) -> int:
        if self.min_inclusive >= self.max_inclusive:
            return self.min_inclusive
        bound = self.max_inclusive - self.min_inclusive + 1
        for _ in range(self.inner + 1):
            bound = random.next_bounded_i32(bound) + 1
        return self.min_inclusive + bound - 1

    def __repr__(self) -> str:
        return (
            f"very_biased_to_bottom({self.min_inclusive}, {self.max_inclusive}, "
            f"inner={self.inner})"
        )


def _float_provider(payload, what: str):
    if isinstance(payload, (int, float)):
        return _ConstantFloat(payload)
    if not isinstance(payload, Mapping):
        raise ValueError(f"carver config field {what!r}: not a float provider: {payload!r}")
    kind = payload.get("type", "minecraft:constant")
    if kind == "minecraft:constant":
        return _ConstantFloat(payload.get("value", 0.0))
    if kind == "minecraft:uniform":
        return _UniformFloat(payload["min_inclusive"], payload["max_exclusive"])
    if kind == "minecraft:trapezoid":
        return _TrapezoidFloat(payload["min"], payload["max"], payload.get("plateau", 0.0))
    raise ValueError(f"carver config field {what!r}: unsupported float provider {kind!r}")


def _int_provider(payload, what: str):
    if isinstance(payload, int):
        return _ConstantInt(payload)
    if not isinstance(payload, Mapping):
        raise ValueError(f"carver config field {what!r}: not an int provider: {payload!r}")
    kind = payload.get("type", "minecraft:constant")
    if kind == "minecraft:constant":
        return _ConstantInt(payload.get("value", 0))
    if kind == "minecraft:uniform":
        return _UniformInt(payload["min_inclusive"], payload["max_inclusive"])
    if kind == "minecraft:very_biased_to_bottom":
        return _VeryBiasedToBottomInt(
            payload["min_inclusive"], payload["max_inclusive"], payload.get("inner", 1)
        )
    raise ValueError(f"carver config field {what!r}: unsupported int provider {kind!r}")


# ==========================================================================
# Height / Y-offset providers
# ==========================================================================


def resolve_y(offset, min_y: int, height: int) -> int:
    """`YOffset.get_y`: absolute, above the bottom, or below the top."""
    if isinstance(offset, (int, float)):
        return int(offset)
    if not isinstance(offset, Mapping):
        raise ValueError(f"not a Y offset: {offset!r}")
    if "absolute" in offset:
        return int(offset["absolute"])
    if "above_bottom" in offset:
        return min_y + int(offset["above_bottom"])
    if "below_top" in offset:
        return min_y + height - 1 - int(offset["below_top"])
    raise ValueError(f"not a Y offset: {offset!r}")


class _HeightProvider:
    """Base of the three `HeightProvider` variants -- Pumpkin's `get_height`."""

    __slots__ = ()

    def get(self, random, min_y: int, height: int) -> int:  # pragma: no cover
        raise NotImplementedError


class _UniformHeight(_HeightProvider):
    __slots__ = ("min_inclusive", "max_inclusive")

    def __init__(self, low, high) -> None:
        self.min_inclusive = low
        self.max_inclusive = high

    def get(self, random, min_y: int, height: int) -> int:
        low = resolve_y(self.min_inclusive, min_y, height)
        high = resolve_y(self.max_inclusive, min_y, height)
        if low > high:
            return low
        if low == high:
            return low
        return low + random.next_bounded_i32(high - low + 1)

    def __repr__(self) -> str:
        return f"uniform({self.min_inclusive}, {self.max_inclusive})"


class _TrapezoidHeight(_HeightProvider):
    __slots__ = ("min_inclusive", "max_inclusive", "plateau")

    def __init__(self, low, high, plateau: int = 0) -> None:
        self.min_inclusive = low
        self.max_inclusive = high
        self.plateau = int(plateau or 0)

    def get(self, random, min_y: int, height: int) -> int:
        low = resolve_y(self.min_inclusive, min_y, height)
        high = resolve_y(self.max_inclusive, min_y, height)
        if low > high:
            return low
        span = high - low
        if self.plateau >= span:
            return low + random.next_bounded_i32(span + 1)
        half = (span - self.plateau) // 2
        other = span - half
        return low + random.next_bounded_i32(other + 1) + random.next_bounded_i32(half + 1)

    def __repr__(self) -> str:
        return f"trapezoid({self.min_inclusive}, {self.max_inclusive}, {self.plateau})"


class _VeryBiasedToBottomHeight(_HeightProvider):
    __slots__ = ("min_inclusive", "max_inclusive", "inner")

    def __init__(self, low, high, inner: int = 1) -> None:
        self.min_inclusive = low
        self.max_inclusive = high
        self.inner = int(inner)

    def get(self, random, min_y: int, height: int) -> int:
        low = resolve_y(self.min_inclusive, min_y, height)
        high = resolve_y(self.max_inclusive, min_y, height)
        if high - low - self.inner < 0:
            return low
        upper = low + self.inner + random.next_bounded_i32(high - low - self.inner + 1)
        biased = low + random.next_bounded_i32(upper - low)
        return low + random.next_bounded_i32(biased - low + self.inner)

    def __repr__(self) -> str:
        return f"very_biased_to_bottom({self.min_inclusive}, {self.max_inclusive})"


def _height_provider(payload, what: str) -> _HeightProvider:
    if not isinstance(payload, Mapping):
        raise ValueError(f"carver config field {what!r}: not a height provider: {payload!r}")
    kind = payload.get("type")
    if kind == "minecraft:uniform":
        return _UniformHeight(payload["min_inclusive"], payload["max_inclusive"])
    if kind == "minecraft:trapezoid":
        return _TrapezoidHeight(
            payload["min_inclusive"], payload["max_inclusive"], payload.get("plateau", 0)
        )
    if kind == "minecraft:very_biased_to_bottom":
        return _VeryBiasedToBottomHeight(
            payload["min_inclusive"], payload["max_inclusive"], payload.get("inner", 1)
        )
    raise ValueError(f"carver config field {what!r}: unsupported height provider {kind!r}")


# ==========================================================================
# The config
# ==========================================================================

#: Fields `CaveWorldCarver` holds in code rather than reading from the codec,
#: for the carvers in `assets/carver.json`. Read off Pumpkin's runtime dump
#: (`Pumpkin/assets/datapack/data/minecraft/worldgen/carver/*.json`), which is
#: the only copy of them in either checkout -- a key present in
#: `assets/carver.json` always wins over this table, so a fuller dump in this
#: repo supersedes it. `room_vertical_radius_multiplier` is deliberately *not*
#: here: it is the config's `yScale`, and Pumpkin's dump naming that same
#: 0.1..0.9 provider `room_vertical_radius_multiplier` is what identifies
#: `yScale` as the room's aspect ratio rather than the tunnel's.
_CODE_SIDE_DEFAULTS: Dict[str, dict] = {
    "cave": {
        "count": {
            "type": "minecraft:very_biased_to_bottom",
            "min_inclusive": 0,
            "max_inclusive": 14,
            "inner": 1,
        },
        "thickness": {"type": "minecraft:trapezoid", "min": 0.0, "max": 3.0, "plateau": 1.0},
        "weird_thickness_bias": True,
        "start_vertical_radius_multiplier": 1.0,
    },
    "cave_extra_underground": {
        "count": {
            "type": "minecraft:very_biased_to_bottom",
            "min_inclusive": 0,
            "max_inclusive": 14,
            "inner": 1,
        },
        "thickness": {"type": "minecraft:trapezoid", "min": 0.0, "max": 3.0, "plateau": 1.0},
        "weird_thickness_bias": True,
        "start_vertical_radius_multiplier": 1.0,
    },
    "nether_cave": {
        "count": {
            "type": "minecraft:very_biased_to_bottom",
            "min_inclusive": 0,
            "max_inclusive": 9,
            "inner": 1,
        },
        "thickness": {"type": "minecraft:trapezoid", "min": 0.0, "max": 6.0, "plateau": 2.0},
        "weird_thickness_bias": False,
        "start_vertical_radius_multiplier": 5.0,
    },
}


class UnsupportedCarverError(RuntimeError):
    """A carver `type` this module does not implement.

    Raised at load time and again at the seam. See the module docstring for why
    skipping is not an option.
    """


@dataclass(frozen=True)
class CarverConfig:
    """One `assets/carver.json` entry, with its providers parsed.

    The codec fields as they appear in the asset, plus the
    `_CODE_SIDE_DEFAULTS` values for the four fields the datapack does not
    carry. Cave-only and canyon-only fields are `None` on the other kind, and
    `load_carvers` is what guarantees the split.

    `y_scale` and `room_vertical_radius_multiplier` are the same provider for a
    cave (the asset has one `yScale`); the two names are kept because that is
    how the two call sites read in the reference.
    """

    name: str
    type: str
    probability: float
    y: _HeightProvider
    y_scale: object
    lava_level: object
    replaceable: str

    # -- cave
    count: Optional[object] = None
    thickness: Optional[object] = None
    weird_thickness_bias: bool = False
    horizontal_radius_multiplier: Optional[object] = None
    vertical_radius_multiplier: Optional[object] = None
    floor_level: Optional[object] = None
    start_vertical_radius_multiplier: Optional[object] = None
    room_vertical_radius_multiplier: Optional[object] = None

    # -- canyon
    vertical_rotation: Optional[object] = None
    distance_factor: Optional[object] = None
    shape_thickness: Optional[object] = None
    width_smoothness: int = 0
    horizontal_radius_factor: Optional[object] = None
    vertical_radius_default_factor: float = 1.0
    vertical_radius_center_factor: float = 0.0

    #: Inert outside the IDE -- see the module docstring.
    debug_settings: Optional[Mapping] = None
    raw: Mapping = field(default_factory=dict, repr=False)

    def resolved_lava_level(self, min_y: int, height: int) -> int:
        """The cell at or below this Y always becomes lava."""
        return resolve_y(self.lava_level, min_y, height)


def _assets_path(name: str) -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, "assets", name)


def _canonical_tag(tag) -> str:
    if tag is None:
        return ""
    name = str(tag)
    if not name.startswith("#"):
        name = "#" + name
    if ":" not in name[1:]:
        name = "#minecraft:" + name[1:]
    return name


def _build_config(name: str, payload: Mapping) -> CarverConfig:
    ctype = str(payload.get("type", ""))
    if ctype not in SUPPORTED_TYPES:
        raise UnsupportedCarverError(
            f"carver {name!r} has type {ctype!r}; this module implements "
            f"{', '.join(sorted(SUPPORTED_TYPES))}"
        )
    config = payload.get("config") or {}
    bare = name[len(_PREFIX):] if name.startswith(_PREFIX) else name
    extra = _CODE_SIDE_DEFAULTS.get(bare, {})

    def pick(field_name: str, default=None):
        if field_name in config:
            return config[field_name]
        if field_name in extra:
            return extra[field_name]
        return default

    if ctype in ("minecraft:cave", "minecraft:nether_cave"):
        kind = dict(
            count=_int_provider(pick("count"), "count"),
            thickness=_float_provider(pick("thickness"), "thickness"),
            weird_thickness_bias=bool(pick("weird_thickness_bias", False)),
            horizontal_radius_multiplier=_float_provider(
                config.get("horizontal_radius_multiplier"), "horizontal_radius_multiplier"
            ),
            vertical_radius_multiplier=_float_provider(
                config.get("vertical_radius_multiplier"), "vertical_radius_multiplier"
            ),
            floor_level=_float_provider(config.get("floor_level"), "floor_level"),
            start_vertical_radius_multiplier=_float_provider(
                pick("start_vertical_radius_multiplier", 1.0), "start_vertical_radius_multiplier"
            ),
            # `config.yScale` is the room's vertical radius as a fraction of
            # the horizontal one; Pumpkin's runtime dump names that same
            # provider `room_vertical_radius_multiplier`.
            room_vertical_radius_multiplier=_float_provider(config.get("yScale"), "yScale"),
        )
    else:
        shape = config.get("shape") or {}
        kind = dict(
            vertical_rotation=_float_provider(config.get("vertical_rotation"), "vertical_rotation"),
            distance_factor=_float_provider(shape.get("distance_factor"), "shape.distance_factor"),
            shape_thickness=_float_provider(shape.get("thickness"), "shape.thickness"),
            width_smoothness=int(shape.get("width_smoothness", 0)),
            horizontal_radius_factor=_float_provider(
                shape.get("horizontal_radius_factor"), "shape.horizontal_radius_factor"
            ),
            vertical_radius_default_factor=float(shape.get("vertical_radius_default_factor", 1.0)),
            vertical_radius_center_factor=float(shape.get("vertical_radius_center_factor", 0.0)),
        )

    return CarverConfig(
        name=name,
        type=ctype,
        probability=float(config.get("probability", 0.0)),
        y=_height_provider(config.get("y"), "y"),
        y_scale=_float_provider(config.get("yScale"), "yScale"),
        lava_level=config.get("lava_level"),
        replaceable=_canonical_tag(config.get("replaceable")),
        debug_settings=config.get("debug_settings"),
        raw=dict(payload),
        **kind,
    )


def load_carvers(path: Optional[str] = None) -> Dict[str, CarverConfig]:
    """`assets/carver.json` as `name -> CarverConfig`, providers parsed.

    Raises `UnsupportedCarverError` for a carver type this module does not
    implement, and `ValueError` for a provider variant it does not understand.
    Both are data problems, and both would otherwise show up as terrain that is
    subtly wrong rather than as a failure.
    """
    with open(path or _assets_path("carver.json"), encoding="utf-8") as handle:
        payload = json.load(handle)
    return {str(name): _build_config(str(name), entry) for name, entry in payload.items()}


def _lookup_config(carver_data: Mapping[str, object], name: str):
    """Find a carver's config under either spelling of its id.

    `biome.json` and `BiomeTable` spell carvers `minecraft:cave`;
    `assets/carver.json` and `registry.worldgen` key them `cave`. Both spellings
    are in use on either side of this seam, and a bare lookup would fail on
    whichever one it did not expect.
    """
    for key in (name, name[len(_PREFIX):] if name.startswith(_PREFIX) else _PREFIX + name):
        if key in carver_data:
            return carver_data[key]
    raise KeyError(
        f"carver {name!r} is not in carver_data ({', '.join(sorted(carver_data))})"
    )


def _as_config(name: str, value: object) -> CarverConfig:
    """Accept a `CarverConfig`, a `registry.worldgen.CarverData`, or a payload.

    `carve` is handed `carver_data` by the generator, and the generator may
    have built it from either `load_carvers()` or the game registry. Both carry
    the same numbers; only the wrapping differs.
    """
    if isinstance(value, CarverConfig):
        return value
    raw = getattr(value, "config", None)
    if isinstance(raw, Mapping):
        return _build_config(name, raw)
    if isinstance(value, Mapping):
        return _build_config(name, value)
    raise TypeError(f"carver_data[{name!r}] is neither a CarverConfig nor a payload: {value!r}")


# ==========================================================================
# Biome carver lists
# ==========================================================================

def _canonical_id(name: str) -> str:
    name = str(name)
    return name if ":" in name else _PREFIX + name


def _normalize_carver_names(entries) -> Tuple[str, ...]:
    """A biome's `carvers` value as a tuple of ids, repairing the string form.

    `biome.json` writes a single carver as a bare string
    (`"carvers": "minecraft:nether_cave"`), and iterating a string yields its
    characters. `registry/worldgen.py` documents that trap; `BiomeTable` in
    this package does not handle it, so `BiomeTable.carvers` for the five
    nether biomes arrives as 21 one-character entries. Joining them back is the
    only repair available at this point in the call chain, and the shape of the
    result is checked before it is used.
    """
    if isinstance(entries, str):
        return (_canonical_id(entries),)
    values = tuple(entries or ())
    if len(values) > 1 and all(
        isinstance(value, str) and len(value) == len(_PREFIX) + 1 and value.startswith(_PREFIX)
        for value in values
    ):
        joined = "".join(value[len(_PREFIX):] for value in values)
        return (_canonical_id(joined),)
    return tuple(_canonical_id(value) for value in values)


def carvers_for_biome(biome_id: int, biome_table) -> Tuple[str, ...]:
    """The carver ids a numeric biome id lists, in order.

    `biome_table` is `world.vanilla.biome.BiomeTable`. The order matters twice
    over: it is the order the carvers run, and the index in the list is the
    salt in the carver's seed.
    """
    name = biome_table.name_of(int(biome_id))
    return _normalize_carver_names(biome_table.carvers.get(name, ()))


# ==========================================================================
# Preset settings
# ==========================================================================

_SETTINGS_CACHE: Dict[str, dict] = {}


def preset_settings(preset: str) -> dict:
    """`sea_level`, `legacy_random_source`, `aquifers_enabled`, `default_fluid`.

    Read out of `assets/chunk_gen_settings.json` rather than passed in, because
    the seam this module is called through fixes its keyword list and every one
    of these is a property of the preset, not of the chunk.
    """
    key = preset[len(_PREFIX):] if preset.startswith(_PREFIX) else preset
    if key not in _SETTINGS_CACHE:
        with open(_assets_path("chunk_gen_settings.json"), encoding="utf-8") as handle:
            payload = json.load(handle)
        for name, entry in payload.items():
            noise = entry.get("noise") or {}
            _SETTINGS_CACHE[name] = {
                "sea_level": int(entry.get("sea_level", 0)),
                "legacy_random_source": bool(entry.get("legacy_random_source", True)),
                "aquifers_enabled": bool(entry.get("aquifers_enabled", False)),
                "default_fluid": entry.get("default_fluid") or {"Name": "minecraft:air"},
                "min_y": int(noise.get("min_y", -64)),
                "height": int(noise.get("height", 384)),
            }
    try:
        return _SETTINGS_CACHE[key]
    except KeyError:
        raise KeyError(
            f"no generator preset {preset!r}; loaded presets are "
            f"{', '.join(sorted(_SETTINGS_CACHE))}"
        ) from None


# ==========================================================================
# Block ids and tags
# ==========================================================================

_BLOCK_IDS: Dict[object, int] = {}


def _block_id(name: str, properties: Optional[Mapping] = None) -> int:
    """A global block state id, resolved from the registry and cached.

    Deliberately lazy: importing this module must not require a booted
    registry, because `world/vanilla` is importable on its own.
    """
    key: object = name if not properties else (name, tuple(sorted(properties.items())))
    cached = _BLOCK_IDS.get(key)
    if cached is None:
        from world.block_states import state_id

        cached = state_id(name, dict(properties) if properties else None)
        _BLOCK_IDS[key] = cached
    return cached


def _block_id_from_payload(payload) -> int:
    """A `{Name, Properties}` payload as a state id (the preset's fluid)."""
    if isinstance(payload, Mapping):
        name = str(payload.get("Name", "minecraft:air"))
        properties = payload.get("Properties")
        return _block_id(name, properties if isinstance(properties, Mapping) else None)
    return _block_id(str(payload))


def _state_ids_of_block(name: str) -> range:
    """Every state id of a block. A tag holds blocks, not states."""
    from registry import get_game_registry

    block = get_game_registry().blocks.get_block(_canonical_id(name))
    if block is None:
        raise KeyError(f"{name!r} is not in the block registry")
    return block.state_ids


def _tag_members(tags, tag_name: str) -> Iterable:
    """Whatever the caller's `tags` object can say about one tag.

    The seam only promises "answers `#minecraft:...` with an iterable of global
    block state ids", so a plain mapping and a callable are both accepted, and
    a `registry.tags.TagRegistry` is understood too, because that is what the
    server will hand over.
    """
    if tags is None:
        raise ValueError("carve() needs a `tags` object; without one nothing is replaceable")
    if callable(tags):
        return tags(tag_name)
    getter = getattr(tags, "states", None)
    if callable(getter):
        return getter(tag_name)
    try:
        return tags[tag_name]  # type: ignore[index]
    except (TypeError, KeyError):
        pass
    bare = tag_name[1:] if tag_name.startswith("#") else tag_name
    if not bare.startswith(_PREFIX):
        bare = _PREFIX + bare
    if hasattr(tags, "entries"):
        return tags.entries("minecraft:block", bare)
    raise TypeError(f"cannot resolve tag {tag_name!r} from {type(tags).__name__}")


def resolve_replaceable(tags, tag_names: Iterable[str]) -> Set[int]:
    """The union of some tags, as global block state ids.

    Members may already be ids or block *names* (a `TagRegistry` answers with
    names), and a name expands to every one of its states -- vanilla tests the
    `BlockState` against a `HolderSet<Block>`, so `water[level=3]` is
    replaceable if `minecraft:water` is tagged.
    """
    ids: Set[int] = set()
    seen: List[str] = []
    for tag_name in tag_names:
        if not tag_name or tag_name in seen:
            continue
        seen.append(tag_name)
        for member in _tag_members(tags, tag_name):
            if isinstance(member, (int, np.integer)):
                ids.add(int(member))
            else:
                ids.update(_state_ids_of_block(str(member)))
    if not ids:
        raise ValueError(
            f"replaceable tags {sorted(seen)} resolved to no blocks; refusing to "
            "carve nothing and call it success"
        )
    return ids


# ==========================================================================
# The RNG
# ==========================================================================

def _to_signed64(value: int) -> int:
    return value - (1 << 64) if value >= _U64_SIGN else value


def get_large_feature_seed(seed: int, chunk_x: int, chunk_z: int) -> int:
    """`WorldgenRandom.setLargeFeatureSeed`: the carver's positional seed.

    An LCG seeded with `seed` supplies two 64-bit scales; the result is
    `chunkX * l ^ chunkZ * m ^ seed`. Both multiplies wrap at 64 bits, and the
    sign only matters through that wrap, so the final mask is what makes this
    match Java.
    """
    random = LegacyRandom(seed)
    x_scale = _to_signed64(random.next_i64())
    z_scale = _to_signed64(random.next_i64())
    result = (chunk_x * x_scale) ^ (chunk_z * z_scale) ^ _to_signed64(seed)
    return result & _MASK64


def new_carver_random(seed: int, legacy: bool, mode: str = "worldgen"):
    """The carver's own generator for one (carver chunk, carver index).

    `legacy` is the preset's `legacy_random_source`, which vanilla turns into
    `WorldgenRandom.Algorithm.LEGACY` / `XOROSHIRO` and then wraps in
    `WorldgenRandom`. `mode="raw"` skips that wrapper, which is what Pumpkin
    does; for the legacy algorithm the two are the same generator, for
    Xoroshiro they are not (see the module docstring).
    """
    if mode == "raw":
        return LegacyRandom(seed) if legacy else Xoroshiro.from_seed(seed)
    if mode != "worldgen":
        raise ValueError(f"unknown random_source {mode!r}; expected 'worldgen' or 'raw'")
    return LegacyRandom(seed) if legacy else WorldgenRandom(Xoroshiro.from_seed(seed))


# ==========================================================================
# The mask
# ==========================================================================

class CarvingMask:
    """Pumpkin's `CarvingMask`: bits indexed `(y - min_y) + (z + x*16) * height`.

    The index order matters only for what `columns()` yields, and that decides
    the order cells are written in. Vanilla's own mask wraps a `BitSet`; the
    layout here is the reference's, and `columns()` reproduces its
    run-splitting exactly -- including the reference's own unit test, which
    `tools/check_carvers.py` re-runs.
    """

    __slots__ = ("min_y", "max_y", "height", "words")

    def __init__(self, min_y: int, max_y: int) -> None:
        self.min_y = int(min_y)
        self.max_y = int(max_y)
        self.height = max(0, self.max_y - self.min_y + 1)
        words = (256 * self.height + 63) // 64
        self.words = np.zeros(max(words, 1), dtype=np.uint64)

    def _index(self, x: int, y: int, z: int) -> int:
        return (y - self.min_y) + ((z + (x << 4)) * self.height)

    def set(self, x: int, y: int, z: int) -> None:
        if y < self.min_y or y > self.max_y or x >= 16 or z >= 16 or x < 0 or z < 0:
            return
        index = self._index(x, y, z)
        self.words[index >> 6] |= np.uint64(1) << np.uint64(index & 63)

    def get(self, x: int, y: int, z: int) -> bool:
        if y < self.min_y or y > self.max_y or x >= 16 or z >= 16 or x < 0 or z < 0:
            return False
        index = self._index(x, y, z)
        return bool((self.words[index >> 6] >> np.uint64(index & 63)) & np.uint64(1))

    def is_empty(self) -> bool:
        return not bool(self.words.any())

    def columns(self) -> List[Tuple[int, int, List[Tuple[int, int]]]]:
        """`(x, z, [(bottom_y, top_y), ...])`, in the reference's order.

        A run is a maximal stretch of set bits. Runs are found per column, in
        flat index order -- x ascending, then z -- and within a column in
        ascending y, which is the order `visit` writes in.
        """
        if self.height <= 0:
            return []
        flat = np.unpackbits(self.words.view(np.uint8), bitorder="little")
        flat = flat[: 256 * self.height].reshape(256, self.height)
        padded = np.zeros((256, self.height + 2), dtype=bool)
        padded[:, 1:-1] = flat
        starts = np.nonzero(padded[:, 1:-1] & ~padded[:, :-2])
        ends = np.nonzero(padded[:, 1:-1] & ~padded[:, 2:])
        result: List[Tuple[int, int, List[Tuple[int, int]]]] = []
        runs: List[Tuple[int, int]] = []
        current = -1
        for column, y_start, y_end in zip(
            starts[0].tolist(), starts[1].tolist(), ends[1].tolist()
        ):
            if column != current:
                runs = []
                result.append((column >> 4, column & 15, runs))
                current = column
            runs.append((y_start + self.min_y, y_end + self.min_y))
        return result

    def visit(self):
        """`(x, z, bottom_y, top_y)` for every contiguous run of set bits."""
        for x, z, runs in self.columns():
            for bottom, top in runs:
                yield x, z, bottom, top

    def __repr__(self) -> str:
        return (
            f"<CarvingMask y={self.min_y}..{self.max_y} "
            f"set={int(np.unpackbits(self.words.view(np.uint8)).sum())}>"
        )


# ==========================================================================
# Ellipsoids
#
# The one genuinely hot loop: every tunnel step carves an ellipsoid, and most
# of the 289 carver chunks' ellipsoids are rejected by the two-line bounding
# test below. `_ellipsoid_kernel` is the same body compiled; the interpreted
# version is the reference, and `tools/check_carvers.py` compares the masks
# they produce bit for bit.
# ==========================================================================

#: Set to False to force the interpreted path; the tool does exactly that.
_USE_NUMBA = _HAS_NUMBA

SKIP_CAVE = 0
SKIP_CANYON = 1


def _ellipsoid_python(
    mask: CarvingMask,
    chunk_x: int,
    chunk_z: int,
    x: float,
    y: float,
    z: float,
    horizontal_radius: float,
    vertical_radius: float,
    skip_kind: int,
    floor_level: float,
    width_factors: np.ndarray,
    min_gen_y: int,
) -> None:
    chunk_min_x = chunk_x << 4
    chunk_min_z = chunk_z << 4
    center_x = chunk_min_x + 8.0
    center_z = chunk_min_z + 8.0
    max_delta = 16.0 + horizontal_radius * 2.0
    if abs(x - center_x) > max_delta or abs(z - center_z) > max_delta:
        return

    min_x_index = max(int(math.floor(x - horizontal_radius)) - chunk_min_x - 1, 0)
    max_x_index = min(max(int(math.floor(x + horizontal_radius)) - chunk_min_x, 0), 15)
    low_y = max(int(math.floor(y - vertical_radius)) - 1, mask.min_y)
    high_y = min(int(math.floor(y + vertical_radius)) + 1, mask.max_y)
    min_z_index = max(int(math.floor(z - horizontal_radius)) - chunk_min_z - 1, 0)
    max_z_index = min(max(int(math.floor(z + horizontal_radius)) - chunk_min_z, 0), 15)
    if max_x_index < min_x_index or max_z_index < min_z_index:
        return

    height = mask.height
    words = mask.words
    for x_index in range(min_x_index, max_x_index + 1):
        world_x = chunk_min_x + x_index
        xd = (world_x + 0.5 - x) / horizontal_radius
        for z_index in range(min_z_index, max_z_index + 1):
            world_z = chunk_min_z + z_index
            zd = (world_z + 0.5 - z) / horizontal_radius
            if xd * xd + zd * zd >= 1.0:
                continue
            for world_y in range(high_y, low_y, -1):
                yd = (world_y - 0.5 - y) / vertical_radius
                if skip_kind == SKIP_CAVE:
                    if yd <= floor_level or xd * xd + yd * yd + zd * zd >= 1.0:
                        continue
                else:
                    y_index = world_y - min_gen_y
                    if y_index == 0:
                        continue
                    if (xd * xd + zd * zd) * float(width_factors[y_index - 1]) + (
                        yd * yd
                    ) / 6.0 >= 1.0:
                        continue
                index = (world_y - mask.min_y) + ((z_index + (x_index << 4)) * height)
                words[index >> 6] |= np.uint64(1) << np.uint64(index & 63)


@njit(cache=True, nogil=True)
def _ellipsoid_kernel(
    words,
    mask_min_y,
    mask_max_y,
    mask_height,
    chunk_min_x,
    chunk_min_z,
    x,
    y,
    z,
    horizontal_radius,
    vertical_radius,
    skip_kind,
    floor_level,
    width_factors,
    min_gen_y,
):
    center_x = chunk_min_x + 8.0
    center_z = chunk_min_z + 8.0
    max_delta = 16.0 + horizontal_radius * 2.0
    if abs(x - center_x) > max_delta or abs(z - center_z) > max_delta:
        return

    min_x_index = max(int(math.floor(x - horizontal_radius)) - chunk_min_x - 1, 0)
    max_x_index = min(max(int(math.floor(x + horizontal_radius)) - chunk_min_x, 0), 15)
    low_y = max(int(math.floor(y - vertical_radius)) - 1, mask_min_y)
    high_y = min(int(math.floor(y + vertical_radius)) + 1, mask_max_y)
    min_z_index = max(int(math.floor(z - horizontal_radius)) - chunk_min_z - 1, 0)
    max_z_index = min(max(int(math.floor(z + horizontal_radius)) - chunk_min_z, 0), 15)
    if max_x_index < min_x_index or max_z_index < min_z_index:
        return

    for x_index in range(min_x_index, max_x_index + 1):
        world_x = chunk_min_x + x_index
        xd = (world_x + 0.5 - x) / horizontal_radius
        for z_index in range(min_z_index, max_z_index + 1):
            world_z = chunk_min_z + z_index
            zd = (world_z + 0.5 - z) / horizontal_radius
            if xd * xd + zd * zd >= 1.0:
                continue
            for world_y in range(high_y, low_y, -1):
                yd = (world_y - 0.5 - y) / vertical_radius
                if skip_kind == 0:
                    if yd <= floor_level or xd * xd + yd * yd + zd * zd >= 1.0:
                        continue
                else:
                    y_index = world_y - min_gen_y
                    if y_index == 0:
                        continue
                    if (xd * xd + zd * zd) * width_factors[y_index - 1] + (
                        yd * yd
                    ) / 6.0 >= 1.0:
                        continue
                index = (world_y - mask_min_y) + ((z_index + (x_index << 4)) * mask_height)
                words[index >> 6] |= np.uint64(1) << np.uint64(index & 63)


def carve_ellipsoid(
    mask: CarvingMask,
    chunk_x: int,
    chunk_z: int,
    x: float,
    y: float,
    z: float,
    horizontal_radius: float,
    vertical_radius: float,
    skip_kind: int,
    floor_level: float,
    width_factors: Optional[np.ndarray],
    min_gen_y: int,
) -> None:
    """Mark the mask cells inside one ellipsoid.

    Everything here is f64 except the radii, which arrive already rounded to
    f32 by their providers. `xd`/`yd`/`zd` are the normalised offsets the skip
    checkers test; `world_y` is a real world Y, not an array row.
    """
    if mask.height <= 0 or horizontal_radius <= 0.0 or vertical_radius <= 0.0:
        return
    factors = _NO_WIDTH_FACTORS if width_factors is None else width_factors
    if _USE_NUMBA:
        _ellipsoid_kernel(
            mask.words,
            mask.min_y,
            mask.max_y,
            mask.height,
            chunk_x << 4,
            chunk_z << 4,
            x,
            y,
            z,
            horizontal_radius,
            vertical_radius,
            skip_kind,
            floor_level,
            factors,
            min_gen_y,
        )
        return
    _ellipsoid_python(
        mask,
        chunk_x,
        chunk_z,
        x,
        y,
        z,
        horizontal_radius,
        vertical_radius,
        skip_kind,
        floor_level,
        factors,
        min_gen_y,
    )


def can_reach(
    chunk_x: int,
    chunk_z: int,
    x: float,
    z: float,
    step: int,
    total_steps: int,
    thickness: float,
) -> bool:
    """`WorldCarver.canReach`: is this step close enough to matter?

    `thickness + 2 + 16` is an f32 sum (the width is an f32 upstream) widened
    to f64, and `total_steps - step` is the *remaining* budget, not the elapsed
    one.
    """
    x_mid = (chunk_x << 4) + 8.0
    z_mid = (chunk_z << 4) + 8.0
    xd = x - x_mid
    zd = z - z_mid
    remaining = float(total_steps - step)
    reach = float(_fadd(_fadd(thickness, F32(2.0)), F32(16.0)))
    return (xd * xd + zd * zd) - (remaining * remaining) <= reach * reach


# ==========================================================================
# The compiled walk
#
# The tunnel step loop is the hot part of the stage: one chunk runs some 38k
# steps through 400-odd tunnels, each step costing five to six RNG draws and
# ten f32 operations. In interpreted Python that is two seconds per chunk, so
# the loop is compiled -- the same trade `perlin.py` makes for the Perlin
# chain.
#
# The RNG is the interesting part. `world.vanilla.random_source` owns the
# generators, and the compiled code must not become a second, drifting copy of
# them: so Python still *seeds* every generator (and draws the probability roll
# and the tunnel seeds), and the compiled code only continues from a state array
# it is handed. The export and the draws live in `world.vanilla.kernel_rng`,
# shared with the ore kernel in `features.py`; `tools/check_carvers.py` replays
# mixed draw sequences through both sides and asserts every draw is identical,
# for all three generators.
from world.vanilla.kernel_rng import (
    RNG_LEGACY as _RNG_LEGACY,
    RNG_RAW as _RNG_RAW,
    RNG_WORLDGEN as _RNG_WORLDGEN,
    next_bits as _rng_next_bits,
    next_bounded_i32 as _rng_next_bounded_i32,
    next_f32 as _rng_next_f32,
    next_i64 as _rng_next_i64,
    next_raw as _rng_raw,
    rotl64 as _rotl64,
    state_from as _rng_state_from,
)

_PI32 = F32(math.pi)


def _rng_state_for_seed(seed: int, legacy: bool, mode: str) -> np.ndarray:
    return _rng_state_from(new_carver_random(seed, legacy, mode))


@njit(cache=True, inline="always")
def _sin_k(sin_t, value):
    index = int(np.float32(value) * np.float32(10430.378)) & 65535
    return sin_t[index]


@njit(cache=True, inline="always")
def _cos_k(sin_t, value):
    index = int(np.float32(value) * np.float32(10430.378) + np.float32(16384.0)) & 65535
    return sin_t[index]


@njit(cache=True, inline="always")
def _can_reach_k(chunk_min_x, chunk_min_z, x, z, step, total_steps, thickness):
    xd = x - (chunk_min_x + 8.0)
    zd = z - (chunk_min_z + 8.0)
    remaining = np.float64(total_steps - step)
    reach = np.float64(np.float32(np.float32(np.float32(thickness + np.float32(2.0)) + np.float32(16.0))))
    return (xd * xd + zd * zd) - (remaining * remaining) <= reach * reach


@njit(cache=True, nogil=True)
def _walk_tunnel_kernel(
    words,
    mask_min_y,
    mask_max_y,
    mask_height,
    chunk_min_x,
    chunk_min_z,
    sin_t,
    factors,
    state,
    x,
    y,
    z,
    horizontal_multiplier,
    vertical_multiplier,
    thickness,
    yaw,
    pitch,
    step0,
    distance,
    y_scale,
    floor_level,
    min_gen_y,
    can_branch,
    branch_values,
    branch_seeds,
):
    """One tunnel's step loop; returns 1 when it spawned two branches.

    Mirrors `_create_tunnel_python` line for line, including the draw order.
    The branch seeds and thicknesses come out through `branch_values` and
    `branch_seeds` rather than being walked here, so the tunnel structure stays
    readable in Python and this stays a straight loop. `branch_values` is
    `[x, y, z, thickness1, yaw1, pitch1, thickness2, yaw2, pitch2, step]`.
    """
    split_point = _rng_next_bounded_i32(state, distance // 2) + distance // 4
    is_steep = _rng_next_bounded_i32(state, 6) == 0
    y_rota = np.float32(0.0)
    x_rota = np.float32(0.0)

    for current_step in range(step0, distance):
        progress = np.float32(
            np.float32(np.float32(_PI32) * np.float32(current_step)) / np.float32(distance)
        )
        horizontal_radius = 1.5 + np.float64(_sin_k(sin_t, progress) * thickness)
        vertical_radius = horizontal_radius * y_scale
        cos_x = _cos_k(sin_t, pitch)
        x += np.float64(np.float32(_cos_k(sin_t, yaw) * cos_x))
        y += np.float64(_sin_k(sin_t, pitch))
        z += np.float64(np.float32(_sin_k(sin_t, yaw) * cos_x))

        if is_steep:
            decay = np.float32(0.92)
        else:
            decay = np.float32(0.7)
        pitch = np.float32(pitch * decay + x_rota * np.float32(0.1))
        yaw = np.float32(yaw + y_rota * np.float32(0.1))
        first = np.float32(_rng_next_f32(state))
        second = np.float32(_rng_next_f32(state))
        third = np.float32(_rng_next_f32(state))
        x_rota = np.float32(x_rota * np.float32(0.9) + np.float32(np.float32(first - second) * third * np.float32(2.0)))
        fourth = np.float32(_rng_next_f32(state))
        fifth = np.float32(_rng_next_f32(state))
        sixth = np.float32(_rng_next_f32(state))
        y_rota = np.float32(y_rota * np.float32(0.75) + np.float32(np.float32(fourth - fifth) * sixth * np.float32(4.0)))

        if can_branch == 1 and current_step == split_point and np.float64(thickness) > 1.0:
            seed_one = _rng_next_i64(state)
            thickness_one = np.float32(
                np.float32(_rng_next_f32(state)) * np.float32(0.5) + np.float32(0.5)
            )
            seed_two = _rng_next_i64(state)
            thickness_two = np.float32(
                np.float32(_rng_next_f32(state)) * np.float32(0.5) + np.float32(0.5)
            )
            branch_vertical = np.float32(pitch / np.float32(3.0))
            branch_values[0] = x
            branch_values[1] = y
            branch_values[2] = z
            branch_values[3] = np.float64(thickness_one)
            branch_values[4] = np.float64(np.float32(yaw - np.float32(1.5707964)))
            branch_values[5] = np.float64(branch_vertical)
            branch_values[6] = np.float64(thickness_two)
            branch_values[7] = np.float64(np.float32(yaw + np.float32(1.5707964)))
            branch_values[8] = np.float64(branch_vertical)
            branch_values[9] = np.float64(current_step)
            branch_seeds[0] = seed_one
            branch_seeds[1] = seed_two
            return 1

        if _rng_next_bounded_i32(state, 4) != 0:
            if _can_reach_k(chunk_min_x, chunk_min_z, x, z, current_step, distance, thickness) == 0:
                return 0
            _ellipsoid_kernel(
                words, mask_min_y, mask_max_y, mask_height, chunk_min_x, chunk_min_z,
                x, y, z,
                horizontal_radius * horizontal_multiplier,
                vertical_radius * vertical_multiplier,
                0, floor_level, factors, min_gen_y,
            )
    return 0


@njit(cache=True, nogil=True)
def _walk_canyon_kernel(
    words,
    mask_min_y,
    mask_max_y,
    mask_height,
    chunk_min_x,
    chunk_min_z,
    sin_t,
    state,
    x,
    y,
    z,
    thickness,
    yaw,
    pitch,
    distance,
    y_scale,
    width_smoothness,
    radius_kind,
    radius_lo,
    radius_hi,
    vertical_default,
    vertical_center,
    min_gen_y,
    depth,
):
    """`_do_carve_python`'s loop, including `initWidthFactors` before it."""
    factors = np.zeros(depth, dtype=np.float32)
    width_factor = np.float32(1.0)
    for y_index in range(depth):
        if y_index == 0 or _rng_next_bounded_i32(state, width_smoothness) == 0:
            first = np.float32(_rng_next_f32(state))
            second = np.float32(_rng_next_f32(state))
            width_factor = np.float32(np.float32(1.0) + np.float32(first * second))
        factors[y_index] = np.float32(width_factor * width_factor)

    y_rota = np.float32(0.0)
    x_rota = np.float32(0.0)

    for current_step in range(0, distance):
        progress = np.float32(
            np.float32(np.float32(_PI32) * np.float32(current_step)) / np.float32(distance)
        )
        horizontal_radius = 1.5 + np.float64(_sin_k(sin_t, progress) * thickness)
        vertical_radius = horizontal_radius * y_scale
        if radius_kind == 1:
            span = np.float32(np.float32(radius_hi) - np.float32(radius_lo))
            drawn = np.float32(np.float32(radius_lo) + np.float32(_rng_next_f32(state)) * span)
            horizontal_radius *= np.float64(drawn)
        else:
            horizontal_radius *= np.float64(radius_lo)

        multiplier = np.float32(
            np.float32(1.0)
            - np.float32(
                abs(np.float32(np.float32(0.5) - np.float32(np.float32(current_step) / np.float32(distance))))
                * np.float32(2.0)
            )
        )
        factor = np.float32(
            np.float32(vertical_default) + np.float32(np.float32(vertical_center) * multiplier)
        )
        jitter = np.float32(np.float32(_rng_next_f32(state)) * np.float32(0.25) + np.float32(0.75))
        vertical_radius = np.float64(factor) * vertical_radius * np.float64(jitter)

        cos_x = _cos_k(sin_t, pitch)
        x += np.float64(np.float32(_cos_k(sin_t, yaw) * cos_x))
        y += np.float64(_sin_k(sin_t, pitch))
        z += np.float64(np.float32(_sin_k(sin_t, yaw) * cos_x))

        pitch = np.float32(pitch * np.float32(0.7) + x_rota * np.float32(0.05))
        yaw = np.float32(yaw + y_rota * np.float32(0.05))
        first = np.float32(_rng_next_f32(state))
        second = np.float32(_rng_next_f32(state))
        third = np.float32(_rng_next_f32(state))
        x_rota = np.float32(x_rota * np.float32(0.8) + np.float32(np.float32(first - second) * third * np.float32(2.0)))
        fourth = np.float32(_rng_next_f32(state))
        fifth = np.float32(_rng_next_f32(state))
        sixth = np.float32(_rng_next_f32(state))
        y_rota = np.float32(y_rota * np.float32(0.5) + np.float32(np.float32(fourth - fifth) * sixth * np.float32(4.0)))

        if _rng_next_bounded_i32(state, 4) != 0:
            if _can_reach_k(chunk_min_x, chunk_min_z, x, z, current_step, distance, thickness) == 0:
                return
            _ellipsoid_kernel(
                words, mask_min_y, mask_max_y, mask_height, chunk_min_x, chunk_min_z,
                x, y, z, horizontal_radius, vertical_radius,
                1, 0.0, factors, min_gen_y,
            )


# ==========================================================================
# The cave carver
# ==========================================================================

def _create_room(
    mask: CarvingMask,
    chunk_x: int,
    chunk_z: int,
    x: float,
    y: float,
    z: float,
    thickness,
    y_scale: float,
    floor_level: float,
    min_gen_y: int,
) -> None:
    """`CaveWorldCarver.createRoom`. Note `sin(FRAC_PI_2)` is the *table's*."""
    horizontal_radius = 1.5 + float(_fmul(_sin(F32(math.pi / 2.0)), thickness))
    vertical_radius = horizontal_radius * y_scale
    carve_ellipsoid(
        mask, chunk_x, chunk_z, x + 1.0, y, z, horizontal_radius, vertical_radius,
        SKIP_CAVE, floor_level, None, min_gen_y,
    )


def _provider_step(provider, what: str) -> Tuple[int, float, float]:
    """`(kind, a, b)` for the providers the compiled walk samples itself."""
    if isinstance(provider, _ConstantFloat):
        return 0, float(provider.value), 0.0
    if isinstance(provider, _UniformFloat):
        return 1, float(provider.min_inclusive), float(provider.max_exclusive)
    if isinstance(provider, _TrapezoidFloat):
        return 2, float(provider.min), float(provider.max)
    raise ValueError(f"{what}: unsupported provider {provider!r}")


def _create_tunnel_python(
    mask: CarvingMask,
    chunk_x: int,
    chunk_z: int,
    tunnel_seed: int,
    x: float,
    y: float,
    z: float,
    horizontal_radius_multiplier: float,
    vertical_radius_multiplier: float,
    thickness,
    horizontal_rotation,
    vertical_rotation,
    step: int,
    distance: int,
    y_scale: float,
    floor_level: float,
    legacy: bool,
    mode: str,
    min_gen_y: int,
    depth: int,
    can_branch: bool = True,
) -> None:
    """One tunnel, branching once about halfway along if it is thick enough.

    Every angle is f32, and every step's position update is an f32 delta added
    to an f64 position. The *order* of the draws below is the algorithm: the
    branch seeds and thicknesses are drawn in argument order in the reference,
    so seed, thickness, seed, thickness.
    """
    random = new_carver_random(tunnel_seed, legacy, mode)
    split_point = random.next_bounded_i32(distance // 2) + distance // 4
    is_steep = random.next_bounded_i32(6) == 0
    y_rota = F32(0.0)
    x_rota = F32(0.0)

    for current_step in range(step, distance):
        progress = _fdiv(_fmul(F32(math.pi), F32(current_step)), F32(distance))
        horizontal_radius = 1.5 + float(_fmul(_sin(progress), thickness))
        vertical_radius = horizontal_radius * y_scale
        cos_x = _cos(vertical_rotation)
        x += float(_fmul(_cos(horizontal_rotation), cos_x))
        y += float(_sin(vertical_rotation))
        z += float(_fmul(_sin(horizontal_rotation), cos_x))

        decay = F32(0.92) if is_steep else F32(0.7)
        vertical_rotation = _fadd(_fmul(vertical_rotation, decay), _fmul(x_rota, F32(0.1)))
        horizontal_rotation = _fadd(horizontal_rotation, _fmul(y_rota, F32(0.1)))
        x_rota = _fadd(
            _fmul(x_rota, F32(0.9)),
            _fmul(
                _fmul(_fsub(F32(random.next_f32()), F32(random.next_f32())), F32(random.next_f32())),
                F32(2.0),
            ),
        )
        y_rota = _fadd(
            _fmul(y_rota, F32(0.75)),
            _fmul(
                _fmul(_fsub(F32(random.next_f32()), F32(random.next_f32())), F32(random.next_f32())),
                F32(4.0),
            ),
        )

        if can_branch and current_step == split_point and float(thickness) > 1.0:
            first_seed = random.next_i64()
            first_thickness = _fadd(_fmul(F32(random.next_f32()), F32(0.5)), F32(0.5))
            second_seed = random.next_i64()
            second_thickness = _fadd(_fmul(F32(random.next_f32()), F32(0.5)), F32(0.5))
            branch_vertical = _fdiv(vertical_rotation, F32(3.0))
            _create_tunnel_python(
                mask, chunk_x, chunk_z, first_seed, x, y, z,
                horizontal_radius_multiplier, vertical_radius_multiplier, first_thickness,
                _fsub(horizontal_rotation, F32(math.pi / 2.0)), branch_vertical,
                current_step, distance, 1.0, floor_level, legacy, mode, min_gen_y, depth, False,
            )
            _create_tunnel_python(
                mask, chunk_x, chunk_z, second_seed, x, y, z,
                horizontal_radius_multiplier, vertical_radius_multiplier, second_thickness,
                _fadd(horizontal_rotation, F32(math.pi / 2.0)), branch_vertical,
                current_step, distance, 1.0, floor_level, legacy, mode, min_gen_y, depth, False,
            )
            return

        if random.next_bounded_i32(4) != 0:
            if not can_reach(chunk_x, chunk_z, x, z, current_step, distance, float(thickness)):
                return
            carve_ellipsoid(
                mask, chunk_x, chunk_z, x, y, z,
                horizontal_radius * horizontal_radius_multiplier,
                vertical_radius * vertical_radius_multiplier,
                SKIP_CAVE, floor_level, None, min_gen_y,
            )


def _create_tunnel_compiled(
    mask: CarvingMask,
    chunk_x: int,
    chunk_z: int,
    tunnel_seed: int,
    x: float,
    y: float,
    z: float,
    horizontal_radius_multiplier: float,
    vertical_radius_multiplier: float,
    thickness,
    horizontal_rotation,
    vertical_rotation,
    step: int,
    distance: int,
    y_scale: float,
    floor_level: float,
    legacy: bool,
    mode: str,
    min_gen_y: int,
    depth: int,
    can_branch: bool = True,
) -> None:
    """The compiled twin of `_create_tunnel_python`.

    The kernel walks one tunnel and, if it branched, hands the two branches'
    parameters back; the branch walks are driven from here so a branch cannot
    branch again (its thickness is at most 1.0, which is what vanilla relies
    on too).
    """
    state = _rng_state_for_seed(tunnel_seed, legacy, mode)
    values = np.zeros(10, dtype=np.float64)
    seeds = np.zeros(2, dtype=np.uint64)
    branched = _walk_tunnel_kernel(
        mask.words, mask.min_y, mask.max_y, mask.height, chunk_x << 4, chunk_z << 4,
        sin_table(), _NO_WIDTH_FACTORS, state,
        float(x), float(y), float(z), float(horizontal_radius_multiplier),
        float(vertical_radius_multiplier), F32(thickness), F32(horizontal_rotation),
        F32(vertical_rotation), int(step), int(distance), float(y_scale), float(floor_level),
        int(min_gen_y), 1 if can_branch else 0, values, seeds,
    )
    if not branched:
        return
    for branch in range(2):
        # `values` is [x, y, z, (thickness, yaw, pitch) * 2, step].
        base = 3 + 3 * branch
        _create_tunnel_compiled(
            mask, chunk_x, chunk_z, int(seeds[branch]),
            values[0], values[1], values[2],
            horizontal_radius_multiplier, vertical_radius_multiplier,
            F32(values[base]), F32(values[base + 1]), F32(values[base + 2]),
            int(values[9]), distance, 1.0, floor_level, legacy, mode, min_gen_y, depth, False,
        )


def _create_tunnel(
    mask: CarvingMask,
    chunk_x: int,
    chunk_z: int,
    tunnel_seed: int,
    x: float,
    y: float,
    z: float,
    horizontal_radius_multiplier: float,
    vertical_radius_multiplier: float,
    thickness,
    horizontal_rotation,
    vertical_rotation,
    step: int,
    distance: int,
    y_scale: float,
    floor_level: float,
    legacy: bool,
    mode: str,
    min_gen_y: int,
    depth: int,
    can_branch: bool = True,
) -> None:
    """Walk one tunnel: compiled when numba is installed, interpreted otherwise."""
    walk = _create_tunnel_compiled if _USE_NUMBA else _create_tunnel_python
    walk(
        mask, chunk_x, chunk_z, tunnel_seed, x, y, z,
        horizontal_radius_multiplier, vertical_radius_multiplier, thickness,
        horizontal_rotation, vertical_rotation, step, distance, y_scale, floor_level,
        legacy, mode, min_gen_y, depth, can_branch,
    )


def carve_cave(
    config: CarverConfig,
    mask: CarvingMask,
    random,
    chunk_x: int,
    chunk_z: int,
    carver_chunk_x: int,
    carver_chunk_z: int,
    min_gen_y: int,
    depth: int,
    legacy: bool,
    mode: str,
) -> None:
    """`CaveWorldCarver.carve`, which `nether_cave` shares."""
    cave_count = config.count.get(random)
    for _ in range(cave_count):
        x = (carver_chunk_x << 4) + random.next_bounded_i32(16)
        y = float(config.y.get(random, min_gen_y, depth))
        z = (carver_chunk_z << 4) + random.next_bounded_i32(16)

        horizontal_multiplier = float(config.horizontal_radius_multiplier.get(random))
        vertical_multiplier = float(config.vertical_radius_multiplier.get(random))
        start_vertical_multiplier = float(config.start_vertical_radius_multiplier.get(random))
        floor_level = float(config.floor_level.get(random))

        if random.next_bounded_i32(4) == 0:
            room_multiplier = float(config.room_vertical_radius_multiplier.get(random))
            thickness = _fadd(F32(1.0), _fmul(F32(random.next_f32()), F32(6.0)))
            _create_room(
                mask, chunk_x, chunk_z, float(x), y, float(z), thickness,
                room_multiplier, floor_level, min_gen_y,
            )
            tunnels = 1 + random.next_bounded_i32(4)
        else:
            tunnels = 1

        for _ in range(tunnels):
            horizontal_rotation = _fmul(_fmul(F32(random.next_f32()), F32(math.pi)), F32(2.0))
            vertical_rotation = _fdiv(_fsub(F32(random.next_f32()), F32(0.5)), F32(4.0))
            thickness = config.thickness.get(random)
            if config.weird_thickness_bias and random.next_bounded_i32(10) == 0:
                thickness = _fmul(
                    thickness,
                    _fadd(
                        _fmul(_fmul(F32(random.next_f32()), F32(random.next_f32())), F32(3.0)),
                        F32(1.0),
                    ),
                )
            distance = MAX_DISTANCE - random.next_bounded_i32(MAX_DISTANCE // 4)
            tunnel_seed = random.next_i64()
            _create_tunnel(
                mask, chunk_x, chunk_z, tunnel_seed, float(x), y, float(z),
                horizontal_multiplier, vertical_multiplier, thickness,
                horizontal_rotation, vertical_rotation, 0, distance,
                start_vertical_multiplier, floor_level, legacy, mode, min_gen_y, depth,
            )


# ==========================================================================
# The canyon carver
# ==========================================================================

def _canyon_width_factors(depth: int, config: CarverConfig, random) -> np.ndarray:
    """`CanyonWorldCarver.initWidthFactors`: one squared width per Y layer.

    The first layer is forced, then a `width_smoothness`-sided die redraws the
    factor. Squaring is what makes the walls concave.
    """
    factors = np.zeros(depth, dtype=np.float32)
    width_factor = F32(1.0)
    for y_index in range(depth):
        if y_index == 0 or random.next_bounded_i32(config.width_smoothness) == 0:
            width_factor = _fadd(F32(1.0), _fmul(F32(random.next_f32()), F32(random.next_f32())))
        factors[y_index] = _fmul(width_factor, width_factor)
    return factors


def _canyon_vertical_radius(
    config: CarverConfig, random, vertical_radius: float, distance: float, current_step: float
) -> float:
    """`updateVerticalRadius`: fattest at the centre of the branch."""
    vertical_multiplier = _fsub(
        F32(1.0),
        _fmul(_fabs(_fsub(F32(0.5), _fdiv(F32(current_step), F32(distance)))), F32(2.0)),
    )
    factor = _fadd(
        F32(config.vertical_radius_default_factor),
        _fmul(F32(config.vertical_radius_center_factor), vertical_multiplier),
    )
    jitter = _fadd(_fmul(F32(random.next_f32()), F32(0.25)), F32(0.75))
    return float(factor) * vertical_radius * float(jitter)


def _do_carve_python(
    config: CarverConfig,
    mask: CarvingMask,
    chunk_x: int,
    chunk_z: int,
    tunnel_seed: int,
    x: float,
    y: float,
    z: float,
    thickness,
    horizontal_rotation,
    vertical_rotation,
    step: int,
    distance: int,
    y_scale: float,
    legacy: bool,
    mode: str,
    min_gen_y: int,
    depth: int,
) -> None:
    """`CanyonWorldCarver.doCarve`: one tunnel that never branches."""
    random = new_carver_random(tunnel_seed, legacy, mode)
    width_factors = _canyon_width_factors(depth, config, random)
    y_rota = F32(0.0)
    x_rota = F32(0.0)

    for current_step in range(step, distance):
        progress = _fdiv(_fmul(F32(current_step), F32(math.pi)), F32(distance))
        horizontal_radius = 1.5 + float(_fmul(_sin(progress), thickness))
        vertical_radius = horizontal_radius * y_scale
        horizontal_radius *= float(config.horizontal_radius_factor.get(random))
        vertical_radius = _canyon_vertical_radius(
            config, random, vertical_radius, float(distance), float(current_step)
        )

        cos_x = _cos(vertical_rotation)
        x += float(_fmul(_cos(horizontal_rotation), cos_x))
        y += float(_sin(vertical_rotation))
        z += float(_fmul(_sin(horizontal_rotation), cos_x))

        vertical_rotation = _fadd(_fmul(vertical_rotation, F32(0.7)), _fmul(x_rota, F32(0.05)))
        horizontal_rotation = _fadd(horizontal_rotation, _fmul(y_rota, F32(0.05)))
        x_rota = _fadd(
            _fmul(x_rota, F32(0.8)),
            _fmul(
                _fmul(_fsub(F32(random.next_f32()), F32(random.next_f32())), F32(random.next_f32())),
                F32(2.0),
            ),
        )
        y_rota = _fadd(
            _fmul(y_rota, F32(0.5)),
            _fmul(
                _fmul(_fsub(F32(random.next_f32()), F32(random.next_f32())), F32(random.next_f32())),
                F32(4.0),
            ),
        )

        if random.next_bounded_i32(4) != 0:
            if not can_reach(chunk_x, chunk_z, x, z, current_step, distance, float(thickness)):
                return
            carve_ellipsoid(
                mask, chunk_x, chunk_z, x, y, z, horizontal_radius, vertical_radius,
                SKIP_CANYON, 0.0, width_factors, min_gen_y,
            )


def _do_carve_compiled(
    config: CarverConfig,
    mask: CarvingMask,
    chunk_x: int,
    chunk_z: int,
    tunnel_seed: int,
    x: float,
    y: float,
    z: float,
    thickness,
    horizontal_rotation,
    vertical_rotation,
    step: int,
    distance: int,
    y_scale: float,
    legacy: bool,
    mode: str,
    min_gen_y: int,
    depth: int,
) -> None:
    """The compiled twin of `_do_carve_python`, width factors included."""
    radius_kind, radius_lo, radius_hi = _provider_step(
        config.horizontal_radius_factor, "shape.horizontal_radius_factor"
    )
    state = _rng_state_for_seed(tunnel_seed, legacy, mode)
    _walk_canyon_kernel(
        mask.words, mask.min_y, mask.max_y, mask.height, chunk_x << 4, chunk_z << 4,
        sin_table(), state,
        float(x), float(y), float(z), F32(thickness), F32(horizontal_rotation),
        F32(vertical_rotation), int(distance), float(y_scale), int(config.width_smoothness),
        radius_kind, radius_lo, radius_hi,
        float(config.vertical_radius_default_factor),
        float(config.vertical_radius_center_factor),
        int(min_gen_y), int(depth),
    )


def _do_carve(
    config: CarverConfig,
    mask: CarvingMask,
    chunk_x: int,
    chunk_z: int,
    tunnel_seed: int,
    x: float,
    y: float,
    z: float,
    thickness,
    horizontal_rotation,
    vertical_rotation,
    step: int,
    distance: int,
    y_scale: float,
    legacy: bool,
    mode: str,
    min_gen_y: int,
    depth: int,
) -> None:
    """Walk one canyon: compiled when numba is installed, interpreted otherwise."""
    walk = _do_carve_compiled if _USE_NUMBA else _do_carve_python
    walk(
        config, mask, chunk_x, chunk_z, tunnel_seed, x, y, z, thickness,
        horizontal_rotation, vertical_rotation, step, distance, y_scale,
        legacy, mode, min_gen_y, depth,
    )


def carve_canyon(
    config: CarverConfig,
    mask: CarvingMask,
    random,
    chunk_x: int,
    chunk_z: int,
    carver_chunk_x: int,
    carver_chunk_z: int,
    min_gen_y: int,
    depth: int,
    legacy: bool,
    mode: str,
) -> None:
    """`CanyonWorldCarver.carve`: exactly one canyon per start chunk."""
    x = (carver_chunk_x << 4) + random.next_bounded_i32(16)
    y = config.y.get(random, min_gen_y, depth)
    z = (carver_chunk_z << 4) + random.next_bounded_i32(16)

    horizontal_rotation = _fmul(_fmul(F32(random.next_f32()), F32(math.pi)), F32(2.0))
    vertical_rotation = config.vertical_rotation.get(random)
    y_scale = float(config.y_scale.get(random))
    thickness = config.shape_thickness.get(random)
    distance = int(F32(MAX_DISTANCE) * F32(config.distance_factor.get(random)))
    tunnel_seed = random.next_i64()

    _do_carve(
        config, mask, chunk_x, chunk_z, tunnel_seed, float(x), float(y), float(z),
        thickness, horizontal_rotation, vertical_rotation, 0, distance, y_scale,
        legacy, mode, min_gen_y, depth,
    )


# ==========================================================================
# What a carved cell becomes
# ==========================================================================

class CarveStatePolicy:
    """Vanilla `WorldCarver.getCarveState`, with the aquifer factored out.

    The order is load-bearing: lava level first, then the aquifer, then the
    no-aquifer fallback. `state()` returning `None` means "do not carve this
    cell" -- that is what an aquifer's barrier answer means in vanilla, and it
    is why this returns an Optional rather than an id.
    """

    __slots__ = (
        "lava_level",
        "aquifer",
        "aquifers_enabled",
        "sea_level",
        "default_fluid",
        "lava",
    )

    def __init__(
        self,
        lava_level: int,
        sea_level: int,
        aquifers_enabled: bool,
        default_fluid: int,
        aquifer: Optional[Callable[[int, int, int], Optional[int]]] = None,
    ) -> None:
        self.lava_level = int(lava_level)
        self.sea_level = int(sea_level)
        self.aquifers_enabled = bool(aquifers_enabled)
        self.default_fluid = int(default_fluid)
        self.aquifer = aquifer
        self.lava = _block_id("minecraft:lava")

    def state(self, x: int, y: int, z: int) -> Optional[int]:
        if y <= self.lava_level:
            return self.lava
        if self.aquifer is not None:
            return self.aquifer(x, y, z)
        if y < self.sea_level and not self.aquifers_enabled:
            return self.default_fluid
        return AIR


def _apply_carving_mask(
    mask: CarvingMask,
    blocks: np.ndarray,
    chunk_x: int,
    chunk_z: int,
    min_y: int,
    policy: CarveStatePolicy,
    replaceable: Set[int],
    top_material: Optional[Callable[[int, int, int, bool, bool], Optional[int]]],
) -> None:
    """Write the mask into the block array.

    One column run at a time, top down. The direction is what lets a cell know
    that the tunnel has already passed grass or mycelium above it, which is
    when vanilla restores the surface material underneath.

    The `replaceable` test is vanilla's `canReplaceBlock`. It is what keeps a
    tunnel from eating obsidian, and it is what keeps carved-out air above the
    surface from being refilled: air is not in the tag.
    """
    bedrock = _block_id("minecraft:bedrock")
    grass = _block_id("minecraft:grass_block")
    mycelium = _block_id("minecraft:mycelium")
    dirt = _block_id("minecraft:dirt")
    liquid_states = (_block_id("minecraft:water"), _block_id("minecraft:lava"))
    restore = top_material is not None and policy.aquifers_enabled
    rows = blocks.shape[0]

    for x, z, runs in mask.columns():
        world_x = (chunk_x << 4) + x
        world_z = (chunk_z << 4) + z
        for bottom_y, top_y in runs:
            has_grass = False
            for world_y in range(top_y, bottom_y - 1, -1):
                row = world_y - min_y
                current = int(blocks[row, z, x])
                if current == bedrock:
                    continue
                if current == grass or current == mycelium:
                    has_grass = True
                if current not in replaceable:
                    continue
                state = policy.state(world_x, world_y, world_z)
                if state is None:
                    continue
                blocks[row, z, x] = state
                if restore and has_grass:
                    below = world_y - 1
                    below_row = below - min_y
                    if 0 <= below_row < rows and int(blocks[below_row, z, x]) == dirt:
                        material = top_material(
                            world_x, below, world_z, state in liquid_states, False
                        )
                        if material is not None:
                            blocks[below_row, z, x] = material


# ==========================================================================
# The stage
# ==========================================================================

def _biome_provider(chunk_x: int, chunk_z: int, biomes: np.ndarray, biome_at):
    """Biome id at a *quart* position, for any carver chunk in the 17x17 area.

    The fallback clamps the query into this chunk's own array, which is exact
    for the centre carver chunk (`quart_x = chunk_x * 4`) and the nearest
    available answer outside it. With the shipped biome data every overworld
    biome lists the same carver set, so the answer is the same either way; see
    the module docstring.
    """
    if biome_at is not None:
        return lambda qx, qy, qz: int(biome_at(qx, qy, qz))
    rows = biomes.shape[0]
    base_x = chunk_x << 2
    base_z = chunk_z << 2

    def clamped(qx: int, qy: int, qz: int) -> int:
        x = min(max(qx - base_x, 0), 3)
        z = min(max(qz - base_z, 0), 3)
        y = min(max(qy, 0), rows - 1)
        return int(biomes[y, z, x])

    return clamped


def carve(
    chunk_x: int,
    chunk_z: int,
    blocks: np.ndarray,
    biomes: np.ndarray,
    *,
    preset: str,
    seed: int,
    min_y: int,
    height: int,
    biome_carvers: Mapping[int, Sequence[str]],
    carver_data: Mapping[str, object],
    tags: object,
    neighbour_blocks: object = None,
    biome_at: Optional[Callable[[int, int, int], int]] = None,
    sea_level: Optional[int] = None,
    aquifer: Optional[Callable[[int, int, int], Optional[int]]] = None,
    top_material: Optional[Callable[[int, int, int, bool, bool], Optional[int]]] = None,
    random_source: str = "worldgen",
) -> None:
    """Run the carver stage over one chunk, in place.

    `blocks` is `(height, 16, 16)` uint16 `[y, z, x]`, `biomes` is
    `(height // 4, 4, 4)` int32 quart cells, and both are the layouts the
    terrain and biome stages already use. `neighbour_blocks` is accepted for
    the caller's convenience and never read -- see the module docstring.
    """
    if random_source not in ("worldgen", "raw"):
        raise ValueError(f"unknown random_source {random_source!r}")
    settings = preset_settings(preset)
    legacy = bool(settings["legacy_random_source"])
    if sea_level is None:
        sea_level = int(settings["sea_level"])

    mask = CarvingMask(min_y + 1, min_y + height - 1 - PROTECTED_BLOCKS_ON_TOP)
    provider = _biome_provider(chunk_x, chunk_z, biomes, biome_at)
    list_cache: Dict[int, Tuple[str, ...]] = {}
    configs: Dict[str, CarverConfig] = {}
    live: List[str] = []

    if mask.height > 0:
        for dx in range(-RANGE, RANGE + 1):
            for dz in range(-RANGE, RANGE + 1):
                carver_chunk_x = chunk_x + dx
                carver_chunk_z = chunk_z + dz
                biome_id = provider(carver_chunk_x << 2, 0, carver_chunk_z << 2)
                names = list_cache.get(biome_id)
                if names is None:
                    names = _normalize_carver_names(biome_carvers.get(biome_id, ()))
                    list_cache[biome_id] = names
                for index, name in enumerate(names):
                    config = configs.get(name)
                    if config is None:
                        try:
                            raw_config = _lookup_config(carver_data, name)
                        except KeyError as error:
                            raise KeyError(f"biome {biome_id}: {error}") from None
                        config = _as_config(name, raw_config)
                        configs[name] = config
                    # The seed salt is the index in the *biome's* list, so a
                    # different biome with a different list shifts every carver
                    # after it. This is why the list must be exactly right.
                    carver_seed = get_large_feature_seed(
                        (seed + index) & _MASK64, carver_chunk_x, carver_chunk_z
                    )
                    random = new_carver_random(carver_seed, legacy, random_source)
                    if random.next_f32() > config.probability:
                        continue
                    if name not in live:
                        live.append(name)
                    if config.type == "minecraft:canyon":
                        carve_canyon(
                            config, mask, random, chunk_x, chunk_z, carver_chunk_x,
                            carver_chunk_z, min_y, height, legacy, random_source,
                        )
                    else:
                        carve_cave(
                            config, mask, random, chunk_x, chunk_z, carver_chunk_x,
                            carver_chunk_z, min_y, height, legacy, random_source,
                        )

    if mask.is_empty():
        return

    # One mask for the whole chunk means one lava level and one replaceable set
    # for the cells in it. The shipped data has a single value per dimension
    # (all three overworld carvers share the tag and `above_bottom: 8`; the
    # nether has one carver), so the union is exact in practice; a datapack
    # that mixed them could not be represented by a shared mask, and the
    # maximum lava level is the conservative reading.
    lava_level = max(
        (configs[name].resolved_lava_level(min_y, height) for name in live), default=min_y
    )
    policy = CarveStatePolicy(
        lava_level=lava_level,
        sea_level=sea_level,
        aquifers_enabled=bool(settings["aquifers_enabled"]),
        default_fluid=_block_id_from_payload(settings["default_fluid"]),
        aquifer=aquifer,
    )
    replaceable = resolve_replaceable(tags, [configs[name].replaceable for name in live])
    _apply_carving_mask(mask, blocks, chunk_x, chunk_z, min_y, policy, replaceable, top_material)


__all__ = [
    "AIR",
    "CARVER_RANGE",
    "CarveStatePolicy",
    "CarverConfig",
    "CarvingMask",
    "MAX_DISTANCE",
    "PROTECTED_BLOCKS_ON_TOP",
    "RANGE",
    "SUPPORTED_TYPES",
    "UnsupportedCarverError",
    "can_reach",
    "carve",
    "carve_canyon",
    "carve_cave",
    "carve_ellipsoid",
    "carvers_for_biome",
    "get_large_feature_seed",
    "load_carvers",
    "new_carver_random",
    "preset_settings",
    "resolve_replaceable",
    "resolve_y",
    "sin_table",
]
