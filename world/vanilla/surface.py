"""
Surface rules: which block the top of the terrain is made of.

The noise stage decides *where* rock is; this decides that the rock is grass,
dirt, sand, gravel, terracotta, snow or bedrock. It is also the last worldgen
stage that can be checked against a vanilla chunk dump exactly, which is why it
is the one worth getting fanatical about.

    from world.vanilla.surface import SurfaceSystem

    system = SurfaceSystem.from_assets("overworld", seed=0, biome_table=table)
    system.build_surface(blocks, biomes, chunk_x=0, chunk_z=0)

`blocks` is `(height, 16, 16)` `uint16` indexed `[y, z, x]` with row 0 at
`min_y`; `biomes` is `(height // 4, 4, 4)` `int32` in the same order. Both are
mutated in place. The pass only ever *replaces* the preset's default block, so
the solid/air silhouette is identical before and after - which is exactly what
the dumps show: `noise_no_blend_no_beard_0_0` and `no_blend_no_beard_surface_0_0`
differ in 18652 cells and every one of them is stone becoming something else.

    python tools/check_surface_rules.py

The rules run in two forms
--------------------------
`build_surface` does not walk the parsed rule tree at runtime: a tree walk is a
dozen Python calls per stone cell and a chunk has tens of thousands of them.
`compile_rule` flattens the same tree into one generated straight-line function
at startup, with conditions inlined, and the hot loop calls that. The tree stays
as the readable form and as the oracle the check tool compares the compiled
matcher against on swept states; both are emitted from the same parse-time
condition objects, so they cannot drift apart silently.

The five things that are easy to get wrong
------------------------------------------

**The pass runs top-down, per column, and the depth counters are incremental.**
`stone_depth_above` counts solid blocks upward from here, `stone_depth_below`
counts down to the bottom of the current solid run, and `fluid_height` is where
the *first* liquid from the top begins. Air resets all three; a liquid resets
`fluid_height` only on first sight and is otherwise skipped, and it does **not**
reset `stone_depth_above`. Running the loop bottom-up, or recomputing the depths
from scratch, changes which cells the rules see and therefore moves the surface.

**`sequence` is first-match-wins, so the order in the file is load-bearing.**
Bedrock is tested before the main branch and wins ties by being first; the roof
bedrock is tested last. Reordering that list silently rewrites the world.

**`run_depth` is f32, truncated toward zero, and drawn once per column.**
`surfaceNoise * 2.75 + 3.0 + random * 0.25`. Widening any part of it to double
shifts the whole depth band by a block in places, which no screenshot will show
and a dump will.

**`above_preliminary_surface` is a lower bound, not the surface.** The estimator
is bilinear over the four corners of the chunk footprint - vanilla evaluates
`initial_density_without_jaggedness` at the corners and joins them itself -
sampled at y = 0, floored, then reduced by 8 and raised by `run_depth`. Reading
it as "the height of the terrain" makes the condition reject nearly everything
and the world turns to stone.

**`random_name` is hashed big-endian, like a noise, not little-endian.**
`md5(name)` read as two 64-bit words; the byte order is invisible in `assets/`
(names only) and only shows up as a bedrock layer that is statistically right
and individually wrong. `tools/check_surface_rules.py` reproduces 1280 bedrock
cells exactly with the big-endian reading and misses 310 with the other one, so
the recipe is pinned rather than argued about.

What is checked against what
----------------------------
`tools/check_surface_rules.py` compares this pass cell for cell against vanilla
chunk dumps in `assets/vanilla_chunks/`: a plains/ocean chunk, a badlands chunk
and a frozen ocean chunk, all at seed 0. Those dumps ship with the *pre-surface*
blocks of the same chunk, so a mismatch can be attributed to this pass rather
than to the terrain underneath it.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from world.vanilla.density import Context as DensityContext
from world.vanilla.density import load_parameters as load_noise_parameters
from world.vanilla.density import parse as parse_density
from world.vanilla.perlin import DoublePerlinNoiseSampler
from world.vanilla.random_source import (
    LegacyRandom,
    Xoroshiro,
    XoroshiroSplitter,
    name_seed,
)
from world.vanilla.simplex import OctaveSimplexNoiseSampler

_ABSOLUTE = "absolute"
_ABOVE_BOTTOM = "above_bottom"
_BELOW_TOP = "below_top"

#: Vanilla's `SurfaceSystem.SURFACE_HEIGHT_OFFSET`. The preliminary-surface
#: estimate trails the real surface where the terrain climbs, so vanilla drops
#: it another 8 blocks and treats everything above as "might be the surface".
_SURFACE_HEIGHT_OFFSET = 8

#: Blocks per quart cell; biomes are stored per quart, so a block lookup is `/4`.
_QUART = 4

#: Air. Fixed at 0 by the registry and by the protocol.
AIR = 0

#: Sentinel for "no fluid seen in this column yet" - vanilla's `Integer.MIN_VALUE`.
_NO_FLUID = -(1 << 31)


def _assets_path(name: str) -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, "assets", name)


def name_hash(name: str) -> Tuple[int, int]:
    """The `(lo, hi)` a `random_name` positional factory derives from.

    `md5(name)` read **big-endian** - the same recipe `random_source.name_seed`
    uses for noise parameters, because it is the same recipe. Until the vanilla
    dumps were compared cell for cell this was read little-endian, on the
    strength of Pumpkin's generated surface-rule constants; the dumps show the
    bedrock layer is wrong that way (310 of 1280 cells) and exact this way (0 of
    1280). Read `tools/check_surface_rules.py` before changing it back.
    """
    return name_seed(name)


# ==================== anchors ====================

class Anchor:
    """A rule's y reference: absolute, above the world bottom, or below the top."""

    __slots__ = ("kind", "value")

    def __init__(self, kind: str, value: int) -> None:
        self.kind = kind
        self.value = int(value)

    @classmethod
    def parse(cls, payload: Any) -> "Anchor":
        if isinstance(payload, Mapping):
            if _ABSOLUTE in payload:
                return cls(_ABSOLUTE, payload[_ABSOLUTE])
            if _ABOVE_BOTTOM in payload:
                return cls(_ABOVE_BOTTOM, payload[_ABOVE_BOTTOM])
            if _BELOW_TOP in payload:
                return cls(_BELOW_TOP, payload[_BELOW_TOP])
        raise ValueError(f"unknown surface-rule anchor {payload!r}")

    def get_y(self, min_y: int, height: int) -> int:
        if self.kind == _ABSOLUTE:
            return self.value
        if self.kind == _ABOVE_BOTTOM:
            return min_y + self.value
        # `below_top` counts down from the *last* block, not from one past it.
        return min_y + height - 1 - self.value

    def __repr__(self) -> str:
        return f"Anchor({self.kind}={self.value})"


# ==================== block ids this pass needs ====================

class StateLookup:
    """Fast, registry-backed answers to the three questions a rule asks of a cell.

    A dict lookup per cell would be affordable but is not free at ~30k stone
    cells per chunk, and the questions are all "what kind of block is id N" -
    which is a property of the id, not of the position. So the id space is
    flattened into three lookup tables once.
    """

    __slots__ = ("liquid", "solid_or_liquid", "water", "total",
                 "liquid_bytes", "solid_or_liquid_bytes")

    def __init__(self) -> None:
        from registry import get_game_registry

        blocks = get_game_registry().blocks
        total = int(blocks.total_states)
        self.total = total
        self.liquid = np.zeros(total, dtype=bool)
        self.solid_or_liquid = np.zeros(total, dtype=bool)
        # Walk the *block* table, not the id space: a block owns a contiguous
        # range of state ids (`first_state_id`..`last_state_id`), so this is 1166
        # iterations instead of 29671. Reading a `states` list off the block
        # would be nicer and does not exist - `BlockType` exposes the range.
        for block in (blocks.blocks.values() if hasattr(blocks, "blocks") else ()):
            first = int(getattr(block, "first_state_id", -1))
            last = int(getattr(block, "last_state_id", -1)) + 1
            if first < 0 or last <= first:
                continue
            name = block.name
            if name in ("minecraft:water", "minecraft:lava"):
                self.liquid[first:last] = True
            elif name != "minecraft:air":
                self.solid_or_liquid[first:last] = True
        # `build_surface` indexes these twice per cell, ~64k times a chunk, and
        # a `bytes` index returns a Python int where a numpy one returns
        # `np.bool_` -- same truthiness, a third of the cost. Kept alongside
        # the arrays because `is_liquid`/`is_solid` still want the bounds
        # behaviour of a numpy index.
        self.liquid_bytes = self.liquid.tobytes()
        self.solid_or_liquid_bytes = self.solid_or_liquid.tobytes()
        self.water = self._default_id("minecraft:water")

    @staticmethod
    def _default_id(name: str) -> int:
        from world.block_states import state_id

        return state_id(name)

    def is_liquid(self, block_id: int) -> bool:
        return bool(self.liquid[block_id]) if 0 <= block_id < self.total else False

    def is_solid(self, block_id: int) -> bool:
        return bool(self.solid_or_liquid[block_id]) if 0 <= block_id < self.total else False


# ==================== biome temperature ====================

class BiomeTemperature:
    """`Biome.getTemperature(pos)` - what the `minecraft:temperature` condition asks.

    Three samplers on *fixed legacy seeds*, because they are static fields on
    `Biome` in vanilla and do not move with the world seed. They are built lazily
    so a preset that never uses the condition never pays for three simplex
    permutation tables.

    The `frozen` modifier is a noise test, not a rectangle: it is what makes
    frozen oceans cold in patches.
    """

    __slots__ = ("_temperature", "_frozen_ocean", "_foliage")

    def __init__(self) -> None:
        self._temperature: Optional[OctaveSimplexNoiseSampler] = None
        self._frozen_ocean: Optional[OctaveSimplexNoiseSampler] = None
        self._foliage: Optional[OctaveSimplexNoiseSampler] = None

    @property
    def temperature_noise(self) -> OctaveSimplexNoiseSampler:
        if self._temperature is None:
            self._temperature = OctaveSimplexNoiseSampler(LegacyRandom(1234), [0])
        return self._temperature

    @property
    def frozen_ocean_noise(self) -> OctaveSimplexNoiseSampler:
        if self._frozen_ocean is None:
            self._frozen_ocean = OctaveSimplexNoiseSampler(LegacyRandom(3456), [-2, -1, 0])
        return self._frozen_ocean

    @property
    def foliage_noise(self) -> OctaveSimplexNoiseSampler:
        if self._foliage is None:
            self._foliage = OctaveSimplexNoiseSampler(LegacyRandom(2345), [0])
        return self._foliage

    def modified(self, x: float, z: float, temperature: float, modifier: str) -> float:
        if modifier != "frozen":
            return temperature
        frozen = self.frozen_ocean_noise.sample(x * 0.05, z * 0.05, False) * 7.0
        foliage = self.foliage_noise.sample(x * 0.2, z * 0.2, False)
        if frozen + foliage < 0.3:
            if self.foliage_noise.sample(x * 0.09, z * 0.09, False) < 0.8:
                return 0.2
        return temperature

    def at(self, x: int, y: int, z: int, base: float, modifier: str, sea_level: int) -> float:
        """Height-adjusted temperature, which is what the condition thresholds.

        Above `sea_level + 17` it falls off with height - the reason a mountain
        is cold without a cold biome under it.
        """
        value = self.modified(float(x), float(z), base, modifier)
        offset = sea_level + 17
        if y > offset:
            noise = float(np.float32(self.temperature_noise.sample(x / 8.0, z / 8.0, False))) * 8.0
            value = value - float(np.float32((noise + y - offset) * 0.05)) / 40.0
        return value


_BIOME_TEMPERATURE = BiomeTemperature()


# ==================== badlands and iceberg terrain ====================

class TerrainBuilder:
    """Per-column surgery the surface pass runs for two biomes, before the rules.

    Badlands get banded terracotta and noise-built plateaus; frozen oceans get
    ice spires that reach above and below the waterline. Neither is expressible
    as a rule: both read the column and write stacks of blocks.

    The 192-entry terracotta band table comes from `minecraft:clay_bands` and is
    per-world. Its construction interleaves four generators and a run of coin
    flips; getting the order wrong produces stripes in the wrong *colours*, which
    is a bug no screenshot can settle.
    """

    __slots__ = ("bands", "bands_offset", "badlands_pillar", "badlands_surface",
                 "badlands_pillar_roof", "iceberg_pillar", "iceberg_pillar_roof",
                 "iceberg_surface", "states", "_colors")

    def __init__(self, splitter: XoroshiroSplitter, states: StateLookup,
                 parameters: Optional[Dict[str, dict]] = None) -> None:
        self.states = states
        self._colors: Dict[str, int] = {}
        self.bands = self._create_terracotta_bands(splitter.split_string("minecraft:clay_bands"))
        self.bands_offset = _noise(splitter, "clay_bands_offset", parameters)
        self.badlands_pillar = _noise(splitter, "badlands_pillar", parameters)
        self.badlands_surface = _noise(splitter, "badlands_surface", parameters)
        self.badlands_pillar_roof = _noise(splitter, "badlands_pillar_roof", parameters)
        self.iceberg_pillar = _noise(splitter, "iceberg_pillar", parameters)
        self.iceberg_pillar_roof = _noise(splitter, "iceberg_pillar_roof", parameters)
        self.iceberg_surface = _noise(splitter, "iceberg_surface", parameters)

    def _color(self, name: str) -> int:
        cached = self._colors.get(name)
        if cached is None:
            from world.block_states import state_id

            cached = state_id(f"minecraft:{name}")
            self._colors[name] = cached
        return cached

    def _create_terracotta_bands(self, random: Xoroshiro) -> List[int]:
        terracotta = self._color("terracotta")
        bands = [terracotta] * 192

        i = 0
        while i < len(bands):
            i += random.next_bounded_i32(5) + 1
            if i >= len(bands):
                break
            bands[i] = self._color("orange_terracotta")
            i += 1

        self._add_bands(random, bands, 1, self._color("yellow_terracotta"))
        self._add_bands(random, bands, 2, self._color("brown_terracotta"))
        self._add_bands(random, bands, 1, self._color("red_terracotta"))

        white = self._color("white_terracotta")
        light_gray = self._color("light_gray_terracotta")
        count = random.next_between(9, 15)
        index = 0
        band = 0
        while band < count and index < len(bands):
            bands[index] = white
            if index > 1 and random.next_bool():
                bands[index - 1] = light_gray
            if index + 1 < len(bands) and random.next_bool():
                bands[index + 1] = light_gray
            index += random.next_bounded_i32(16) + 4
            band += 1
        return bands

    @staticmethod
    def _add_bands(random: Xoroshiro, bands: List[int], min_size: int, state: int) -> None:
        for _ in range(random.next_between(6, 15)):
            width = min_size + random.next_bounded_i32(3)
            start = random.next_bounded_i32(len(bands))
            for m in range(width):
                if start + m >= len(bands):
                    break
                bands[start + m] = state

    def terracotta_at(self, x: int, y: int, z: int) -> int:
        """`SurfaceSystem.getBand`: the band index is `y + round(noise * 4)`, mod 192."""
        offset = int(np.round(float(self.bands_offset.get(float(x), 0.0, float(z))) * 4.0))
        return self.bands[(y + offset) % len(self.bands)]

    def place_badlands_pillar(self, blocks: np.ndarray, local_x: int, local_z: int,
                              world_x: int, world_z: int, surface_y: int, min_y: int,
                              default_block: int) -> None:
        """Raise a badlands plateau to its noise-defined top, or do nothing.

        The scan before the fill is not an optimisation - it is the rule. Vanilla
        walks down from the top and gives up if it meets the default block (the
        plateau is already there) or water (never build a pillar out of an
        ocean). Filling first and checking later produces islands in the sea.
        """
        surface_noise = abs(float(self.badlands_surface.get(float(world_x), 0.0, float(world_z))) * 8.25)
        pillar_noise = float(self.badlands_pillar.get(world_x * 0.2, 0.0, world_z * 0.2)) * 15.0
        threshold = min(surface_noise, pillar_noise)
        if threshold <= 0.0:
            return

        roof = abs(float(self.badlands_pillar_roof.get(world_x * 0.75, 0.0, world_z * 0.75)) * 1.5)
        scaled = threshold * threshold * 2.5
        transformed = float(np.ceil(roof * 50.0)) + 24.0
        elevation = int(np.floor(64.0 + min(scaled, transformed)))
        if surface_y > elevation:
            return

        column = blocks[:, local_z, local_x]
        water = self.states.water
        for y in range(elevation, min_y - 1, -1):
            block = int(column[y - min_y])
            if block == default_block:
                break
            if block == water:
                return

        for y in range(elevation, min_y - 1, -1):
            if int(column[y - min_y]) != AIR:
                break
            column[y - min_y] = np.uint16(default_block)

    def place_iceberg(self, blocks: np.ndarray, local_x: int, local_z: int,
                      world_x: int, world_z: int, estimated_surface_y: int,
                      current_top_y: int, sea_level: int, biome: str,
                      base_temperature: float, modifier: str,
                      splitter: XoroshiroSplitter, min_y: int) -> None:
        """Frozen ocean ice spires, above and below the waterline."""
        surface_noise = abs(float(self.iceberg_surface.get(float(world_x), 0.0, float(world_z))) * 8.25)
        pillar_noise = float(self.iceberg_pillar.get(world_x * 1.28, 0.0, world_z * 1.28)) * 15.0
        threshold = min(surface_noise, pillar_noise)
        if threshold <= 1.8:
            return

        roof = abs(float(self.iceberg_pillar_roof.get(world_x * 1.17, 0.0, world_z * 1.17)) * 1.5)
        block_threshold = min(threshold * threshold * 1.2, float(np.ceil(roof * 40.0)) + 14.0)
        temperature = _BIOME_TEMPERATURE.at(world_x, sea_level, world_z, base_temperature,
                                            modifier, sea_level)
        if temperature > 0.1:
            block_threshold -= 2.0

        if block_threshold > 2.0:
            top_block = int(block_threshold) + sea_level
            bottom_block = int(sea_level - block_threshold - 7.0)
        else:
            top_block = 0
            bottom_block = 0

        random = splitter.split_pos(world_x, 0, world_z)
        snow_count = 2 + random.next_bounded_i32(4)
        snow_bottom = sea_level + 18 + random.next_bounded_i32(10)
        placed_snow = 0

        column = blocks[:, local_z, local_x]
        water = self.states.water
        snow = self._color("snow_block")
        ice = self._color("packed_ice")
        top_y = max(current_top_y, top_block + 1)
        for y in range(top_y, estimated_surface_y - 1, -1):
            block = int(column[y - min_y])
            if (block == AIR and y < top_block and random.next_f64() > 0.01) or (
                block == water and y > bottom_block and y < sea_level and bottom_block != 0
                and random.next_f64() > 0.15
            ):
                if placed_snow <= snow_count and y > snow_bottom:
                    column[y - min_y] = np.uint16(snow)
                    placed_snow += 1
                else:
                    column[y - min_y] = np.uint16(ice)


def _noise(splitter: XoroshiroSplitter, name: str,
           parameters: Optional[Dict[str, dict]] = None) -> DoublePerlinNoiseSampler:
    """A `NormalNoise` by name, derived exactly as `density.py`'s library does.

    `assets/noise_parameters.json` keys are bare (`surface`) while the rules
    spell the same noise `minecraft:surface`; both are accepted because the
    sampler is derived from the *qualified* name either way.
    """
    bare = name[len("minecraft:"):] if name.startswith("minecraft:") else name
    table = parameters if parameters is not None else load_noise_parameters()
    param = table.get(bare)
    if param is None:
        raise KeyError(f"no noise parameters for {name!r}")
    amplitudes = param.get("amplitudes") or [1.0]
    random = splitter.split_string(f"minecraft:{bare}")
    return DoublePerlinNoiseSampler(
        random, int(param.get("firstOctave", -7)), [float(a) for a in amplitudes]
    )


# ==================== rule context ====================

class SurfaceContext:
    """The state a rule reads. One instance per chunk, rewritten per column.

    The hot loop hands per-cell state (y, depths, fluid, biome) to the compiled
    matcher as arguments; this context carries the per-column and per-world
    state. The per-cell fields below stay because the rule *tree* still reads
    them - it is the oracle `tools/check_surface_rules.py` fuzzes the compiled
    matcher against.

    A fresh object per block would be correct and much slower: the rules are a
    straight-line walk and the context is the only thing they share.
    """

    __slots__ = ("min_y", "height", "sea_level", "splitter", "terrain", "run_depth",
                 "secondary_depth", "block_x", "block_y", "block_z", "biome",
                 "biome_temperature", "biome_modifier", "fluid_height", "stone_depth_above",
                 "stone_depth_below", "surface_min_y", "steep", "run_bottom",
                 "world_min_y", "world_height")

    def __init__(self, min_y: int, height: int, sea_level: int,
                 splitter: XoroshiroSplitter, terrain: TerrainBuilder,
                 world_min_y: Optional[int] = None,
                 world_height: Optional[int] = None) -> None:
        self.min_y = min_y
        self.height = height
        # Anchors are resolved against the dimension, which is not the same as
        # the array when a caller hands over a window of it: server chunks are
        # 256 blocks tall inside a 384-block overworld, and `above_bottom: 0`
        # has to mean -64 rather than 0.
        self.world_min_y = min_y if world_min_y is None else int(world_min_y)
        self.world_height = height if world_height is None else int(world_height)
        self.sea_level = sea_level
        self.splitter = splitter
        self.terrain = terrain
        self.run_depth = 0
        self.secondary_depth = 0.0
        self.block_x = 0
        self.block_y = 0
        self.block_z = 0
        self.biome = "minecraft:plains"
        self.biome_temperature = 0.0
        self.biome_modifier = ""
        self.fluid_height = _NO_FLUID
        self.stone_depth_above = 0
        self.stone_depth_below = 0
        self.surface_min_y = 0
        self.steep = False
        self.run_bottom = 0


# ==================== conditions ====================

class Condition:
    """A rule condition, readable two ways.

    `__call__` is the tree walker's oracle. `emit` appends its share of a flat
    matcher to a `_Compiler` and returns the boolean expression it left behind.
    Both readings are built from the same parse-time constants, so the compiled
    matcher cannot drift from the tree; `tools/check_surface_rules.py` then
    asserts the two agree on swept states, on top of the dump comparison that
    pins both to vanilla.
    """

    __slots__ = ()

    def __call__(self, ctx: SurfaceContext) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def emit(self, gen: "_Compiler", indent: int) -> str:  # pragma: no cover - interface
        raise NotImplementedError


def _canonical(name: Any) -> str:
    name = str(name)
    return name if name.startswith("minecraft:") else f"minecraft:{name}"


def _anchor_expr(anchor: Anchor) -> str:
    """`anchor.get_y(world_min_y, world_height)` as source, with the window read
    from the `wmin`/`wh` locals the compiled matcher hoists."""
    if anchor.kind == _ABSOLUTE:
        return f"({anchor.value})"
    if anchor.kind == _ABOVE_BOTTOM:
        return f"(wmin + {anchor.value})"
    return f"(wmin + wh - 1 - {anchor.value})"


class VerticalGradient(Condition):
    """Bedrock: certain below `true_at_and_below`, never above `false_at_and_above`.

    In between the chance falls linearly from 1 to 0 and the draw is positional,
    so the transition is a rough deterministic band rather than a smooth ramp.
    `random_name` is what makes each gradient in the file independent.
    """

    __slots__ = ("true_at", "false_at", "lo", "hi")

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.true_at = Anchor.parse(payload["true_at_and_below"])
        self.false_at = Anchor.parse(payload["false_at_and_above"])
        self.lo, self.hi = name_hash(_canonical(payload.get("random_name", "minecraft:bedrock_floor")))

    def __call__(self, ctx: SurfaceContext) -> bool:
        y = ctx.block_y
        low = self.true_at.get_y(ctx.world_min_y, ctx.world_height)
        if y <= low:
            return True
        high = self.false_at.get_y(ctx.world_min_y, ctx.world_height)
        if y >= high:
            return False
        # Vanilla maps low -> 1.0, high -> 0.0; writing the rise instead would
        # put solid bedrock in the sky and none at the bottom.
        chance = 1.0 - (y - low) / (high - low)
        random = ctx.splitter.from_lo_and_hi(self.lo, self.hi).next_splitter().split_pos(
            ctx.block_x, y, ctx.block_z)
        return random.next_f32() < chance

    def emit(self, gen: "_Compiler", indent: int) -> str:
        # Outside the transition band this is two compares and no call; the
        # positional draw is a helper so the common paths stay inline.
        low = gen.fresh("_low")
        high = gen.fresh("_high")
        flag = gen.fresh("_c")
        chance = gen.fresh("_chance")
        pad = gen.pad(indent)
        gen.line(f"{pad}{low} = {_anchor_expr(self.true_at)}")
        gen.line(f"{pad}{high} = {_anchor_expr(self.false_at)}")
        gen.line(f"{pad}if y <= {low}:")
        gen.line(f"{pad}    {flag} = True")
        gen.line(f"{pad}elif y >= {high}:")
        gen.line(f"{pad}    {flag} = False")
        gen.line(f"{pad}else:")
        gen.line(f"{pad}    {chance} = 1.0 - (y - {low}) / ({high} - {low})")
        if gen.splitter is not None:
            # `from_lo_and_hi(...).next_splitter()` is a pure derivation from
            # the system's splitter, so bake it and pay only the per-cell
            # `split_pos` draw at match time.
            derived = gen.const(gen.splitter.from_lo_and_hi(self.lo, self.hi).next_splitter())
            roll = f"{derived}.split_pos(ctx.block_x, y, ctx.block_z).next_f32()"
        else:
            roll = f"_vgrad_roll(ctx, y, {self.lo!r}, {self.hi!r})"
        gen.line(f"{pad}    {flag} = {roll} < {chance}")
        return flag


class StoneDepth(Condition):
    __slots__ = ("offset", "ceiling", "add_surface_depth", "secondary_range")

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.offset = int(payload.get("offset", 0))
        self.ceiling = payload.get("surface_type") == "ceiling"
        self.add_surface_depth = bool(payload.get("add_surface_depth", False))
        self.secondary_range = int(payload.get("secondary_depth_range", 0))

    def __call__(self, ctx: SurfaceContext) -> bool:
        depth = ctx.stone_depth_below if self.ceiling else ctx.stone_depth_above
        extra = ctx.run_depth if self.add_surface_depth else 0
        if self.secondary_range:
            # `Mth.map` does not clamp, and the secondary field does exceed 1, so
            # this is an extrapolation rather than a rescale.
            extra += int((ctx.secondary_depth + 1.0) * 0.5 * self.secondary_range)
        return depth <= 1 + self.offset + extra

    def emit(self, gen: "_Compiler", indent: int) -> str:
        depth = "sdb" if self.ceiling else "sda"
        extra = "ctx.run_depth" if self.add_surface_depth else "0"
        if self.secondary_range:
            extra = f"({extra} + int((ctx.secondary_depth + 1.0) * 0.5 * {self.secondary_range}))"
        return f"({depth} <= 1 + {self.offset} + {extra})"


class BiomeIs(Condition):
    __slots__ = ("wanted",)

    def __init__(self, wanted: frozenset) -> None:
        self.wanted = wanted

    def __call__(self, ctx: SurfaceContext) -> bool:
        return ctx.biome in self.wanted

    def emit(self, gen: "_Compiler", indent: int) -> str:
        return f"(biome in {gen.const(self.wanted)})"


class YAbove(Condition):
    __slots__ = ("anchor", "multiplier", "add_stone_depth")

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.anchor = Anchor.parse(payload["anchor"])
        self.multiplier = int(payload.get("surface_depth_multiplier", 0))
        self.add_stone_depth = bool(payload.get("add_stone_depth", False))

    def __call__(self, ctx: SurfaceContext) -> bool:
        depth = ctx.stone_depth_above if self.add_stone_depth else 0
        return (ctx.block_y + depth
                >= self.anchor.get_y(ctx.world_min_y, ctx.world_height) + ctx.run_depth * self.multiplier)

    def emit(self, gen: "_Compiler", indent: int) -> str:
        depth = "sda" if self.add_stone_depth else "0"
        # `run_depth` is an int, so a zero multiplier folds away exactly.
        rise = f" + ctx.run_depth * {self.multiplier}" if self.multiplier else ""
        return f"(y + {depth} >= {_anchor_expr(self.anchor)}{rise})"


class Water(Condition):
    __slots__ = ("offset", "multiplier", "add_stone_depth")

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.offset = int(payload.get("offset", 0))
        self.multiplier = int(payload.get("surface_depth_multiplier", 0))
        self.add_stone_depth = bool(payload.get("add_stone_depth", False))

    def __call__(self, ctx: SurfaceContext) -> bool:
        if ctx.fluid_height == _NO_FLUID:
            # No fluid anywhere below this cell, so there is nothing to be under.
            # The alternative reading - let `MIN_VALUE` into the arithmetic, where
            # it always loses - was measured against the dumps and is worse
            # everywhere: it fixes the deep sand under caves in one chunk and
            # breaks the entire surface of another. See
            # `tools/check_surface_rules.py`, which records both.
            return True
        depth = ctx.stone_depth_above if self.add_stone_depth else 0
        return (ctx.block_y + depth
                >= ctx.fluid_height + self.offset + ctx.run_depth * self.multiplier)

    def emit(self, gen: "_Compiler", indent: int) -> str:
        depth = "sda" if self.add_stone_depth else "0"
        rise = f" + ctx.run_depth * {self.multiplier}" if self.multiplier else ""
        return f"(fluid == _NO_FLUID or y + {depth} >= fluid + {self.offset}{rise})"


class NoiseThreshold(Condition):
    __slots__ = ("sampler", "low", "high")

    def __init__(self, payload: Mapping[str, Any],
                 noise_of: Callable[[str], DoublePerlinNoiseSampler]) -> None:
        self.sampler = noise_of(_canonical(payload["noise"]))
        self.low = float(payload.get("min_threshold", -1e30))
        self.high = float(payload.get("max_threshold", 1e30))

    def __call__(self, ctx: SurfaceContext) -> bool:
        # Sampled at y = 0: it is a 2D field deciding *where* the pattern is.
        value = float(self.sampler.get(float(ctx.block_x), 0.0, float(ctx.block_z)))
        return self.low <= value <= self.high

    def emit(self, gen: "_Compiler", indent: int) -> str:
        # A chained comparison evaluates the sample once, like `__call__`.
        sampler = gen.const(self.sampler)
        return f"({self.low!r} <= {sampler}.get(float(ctx.block_x), 0.0, float(ctx.block_z)) <= {self.high!r})"


class Temperature(Condition):
    __slots__ = ()

    def __call__(self, ctx: SurfaceContext) -> bool:
        value = _BIOME_TEMPERATURE.at(ctx.block_x, ctx.block_y, ctx.block_z,
                                      ctx.biome_temperature, ctx.biome_modifier,
                                      ctx.sea_level)
        return value < 0.15

    def emit(self, gen: "_Compiler", indent: int) -> str:
        return "(_temp_at(ctx, y, temp, mod) < 0.15)"


class Not(Condition):
    __slots__ = ("inner",)

    def __init__(self, inner: Condition) -> None:
        self.inner = inner

    def __call__(self, ctx: SurfaceContext) -> bool:
        return not self.inner(ctx)

    def emit(self, gen: "_Compiler", indent: int) -> str:
        return f"(not ({self.inner.emit(gen, indent)}))"


class AbovePreliminarySurface(Condition):
    __slots__ = ()

    def __call__(self, ctx: SurfaceContext) -> bool:
        return ctx.block_y >= ctx.surface_min_y

    def emit(self, gen: "_Compiler", indent: int) -> str:
        return "(y >= ctx.surface_min_y)"


class Hole(Condition):
    """"Is this cell inside a carved hole rather than under the crust?" A rule
    has no way to see the carver's mask, so vanilla answers with the only
    proxy it has: a column whose surface depth came out at or below zero is
    one the crust guard would have carved through. Pinning this to a
    constant instead makes `not(hole)` unconditionally true, which bands
    terracotta through every badlands and lets the deep sand rules fire
    under caves.
    """

    __slots__ = ()

    def __call__(self, ctx: SurfaceContext) -> bool:
        return ctx.run_depth <= 0

    def emit(self, gen: "_Compiler", indent: int) -> str:
        return "(ctx.run_depth <= 0)"


class Steep(Condition):
    __slots__ = ()

    def __call__(self, ctx: SurfaceContext) -> bool:
        return ctx.steep

    def emit(self, gen: "_Compiler", indent: int) -> str:
        return "ctx.steep"


def _compile_condition(payload: Mapping[str, Any],
                       noise_of: Callable[[str], DoublePerlinNoiseSampler]) -> Condition:
    kind = payload.get("type", "")
    if kind == "minecraft:vertical_gradient":
        return VerticalGradient(payload)
    if kind == "minecraft:above_preliminary_surface":
        return AbovePreliminarySurface()
    if kind == "minecraft:stone_depth":
        return StoneDepth(payload)
    if kind == "minecraft:biome":
        wanted = frozenset(_canonical(b) for b in payload.get("biome_is") or ())
        return BiomeIs(wanted)
    if kind == "minecraft:y_above":
        return YAbove(payload)
    if kind == "minecraft:noise_threshold":
        return NoiseThreshold(payload, noise_of)
    if kind == "minecraft:water":
        return Water(payload)
    if kind == "minecraft:not":
        return Not(_compile_condition(payload["invert"], noise_of))
    if kind == "minecraft:hole":
        return Hole()
    if kind == "minecraft:temperature":
        return Temperature()
    if kind == "minecraft:steep":
        return Steep()
    raise ValueError(f"unknown surface-rule condition {kind!r}")


# ==================== rules ====================

class Rule:
    """A surface rule. `match` returns a block-state id, or -1 for no match.

    `emit` is the compile side: it appends this rule's share of a flat matcher
    to a `_Compiler`, in the same first-match-wins order `match` walks.
    """

    __slots__ = ()

    def match(self, ctx: SurfaceContext) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def emit(self, gen: "_Compiler", indent: int) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class BlockRule(Rule):
    __slots__ = ("state",)

    def __init__(self, state: int) -> None:
        self.state = state

    def match(self, ctx: SurfaceContext) -> int:
        return self.state

    def emit(self, gen: "_Compiler", indent: int) -> None:
        gen.line(f"{gen.pad(indent)}return {self.state}")

    def __repr__(self) -> str:
        return f"BlockRule({self.state})"


class SequenceRule(Rule):
    """First match wins - the order in the file is the rule."""

    __slots__ = ("rules",)

    def __init__(self, rules: Sequence[Rule]) -> None:
        self.rules = tuple(rules)

    def match(self, ctx: SurfaceContext) -> int:
        for rule in self.rules:
            result = rule.match(ctx)
            if result >= 0:
                return result
        return -1

    def emit(self, gen: "_Compiler", indent: int) -> None:
        # A nested sequence that matches nothing simply falls through to the
        # next entry of the outer one, which is exactly `match` returning -1.
        for rule in self.rules:
            rule.emit(gen, indent)


class ConditionRule(Rule):
    __slots__ = ("condition", "then_run")

    def __init__(self, condition: Condition, then_run: Rule) -> None:
        self.condition = condition
        self.then_run = then_run

    def match(self, ctx: SurfaceContext) -> int:
        if self.condition(ctx):
            return self.then_run.match(ctx)
        return -1

    def emit(self, gen: "_Compiler", indent: int) -> None:
        expr = self.condition.emit(gen, indent)
        gen.line(f"{gen.pad(indent)}if {expr}:")
        self.then_run.emit(gen, indent + 1)


class BadlandsRule(Rule):
    """`minecraft:bandlands` - the banded terracotta, indexed by y."""

    __slots__ = ()

    def match(self, ctx: SurfaceContext) -> int:
        return ctx.terrain.terracotta_at(ctx.block_x, ctx.block_y, ctx.block_z)

    def emit(self, gen: "_Compiler", indent: int) -> None:
        gen.line(f"{gen.pad(indent)}return ctx.terrain.terracotta_at(ctx.block_x, y, ctx.block_z)")


class BlockResolver:
    """`{Name, Properties}` -> a global state id, memoised.

    Rules name 100 blocks and are parsed once, so the cache is for the benefit
    of repeated names rather than repeated parses - but a world built twice in
    one process (a test, a reload) would otherwise re-resolve every one.
    """

    __slots__ = ("_cache",)

    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], int] = {}

    def __call__(self, payload: Mapping[str, Any]) -> int:
        name = _canonical(payload.get("Name") or payload.get("name") or "")
        properties = payload.get("Properties") or {}
        key = (name, tuple(sorted((str(k), str(v)) for k, v in properties.items())))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        from world.block_states import state_id

        resolved = state_id(name, dict(key[1]) or None)
        self._cache[key] = resolved
        return resolved


def parse_surface_rule(payload: Any, *,
                       noise_of: Optional[Callable[[str], DoublePerlinNoiseSampler]] = None) -> Rule:
    """Compile one `surface_rule` payload into a walkable tree."""
    if payload is None:
        raise ValueError("no surface rule payload")
    if noise_of is None:
        noise_of = lambda name: _noise(XoroshiroSplitter(0, 0), name)  # noqa: E731
    return _parse_rule(payload, BlockResolver(), noise_of)


def _parse_rule(payload: Mapping[str, Any], resolver: BlockResolver,
                noise_of: Callable[[str], DoublePerlinNoiseSampler]) -> Rule:
    kind = payload.get("type", "")
    if kind == "minecraft:block":
        return BlockRule(resolver(payload["result_state"]))
    if kind == "minecraft:sequence":
        return SequenceRule([
            _parse_rule(entry, resolver, noise_of) for entry in payload["sequence"]
        ])
    if kind == "minecraft:condition":
        return ConditionRule(
            _compile_condition(payload["if_true"], noise_of),
            _parse_rule(payload["then_run"], resolver, noise_of),
        )
    if kind == "minecraft:bandlands":
        return BadlandsRule()
    raise ValueError(f"unknown surface-rule node {kind!r}")


# ==================== compiled matcher ====================

def _vgrad_roll(ctx: SurfaceContext, y: int, lo: int, hi: int) -> float:
    """The positional draw inside a `vertical_gradient` transition band.

    Split out so the compiled matcher's common paths (outside the band, a
    handful of rows per column) stay two compares and no calls; this runs only
    where the gradient is fading.
    """
    return ctx.splitter.from_lo_and_hi(lo, hi).next_splitter().split_pos(
        ctx.block_x, y, ctx.block_z).next_f32()


def _temp_at(ctx: SurfaceContext, y: int, temperature: float, modifier: str) -> float:
    """`Biome.getTemperature` at a cell, for the compiled `temperature` rule."""
    return _BIOME_TEMPERATURE.at(ctx.block_x, y, ctx.block_z, temperature,
                                 modifier, ctx.sea_level)


class _Compiler:
    """Accumulates one flat matcher function from a rule tree.

    The tree walker spends most of a cell's budget on Python calls - roughly a
    dozen `match`/condition invocations per stone cell, each a frame. The
    compiler walks the same tree once at startup and emits straight-line source
    with the conditions inlined, so a cell costs one call instead of a dozen.
    First-match-wins order is preserved by construction: rules emit `return`s
    and a condition gates its branch with a plain `if`.
    """

    __slots__ = ("lines", "consts", "_counter", "splitter")

    def __init__(self, splitter: Optional[XoroshiroSplitter] = None) -> None:
        self.lines: List[str] = []
        self.consts: List[Any] = []
        self._counter = 0
        #: When set, `vertical_gradient` draws bake their derived splitter at
        #: compile time instead of re-deriving it on every transition-band cell.
        self.splitter = splitter

    def line(self, text: str) -> None:
        self.lines.append(text)

    def const(self, value: Any) -> str:
        self.consts.append(value)
        return f"_K[{len(self.consts) - 1}]"

    def fresh(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}{self._counter}"

    @staticmethod
    def pad(indent: int) -> str:
        return "    " * indent


def compile_rule(rule: Rule, *, splitter: Optional[XoroshiroSplitter] = None) -> Callable[..., int]:
    """Compile a parsed rule tree into one flat Python matcher.

    The result takes the cell state as arguments - `match(ctx, y, fluid, sda,
    sdb, biome, temp, mod)` where `sda`/`sdb` are `stone_depth_above`/`below` -
    so the hot loop can pass locals instead of writing them into a context, and
    the generated body reads them as fast locals. Per-column state (positions,
    `run_depth`, the preliminary-surface floor) is still read off `ctx`, which
    is written once per column.

    Pass `splitter` to bake each `vertical_gradient`'s derived draw splitter at
    compile time; the derivation is pure, but a context carrying a *different*
    splitter would then draw different bedrock than the tree does, so only do
    this when the context is known to carry the same one.

    The tree itself is kept: it is the oracle `tools/check_surface_rules.py`
    compares this function against, and `match` remains the readable form.
    """
    gen = _Compiler(splitter)
    rule.emit(gen, 1)
    gen.line("    return -1")
    source = "\n".join([
        "def _compiled_match(ctx, y, fluid, sda, sdb, biome, temp, mod):",
        "    wmin = ctx.world_min_y",
        "    wh = ctx.world_height",
        *gen.lines,
    ])
    namespace: Dict[str, Any] = {
        "_K": gen.consts,
        "_vgrad_roll": _vgrad_roll,
        "_temp_at": _temp_at,
        "_NO_FLUID": _NO_FLUID,
    }
    exec(compile(source, "<surface_rule>", "exec"), namespace)
    matcher = namespace["_compiled_match"]
    #: The generated source, for debugging a rule that matches where it should not.
    matcher.source = source  # type: ignore[attr-defined]
    return matcher


# ==================== helpers over the block volume ====================

def world_surface_heightmap(blocks: np.ndarray, min_y: int) -> np.ndarray:
    """Topmost non-air y per column, plus one - `Heightmap.Types.WORLD_SURFACE_WG`.

    The same predicate the dump's `step_to_noise` leaves behind, and the one the
    surface pass iterates down from. `OCEAN_FLOOR` or `MOTION_BLOCKING` here
    moves every ocean column's surface one or more blocks and the rules then
    disagree with vanilla on the sea floor.
    """
    occupied = blocks != np.uint16(AIR)
    # The `-1` has to be per *cell*, not per column: masking the row axis with
    # `occupied.any(axis=0)` replaces the whole column with the row index vector
    # and every non-empty column then reports the top of the world.
    rows = np.where(occupied, np.arange(blocks.shape[0])[:, None, None], -1).max(axis=0)
    # `rows` is [z, x]; the heightmap is [x, z].
    return (rows.T + min_y + 1).astype(np.int32)


def _steep(heightmap_rows: List[List[int]], local_x: int, local_z: int) -> bool:
    """`SurfaceSystem.steep`: a four-block step to a neighbour, Z first.

    The local (clamped) neighbours are deliberate: vanilla reads the chunk's own
    heightmap, not the neighbouring chunk's, so a cliff exactly on a chunk border
    can be missed. Reproducing the oversight is the point. `heightmap_rows` is
    the `[x][z]` tolist of the heightmap the pass iterates from.
    """
    z_sub = max(0, local_z - 1)
    z_add = min(15, local_z + 1)
    if heightmap_rows[local_x][z_add] >= heightmap_rows[local_x][z_sub] + 4:
        return True
    x_sub = max(0, local_x - 1)
    x_add = min(15, local_x + 1)
    return heightmap_rows[x_sub][local_z] >= heightmap_rows[x_add][local_z] + 4


# ==================== the system ====================

class SurfaceStats:
    """What one chunk's surface pass did. Cheap, and the only honest evidence."""

    __slots__ = ("columns", "cells_visited", "cells_changed", "rules_evaluated",
                 "pillars", "icebergs")

    def __init__(self) -> None:
        self.columns = 0
        self.cells_visited = 0
        self.cells_changed = 0
        self.rules_evaluated = 0
        self.pillars = 0
        self.icebergs = 0

    def as_dict(self) -> Dict[str, int]:
        return {slot: getattr(self, slot) for slot in self.__slots__}

    def __repr__(self) -> str:
        return (f"<SurfaceStats columns={self.columns} visited={self.cells_visited} "
                f"changed={self.cells_changed} pillars={self.pillars} icebergs={self.icebergs}>")


class SurfaceSystem:
    """One preset's surface pass, ready to run on chunks."""

    __slots__ = ("preset", "seed", "min_y", "height", "sea_level", "default_block",
                 "rule", "compiled", "states", "splitter", "terrain", "biome_table",
                 "surface_noise", "secondary_noise", "noise_parameters",
                 "world_min_y", "world_height",
                 "_estimator", "_estimate_cache", "diagnostics")

    def __init__(self, preset: str, seed: int, *, rule: Any, min_y: int, height: int,
                 sea_level: int, default_block: int, biome_table=None,
                 noise_parameters: Optional[Dict[str, dict]] = None,
                 density_function_path: Optional[str] = None,
                 world_min_y: Optional[int] = None,
                 world_height: Optional[int] = None) -> None:
        self.preset = preset
        self.seed = seed
        self.min_y = min_y
        self.height = height
        # Anchors are resolved against the *dimension*, which is not the same as
        # `min_y`/`height` when the caller's array is a window of it: server
        # chunks are 256 blocks tall inside a 384-block overworld, and
        # `above_bottom: 0` has to mean -64 rather than 0.
        self.world_min_y = min_y if world_min_y is None else int(world_min_y)
        self.world_height = height if world_height is None else int(world_height)
        self.sea_level = sea_level
        self.default_block = default_block
        self.biome_table = biome_table
        self.diagnostics: Dict[str, int] = {}

        self.noise_parameters = (noise_parameters if noise_parameters is not None
                                 else load_noise_parameters())
        base = Xoroshiro.from_seed(seed).next_splitter()
        self.splitter = base
        self.surface_noise = _noise(base, "surface", self.noise_parameters)
        self.secondary_noise = _noise(base, "surface_secondary", self.noise_parameters)
        self.states = StateLookup()
        self.terrain = TerrainBuilder(base, self.states, self.noise_parameters)
        self.rule = parse_surface_rule(
            rule, noise_of=lambda name: _noise(base, name, self.noise_parameters))
        self.compiled = compile_rule(self.rule, splitter=self.splitter)

        # The preliminary-surface estimator is parsed in the proto phase, which
        # is what vanilla's estimator sees. `initialDensityWithoutJaggedness`
        # contains no `Interpolated` node in this dump - only `finalDensity` and
        # the vein slots do - so the phase cannot change the value; the corners
        # are exact either way and the field between them is built here.
        self._estimator = self._parse_estimator(density_function_path)
        self._estimate_cache: Dict[Tuple[int, int], Tuple[float, float, float, float]] = {}

    @classmethod
    def from_assets(cls, preset: str = "overworld", seed: int = 0, *,
                    biome_table=None, settings: Optional[Mapping[str, Any]] = None,
                    noise_parameters: Optional[Dict[str, dict]] = None,
                    density_function_path: Optional[str] = None,
                    window_min_y: Optional[int] = None,
                    window_height: Optional[int] = None) -> "SurfaceSystem":
        """Build from `assets/chunk_gen_settings.json`.

        `window_min_y` / `window_height` describe the array a caller will pass to
        `build_surface` when it is not the whole dimension - the live server
        stores a 256-block chunk inside a 384-block overworld. The rules still
        anchor to the preset's own extent; only the array shape changes.
        """
        if settings is None:
            with open(_assets_path("chunk_gen_settings.json"), encoding="utf-8") as handle:
                presets = json.load(handle)
            if preset not in presets:
                known = ", ".join(sorted(presets))
                raise KeyError(f"unknown generator preset {preset!r}; have: {known}")
            settings = presets[preset]
        noise = settings.get("noise") or {}
        world_min_y = int(noise.get("min_y", -64))
        world_height = int(noise.get("height", 384))
        return cls(
            preset, seed,
            rule=settings.get("surface_rule"),
            min_y=world_min_y if window_min_y is None else int(window_min_y),
            height=world_height if window_height is None else int(window_height),
            sea_level=int(settings.get("sea_level", 63)),
            default_block=BlockResolver()(settings.get("default_block") or {"Name": "minecraft:stone"}),
            biome_table=biome_table,
            noise_parameters=noise_parameters,
            density_function_path=density_function_path,
            world_min_y=world_min_y,
            world_height=world_height,
        )

    def _parse_estimator(self, path: Optional[str]):
        """The `initial_density_without_jaggedness` node, in its proto phase.

        Stored with the `Context` it was parsed against, because a node does not
        carry its own - `eval` takes one - and rebuilding it per chunk would
        re-derive every noise sampler in the tree.
        """
        with open(path or _assets_path("density_function.json"), encoding="utf-8") as handle:
            trees = json.load(handle)
        preset = trees.get(self.preset)
        if preset is None:
            return None
        node = preset.get("initialDensityWithoutJaggedness")
        if node is None:
            return None
        context = DensityContext(self.seed, self.noise_parameters)
        return parse_density(node, context, pass_through_wrappers=True), context

    # -- the estimator -----------------------------------------------------

    def _corner_estimates(self, chunk_x: int, chunk_z: int) -> Tuple[float, float, float, float]:
        """The four chunk-corner values of the preliminary-surface function.

        `estimate_height` aligns to a quart first, which is why the corners are
        `(chunk << 4)` masked rather than offset by one block.
        """
        cached = self._estimate_cache.get((chunk_x, chunk_z))
        if cached is not None:
            return cached
        if self._estimator is None:
            corners = (float(self.min_y),) * 4
        else:
            node, context = self._estimator
            base_x = (chunk_x << 4) & ~3
            base_z = (chunk_z << 4) & ~3
            estimate = [
                float(np.floor(node.eval(context, base_x + dx * 16, 0, base_z + dz * 16)))
                for dz in (0, 1) for dx in (0, 1)
            ]
            # `lerp2(dx, dz, v00, v10, v01, v11)`, i.e. x varies inside z.
            corners = (estimate[0], estimate[1], estimate[2], estimate[3])
        if len(self._estimate_cache) > 256:
            self._estimate_cache.clear()
        self._estimate_cache[(chunk_x, chunk_z)] = corners
        return corners

    def preliminary_surface_levels(self, chunk_x: int, chunk_z: int) -> np.ndarray:
        """Per-column lower bound, `(16, 16)` indexed `[x, z]`."""
        v00, v10, v01, v11 = self._corner_estimates(chunk_x, chunk_z)
        fx = np.arange(16, dtype=np.float64)[:, None] / 16.0
        fz = np.arange(16, dtype=np.float64)[None, :] / 16.0
        top = v00 + (v10 - v00) * fx
        bottom = v01 + (v11 - v01) * fx
        return np.floor(top + (bottom - top) * fz).astype(np.int32)

    # -- the pass ----------------------------------------------------------

    def build_surface(self, blocks: np.ndarray, biomes: np.ndarray, chunk_x: int,
                      chunk_z: int, *, heightmap: Optional[np.ndarray] = None,
                      preliminary: Optional[np.ndarray] = None,
                      world_min_y: Optional[int] = None,
                      world_height: Optional[int] = None) -> SurfaceStats:
        """Dress one chunk's terrain.

        `heightmap` is `(16, 16)` `[x, z]`, topmost non-air y plus one; derived
        from `blocks` when omitted. `preliminary` is `(16, 16)` `[x, z]`, the
        per-column estimate; derived when omitted.
        """
        stats = SurfaceStats()
        expected = (self.height, 16, 16)
        if blocks.shape != expected:
            raise ValueError(f"blocks must be {expected}, got {blocks.shape}")
        quart = (self.height // 4, 4, 4)
        if biomes.shape != quart:
            raise ValueError(f"biomes must be {quart}, got {biomes.shape}")
        if world_min_y is not None:
            self.world_min_y = int(world_min_y)
        if world_height is not None:
            self.world_height = int(world_height)

        if heightmap is None:
            heightmap = world_surface_heightmap(blocks, self.min_y)
        if preliminary is None:
            preliminary = self.preliminary_surface_levels(chunk_x, chunk_z)

        table = self.biome_table
        names = table.names if table is not None else {}
        temperatures = table.temperature if table is not None else {}
        modifiers = table.modifier if table is not None else {}

        ctx = SurfaceContext(self.min_y, self.height, self.sea_level, self.splitter,
                             self.terrain, self.world_min_y, self.world_height)
        min_y = self.min_y
        default_block = self.default_block
        liquids = self.states.liquid_bytes
        solids = self.states.solid_or_liquid_bytes
        total = self.states.total
        match = self.compiled
        # The cell loop reads all three of these once per cell, and a numpy
        # scalar costs about four times a list index for the same integer. All
        # three are read-only for the length of this pass: it writes only
        # `blocks`, through the pillar/iceberg surgery and the rules.
        heightmap_rows = heightmap.tolist()
        preliminary_rows = preliminary.tolist()
        biome_rows = biomes.tolist()
        biome_row_count = len(biome_rows)

        columns = 0
        cells_visited = 0
        cells_changed = 0
        rules_evaluated = 0
        pillars = 0
        icebergs = 0

        for local_x in range(16):
            for local_z in range(16):
                world_x = chunk_x * 16 + local_x
                world_z = chunk_z * 16 + local_z
                base = min_y
                top_block = heightmap_rows[local_x][local_z]

                # Vanilla resolves the column's biome at the *pre-pillar* surface
                # height, which is what decides the pillar and iceberg special
                # cases, and then per block for the rules. A filled pillar is
                # therefore never re-tested and stays unbanded stone.
                qx = local_x // _QUART
                qz = local_z // _QUART
                biome_at_surface = names.get(biome_rows[
                    min(max((top_block - min_y) // _QUART, 0), biome_row_count - 1)
                ][qz][qx], "minecraft:plains")

                if biome_at_surface == "minecraft:eroded_badlands":
                    self.terrain.place_badlands_pillar(
                        blocks, local_x, local_z, world_x, world_z, top_block,
                        min_y, default_block)
                    pillars += 1

                # Read the column through a Python list. Nothing below the rules
                # looks at the volume -- the pillar is the last thing to write
                # it, which is why this is taken after it -- so a snapshot here
                # is the column the walk sees.
                column = blocks[:, local_z, local_x].tolist()

                ctx.block_x = world_x
                ctx.block_z = world_z
                ctx.run_depth = self._run_depth(world_x, world_z)
                # `secondaryDepth` is a once-per-column noise read; `stone_depth`
                # rules with a non-zero `secondary_depth_range` widen their band
                # by it.
                ctx.secondary_depth = float(
                    self.secondary_noise.get(float(world_x), 0.0, float(world_z)))
                ctx.surface_min_y = (preliminary_rows[local_x][local_z]
                                     + ctx.run_depth - _SURFACE_HEIGHT_OFFSET)
                ctx.steep = _steep(heightmap_rows, local_x, local_z)
                columns += 1

                stone_depth_above = 0
                fluid_height = _NO_FLUID
                run_top = None
                run_bottom = 0
                # The biome lives on a `_QUART` grid, so it is resolved once per
                # band instead of per rule cell: three dict lookups for four
                # cells rather than twelve.
                band = -1
                biome = biome_at_surface
                temperature = 0.0
                modifier = ""

                for y in range(top_block - 1, min_y - 1, -1):
                    block = column[y - base]
                    if block == AIR:
                        stone_depth_above = 0
                        fluid_height = _NO_FLUID
                        run_top = None
                        continue
                    if liquids[block] if block < total else False:
                        if fluid_height == _NO_FLUID:
                            fluid_height = y + 1
                        continue

                    if run_top != y + 1:
                        # Entering a new solid run. Find its extent once: the
                        # per-cell depth below is then an index, not a scan, and
                        # every cell in the run agrees with the top-down scan it
                        # replaces.
                        run_bottom = y
                        while run_bottom - 1 >= min_y:
                            below = column[run_bottom - 1 - base]
                            if not (solids[below] if below < total else False):
                                break
                            run_bottom -= 1
                    # Track the previous solid cell, not the run's top: the test
                    # above means "was the cell directly above me solid", and only
                    # then is the run below already known. Anchoring `run_top` at
                    # the run entry instead re-triggers the scan every other cell
                    # and turns the walk quadratic.
                    run_top = y
                    stone_depth_above += 1
                    cells_visited += 1

                    if block != default_block:
                        continue

                    # y >= min_y by the loop bounds, so the index is never
                    # negative and only the top clamp survives from the
                    # `min(max(...))` the biome-at-surface lookup still uses.
                    row = (y - min_y) // _QUART
                    if row >= biome_row_count:
                        row = biome_row_count - 1
                    if row != band:
                        band = row
                        biome = names.get(biome_rows[row][qz][qx])
                        if biome is None:
                            biome = biome_at_surface
                        temperature = temperatures.get(biome, 0.0)
                        modifier = modifiers.get(biome, "")
                    rules_evaluated += 1

                    new_state = match(ctx, y, fluid_height, stone_depth_above,
                                      y - run_bottom + 1, biome, temperature, modifier)
                    if 0 <= new_state != block:
                        column[y - base] = new_state
                        blocks[y - base, local_z, local_x] = new_state
                        cells_changed += 1

                if biome_at_surface in ("minecraft:frozen_ocean", "minecraft:deep_frozen_ocean"):
                    icebergs += 1
                    self.terrain.place_iceberg(
                        blocks, local_x, local_z, world_x, world_z,
                        preliminary_rows[local_x][local_z], top_block, self.sea_level,
                        biome_at_surface, temperatures.get(biome_at_surface, 0.0),
                        modifiers.get(biome_at_surface, ""), self.splitter, min_y)

        stats.columns = columns
        stats.cells_visited = cells_visited
        stats.cells_changed = cells_changed
        stats.rules_evaluated = rules_evaluated
        stats.pillars = pillars
        stats.icebergs = icebergs
        self.diagnostics = stats.as_dict()
        return stats

    def _run_depth(self, world_x: int, world_z: int) -> int:
        """`SurfaceSystem.getSurfaceDepth`: f32 throughout, truncated toward zero."""
        noise = np.float32(self.surface_noise.get(float(world_x), 0.0, float(world_z)))
        extra = np.float32(self.splitter.split_pos(world_x, 0, world_z).next_f64() * 0.25)
        total = np.float32(np.float32(noise * np.float32(2.75)) + np.float32(3.0) + extra)
        return int(total)


__all__ = [
    "AIR",
    "Anchor",
    "BadlandsRule",
    "BiomeTemperature",
    "BlockResolver",
    "BlockRule",
    "Condition",
    "ConditionRule",
    "Rule",
    "SequenceRule",
    "StateLookup",
    "SurfaceContext",
    "SurfaceStats",
    "SurfaceSystem",
    "TerrainBuilder",
    "compile_rule",
    "name_hash",
    "parse_surface_rule",
    "world_surface_heightmap",
]
