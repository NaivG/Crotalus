"""
Vanilla-compatible world generation primitives: RNG, noise, density, and the
four layers that turn a density field into a world.

This package is the part of terrain generation that has to agree with
Minecraft exactly rather than merely look plausible. `world/legacy/` carries a
fast Perlin field of its own; this one is the reference, and it is what
`[world] generator = "vanilla"` selects.

    from world.vanilla import NoiseRouter

    router = NoiseRouter.from_assets("overworld", seed=1234)
    router.begin_chunk(0, 0)
    router.surface_height(x, z, -64, 320)

`world.vanilla.terrain` turns that router into actual chunks, but it is not
re-exported here: it is a driver rather than a primitive, it writes to a
`Chunk`, and importing it would pull `world.generator` into every tool that
imports this package. Get it through `world.generator.create_generator()`.

The layers, in the order a chunk is generated:

`random_source`  Xoroshiro128+, the splitter, Java's `WorldgenRandom`, the
                 48-bit LCG, and `Mth.getSeed`. Every draw verified against Java.
`perlin`         Perlin, the Y-quantised ("smeared") variant, and the octave
                 stack with its three nested normalisation factors.
`simplex`        The 2D/3D simplex sampler the biome temperature needs.
`density`        The density-function interpreter: sixteen node types and four
                 wrappers, including the `Interpolated` cell blend.
`router`         The fifteen-slot noise router, assembled and sampleable.
`biome`          The multi-noise biome source: six climate values, quantized,
                 resolved through the parameter tree.
`surface`        The surface-rule interpreter: what the top of the terrain is
                 made of, plus the badlands and iceberg column surgery.

Two more modules are importable but deliberately *not* re-exported here, because
they are large and neither is needed to sample a density field:

`world.vanilla.carver`    Minecraft's cave and canyon carvers.
`world.vanilla.features`  The decorated-feature stage: vanilla's fifteen
                          placement modifiers and the configured-feature types
                          the biome data references.

    from world.vanilla import carver, features

`tools/check_density_functions.py` is the acceptance test for the first six and
compares against values verified against a real Minecraft.
`tools/check_biomes.py`, `check_surface_rules.py`, `check_carvers.py` and
`check_features.py` cover the rest. The carver and feature layers have **no
outside oracle** - no vanilla dump of either stage exists - so those two tools
check determinism, dispatch and measured distributions rather than parity, and
say so in their own docstrings. Run all five after any change here.
`tools/check_terrain_shape.py` covers `terrain.py`, and is the only one with a
whole-chunk oracle: it replays the `before` arrays in `assets/vanilla_chunks/`.
"""

from world.vanilla.random_source import (
    LegacyRandom,
    WorldgenRandom,
    Xoroshiro,
    XoroshiroSplitter,
    mix_stafford_13,
    name_seed,
)
from world.vanilla.perlin import (
    DoublePerlinNoiseSampler,
    NormalNoise,
    PerlinNoise,
    SmearedPerlinNoise,
    create_fbm,
    fbm_range_max,
)
from world.vanilla.simplex import OctaveSimplexNoiseSampler, SimplexNoise
from world.vanilla.density import (
    Context,
    Interpolated,
    Node,
    cell_size_for,
    collect_flat_caches,
    load_corpus,
    load_parameters,
    parse,
)
from world.vanilla.router import NoiseRouter
from world.vanilla.biome import (
    BiomeTable,
    BiomeTree,
    ClimateSampler,
    EndBiomeSource,
    FixedBiomeSource,
    MultiNoiseBiomeSource,
    biome_ids_from_assets,
    biome_source,
    biome_table,
    quantize,
    unquantize,
)
from world.vanilla.surface import (
    SurfaceSystem,
    parse_surface_rule,
    world_surface_heightmap,
)

__all__ = [
    "BiomeTable",
    "BiomeTree",
    "ClimateSampler",
    "Context",
    "DoublePerlinNoiseSampler",
    "EndBiomeSource",
    "FixedBiomeSource",
    "Interpolated",
    "LegacyRandom",
    "MultiNoiseBiomeSource",
    "Node",
    "NoiseRouter",
    "NormalNoise",
    "OctaveSimplexNoiseSampler",
    "PerlinNoise",
    "SimplexNoise",
    "SmearedPerlinNoise",
    "SurfaceSystem",
    "WorldgenRandom",
    "Xoroshiro",
    "XoroshiroSplitter",
    "biome_ids_from_assets",
    "biome_source",
    "biome_table",
    "cell_size_for",
    "collect_flat_caches",
    "create_fbm",
    "fbm_range_max",
    "load_corpus",
    "load_parameters",
    "mix_stafford_13",
    "name_seed",
    "parse",
    "parse_surface_rule",
    "quantize",
    "unquantize",
    "world_surface_heightmap",
]
