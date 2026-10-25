"""
`world/feature` — the decoration layer, shared by every generator backend.

A generator backend decides what the terrain *is*: `world/legacy` extrudes a
Perlin heightmap, `world/vanilla` reads a density function. Neither of them
decides what grows on it. This package does, in four passes, in this order:

    biomes      -> world/vanilla/biome.py     which biome each cell is
    surface     -> world/vanilla/surface.py   grass, sand, terracotta, bedrock
    carvers     -> world/vanilla/carver.py    caves and canyons
    features    -> world/vanilla/features.py  trees, ore, lakes, monster rooms

The order is not arbitrary. The carvers ask the biome volume which caves a
biome has; the surface rules ask it what its top block is; the features ask it
what to scatter. So biomes comes first and the other three read what it wrote.

Keeping these here rather than inside a generator is the point of the split.
The terrain algorithm is the only thing `[world] generator` switches, and these
four passes do not care which one is running - they read and write a `Chunk`
and nothing else about where its stone came from.

    layers = FeatureLayers(preset="overworld", seed=1234)
    table = layers.table                      # resolved once, shared
    layers.biomes.apply(chunk)
    layers.surface.apply(chunk, table)
    layers.carvers.apply(chunk, table, sea_level=63)
    layers.features.apply(chunk, table)

The caller drives the order rather than this module, because the legacy
backend has a fifth pass - its own ore veins - and it runs *between* surface
and carvers. Hiding the sequence behind one `run()` would have meant either
moving the ore pass or accepting a different chunk than the legacy backend
produces today.

Every layer is built once and reused, and every layer degrades rather than
raises: a missing or broken vanilla layer logs one warning naming what the
world loses and is not retried per chunk. A terrain generator that died
because a decoration table failed to parse would be a poor trade.
"""

from typing import Optional

from world.chunk import Chunk
from world.feature.biomes import BiomeLayer
from world.feature.carvers import CarverLayer
from world.feature.features import FeatureLayer
from world.feature.surface import SurfaceLayer

__all__ = ["FeatureLayers", "BiomeLayer", "SurfaceLayer", "CarverLayer", "FeatureLayer"]


class FeatureLayers:
    """The four decoration passes, assembled once per generator.

    The layer objects are created eagerly - they are four small holders, and
    nothing expensive runs until a chunk asks - but each one's *vanilla*
    counterpart is built lazily on its first chunk, so a flat world that never
    reaches the surface pass never parses the surface rules.
    """

    def __init__(self, preset: Optional[str], seed: int) -> None:
        self.preset = preset
        self.seed = seed
        self.biomes = BiomeLayer(preset, seed)
        self.surface = SurfaceLayer(preset, seed)
        self.carvers = CarverLayer(preset, seed)
        self.features = FeatureLayer(preset, seed)
        self._table = None
        self._shared = False

    @property
    def table(self):
        """The `BiomeTable` every layer resolves biome names through.

        Loaded on first use and handed to the three downstream layers, so the
        carver and feature passes do not each parse the biome file again.
        """
        if self._table is None:
            self._table = self.biomes.biome_table()
        if not self._shared:
            self.surface.bind(self._table)
            self._shared = True
        return self._table
