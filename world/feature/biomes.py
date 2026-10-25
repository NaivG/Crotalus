"""
Biome volume: which biome each cell is.

Drives `world/vanilla/biome.py`, which is the multi-noise biome source: six
climate values (temperature, vegetation, continentalness, erosion, depth,
ridges) sampled from the same noise router the terrain comes from, quantized
and resolved through a parameter tree into a biome id.

This layer is the *input* to the three after it. The carvers ask it which
caves a biome has, the surface rules ask it what its top block is made of, and
the features ask it what to scatter on it -- so a chunk with no biome volume
silently loses all three, which is why a broken biome source downgrades to
"every cell is whatever the caller set" rather than raising.
"""

from typing import Optional

import numpy as np
from loguru import logger

from world.chunk import Chunk


class BiomeLayer:
    """The biome pass. Built once per generator, reused for every chunk."""

    def __init__(self, preset: Optional[str], seed: int) -> None:
        self.preset = preset
        self.seed = seed
        self._source = None
        #: Shared with the carver/surface/feature passes, which all resolve
        #: biome *names* through the same table.
        self.table = None
        self._failed = False

    def biome_table(self):
        """The `BiomeTable`, loaded on first use and shared with the layers
        that need to turn a biome id back into its name."""
        if self.table is None:
            from world.vanilla.biome import BiomeTable

            self.table = BiomeTable.from_assets()
        return self.table

    def source(self):
        """The multi-noise biome source for this preset, built once and reused.

        Building it parses a 9000-node tree and six density functions, so it is
        per-generator, not per-chunk. `None` when the registry has no preset -
        a custom dimension - in which case biomes stay whatever the caller set.
        """
        if self._source is not None:
            return self._source
        if self._failed:
            return None
        try:
            from world.vanilla.biome import biome_source

            table = self.biome_table()
            self._source = biome_source(self.preset, self.seed, table.ids)
        except Exception as error:  # pragma: no cover - only a broken install
            logger.warning(
                f"Vanilla biome source unavailable, chunks ship as plains: {error}")
            self._failed = True
            self._source = None
        return self._source

    def apply(self, chunk: Chunk) -> None:
        """Fill `Chunk.biomes` from the multi-noise source.

        The chunk's section `j` holds world Y `chunk.min_y + j * 16`, which is
        also how `chunk_codec` addresses the wire, so the window is
        `(chunk.min_y, 16 * len(sections))`. The sampler is a batched call over
        the whole 4x4x(16/4) grid: one call per chunk, because the tree keeps a
        last-result cache across samples and splitting the batch changes who
        wins a tie.
        """
        source = self.source()
        if source is None:
            return
        height = 16 * len(chunk.sections)
        volume = source.fill_volume(chunk.position.x, chunk.position.z,
                                    chunk.min_y, height)
        chunk.set_biome_volume(volume)
