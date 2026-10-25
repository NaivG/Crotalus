"""
Caves and canyons: `world/vanilla/carver.py`.

This is the one layer with no oracle. Pumpkin's test data covers only the noise
and surface stages, so unlike everything else here there is no vanilla dump to
compare against - the module reproduces the tunnel *geometry* (the start
placement, the random walk, the ellipsoid chain) but two questions about
Minecraft's carvers are recorded unresolved in its own docstring. What it does
do is walk a 17x17 chunk neighbourhood, so the same tunnel is carved from
whichever chunk asks for it first and the seams line up.

The legacy alternative is `world/legacy/cave_carver.py`, a wandering-tunnel
carver of this project's own design: fast, connected, and nothing to do with
vanilla. `[world] generator` picks the terrain; `cave_mode` picks the carver.
"""

import numpy as np
from loguru import logger

from world.chunk import Chunk


class CarverLayer:
    """The carver pass. Built once per generator, reused for every chunk."""

    def __init__(self, preset, seed: int) -> None:
        self.preset = preset
        self.seed = seed
        self._carver_data = None
        self._tag_cache = None
        self._failed = False

    def tags(self):
        """The `#minecraft:*_carver_replaceables` tags, resolved once."""
        if self._tag_cache is None:
            try:
                from registry import get_game_registry

                self._tag_cache = get_game_registry().tags
            except Exception:  # pragma: no cover - only a broken install
                self._tag_cache = None
        return self._tag_cache

    def apply(self, chunk: Chunk, table=None, sea_level: int = 63) -> None:
        """Carve with `world/vanilla/carver.py`.

        That module walks a 17x17 chunk neighbourhood per chunk, so the same
        tunnel is carved from whichever chunk asks for it first and the seams
        line up. Here it only ever sees this chunk: `neighbour_blocks` is left
        out and every column outside reads as air, which means a tunnel started
        next door cannot *continue* into this one - it is carved only by the
        chunks around it, when they generate. The carver counts what it drops;
        see `CarveStats`.

        `sea_level` decides what counts as underwater while the carver walks,
        so it has to be the *dimension's* level - the nether's is 32, not the
        overworld's 63 - which is why the caller passes it in rather than this
        module looking it up.
        """
        try:
            from world.vanilla import carver as vanilla_carver
        except Exception as error:  # pragma: no cover - only a broken install
            logger.warning(f"Vanilla carvers unavailable, no caves carved: {error}")
            return

        if self._carver_data is None:
            try:
                if table is None:
                    from world.vanilla.biome import BiomeTable

                    table = BiomeTable.from_assets()
                self._carver_data = vanilla_carver.load_carvers()
            except Exception as error:  # pragma: no cover - only a broken install
                logger.warning(f"Vanilla carver data unavailable, no caves carved: {error}")
                self._carver_data = {}
                self._failed = True
                return

        volume = chunk.biome_volume()
        biome_carvers = {
            int(biome_id): vanilla_carver.carvers_for_biome(int(biome_id), table)
            for biome_id in np.unique(volume)
        }
        blocks = chunk.blocks_view()
        stats = vanilla_carver.carve(
            chunk.position.x, chunk.position.z, blocks, volume,
            preset=self.preset,
            seed=self.seed,
            min_y=chunk.min_y,
            height=16 * len(chunk.sections),
            biome_carvers=biome_carvers,
            carver_data=self._carver_data,
            tags=self.tags(),
            sea_level=sea_level,
        )
        chunk.commit_blocks_view(blocks)
        chunk.is_dirty = True
        logger.debug(f"Carvers chunk {chunk.position}: {stats}")
