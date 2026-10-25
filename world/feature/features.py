"""
Decorated features: trees, grass, ore, lakes, geodes, monster rooms.

Drives `world/vanilla/features.py` - the biome's per-step feature lists, placed
with vanilla's fifteen placement modifiers. This is the stage that makes a world
look inhabited rather than merely correct.

It is also where **ores come from** on a vanilla-backend world. The legacy
backend places them itself with `_generate_ores`; here they are
`minecraft:ore_vein` configured features in the biome's list. Turning this
layer on while also running the legacy ore pass would place every vein twice,
which is why `BaseGenerator._generate_ores` is a no-op hook that only the
legacy backend fills in.

Two thirds of the configured-feature types the data references are implemented;
the rest are enumerated by `world.vanilla.features.missing_types()` rather than
silently skipped.
"""

from loguru import logger

from world.chunk import Chunk


class FeatureLayer:
    """The feature pass. Built once per generator, reused for every chunk."""

    def __init__(self, preset, seed: int) -> None:
        self.preset = preset
        self.seed = seed
        self._data = None
        self._failed = False

    def apply(self, chunk: Chunk, table=None) -> None:
        """Decorate with `world/vanilla/features.py`.

        This reads and writes only this chunk, so a feature that spans a border
        is clipped and the clipped writes are *counted* in
        `FeatureStats.writes_dropped` rather than quietly lost - the server
        streams chunks independently and has no neighbour to hand a straddling
        tree to.
        """
        try:
            from world.vanilla import features as vanilla_features
        except Exception as error:  # pragma: no cover - only a broken install
            logger.warning(
                f"Vanilla features unavailable, nothing decorated: {error}")
            return

        if self._data is None:
            try:
                if table is None:
                    from world.vanilla.biome import BiomeTable

                    table = BiomeTable.from_assets()
                self._data = (
                    vanilla_features.load_placed_features(),
                    vanilla_features.load_configured_features(),
                )
            except Exception as error:  # pragma: no cover - only a broken install
                logger.warning(
                    f"Vanilla feature data unavailable, nothing decorated: {error}")
                self._data = (None, None)
                self._failed = True
                return

        placed, configured = self._data
        blocks = chunk.blocks_view()
        stats = vanilla_features.decorate(
            chunk.position.x, chunk.position.z, blocks, chunk.biome_volume(),
            preset=self.preset,
            seed=self.seed,
            min_y=chunk.min_y,
            height=16 * len(chunk.sections),
            biome_table=table,
            placed_features=placed,
            configured_features=configured,
            heightmap=None,
        )
        chunk.commit_blocks_view(blocks)
        chunk.is_dirty = True
        logger.debug(f"Features chunk {chunk.position}: {stats}")
