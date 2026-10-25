"""
Surface rules: what the top of the terrain is made of.

Drives `world/vanilla/surface.py`. This is the pass that turns a grey stone
mass into a Minecraft one - grass and dirt over stone, sand at every shore,
podzol in the taiga, the badlands' terracotta banding, snow above the freeze
line, and the bedrock floor at the bottom of the world.

It also owns the bedrock. Nothing else writes `vertical_gradient`'s
`above_bottom: 0` band, so a world generated with this layer off has a bare
stone floor. That is why neither terrain backend claims bedrock itself: vanilla
leaves the world's bottom row as stone at the noise stage, exactly as
`NoiseBasedChunkGenerator` does, and lets the rules here dress it.
"""

from typing import Optional

from loguru import logger

from world.chunk import Chunk


class SurfaceLayer:
    """The surface pass. Built once per generator, reused for every chunk."""

    def __init__(self, preset: Optional[str], seed: int) -> None:
        self.preset = preset
        self.seed = seed
        self._system = None
        self._failed = False
        self._table = None
        #: The vertical window the rules were built for. The first chunk sets
        #: it, and every later chunk in the dimension has the same one -- a
        #: chunk spans the whole dimension, so there is only ever one.
        self._window: Optional[tuple[int, int]] = None

    def system(self, window_min_y: int, window_height: int):
        """The surface pass for this preset, built once and reused."""
        if self._system is not None:
            return self._system
        if self._failed:
            return None
        try:
            from world.vanilla.surface import SurfaceSystem

            if self._table is None:
                from world.vanilla.biome import BiomeTable

                self._table = BiomeTable.from_assets()
            # The array window is the chunk's own vertical window; the rules
            # anchor to the preset's noise shape (the overworld's is the same
            # -64..319, the nether's is shorter than its dimension). Since the
            # fix that chunks span the dimension, a full-window chunk needs no
            # offset between the two.
            self._system = SurfaceSystem.from_assets(
                self.preset, self.seed, biome_table=self._table,
                window_min_y=window_min_y, window_height=window_height)
            self._window = (window_min_y, window_height)
        except Exception as error:  # pragma: no cover - only a broken install
            logger.warning(
                f"Vanilla surface rules unavailable, terrain stays bare: {error}")
            self._failed = True
            self._system = None
        return self._system

    def bind(self, table) -> None:
        """Share the `BiomeTable` the biome pass already loaded."""
        self._table = table

    def apply(self, chunk: Chunk, table=None) -> None:
        """Dress the terrain with `world/vanilla/surface.py`.

        The pass works on a `(height, 16, 16)` `[y, z, x]` view of the chunk
        with row 0 at the chunk's own `min_y`, which is where the sections
        live. Since the chunk spans the dimension, the rules' anchors
        (`above_bottom: 0` is bedrock at the bottom of the world) resolve onto
        rows of the same array - the `above_bottom` bedrock band lands at the
        window floor and is actually placed.
        """
        if self._table is None and table is not None:
            self.bind(table)
        height = 16 * len(chunk.sections)
        system = self.system(chunk.min_y, height)
        if system is None:
            return
        blocks = chunk.blocks_view()
        stats = system.build_surface(
            blocks, chunk.biome_volume(), chunk.position.x, chunk.position.z)
        chunk.commit_blocks_view(blocks)
        chunk.is_dirty = True
        logger.debug(f"Surface chunk {chunk.position}: {stats}")
