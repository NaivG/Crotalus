"""
`world/vanilla/terrain.py` — terrain from Minecraft's density functions.

Every other module in `world/vanilla/` computes something; this one turns that
into a chunk. It is the backend `[world] generator = "vanilla"` selects, and it
exists because until now nothing did: the density interpreter could evaluate
`finalDensity` over a whole chunk in about 140 ms, and the result was thrown
away in favour of a Perlin heightmap. The stone you walk on is that field now.

    from world.vanilla.terrain import VanillaWorldGenerator

`final_density` is a function of position that is **positive inside stone and
negative in air**, so the whole fill is one threshold. Vanilla then dresses it:
`world/feature/` adds the biome volume, the surface rules (which is also what
lays the bedrock floor and turns stone into grass), the carvers and the
features. Nothing here claims a dirt band or a grass block, for the same reason
`NoiseBasedChunkGenerator` does not - the surface rules are the stage that owns
"what the top of the terrain is made of".

**Water is the one place this is an approximation, and it is a deliberate one.**
Vanilla fills water from *aquifers*, which read the router's `fluid_level_*`
and `lava` slots to decide where groundwater goes. Aquifers are not ported. The
rule here is the cheap stand-in: fluid fills each column from just above its
topmost stone up to the preset's sea level. That puts water in the ocean above
the sea floor, which is most of what you see, and leaves enclosed caves as air
rather than flooding them - the obvious alternative, "anything below sea level
that is not stone is water", turns every cave in the world into a flooded one.
A real aquifer would also fill some of those caves, and would not fill others;
this rule fills none of them. `world/DESIGN-density-preset-gaps.md` records
what is still missing.

**Two presets do not reproduce vanilla yet** - the nether comes out nearly
empty and the end comes out as air, for reasons that are about the data and the
interpreter rather than about this file. That document explains both, and
`tools/check_terrain_shape.py` measures them. The overworld, `large_biomes`,
`amplified`, `caves` and `floating_islands` all work.
"""

from typing import Optional

import numpy as np
from loguru import logger

from world.block_states import BLOCKS
from world.chunk import Chunk
from world.generator import BaseGenerator, GeneratorSettings


class VanillaWorldGenerator(BaseGenerator):
    """Terrain from the density router; decoration from `world/feature/`."""

    def __init__(self, settings: Optional[GeneratorSettings] = None, *,
                 apply_registry: bool = True):
        super().__init__(settings, apply_registry=apply_registry)
        #: Built on the first non-flat chunk. Parsing `density_function.json`
        #: costs ~360 ms and 15 slots of tree, so it is per-generator and not
        #: per-chunk; the `_failed` latch keeps a broken install from paying
        #: that 360 ms again for every chunk that follows.
        self._router = None
        self._router_failed = False

    def router(self, preset: Optional[str]):
        """The `NoiseRouter` for this preset, built once and reused.

        Safe to share between chunks only because the whole `generate_chunk`
        body is under the base class's lock: `NoiseRouter` carries per-chunk
        `FlatCache` state, and two threads evaluating against each other's
        caches smears the terrain along Z without raising anything.
        """
        if self._router is not None:
            return self._router
        if self._router_failed:
            return None
        try:
            from world.vanilla.router import NoiseRouter

            self._router = NoiseRouter.from_assets(preset, seed=self._seed)
        except Exception as error:  # pragma: no cover - only a broken install
            logger.warning(
                f"Vanilla density router unavailable, no terrain generated: {error}")
            self._router_failed = True
            self._router = None
        return self._router

    def set_seed(self, seed: int) -> None:
        super().set_seed(seed)
        # The router bakes the seed into every sampler, so a seed change has
        # to throw it away rather than reseed it.
        self._router = None
        self._router_failed = False

    def _generate_terrain(self, chunk: Chunk) -> None:
        """Fill the chunk from `final_density`.

        `NoiseRouter.fill_chunk` evaluates the whole `(height, 16, 16)` volume
        in one batched call and manages the router's chunk state itself, so
        this is a threshold and a write.
        """
        router = self.router(self.settings.preset)
        if router is None:
            return

        min_y = chunk.min_y
        height = 16 * len(chunk.sections)
        density = router.fill_chunk(chunk.position.x, chunk.position.z,
                                    min_y, min_y + height)

        solid = density > 0
        y = np.arange(min_y, min_y + height, dtype=np.int32)[:, None, None]

        # Per column, the world Y of the topmost stone. -1 becomes the row
        # below the window, so a column with no stone at all reads as "ground
        # at the floor" and can still take water - which is what an ocean
        # column with no sea floor under it should look like.
        local = np.arange(height, dtype=np.int32)[:, None, None]
        top_row = np.where(solid, local, -1).max(axis=0)
        column_top = np.where(top_row >= 0, min_y + np.maximum(top_row, 0), min_y - 1)

        # See the module docstring: fluid sits between the ground and the sea
        # level, and nowhere below the ground. That second half is what keeps
        # caves dry.
        sea_level = self.settings.sea_level
        flooded = ((~solid) & (y > column_top[None, :, :]) & (y <= sea_level))

        blocks = chunk.blocks_view()
        np.copyto(blocks, np.uint16(BLOCKS.AIR))
        blocks[solid] = np.uint16(self.settings.default_block)
        blocks[flooded] = np.uint16(self.settings.default_fluid)
        chunk.commit_blocks_view(blocks)

        logger.debug(
            f"Terrain chunk {chunk.position}: "
            f"{float(solid.mean())*100:.1f}% stone, "
            f"{float(flooded.mean())*100:.1f}% {self.settings.default_fluid}"
        )
