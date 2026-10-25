"""
`world/legacy/generator.py` — the self-developed terrain generator.

This is the backend that predates the density-function port, and it is still
worth having: it extrudes a numba Perlin heightmap into stone, which costs
about 20 ms a chunk where the vanilla pipeline costs about a second. On a
large view distance that difference is the difference between a world you can
walk around in and a loading screen.

What it is not is Minecraft. The terrain is a heightmap, not a density
function, so the shapes are its own; the caves are a wandering-tunnel carver of
this project's design; the ore veins are a hardcoded table. Everything *on*
that stone - biomes, surface rules, the vanilla carver, trees, geodes - comes
from `world/feature/` and is identical to what the vanilla backend produces.
That is the whole point of the split: this file decides the shape, and
nothing else.

The knobs are in `GeneratorSettings` (`noise_scale`, `base_height`,
`height_variance`, `generate_ores`) and in `CaveSettings` below. The vanilla
backend ignores every one of them.

    from world.legacy.generator import LegacyWorldGenerator
    generator = LegacyWorldGenerator(GeneratorSettings(seed=1234, preset="overworld"))
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger

from world.block_states import BLOCKS
from world.chunk import Chunk
from world.generator import BaseGenerator, GeneratorSettings
from world.legacy.cave_carver import carve_chunk_caves, carve_cheese
from world.legacy.noise import PerlinNoise, generate_heightmap_fast, init_perlin, _perm


@dataclass
class CaveSettings:
    """Tuning for the tunnel carver in `world/legacy/cave_carver.py`.

    Every field here is a taste knob, not a correctness one -- the carver
    produces connected caves across a wide range of these. The two worth
    understanding before touching anything else are `radius` and
    `cave_count`, because cave density scales with radius^2 * cave_count.
    """
    # Carve window. `max_y` is a hard cap; the effective ceiling is derived
    # from the terrain settings, so this does not carve tunnels through thin
    # air above the surface.
    min_y: int = 5
    max_y: int = 255
    # Headroom above the derived terrain ceiling. Caves are allowed into it
    # because the surface guard is per column and some columns dip well below
    # the chunk's high point.
    max_y_margin: int = 8
    # Start points per 16 x 16 carver chunk. The carver runs over a
    # (2*radius+1)^2 neighbourhood, so 4 lands at roughly 9% of the rock below
    # the surface being opened up, which is about what the vanilla overworld
    # does. This is the dial to turn for "more caves" or "sparser caves";
    # measured means for 3 / 5 / 8 / 12 are 7% / 13% / 22% / 31%.
    cave_count: int = 2
    # How many carver chunks either side of this one start tunnels. Tunnels
    # that could still reach the chunk are the only ones the walk bothers to
    # finish, so a larger radius costs bookkeeping and no real work -- but it
    # must be large enough to not clip a tunnel that *would* have arrived,
    # which needs 16 * radius + 16 <= max_length + 20.
    radius: int = 6
    # Longest tunnel, in blocks. Longer tunnels reach further, so they need a
    # bigger radius to be safe.
    max_length: int = 112
    # Tunnel cross-section, as multipliers on vanilla's own range (a half-width
    # of 0.75-1.25 blocks and a half-height of half that, so tunnels come out
    # roughly 3-6 blocks across and 1.5-3 tall). 1.0 reproduces the reference.
    # This is the dial for "I want to be able to walk/crawl/fly through it".
    tunnel_width: float = 1.0
    tunnel_height: float = 0.8
    # Where tunnel starts are placed. The bottom bias is built into the
    # carver, so `spawn_max_y` is the shallowest a start is likely to land.
    spawn_min_y: int = 8
    spawn_max_y: int = 120
    # Keep the crust this many blocks below the surface. Must not go below
    # the 4-block dirt band the terrain pass leaves, or caves would undercut
    # it and leave dirt hanging over open air.
    surface_margin: int = 5
    # Open chambers on top of the tunnels, from a large-scale noise field.
    cheese: bool = True
    cheese_scale: float = 96.0
    cheese_threshold: float = 0.42


class LegacyWorldGenerator(BaseGenerator):
    """Perlin terrain, its own ore veins, and its own tunnel carver.

    Everything else -- biomes, surface rules, the vanilla carver, features --
    is inherited from `BaseGenerator` and `world/feature/`, unchanged.
    """

    def __init__(self, settings: Optional[GeneratorSettings] = None, *,
                 apply_registry: bool = True):
        super().__init__(settings, apply_registry=apply_registry)
        self.caves = CaveSettings()
        self._init_noise(self._seed)

    def _init_noise(self, seed: int) -> None:
        """(Re)build the Perlin fields this backend shapes terrain with."""
        self._terrain_noise = PerlinNoise(
            seed=seed,
            octaves=self.settings.noise_octaves,
            persistence=self.settings.noise_persistence,
            lacunarity=self.settings.noise_lacunarity,
            scale=self.settings.noise_scale,
        )

    def set_seed(self, seed: int) -> None:
        super().set_seed(seed)
        init_perlin(seed)
        self._init_noise(seed)
        logger.info(f"Generator seed changed to {seed}")

    # -- terrain ------------------------------------------------------------

    def _generate_terrain(self, chunk: Chunk) -> None:
        """Generate normal terrain using noise."""
        pos = chunk.position
        sea_level = self.settings.sea_level
        base_height = self.settings.base_height
        height_variance = self.settings.height_variance

        # Generate heightmap using fast numba function
        heightmap = generate_heightmap_fast(
            pos.x, pos.z, 16, 16,
            self.settings.noise_scale,
            self.settings.noise_octaves,
            self.settings.noise_persistence,
            self.settings.noise_lacunarity,
            float(base_height),
            float(height_variance),
            _perm
        )

        # Fill terrain.
        #
        # This used to be a 16 x 16 x 256 loop calling `chunk.set_block`:
        # ~66k locked, bounds-checked Python calls per chunk, which dominated
        # generation outright (744 ms/chunk -- 650x the cost of *encoding* the
        # finished chunk). Nothing about a column depends on its neighbours,
        # so the whole volume is one broadcast comparison instead of a loop.
        #
        # The fill spans the chunk's whole window (world Y min_y ..
        # min_y+height-1) and claims no bedrock: the bedrock floor is the
        # surface rules' job (`vertical_gradient` with `above_bottom: 0`, which
        # unconditionally rewrites the world's bottom row and statistically
        # the band above it). This matches vanilla's noise stage, which also
        # leaves stone there for the surface pass to dress. A world with the
        # surface pass off gets a bare stone floor - bare Perlin is documented
        # as not-Minecraft anyway.
        #
        # The condition order below mirrors the if/elif chain it replaces, and
        # the order matters: the floor band is claimed first, and the surface
        # block is tested before the dirt band, otherwise `y >= surface - 4`
        # would swallow it.
        min_y = chunk.min_y
        height = 16 * len(chunk.sections)
        y = np.arange(min_y, min_y + height, dtype=np.int32)[:, None, None]
        surface = heightmap.astype(np.int32)[None, :, :]   # [1, 16, 16] over (z, x)
        solid = y < surface
        is_surface = solid & (y == surface - 1)

        is_low_stone = solid & (y < min_y + 5)
        is_dirt = solid & ~is_surface & ~is_low_stone & (y >= surface - 4)
        is_stone = solid & ~is_surface & ~is_low_stone & ~is_dirt
        is_water = ~solid & (y <= sea_level)

        block = np.select(
            [is_low_stone, is_surface, is_dirt, is_stone, is_water],
            [
                np.uint16(self.settings.default_block),
                # Below sea level the top block is sand, above it grass.
                np.where(surface <= sea_level,
                         np.uint16(BLOCKS.SAND), np.uint16(BLOCKS.GRASS)),
                np.uint16(BLOCKS.DIRT),
                np.uint16(self.settings.default_block),
                np.uint16(self.settings.default_fluid),
            ],
            default=np.uint16(BLOCKS.AIR),
        )

        # block is [y, z, x]; ChunkSection.blocks is flat with
        # index = x + z*16 + y*256, which is exactly that C-order flatten.
        for section_index, section in enumerate(chunk.sections):
            start = section_index * 16
            section.blocks[:] = block[start:start + 16].reshape(-1)
            # Wholesale assignment bypasses `set_block`, which is what keeps
            # this count current -- see ChunkSection.recompute_non_air_count.
            section.recompute_non_air_count()
        chunk.is_dirty = True

    # -- ore ----------------------------------------------------------------

    def _generate_ores(self, chunk: Chunk) -> None:
        """Generate ore deposits.

        A vanilla-backend world gets its ore from `world/feature/features.py`
        as `minecraft:ore_vein` configured features, so the base class leaves
        this alone; overriding it here is what keeps the two worlds from
        placing every vein twice.
        """
        import random

        pos = chunk.position
        random.seed(self._seed + pos.x * 31337 + pos.z * 7919)

        ore_veins = {
            BLOCKS.COAL_ORE: (5, 128, 20, 17),
            BLOCKS.IRON_ORE: (5, 64, 20, 9),
            BLOCKS.GOLD_ORE: (5, 32, 2, 9),
            BLOCKS.DIAMOND_ORE: (5, 16, 1, 8),
            BLOCKS.LAPIS_ORE: (5, 32, 1, 8),
            BLOCKS.REDSTONE_ORE: (5, 16, 8, 8),
        }

        # Ore bands are absolute world Y; clamp them into this chunk's window
        # so a dimension that reaches below 0 still gets its band where the
        # data says rather than where a 0..255 column happened to end.
        window_lo = chunk.min_y
        window_hi = chunk.min_y + chunk.height - 1

        for block_id, (min_y, max_y, veins, size) in ore_veins.items():
            for _ in range(veins):
                # Random position for vein center
                center_x = random.randint(0, 15)
                center_y = random.randint(max(window_lo, min_y), min(window_hi, max_y))
                center_z = random.randint(0, 15)

                # Generate vein
                self._generate_ore_vein(chunk, center_x, center_y, center_z, block_id, size)

    def _generate_ore_vein(
        self,
        chunk: Chunk,
        cx: int,
        cy: int,
        cz: int,
        block_id: int,
        size: int
    ) -> None:
        """Generate a single ore vein using random walk."""
        import random

        random.seed(hash((cx, cy, cz, self._seed)))

        # Walk the whole path first, touch the chunk afterwards.
        #
        # The path depends only on this vein's own seed, never on what is
        # already in the chunk -- the `== STONE` test only decides whether a
        # step is *written*. Reading and writing per step therefore bought
        # nothing and cost a locked `set_block` (plus four IntEnum lookups)
        # on every step, which made this the most expensive part of
        # generation by a wide margin.
        #
        # "Stone" here is the dimension's own default block, so a vein in the
        # nether replaces netherrack rather than leaving a stone seam.
        path: List[Tuple[int, int, int]] = []
        x, y, z = cx, cy, cz

        for _ in range(size):
            path.append((x, y, z))
            x += random.randint(-1, 1)
            y += random.randint(-1, 1)
            z += random.randint(-1, 1)

        placed = False
        # Flat index is x + z*16 + y*256, i.e. [y, z, x] in C order, so
        # reshape hands back a writable view of the same buffer.
        views = [s.blocks.reshape(16, 16, 16) for s in chunk.sections]
        host = np.uint16(self.settings.default_block)
        window_lo = chunk.min_y
        window_hi = chunk.min_y + chunk.height
        for x, y, z in path:
            if not (0 <= x < 16 and window_lo <= y < window_hi and 0 <= z < 16):
                continue
            offset = y - chunk.min_y
            cell = views[offset // 16][offset % 16, z, x]
            # Only replace the default block.
            if cell == host:
                views[offset // 16][offset % 16, z, x] = np.uint16(block_id)
                placed = True

        if placed:
            # Swapping stone for ore leaves the non-air count untouched --
            # both sides are solid -- so there is nothing to recompute here.
            chunk.is_dirty = True

    # -- caves --------------------------------------------------------------

    def _generate_caves(self, chunk: Chunk, vanilla: bool) -> None:
        """Pick a carver, then carve.

        `cave_mode` is honoured here and nowhere else. `"vanilla"` uses the
        ported `world/vanilla/carver.py`, which reproduces Minecraft's tunnel
        geometry but has two unresolved questions recorded in its docstring;
        `"legacy"` is this package's own wandering-tunnel carver, which is
        fast, connected and has nothing to do with vanilla; `"none"` carves
        nothing. A generator with no preset has no carver data to read, so it
        falls back to the legacy carver rather than doing nothing.
        """
        if vanilla and self.settings.cave_mode == "vanilla":
            super()._generate_caves(chunk, vanilla)
            return
        if self.settings.cave_mode == "legacy" or not vanilla:
            self._carve_legacy(chunk)

    def _carve_legacy(self, chunk: Chunk) -> None:
        """Carve connected cave systems: wandering tunnels plus open chambers.

        The carve window stops just above the terrain's own ceiling, and the
        ceiling is derived from settings rather than from this chunk's
        heightmap. That is a correctness constraint, not an optimisation:
        tunnel starts are drawn from a y range, so a per-chunk heightmap would
        make neighbouring chunks choose *different* y for the same start and
        the caves would break along the border. The per-column surface guard
        below has no such problem, because a column's surface height is a
        function of world position and both neighbours agree on it.
        """
        c = self.caves
        pos = chunk.position

        # Window the carve to roughly where the terrain actually reaches.
        ceiling = min(
            c.max_y,
            self.settings.base_height
            + int(self.settings.height_variance * 0.4)
            + c.max_y_margin,
        )

        y0 = c.min_y
        y1 = ceiling
        if y1 - y0 < 16:
            return

        spawn_min = min(c.spawn_min_y, y1 - 16)
        spawn_max = min(c.spawn_max_y, y1)

        mask = np.zeros((y1 - y0 + 1, 16, 16), dtype=np.uint8)

        carve_chunk_caves(
            mask, y0, y1, pos.x, pos.z, self._seed,
            radius=c.radius, cave_count=c.cave_count, max_length=c.max_length,
            y_min=spawn_min, y_max=spawn_max,
            hr_mult=c.tunnel_width, vr_mult=c.tunnel_height,
        )

        if c.cheese:
            carve_cheese(
                mask, y0, y1, pos.x, pos.z, self._seed + 3,
                scale=c.cheese_scale, threshold=c.cheese_threshold,
            )

        # Find each column's topmost solid block, top-down, so the surface
        # guard below knows where the ground is. Columns still unresolved
        # when the scan runs out of sections keep the value they started with,
        # which is the window floor -- a column of pure air has no ground to
        # protect, and carving under one is harmless.
        rock = np.zeros((16, 16), dtype=np.int32)
        unresolved = np.ones((16, 16), dtype=bool)
        local_y = np.arange(16, dtype=np.int32)[:, None, None]
        fluid = np.uint16(self.settings.default_fluid)

        lo_sec = max(0, (y0 - chunk.min_y) // 16)
        hi_sec = min(len(chunk.sections) - 1, (y1 - chunk.min_y) // 16)
        for section_index in range(hi_sec, lo_sec - 1, -1):
            if not unresolved.any():
                break
            view = chunk.sections[section_index].blocks.reshape(16, 16, 16)
            solid = (view != np.uint16(BLOCKS.AIR)) & (view != fluid)
            hit = solid.any(axis=0) & unresolved
            if not hit.any():
                continue
            top = np.where(solid, local_y, np.int32(-1)).max(axis=0)
            rock[hit] = chunk.min_y + section_index * 16 + top[hit]
            unresolved &= ~hit

        ys = np.arange(y0, y1 + 1, dtype=np.int32)[:, None, None]

        # Keep the crust: nothing within `surface_margin` of the ground opens.
        mask &= ys < rock[None, :, :] - c.surface_margin

        rows = np.nonzero(mask.any(axis=(1, 2)))[0]
        if rows.size == 0:
            return

        world_rows = rows + y0
        chunk_rows = world_rows - chunk.min_y

        for section_index in np.unique(chunk_rows // 16):
            selected = chunk_rows // 16 == section_index

            mask_rows = rows[selected]
            block_rows = chunk_rows[selected] % 16

            view = chunk.sections[int(section_index)].blocks.reshape(16, 16, 16)
            blocks = view[block_rows]

            # Only the dimension's own stone opens; a carve that lands on a
            # block the surface rules already dressed is left alone.
            hit = blocks == np.uint16(self.settings.default_block)
            hit &= mask[mask_rows].astype(bool)
            if not hit.any():
                continue

            blocks[hit] = np.uint16(BLOCKS.AIR)
            view[block_rows] = blocks

            chunk.sections[int(section_index)].recompute_non_air_count()

        chunk.is_dirty = True

    def generate_spawn_point(self) -> tuple[int, int, int]:
        """Spawn from the Perlin field, which is what this terrain is made of.

        The live spawn path is `World.get_spawn_point`, which reads a real
        generated chunk; this is the answer for a caller that wants a position
        before any chunk exists.
        """
        import random
        random.seed(self._seed)

        # Find a suitable spawn point
        for _ in range(100):
            x = random.randint(-100, 100)
            z = random.randint(-100, 100)

            # Get height at this position
            noise_val = self._terrain_noise.noise2d(x, z)
            y = int((noise_val + 1) * 0.5 * self.settings.height_variance + self.settings.base_height)

            if y > self.settings.sea_level:
                return (x, y + 1, z)

        # Default spawn
        return (0, self.settings.sea_level + 1, 0)


class TerrainFeature:
    """Base class for terrain features."""

    def generate(self, chunk: Chunk, seed: int) -> None:
        """Generate the feature in a chunk."""
        raise NotImplementedError


class TreeFeature(TerrainFeature):
    """Tree generation feature."""

    def generate(self, chunk: Chunk, seed: int) -> None:
        """Generate trees in a chunk."""
        import random
        random.seed(seed + chunk.position.x * 12345 + chunk.position.z * 67890)

        if random.random() > 0.1:  # 10% chance per chunk
            return

        for _ in range(random.randint(1, 3)):
            x = random.randint(4, 11)
            z = random.randint(4, 11)
            height = random.randint(4, 6)

            # Check if can place tree
            surface = chunk.heightmap[x, z]
            if surface < 64:  # Don't place below sea level
                continue

            # Check space for tree
            can_place = True
            top_y = chunk.min_y + chunk.height
            for dx in range(-1, 2):
                for dz in range(-1, 2):
                    for dy in range(height - 2, height + 1):
                        bx, by, bz = x + dx, surface + dy, z + dz
                        if 0 <= bx < 16 and 0 <= bz < 16 and by < top_y:
                            block = chunk.get_block(bx, by, bz)
                            if block not in (BLOCKS.AIR, BLOCKS.LEAVES):
                                can_place = False
                                break

            if can_place:
                self._place_tree(chunk, x, surface, z, height)

    def _place_tree(
        self,
        chunk: Chunk,
        x: int,
        base_y: int,
        z: int,
        height: int
    ) -> None:
        """Place a tree at the given position."""
        # Trunk
        for y in range(base_y + 1, base_y + height + 1):
            chunk.set_block(x, y, z, BLOCKS.LOG)

        # Leaves
        for dx in range(-2, 3):
            for dz in range(-2, 3):
                for dy in range(height - 1, height + 2):
                    if abs(dx) == 2 and abs(dz) == 2:
                        continue
                    bx, by, bz = x + dx, base_y + dy, z + dz
                    if 0 <= bx < 16 and 0 <= bz < 16 and by < chunk.min_y + chunk.height:
                        if chunk.get_block(bx, by, bz) == BLOCKS.AIR:
                            chunk.set_block(bx, by, bz, BLOCKS.LEAVES)


class FlowerFeature(TerrainFeature):
    """Flower and tall grass generation."""

    def generate(self, chunk: Chunk, seed: int) -> None:
        """Generate flowers and grass in a chunk."""
        import random
        random.seed(seed + chunk.position.x * 111 + chunk.position.z * 222)

        for _ in range(100):
            x = random.randint(0, 15)
            z = random.randint(0, 15)
            surface = chunk.heightmap[x, z]

            if surface < 64 or surface >= chunk.min_y + chunk.height:
                continue

            block = chunk.get_block(x, surface, z)
            if block == BLOCKS.GRASS and random.random() < 0.8:
                chunk.set_block(x, surface + 1, z, BLOCKS.AIR)
