"""Prove the vectorised terrain fill matches the loop it replaced.

A terrain regression is nearly invisible in-game -- a lost ore band, a
shifted waterline, one missing bedrock corner -- so this compares the two
implementations block for block instead of eyeballing the result.

The reference loop spells out the *current* spec: the fill spans the chunk's
whole vertical window (the overworld's -64..319), and it claims no bedrock --
the floor band is the surface rules' job (`above_bottom: 0`), which runs after
this pass and is checked separately by check_surface_rules.py.

Run: python tools/check_terrain_equivalence.py
"""

import sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from loguru import logger
logger.remove()

from world.block_states import BLOCKS
from world.legacy.generator import LegacyWorldGenerator
from world.legacy.noise import _perm, generate_heightmap_fast
from world.generator import GeneratorSettings, GeneratorType
from world.world import World

tmp = Path(tempfile.mkdtemp())
# Pinned to the legacy backend. This tool exists to compare the Perlin fill
# against a plain-loop reference, and `world.overworld.generator` is whatever
# `[world] generator` selected - the vanilla one, whose `_generate_terrain`
# builds a completely different shape and has no `noise_scale` to read.
gen = LegacyWorldGenerator(GeneratorSettings(
    generator_type=GeneratorType.NORMAL, seed=99, preset="overworld"))
world = World(name="eq", seed=99, world_dir=tmp / "eq", generator=gen)
sea_level = gen.settings.sea_level
MIN_Y = world.overworld.settings.min_y
HEIGHT = world.overworld.settings.height


def reference(heightmap):
    """The spec as a plain 16x16xwindow loop, returning [y, z, x].

    Row r holds world Y `MIN_Y + r`, matching what the section arrays hold.
    """
    out = np.full((HEIGHT, 16, 16), np.int32(BLOCKS.AIR), dtype=np.int32)
    for local_x in range(16):
        for local_z in range(16):
            surface_height = int(heightmap[local_z, local_x])
            for y in range(MIN_Y, MIN_Y + HEIGHT):
                block_id = BLOCKS.AIR
                if y < surface_height:
                    if y < MIN_Y + 5:
                        block_id = BLOCKS.STONE
                    elif y == surface_height - 1:
                        block_id = BLOCKS.SAND if surface_height <= sea_level else BLOCKS.GRASS
                    elif y >= surface_height - 4:
                        block_id = BLOCKS.DIRT
                    else:
                        block_id = BLOCKS.STONE
                elif y <= sea_level:
                    block_id = BLOCKS.WATER
                out[y - MIN_Y, local_z, local_x] = block_id
    return out


def current(heightmap):
    """What `_generate_terrain` writes, read back the same way."""
    y = np.arange(MIN_Y, MIN_Y + HEIGHT, dtype=np.int32)[:, None, None]
    surface = heightmap.astype(np.int32)[None, :, :]
    solid = y < surface
    is_surface = solid & (y == surface - 1)
    is_low_stone = solid & (y < MIN_Y + 5)
    is_dirt = solid & ~is_surface & ~is_low_stone & (y >= surface - 4)
    is_stone = solid & ~is_surface & ~is_low_stone & ~is_dirt
    is_water = ~solid & (y <= sea_level)
    return np.select(
        [is_low_stone, is_surface, is_dirt, is_stone, is_water],
        [
            np.int32(BLOCKS.STONE),
            np.where(surface <= sea_level,
                     np.int32(BLOCKS.SAND), np.int32(BLOCKS.GRASS)),
            np.int32(BLOCKS.DIRT),
            np.int32(BLOCKS.STONE),
            np.int32(BLOCKS.WATER),
        ],
        default=np.int32(BLOCKS.AIR),
    )


fails = 0
for cx, cz in [(0, 0), (1, 0), (-3, 7), (100, -200), (5000, 5000), (-9999, 12)]:
    hm = generate_heightmap_fast(
        cx, cz, 16, 16,
        gen.settings.noise_scale, gen.settings.noise_octaves,
        gen.settings.noise_persistence, gen.settings.noise_lacunarity,
        float(gen.settings.base_height), float(gen.settings.height_variance), _perm,
    )
    a, b = reference(hm), current(hm)
    same = np.array_equal(a, b)
    fails += not same
    if not same:
        bad = np.argwhere(a != b)
        print(f"chunk ({cx},{cz}): MISMATCH at {len(bad)} cells, e.g. {bad[:5].tolist()}")
        for yy, zz, xx in bad[:5]:
            print(f"   y={yy} z={zz} x={xx}: ref={a[yy,zz,xx]} new={b[yy,zz,xx]}")
    else:
        print(f"chunk ({cx},{cz}): identical ({a.size} cells)")

# And the real path: run _generate_terrain alone and confirm the section
# layout matches. Not generate_chunk -- that also runs ores and caves, which
# rewrite stone the pure-terrain reference knows nothing about.
from world.chunk import ChunkPos
c = world.overworld.chunk_manager.load_chunk(ChunkPos(0, 0))
gen._generate_terrain(c)
ok_layout = True
ref_terrain = reference(generate_heightmap_fast(
    0, 0, 16, 16, gen.settings.noise_scale, gen.settings.noise_octaves,
    gen.settings.noise_persistence, gen.settings.noise_lacunarity,
    float(gen.settings.base_height), float(gen.settings.height_variance), _perm))
for si, section in enumerate(c.sections):
    got = section.blocks.reshape(16, 16, 16)   # [y, z, x]
    want = ref_terrain[si * 16:si * 16 + 16]
    if not np.array_equal(got, want):
        ok_layout = False
        bad = np.argwhere(got != want)
        print(f"section {si}: layout mismatch at {len(bad)} cells, e.g. {bad[:3].tolist()}")
    # non_air_count must survive the wholesale assignment
    expect = int(np.count_nonzero(want))
    if section.non_air_count != expect:
        ok_layout = False
        print(f"section {si}: non_air_count {section.non_air_count} != {expect}")

print("\nRESULT:", "PASS" if fails == 0 and ok_layout else "FAIL")
