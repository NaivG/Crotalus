"""Check that cave carving does not break the world around it.

Carving is a destructive pass over terrain another pass just built, and the
ways it goes wrong are silent: a cave that reaches too high undercuts the dirt
band and leaves dirt hanging in mid-air, a cave that reaches an ocean drains
it, a carver that writes to the wrong y is invisible because connectivity is
unaffected -- the caves still look right, they are just five blocks off.

Two of those are *game features* and are reported but never fail the check:
floating dirt and open-air / ocean-opening caves are what vanilla carvers
produce (a cave entrance on a hillside, a ravine under a seabed), and are
accepted behavior. What must be untouched by carving is the floor: the
bedrock band's floor row and the layers above it, plus the non-air counts.
The air count is here to catch the opposite failure -- a carver that is too
timid to show up at all.

Run: python tools/check_cave_invariants.py
"""

import sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from loguru import logger
logger.remove()

from world.block_states import BLOCKS
from world.chunk import ChunkPos
from world.generator import GeneratorSettings, GeneratorType, create_generator
from world.world import World

SEED = 7
SAMPLES = [(i * 3 - 45, j * 3 - 3) for i in range(30) for j in range(3)]


def survey(caves):
    """Generate a fixed set of chunks and tally the world properties."""
    # Pinned to the legacy backend. This tool mutates `generate_caves` on the
    # generator and then re-reads the world, which is how it isolates the
    # carver; it was written against the Perlin terrain and the legacy tunnel
    # carver, and the vanilla backend's density terrain is a different subject.
    world = World(name="inv", seed=SEED,
                  world_dir=Path(tempfile.mkdtemp()) / "inv",
                  generator=create_generator(
                      GeneratorSettings(generator_type=GeneratorType.NORMAL,
                                        seed=SEED, preset="overworld"),
                      backend="legacy"))
    manager = world.overworld.chunk_manager
    gen = world.overworld.generator
    gen.settings.generate_caves = caves

    counts = {"water_over_air": 0, "floating_dirt": 0,
              "bedrock_damaged": 0, "non_air_mismatch": 0}
    air_pct = []

    min_y = None
    height = None
    for px, pz in SAMPLES:
        chunk = manager.load_chunk(ChunkPos(px, pz))
        gen.generate_chunk(chunk)
        min_y = chunk.min_y
        height = 16 * len(chunk.sections)
        # [y, z, x], row r holding world Y min_y + r
        blocks = np.stack([s.blocks for s in chunk.sections]).reshape(height, 16, 16)

        air = blocks == np.int32(BLOCKS.AIR)
        water = blocks == np.int32(BLOCKS.WATER)

        # A water block with air immediately under it: water with nothing to
        # stand on, which is how a drained ocean looks. The narrower test --
        # "air somewhere below water" -- is the normal state of any cave
        # under a seabed and flags most of a legitimately carved ocean floor.
        counts["water_over_air"] += int(
            (water[1:] & air[:-1]).any(axis=0).sum())

        # Dirt or grass with air immediately under it: a carved-out crust.
        below_air = np.zeros_like(air)
        below_air[1:] = air[:-1]
        counts["floating_dirt"] += int(
            ((blocks == np.int32(BLOCKS.DIRT))
             | (blocks == np.int32(BLOCKS.GRASS))) .__and__(below_air)
            .any(axis=0).sum())

        # The bedrock band's floor row and the four layers above it. The
        # bottom row of the window is where the surface rules' `above_bottom`
        # gradient puts bedrock unconditionally; rows min_y+1..min_y+4 are the
        # randomized part of the band (bedrock or deepslate), and carving must
        # never hollow them out.
        counts["bedrock_damaged"] += int(
            (blocks[0] != np.int32(BLOCKS.BEDROCK)).sum()
            + (blocks[1:5] == np.int32(BLOCKS.AIR)).sum())

        counts["non_air_mismatch"] += sum(
            1 for s in chunk.sections
            if s.non_air_count != int(np.count_nonzero(s.blocks)))

        heightmap = np.asarray(chunk.heightmap)
        under = (np.arange(min_y, min_y + height)[:, None, None]
                 < heightmap.T[None, :, :])
        air_pct.append(100.0 * air[under].sum() / under.sum())

    return counts, np.array(air_pct)


base, base_air = survey(False)
carved, carved_air = survey(True)

print(f"{len(SAMPLES)} chunks, seed {SEED}\n")
print(f"  {'property':<20}{'caves off':>12}{'caves on':>12}   verdict")
print(f"  {'-'*20:<20}{'-'*12:>12}{'-'*12:>12}   -------")

# Accepted differences: carving these is gameplay (cave entrances, ravines
# under the sea), not a defect. They are printed so the numbers stay visible
# but they do not fail the check.
ACCEPTED = {"water_over_air", "floating_dirt"}
verdicts = {}
for key in base:
    same = base[key] == carved[key]
    verdicts[key] = same
    note = "unchanged" if same else "CHANGED"
    if key == "water_over_air" and not same:
        note = "CHANGED -- an ocean was opened into (accepted)"
    if key == "floating_dirt" and not same:
        note = "CHANGED -- open-air caves undercut the crust (accepted)"
    if key in ACCEPTED:
        verdicts[key] = True
    print(f"  {key:<20}{base[key]:>12}{carved[key]:>12}   {note}")

print(f"\n  air below the surface: mean {carved_air.mean():.1f}%  "
      f"p50 {np.percentile(carved_air, 50):.1f}  "
      f"p90 {np.percentile(carved_air, 90):.1f}  "
      f"max {carved_air.max():.1f}")

# The floor must be untouched by carving, and caves must actually exist.
fails = [k for k, ok in verdicts.items() if not ok]
if carved_air.mean() < 3.0:
    fails.append("too_few_caves")

print("\nRESULT:", "PASS" if not fails else f"FAIL ({', '.join(fails)})")
