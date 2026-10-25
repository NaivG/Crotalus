"""Acceptance check: no decoration-layer plant is left without support.

    python tools/check_plant_support.py [--chunks "cx,cz ..."] [--threads N] [--seed S]

The defect this exists for
--------------------------
`_place_simple_block` lands wherever a placed feature's modifier chain puts it,
and vanilla relies on the state's own `canSurvive` to reject a position with no
soil under it. Crotalus's `_can_place_at` answered True for every block except
saplings, sugar cane and firefly bushes, so `minecraft:grass` -- placed by
`patch_grass_*` at the `WORLD_SURFACE_WG` heightmap, which counts water -- was
written on top of the ocean and into open air. On seed 0 chunk (0,0) that was
36 of 37 plants.

What this asserts
-----------------
Every plant-family block in a generated chunk must rest on a substrate that is
neither air nor a fluid. Hanging blocks (spore blossom, cave vines, vines,
glow lichen, hanging roots) are excluded: having nothing below them is correct.
Stalk blocks count once per column root, not once per segment. Non-plant blocks
with no collision (`resin_clump`, `redstone_torch`) are *not* counted, which is
what a naive "no collision shape" scan gets wrong.

    RESULT: PASS   every plant rests on a non-air, non-fluid block
    RESULT: FAIL   n plants have air or a fluid directly below

Run it after any change to `world/vanilla/features.py`, `assets/tagged` data or
the configured-feature tables. It is a property check, not a golden-output
comparison, so it stays meaningful when a legitimate change moves decoration.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

logger.remove()

DEFAULT_SEED = 0
DEFAULT_PRESET = "overworld"

#: Chosen for habitat coverage rather than adjacency: ocean and river shore,
#: plains and forest, swamp, badlands, frozen ocean, and far-away coordinates.
DEFAULT_CHUNKS = [
    (0, 0), (2, 0), (16, -3), (128, 64), (60, 60), (-320, 0), (3, -2),
    (-5, 7), (12, -8), (-1, -61), (64, 64), (-96, 96), (200, 8), (-200, -8),
    (1, -56), (100, 100), (0, -320), (96, 352),
]

_AIR = {"air", "cave_air", "void_air"}
_FLUID = {"water", "lava", "bubble_column", "flowing_water", "flowing_lava"}

#: Blocks that are *supposed* to have nothing under them.
_HANGING = {
    "spore_blossom", "cave_vines", "cave_vines_plant", "weeping_vines",
    "weeping_vines_plant", "twisting_vines", "twisting_vines_plant",
    "glow_lichen", "vine", "sculk_vein", "hanging_roots", "pale_hanging_moss",
}

#: Not checked, and why. A support assertion is only meaningful for a block
#: vanilla grows *on top of* something.
#:   fire / soul_fire -- placed on a flammable neighbour, not a floor.
#:   melon / pumpkin -- `_plant_rule` leaves them unmodelled on purpose; the
#:     engine places them anywhere, so failing this check would report a gap
#:     that is deliberate rather than a floater.
_NOT_SUPPORTED_BLOCKS = {"fire", "soul_fire", "melon", "pumpkin"}

#: Ground cover that is placed against a face as often as on a floor
#: (`vegetation_patch` writes moss on lush-cave ceilings *and* floors), so it
#: counts as supported when either neighbour in the column is non-air.
_FACE_ATTACHED = {"moss_block", "pale_moss_block"}

#: Stalk blocks whose segments stack, so only the bottom segment is a support
#: question.
_STALKS = {"bamboo", "sugar_cane", "cactus", "kelp", "kelp_plant"}

#: Every block the worldgen data writes through `simple_block`, plus the other
#: plants the feature pass places that a support check applies to. Derived from
#: `assets/configured_features.json` (see `world/vanilla/features.py`
#: `_plant_rule`), not from memory: anything absent here is not checked.
_PLANTS = {
    "short_grass", "tall_grass", "fern", "large_fern", "bush",
    "short_dry_grass", "tall_dry_grass", "dead_bush", "firefly_bush",
    "sweet_berry_bush", "azalea", "flowering_azalea",
    "dandelion", "poppy", "blue_orchid", "allium", "azure_bluet",
    "red_tulip", "orange_tulip", "white_tulip", "pink_tulip", "oxeye_daisy",
    "cornflower", "lily_of_the_valley", "open_eyeblossom", "closed_eyeblossom",
    "lilac", "rose_bush", "peony", "sunflower", "pitcher_plant", "torchflower",
    "pink_petals", "wildflowers", "leaf_litter", "lily_pad",
    "brown_mushroom", "red_mushroom", "crimson_roots", "warped_roots",
    "nether_sprouts", "moss_carpet", "pale_moss_carpet", "small_dripleaf",
    "cactus", "sugar_cane", "bamboo",
    "oak_sapling", "birch_sapling", "spruce_sapling", "jungle_sapling",
    "acacia_sapling", "dark_oak_sapling", "cherry_sapling", "pale_oak_sapling",
    "mangrove_propagule", "melon", "pumpkin", "melon_stem", "pumpkin_stem",
} - _NOT_SUPPORTED_BLOCKS


def census(coord, seed: int, preset: str):
    from world.chunk import Chunk, ChunkPos
    from world.generator import GeneratorSettings, create_generator
    from world.vanilla import features as vf

    cx, cz = coord
    gen = create_generator(GeneratorSettings(seed=seed, preset=preset))
    gen.set_seed(seed)
    chunk = Chunk(ChunkPos(cx, cz))
    gen.generate_chunk(chunk)

    blocks = chunk.blocks_view()
    min_y, height = chunk.min_y, 16 * len(chunk.sections)
    facts = vf.block_facts()
    names = facts.names

    plants = 0
    floating = Counter()
    supports = Counter()
    for y in range(min_y + 1, min_y + height):
        layer = blocks[y - min_y]
        below_layer = blocks[y - 1 - min_y]
        for z in range(16):
            for x in range(16):
                name = names[int(layer[z, x])].removeprefix("minecraft:")
                if name not in _PLANTS or name in _HANGING:
                    continue
                plant_y = y
                if name in _STALKS:
                    # Only the root of a stalk is a support question; the
                    # segments above it rest on the stalk.
                    while plant_y - 1 >= min_y and names[
                            int(blocks[plant_y - 1 - min_y, z, x])
                    ].removeprefix("minecraft:") == name:
                        plant_y -= 1
                plants += 1
                below = names[int(blocks[plant_y - 1 - min_y, z, x])].removeprefix(
                    "minecraft:") if plant_y - 1 >= min_y else "bedrock"
                supports[below] += 1
                if below not in _AIR and below not in _FLUID:
                    continue
                if name in _FACE_ATTACHED:
                    # Attached to a face: supported if the block above is solid
                    # ground (a ceiling) rather than open space or another plant.
                    above_state = (int(blocks[plant_y + 1 - min_y, z, x])
                                   if plant_y + 1 < min_y + height else 0)
                    if not facts.is_air(above_state) and \
                            not facts.blocks_motion(above_state):
                        continue
                floating[name] += 1
    return coord, plants, floating, supports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", default=None,
                        help='space-separated "cx,cz" list; default is the built-in corpus')
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--preset", default=DEFAULT_PRESET)
    args = parser.parse_args()

    if args.chunks:
        chunks = []
        for item in args.chunks.replace(",", " ").split():
            pass
        pairs = [p for p in args.chunks.split() if p]
        chunks = [(int(p.split(",")[0]), int(p.split(",")[1])) for p in pairs]
    else:
        chunks = DEFAULT_CHUNKS

    print(f"check_plant_support  preset={args.preset} seed={args.seed} "
          f"chunks={len(chunks)} threads={args.threads}")
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        results = list(pool.map(lambda c: census(c, args.seed, args.preset), chunks))

    total_plants = 0
    total_floating = Counter()
    total_supports = Counter()
    print(f"\n  {'chunk':>14s} {'plants':>8s} {'floating':>9s}")
    for coord, plants, floating, supports in results:
        total_plants += plants
        total_floating.update(floating)
        total_supports.update(supports)
        print(f"  {str(coord):>12s} {plants:8d} {sum(floating.values()):9d}")

    print(f"\n  plants {total_plants}  floating {sum(total_floating.values())}")
    print(f"  substrates: "
          f"{ {k: v for k, v in total_supports.most_common(10)} }")

    failures = sum(total_floating.values())
    if failures:
        print(f"\n  floating by block: {dict(total_floating.most_common())}")
        print(f"\nRESULT: FAIL ({failures} plant(s) with air or fluid directly below)")
        sys.exit(1)
    if total_plants == 0:
        print("\nRESULT: FAIL (no plants placed at all -- the gate may reject everything)")
        sys.exit(1)
    print("\nRESULT: PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
