"""
Surface-rule acceptance: does the pass still dress terrain the way vanilla does?

    python tools/check_surface_rules.py
    python tools/check_surface_rules.py --chunk overworld_0_0

What this guards
----------------
Surface rules fail *quietly and diffusely*: a wrong `stone_depth` threshold moves
the dirt band four blocks, a wrong `random_name` hash scatters bedrock through
the deepslate, a wrong `run_depth` rounds a band away. None of it raises, and all
of it is invisible in a screenshot. The only test worth having is a cell-for-cell
comparison against a chunk the real game generated.

`assets/vanilla_chunks/` holds those chunks. Each file carries the same chunk
twice: `before` is the noise stage and `after` is noise plus surface, so a
mismatch is attributable to *this* pass rather than to the terrain that fed it.

The layers, strongest first
---------------------------
1. **The primitive recipes.** `Mth.getSeed` (the position hash), the `md5` byte
   order for `random_name`, and simplex noise. These come from vectors checked
   against a real Minecraft, and everything below them is meaningless if they
   are wrong - so they run first and short-circuit nothing but themselves.

2. **The vanilla chunks.** Seven chunks across three dimensions and two seeds,
   compared cell for cell. This is the layer that says whether the pass is
   *right* rather than merely stable.

3. **Determinism and shape.** Running the same chunk twice must give the same
   bytes, the pass must not touch anything but the preset's default block where
   the rules say it may, and the biome volume must not be mutated.

A note on the residuals
-----------------------
`RESIDUALS` below records the mismatch count each chunk is currently allowed.
Two of the seven are exact. The rest are not, and the numbers are here rather
than in a commit message so that a regression is visible as a *number going up*
even in a chunk that was already imperfect. Do not raise a ceiling without
saying which cells moved and why.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402

logger.remove()

from world.vanilla import biome as biome_module  # noqa: E402
from world.vanilla.random_source import _hash_block_pos  # noqa: E402
from world.vanilla.simplex import OctaveSimplexNoiseSampler, SimplexNoise  # noqa: E402
from world.vanilla.surface import (  # noqa: E402
    _NO_FLUID,
    SurfaceContext,
    SurfaceSystem,
    name_hash,
    world_surface_heightmap,
)
from world.vanilla.random_source import LegacyRandom, Xoroshiro  # noqa: E402

CHUNK_DIR = ROOT / "assets" / "vanilla_chunks"

#: Mismatch ceilings per chunk, out of 98304 cells unless the chunk is 128 tall.
#: `None` means the chunk has not been measured yet and any mismatch fails.
RESIDUALS: Dict[str, Optional[int]] = {
    "overworld_0_0": 0,
    "overworld_7_4": 1203,
    "overworld_13579_-2_15": 4067,
    "overworld_badlands_-595_544": 4700,
    "overworld_frozen_ocean_-119_183": 7,
    "nether_0_0": 803,
    "end_0_0": 0,
}

failures: List[str] = []


def check(label: str, condition: bool, detail: str = "") -> bool:
    marker = "OK  " if condition else "FAIL"
    print(f"  {marker} {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(label)
    return condition


# ==================== layer 1: the primitives ====================

#: `Mth.getSeed` vectors from a real Minecraft (Pumpkin's `block_position_hash`).
HASH_VECTORS = [
    ((0, 0, 0), 0),
    ((1, 1, 1), 60311958971344),
    ((4, 4, 4), 120566413180880),
    ((25, 25, 25), 111753446486209),
    ((676, 676, 676), 75210837988243),
    ((458329, 458329, 458329), -43764888250),
    ((-387008604, -387008604, -387008604), 8437923733503),
    ((176771161, 176771161, 176771161), 18421337580760),
]


def check_primitives() -> None:
    print("1. primitives")
    bad = []
    for position, expected in HASH_VECTORS:
        got = _hash_block_pos(*position)
        signed = got - (1 << 64) if got >= (1 << 63) else got
        if signed != expected:
            bad.append((position, signed, expected))
    check("Mth.getSeed matches vanilla", not bad,
          f"{len(HASH_VECTORS)} vectors" if not bad else f"first bad: {bad[0]}")

    # `random_name` factories hash md5 **big-endian**. The little-endian reading
    # is what Pumpkin's generated constants carry and it misses 310 of the 1280
    # bedrock cells in chunk (0,0); this assertion is the only thing standing
    # between that and a silent revert.
    lo, hi = name_hash("minecraft:bedrock_floor")
    check("random_name md5 is big-endian", (lo, hi) == (13544455532117611141,
                                                       14185350335435586452),
          f"lo={lo}")

    # Simplex noise, against vectors taken from a real Minecraft.
    random = Xoroshiro.from_seed(111)
    random.next_i32()
    simplex = SimplexNoise(random)
    check("SimplexNoise origins", (simplex.origin_x, simplex.origin_y, simplex.origin_z)
          == (48.58072036717974, 110.73235882678037, 65.26438852860176))
    samples = [((-50000.0, 0.0), -0.013008608535752102),
               ((-49996.0, 4000.0), 0.5015373706471664),
               (-3.134738528791615e8, 5.676610095659718e7, 0.018940199193618792)]
    worst = 0.0
    for case in samples[:2]:
        worst = max(worst, abs(simplex.sample_2d(*case[0]) - case[1]))
    worst = max(worst, abs(simplex.sample_2d(samples[2][0], samples[2][1]) - samples[2][2]))
    check("SimplexNoise 2D values", worst == 0.0, f"worst |diff| {worst:g}")

    random = Xoroshiro.from_seed(450)
    random.next_i32()
    octaves = OctaveSimplexNoiseSampler(random, [-1, 1, 0])
    check("OctaveSimplexNoiseSampler scaling",
          octaves.lacunarity == 2.0 and abs(octaves.persistence - 1 / 7) < 1e-15)
    value = octaves.sample(-1.3127900550351206e7, 792897.4979227383, False)
    check("OctaveSimplexNoiseSampler value", abs(value - -0.4321152413690901) == 0.0,
          f"got {value!r}")

    # The estimator is parsed in the proto phase; assert that the phase cannot
    # matter for it, so nobody "fixes" a difference that is not there. Only
    # `finalDensity` and the vein slots carry an `Interpolated` wrapper.
    import collections

    trees = json.load(open(ROOT / "assets" / "density_function.json", encoding="utf-8"))
    for preset in ("overworld", "nether", "end"):
        seen = collections.Counter()

        def walk(node):
            if isinstance(node, dict):
                if "type" in node:
                    seen[node["type"]] += 1
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(trees[preset].get("initialDensityWithoutJaggedness"))
        check(f"{preset}: the surface estimator is not Interpolated",
              not seen.get("Interpolated"),
              "so the density phase cannot change the estimate")


# ==================== layer 2: the vanilla chunks ====================

def load_chunk(name: str) -> dict:
    with open(CHUNK_DIR / f"{name}.json", encoding="utf-8") as handle:
        return json.load(handle)


def to_blocks(payload: dict, key: str, palette: Dict[int, int], height: int) -> np.ndarray:
    lookup = np.zeros(65536, dtype=np.uint16)
    for dump_id, resolved in palette.items():
        lookup[dump_id] = resolved
    raw = np.asarray(payload[key], dtype=np.uint16).reshape(16, height, 16)
    # Dump order is (x * height + local_y) * 16 + z; storage wants [y, z, x].
    return lookup[raw].transpose(1, 2, 0).copy()


def check_chunks(only: Optional[str], verbose: bool) -> None:
    print()
    print("2. vanilla chunks")
    from world.block_states import state_id

    index = json.load(open(CHUNK_DIR / "index.json", encoding="utf-8"))
    table = biome_module.biome_table()
    totals = []
    for entry in index:
        name = entry["name"]
        if only and name != only:
            continue
        payload = load_chunk(name)
        min_y, height = payload["min_y"], payload["height"]
        cx, cz, seed, preset = (payload["chunk_x"], payload["chunk_z"],
                                payload["seed"], payload["preset"])
        palette = {int(k): state_id(v) for k, v in payload["palette"].items()}

        before = to_blocks(payload, "before", palette, height)
        after = to_blocks(payload, "after", palette, height)

        source = biome_module.biome_source(preset, seed, table.ids)
        biomes = source.fill_volume(cx, cz, min_y, height)
        biomes_before = biomes.copy()

        system = SurfaceSystem.from_assets(preset, seed, biome_table=table)
        blocks = before.copy()
        started = time.perf_counter()
        stats = system.build_surface(blocks, biomes, cx, cz)
        elapsed = (time.perf_counter() - started) * 1000

        mismatch = int((blocks != after).sum())
        ceiling = RESIDUALS.get(name)
        ok = ceiling is not None and mismatch <= ceiling
        detail = (f"{mismatch}/{after.size} cells, ceiling {ceiling}, "
                  f"{elapsed:.0f} ms, changed {stats.cells_changed}")
        check(f"{name} ({preset}, seed {seed}, chunk ({cx},{cz}))", ok, detail)
        totals.append((name, mismatch, after.size))

        if verbose and mismatch:
            _report_mismatches(before, blocks, after, min_y, height)

        # The biome volume is an input, not an output.
        check(f"  {name}: biome volume untouched", np.array_equal(biomes, biomes_before))

        # Determinism: a second pass on the same input must give the same bytes.
        again = before.copy()
        system.build_surface(again, biomes, cx, cz)
        check(f"  {name}: deterministic", np.array_equal(again, blocks))

    if totals:
        exact = sum(1 for _, m, _ in totals if m == 0)
        total_cells = sum(n for _, _, n in totals)
        total_bad = sum(m for _, m, _ in totals)
        print()
        print(f"  {exact}/{len(totals)} chunks exact; "
              f"{total_bad}/{total_cells} cells differ overall "
              f"({100.0 * total_bad / total_cells:.3f}%)")


def _report_mismatches(before: np.ndarray, got: np.ndarray, want: np.ndarray,
                       min_y: int, height: int, limit: int = 6) -> None:
    from collections import Counter

    from world.block_states import name_of

    diff = got != want
    coords = np.argwhere(diff)
    pairs: Counter = Counter()
    for y, z, x in coords:
        pairs[(name_of(int(before[y, z, x])), name_of(int(got[y, z, x])),
               name_of(int(want[y, z, x])))] += 1
    for (b, g, w), count in pairs.most_common(limit):
        print(f"       {count:6d}  before={b} got={g} want={w}")
    ys = coords[:, 0] + min_y
    print(f"       y range {ys.min()}..{ys.max()}, median {int(np.median(ys))}")


# ==================== layer 3: shape and invariants ====================

def check_invariants() -> None:
    print()
    print("3. invariants")
    table = biome_module.biome_table()
    system = SurfaceSystem.from_assets("overworld", 0, biome_table=table)
    source = biome_module.biome_source("overworld", 0, table.ids)
    biomes = source.fill_volume(0, 0, -64, 384)

    # An all-air chunk must stay all air: the rules only ever replace the default
    # block, and there is none to replace.
    empty = np.zeros((384, 16, 16), dtype=np.uint16)
    stats = system.build_surface(empty, biomes, 0, 0)
    check("an empty chunk stays empty", int(empty.sum()) == 0,
          f"changed {stats.cells_changed}")

    # A solid chunk must stay solid, and the surface rules must fire in it.
    solid = np.full((384, 16, 16), system.default_block, dtype=np.uint16)
    stats = system.build_surface(solid, biomes, 0, 0)
    check("a solid chunk is dressed, not erased", int((solid == 0).sum()) == 0,
          f"visited {stats.cells_visited}, changed {stats.cells_changed}")

    # The heightmap predicate is the one the pass iterates from.
    heightmap = world_surface_heightmap(solid, -64)
    check("WORLD_SURFACE heightmap is one past the top",
          int(heightmap.min()) == 320 and int(heightmap.max()) == 320,
          f"{heightmap.min()}..{heightmap.max()}")

    # A one-block column: the topmost non-air, not the top of the world. This is
    # the bug the dumps caught: masking the row axis by `any(axis=0)` made every
    # non-empty column report 320 and the pass started 256 blocks too high.
    sparse = np.zeros((384, 16, 16), dtype=np.uint16)
    sparse[100, 3, 5] = system.default_block
    sparse_height = world_surface_heightmap(sparse, -64)
    check("a sparse column reports its own top", int(sparse_height[5, 3]) == 37,
          f"got {int(sparse_height[5, 3])}, want 37")
    check("an empty column reports min_y", int(sparse_height[0, 0]) == -64)

    check_compiled_matcher(system)


# ==================== compiled matcher ====================

def check_compiled_matcher(system: SurfaceSystem) -> None:
    """The flat matcher `build_surface` runs must agree with the rule tree.

    The tree is the readable form and the dumps pin *it* to vanilla, but the
    hot loop no longer calls it - so the dumps alone would not notice a
    compiler slip. Sweep the state space the two share: full y sweeps (which
    cross every `vertical_gradient` branch, including its transition band)
    then random states on the depth, fluid, run-depth and biome axes.
    """
    import random as _random

    table = biome_module.biome_table()
    ctx = SurfaceContext(system.min_y, system.height, system.sea_level,
                         system.splitter, system.terrain,
                         system.world_min_y, system.world_height)
    tree, compiled = system.rule, system.compiled
    names = sorted(set(table.names.values())) or ["minecraft:plains"]
    temperatures = table.temperature
    modifiers = table.modifier

    bad = []
    tested = 0

    def sweep(state, y, fluid, sda, sdb, run_depth, secondary, surface_min_y,
              steep, biome):
        nonlocal tested
        tested += 1
        ctx.block_x, ctx.block_y, ctx.block_z = state[0], y, state[2]
        ctx.fluid_height = fluid
        ctx.stone_depth_above = sda
        ctx.stone_depth_below = sdb
        ctx.run_depth = run_depth
        ctx.secondary_depth = secondary
        ctx.surface_min_y = surface_min_y
        ctx.steep = steep
        ctx.biome = biome
        temp = temperatures.get(biome, 0.0)
        mod = modifiers.get(biome, "")
        ctx.biome_temperature = temp
        ctx.biome_modifier = mod
        want = tree.match(ctx)
        got = compiled(ctx, y, fluid, sda, sdb, biome, temp, mod)
        if want != got and len(bad) < 5:
            bad.append((state, y, fluid, sda, sdb, run_depth, want, got))

    # Full y sweeps at a few positions: every row of the window, which crosses
    # the bedrock floor band, the deepslate transition and the world top.
    for x, z, run_depth in ((0, 0, 6), (15, 15, 0), (-32, 7, -1)):
        for y in range(system.world_min_y, system.world_min_y + system.world_height):
            for fluid, sda, sdb in ((_NO_FLUID, 3, 12), (64, 1, 1)):
                sweep((x, y, z), y, fluid, sda, sdb, run_depth, 0.25, 40,
                      False, "minecraft:plains")

    rng = _random.Random(12345)
    for _ in range(20000):
        y = rng.randrange(system.world_min_y, system.world_min_y + system.world_height)
        sweep((rng.randrange(-4096, 4096), y, rng.randrange(-4096, 4096)), y,
              _NO_FLUID if rng.random() < 0.3 else rng.randrange(-64, 320),
              rng.randrange(0, 80), rng.randrange(0, 80),
              rng.randrange(-2, 12), rng.uniform(-1.5, 1.5),
              rng.randrange(-64, 200), rng.random() < 0.25,
              rng.choice(names))

    detail = f"{tested} states swept"
    if bad:
        detail += f"; first disagreements: {bad}"
    check("the compiled matcher agrees with the rule tree", not bad, detail)


# ==================== main ====================

def main() -> int:
    only = None
    verbose = False
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--chunk" and i + 1 < len(args):
            only = args[i + 1]
        elif arg == "--verbose":
            verbose = True

    if not CHUNK_DIR.exists():
        print(f"  FAIL  {CHUNK_DIR} is missing; the vanilla chunk dumps are required")
        return 1

    started = time.perf_counter()
    check_primitives()
    check_chunks(only, verbose)
    if only is None:
        check_invariants()

    print()
    if failures:
        print(f"FAILED ({len(failures)}): " + "; ".join(failures[:5]))
        return 1
    print(f"surface rules OK in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
