"""Profile where chunk-generation time actually goes, and inside the feature layer.

Three measurements, so the feature share is a number rather than an assumption:
  1. end-to-end `generate_chunk` for N chunks (terrain + carvers + features)
  2. `vanilla_features.decorate` alone, on one chunk's arrays
  3. a cProfile of (2), cumulative and self time

Run: py -3.14t tools/profile_features.py [chunks] [cx] [cz]
"""

import cProfile
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

logger.remove()

from world.chunk import Chunk, ChunkPos
from world.generator import GeneratorSettings, create_generator


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    cx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    cz = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    gil = getattr(sys, "_is_gil_enabled", None)
    state = "enabled" if (gil and gil()) else "disabled"
    print(f"python {sys.version.split()[0]}  GIL {state}\n")

    gen = create_generator(GeneratorSettings(seed=0, preset="overworld"))
    gen.set_seed(0)
    preset = gen.settings.preset

    # warm-up: registry load, FlatCache population, carver JIT
    gen.generate_chunk(Chunk(ChunkPos(9999, 9999)))

    # ---- 1. end-to-end ----
    chunks = [Chunk(ChunkPos(cx + i, cz)) for i in range(n)]
    t0 = time.perf_counter()
    for chunk in chunks:
        gen.generate_chunk(chunk)
    e2e = time.perf_counter() - t0

    # ---- 2. feature layer alone ----
    from world.vanilla import features as vf
    from world.vanilla.biome import BiomeTable

    table = BiomeTable.from_assets()
    placed = vf.load_placed_features()
    configured = vf.load_configured_features()
    print(f"preset={preset}  placed={len(placed)}  configured={len(configured)}")

    src = chunks[0]
    args = dict(preset=preset, seed=0, min_y=src.min_y,
                height=16 * len(src.sections), biome_table=table,
                placed_features=placed, configured_features=configured,
                heightmap=None)
    biomes = src.biome_volume().copy()

    t0 = time.perf_counter()
    stats = vf.decorate(src.position.x, src.position.z, src.blocks_view().copy(),
                        biomes, **args)
    feat = time.perf_counter() - t0
    print(f"\ne2e     {e2e:7.3f}s  ({e2e / n:.3f}s/chunk)")
    print(f"feature {feat:7.3f}s  ({feat:.3f}s/chunk, "
          f"{feat / e2e * 100:.1f}% of e2e, one call)")

    # ---- 2b. the four decoration passes, one at a time ----
    layers = gen.layers
    table = layers.table
    probes = [Chunk(ChunkPos(cx + i, cz)) for i in range(n)]
    for probe in probes:
        gen.generate_chunk(probe)
    for name, call in (
        ("biomes", lambda c: layers.biomes.apply(c)),
        ("surface", lambda c: layers.surface.apply(c, table)),
        ("carvers", lambda c: layers.carvers.apply(c, table, sea_level=63)),
        ("features", lambda c: layers.features.apply(c, table)),
    ):
        t0 = time.perf_counter()
        for probe in probes:
            call(probe)
        dt = time.perf_counter() - t0
        print(f"  layer {name:9s} {dt:7.3f}s  ({dt / n:.3f}s/chunk, "
              f"{dt / e2e * 100:5.1f}% of e2e)")

    # ---- 3. profile decorate ----
    prof = cProfile.Profile()
    prof.enable()
    for i in range(n):
        vf.decorate(src.position.x, src.position.z + i, src.blocks_view().copy(),
                    biomes, **args)
    prof.disable()

    print("\n=== decorate: cumulative ===")
    pstats.Stats(prof).sort_stats("cumulative").print_stats(40)

    print("\n=== decorate: self time ===")
    pstats.Stats(prof).sort_stats("tottime").print_stats(30)

    print(f"\nstats: {stats.summary()}")


if __name__ == "__main__":
    main()
