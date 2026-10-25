"""Profile the four decoration passes (`world/feature/`) separately.

Each layer is timed on the same chunk coordinates, then cProfile'd on its own,
so the ranking is measured rather than guessed. `generate_chunk` e2e is
printed too, for the share each layer is of the whole.

Run: py -3.14t tools/profile_decoration.py [chunks] [cx] [cz] [layer...]
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

LAYERS = {
    "biomes": lambda layers, c: layers.biomes.apply(c),
    "surface": lambda layers, c: layers.surface.apply(c, layers.table),
    "carvers": lambda layers, c: layers.carvers.apply(c, layers.table, sea_level=63),
    "features": lambda layers, c: layers.features.apply(c, layers.table),
}
# The e2e order in `BaseGenerator._generate_chunk_locked`. To time a pass, the
# chunk must carry only the passes *before* it - handing a fully generated chunk
# to a second surface pass re-dresses already-dressed terrain, which is not the
# work the e2e does and inflates the number.
PRECEDING = {
    "biomes": (),
    "surface": ("biomes",),
    "carvers": ("biomes", "surface"),
    "features": ("biomes", "surface", "carvers"),
}


def main() -> None:
    argv = sys.argv[1:]
    n = int(argv[0]) if argv else 4
    cx = int(argv[1]) if len(argv) > 1 else 0
    cz = int(argv[2]) if len(argv) > 2 else 0
    wanted = argv[3:] or list(LAYERS)

    gil = getattr(sys, "_is_gil_enabled", None)
    state = "enabled" if (gil and gil()) else "disabled"
    print(f"python {sys.version.split()[0]}  GIL {state}  "
          f"chunks={n} at ({cx},{cz})\n")

    gen = create_generator(GeneratorSettings(seed=0, preset="overworld"))
    gen.set_seed(0)
    layers = gen.layers
    gen.generate_chunk(Chunk(ChunkPos(9999, 9999)))  # warm-up

    probes = [Chunk(ChunkPos(cx + i, cz)) for i in range(n)]
    t0 = time.perf_counter()
    for probe in probes:
        gen.generate_chunk(probe)
    e2e = time.perf_counter() - t0
    print(f"e2e generate_chunk  {e2e:7.3f}s  "
          f"({e2e / n:.3f}s/chunk)\n")

    for name in wanted:
        call = LAYERS[name]
        # Fresh terrain carrying only the passes that precede this one, so the
        # timing is the work the e2e actually does at this point.
        work = [Chunk(ChunkPos(cx + i, cz)) for i in range(n)]
        for probe in work:
            gen._generate_terrain(probe)
            for before in PRECEDING[name]:
                LAYERS[before](layers, probe)
        t0 = time.perf_counter()
        for probe in work:
            call(layers, probe)
        dt = time.perf_counter() - t0
        print(f"=== {name}: {dt:.3f}s  ({dt / n:.3f}s/chunk, "
              f"{dt / e2e * 100:.1f}% of e2e) ===")

        # Fresh chunks again for the profile: the timed pass above just dressed
        # `work`, and a second pass over dressed terrain is a much lighter
        # workload (surface rules skip every non-default cell) - profiling it
        # understates the calls the e2e actually makes.
        work = [Chunk(ChunkPos(cx + i, cz)) for i in range(n)]
        for probe in work:
            gen._generate_terrain(probe)
            for before in PRECEDING[name]:
                LAYERS[before](layers, probe)
        prof = cProfile.Profile()
        prof.enable()
        for probe in work:
            call(layers, probe)
        prof.disable()
        pstats.Stats(prof).sort_stats("tottime").print_stats(14)
        print()


if __name__ == "__main__":
    main()
