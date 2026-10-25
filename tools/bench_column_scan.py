"""Headroom for the two pure-Python hotspots in the decoration layer.

Both are the same mistake in different clothes: a per-cell Python loop whose
body is dominated by numpy *scalar* indexing (~100ns per element) where a
Python list / bytes index (~25ns) or one vectorized pass would do.

  A. `world/vanilla/features.py: Surface.top` - walks one column from the top
     of the world down to the terrain, calling a predicate per cell.
  B. `world/vanilla/surface.py: build_surface` - walks the same column again
     with the surface rules, indexing `liquids`/`solids` per cell.

Run: py -3.14t tools/bench_column_scan.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from loguru import logger

logger.remove()

HEIGHT = 384
MIN_Y = -64
N_COLUMNS = 256          # one chunk
REPEAT = 20


def timed(fn, repeat=REPEAT):
    fn()
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def make_columns(surface_y=70):
    """A chunk-shaped `(height, 16, 16)` volume: air above, stone below."""
    blocks = np.zeros((HEIGHT, 16, 16), dtype=np.uint16)
    stone = 1
    water = 9
    blocks[:surface_y] = stone
    blocks[surface_y:surface_y + 3] = water
    return blocks


def bench_surface_top(blocks):
    """A. The current top-down Python walk, vs a per-kind state-id table."""
    states = np.arange(1, 4096, dtype=np.uint16)
    # `not is_air` over the id space, as a lookup table indexed by state id.
    not_air = np.ones(4096, dtype=bool)
    not_air[0] = False

    def python_walk():
        total = 0
        for lz in range(16):
            for lx in range(16):
                column = blocks[:, lz, lx]
                for row in range(HEIGHT - 1, -1, -1):
                    if not_air[int(column[row])]:
                        total += MIN_Y + row + 1
                        break
        return total

    def numpy_walk():
        total = 0
        for lz in range(16):
            for lx in range(16):
                column = blocks[:, lz, lx]
                mask = not_air[column]
                if mask.any():
                    total += MIN_Y + HEIGHT - int(mask[::-1].argmax())
                else:
                    total += MIN_Y
        return total

    def numpy_walk_batched():
        mask = not_air[blocks]                       # (384, 16, 16)
        # Highest True per column, -1 where the column is all False.
        idx = HEIGHT - 1 - mask[::-1].argmax(axis=0)
        hit = mask.any(axis=0)
        return int(np.where(hit, MIN_Y + idx + 1, MIN_Y).sum())

    ref = python_walk()
    a = timed(python_walk)
    b = timed(numpy_walk)
    c = timed(numpy_walk_batched, repeat=5)
    print(f"  A. Surface.top, {N_COLUMNS} columns, ~{HEIGHT - 70} cells each")
    print(f"     python walk (current) {a * 1e3:8.2f} ms")
    print(f"     numpy per column     {b * 1e3:8.2f} ms   {a / b:5.1f}x")
    print(f"     numpy batched        {c * 1e3:8.2f} ms   {a / c:5.1f}x")
    print(f"     agree: {numpy_walk() == ref == numpy_walk_batched()}")


def bench_build_surface(blocks):
    """B. numpy scalar indexing per cell, vs list + bytes indexing."""
    total = 4096
    liquid = np.zeros(total, dtype=bool)
    solid = np.zeros(total, dtype=bool)
    liquid[9] = True
    solid[1] = True
    liquid_b = liquid.tobytes()
    solid_b = solid.tobytes()
    AIR = 0
    default_block = 1

    def numpy_inner():
        hits = 0
        for lz in range(16):
            for lx in range(16):
                column = blocks[:, lz, lx]
                run_top = None
                run_bottom = 0
                for y in range(70 - 1, MIN_Y - 1, -1):
                    block = int(column[y - MIN_Y])
                    if block == AIR:
                        run_top = None
                        continue
                    if liquid[block] if block < total else False:
                        continue
                    if run_top != y + 1:
                        run_top = y
                        run_bottom = y
                        while run_bottom - 1 >= MIN_Y:
                            below = int(column[run_bottom - 1 - MIN_Y])
                            if not (solid[below] if below < total else False):
                                break
                            run_bottom -= 1
                    hits += 1
                    if block != default_block:
                        continue
        return hits

    def list_inner():
        hits = 0
        for lz in range(16):
            for lx in range(16):
                column = blocks[:, lz, lx].tolist()
                run_top = None
                run_bottom = 0
                for y in range(70 - 1, MIN_Y - 1, -1):
                    block = column[y - MIN_Y]
                    if block == AIR:
                        run_top = None
                        continue
                    if block < total and liquid_b[block]:
                        continue
                    if run_top != y + 1:
                        run_top = y
                        run_bottom = y
                        while run_bottom - 1 >= MIN_Y:
                            below = column[run_bottom - 1 - MIN_Y]
                            if not (below < total and solid_b[below]):
                                break
                            run_bottom -= 1
                    hits += 1
                    if block != default_block:
                        continue
        return hits

    ref = numpy_inner()
    a = timed(numpy_inner)
    b = timed(list_inner)
    print(f"\n  B. build_surface scan, {N_COLUMNS} columns, ~{70 + 64} cells each")
    print(f"     numpy scalars (current) {a * 1e3:8.2f} ms")
    print(f"     list + bytes             {b * 1e3:8.2f} ms   {a / b:5.1f}x")
    print(f"     agree: {list_inner() == ref}")


def main() -> None:
    blocks = make_columns()
    print(f"volume {blocks.shape} {blocks.dtype}  "
          f"{blocks.nbytes / 1e3:.0f} KB\n")
    bench_surface_top(blocks)
    bench_build_surface(blocks)


if __name__ == "__main__":
    main()
