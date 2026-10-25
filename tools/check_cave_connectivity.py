"""Measure whether the cave carver produces connected caves.

The carver this replaced thresholded one 3D noise field at a constant, and
that failure mode is invisible in an air count: the superlevel set of a
scalar field can only ever be as connected as the field is, so a 32-block
noise thresholded at 0.6 gave a scatter of tiny isolated ellipsoids that
looked almost plausible in a volume statistic. Over the region below, the old
carver produced 2 blobs totalling 82 cells.

So this reports, over a 7x7 block of chunks:

  carved   fraction of the sampled volume that became air
  blobs    count of 6-connected air components
  largest  size of the biggest one, in cells and as a share of all the air
  depth    the shallowest and deepest y that any cave reaches

`blobs` is the number to watch. Thousands of tiny ones is the old behaviour.

Run: python tools/check_cave_connectivity.py [cave_count]
"""

import sys, time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from loguru import logger
logger.remove()

from world.legacy.cave_carver import carve_chunk_caves, carve_cheese

SEED = 20261004
MIN_Y, MAX_Y = 5, 120
RADIUS = 3          # chunks each way, for the connectivity scan
DEPTH = 64          # y range used for the scan
CHEESE_SCALE = 96.0
CHEESE_THRESHOLD = 0.42


def carve_region(cx0, cz0, radius):
    """Carve a (radius*2+1)^2 block of chunks into one volume.

    Each chunk is carved independently through the public entry points, which
    is the point: if the neighbourhood handling is right, a tunnel owned by a
    neighbouring carver chunk arrives here intact rather than being clipped at
    the 16-block border.

    No surface guard here. This measures the carver, not the terrain, so the
    percentages are of the raw carve window.
    """
    n = radius * 2 + 1
    volume = np.zeros((DEPTH, n * 16, n * 16), dtype=bool)
    for dx in range(-radius, radius + 1):
        for dz in range(-radius, radius + 1):
            mask = np.zeros((MAX_Y - MIN_Y + 1, 16, 16), dtype=np.uint8)
            carve_chunk_caves(mask, MIN_Y, MAX_Y, cx0 + dx, cz0 + dz, SEED,
                              radius=6, cave_count=CAVE_COUNT)
            carve_cheese(mask, MIN_Y, MAX_Y, cx0 + dx, cz0 + dz, SEED + 3,
                         scale=CHEESE_SCALE, threshold=CHEESE_THRESHOLD)
            volume[:, (dz + radius) * 16:(dz + radius) * 16 + 16,
                   (dx + radius) * 16:(dx + radius) * 16 + 16] = \
                mask[:DEPTH].astype(bool)
    return volume


def count_blobs(volume):
    """6-connected component count via BFS. Iterative -- the volume is deep."""
    seen = np.zeros_like(volume)
    blobs = 0
    largest = 0
    air = volume.copy()
    for y, z, x in zip(*np.nonzero(air)):
        if seen[y, z, x]:
            continue
        blobs += 1
        size = 0
        queue = deque([(y, z, x)])
        seen[y, z, x] = True
        while queue:
            cy, cz, cx = queue.popleft()
            size += 1
            for ny, nz, nx in ((cy+1, cz, cx), (cy-1, cz, cx),
                               (cy, cz+1, cx), (cy, cz-1, cx),
                               (cy, cz, cx+1), (cy, cz, cx-1)):
                if (0 <= ny < volume.shape[0] and 0 <= nz < volume.shape[1]
                        and 0 <= nx < volume.shape[2] and volume[ny, nz, nx]
                        and not seen[ny, nz, nx]):
                    seen[ny, nz, nx] = True
                    queue.append((ny, nz, nx))
        largest = max(largest, size)
    return blobs, largest


def depth_reached(volume):
    """Shallowest and deepest y with air, so a dead band would show up here."""
    rows = np.nonzero(volume.any(axis=(1, 2)))[0]
    if rows.size == 0:
        return 0, 0
    return int(rows[0]), int(rows[-1])


CAVE_COUNT = int(sys.argv[1]) if len(sys.argv) > 1 else 4

# Warm the JIT before timing. numba caches the compiled carver to disk, but
# the first call in a fresh process still pays to load it, and averaging that
# across the region would report it as generation cost.
_warm = np.zeros((MAX_Y - MIN_Y + 1, 16, 16), dtype=np.uint8)
carve_chunk_caves(_warm, MIN_Y, MAX_Y, 9000, 9000, SEED, cave_count=CAVE_COUNT)
carve_cheese(_warm, MIN_Y, MAX_Y, 9000, 9000, SEED + 3,
             scale=CHEESE_SCALE, threshold=CHEESE_THRESHOLD)

start = time.perf_counter()
volume = carve_region(0, 0, RADIUS)
carve_ms = (time.perf_counter() - start) * 1000

blobs, largest = count_blobs(volume)
total_air = int(volume.sum())
shallow, deep = depth_reached(volume)

print(f"cave_count = {CAVE_COUNT}   ({RADIUS*2+1}x{RADIUS*2+1} chunks, "
      f"{volume.shape[1]}x{volume.shape[2]} blocks, y {MIN_Y}..{MIN_Y+DEPTH})")
print(f"  carved   {100.0*total_air/volume.size:5.1f}% of volume")
print(f"  blobs    {blobs:6d} connected air components")
print(f"  largest  {largest:6d} cells  ({100.0*largest/max(1,total_air):.1f}% of all air)")
print(f"  depth    y {MIN_Y+shallow} .. y {MIN_Y+deep}  "
      f"({(deep - shallow + 1) * 100 // DEPTH}% of the sampled depth has caves)")
print(f"  carve    {carve_ms/RADIUS/RADIUS:.1f} ms per chunk (incl. region setup)")

# A mid-depth slice, so the shape is visible rather than just counted.
y = DEPTH // 2
print(f"\nslice y={MIN_Y+y} (one line per z, every other row):")
for z in range(0, volume.shape[1], 2):
    print("  " + "".join("#" if volume[y, z, x] else "." for x in range(volume.shape[2])))

# The old carver, for reference: 2 blobs, 82 cells, 0.007% of volume.
ok = (blobs < 400
      and largest > 0.15 * total_air
      and 0.04 < total_air / volume.size < 0.18
      and deep - shallow > DEPTH * 0.7)
print("\nRESULT:", "PASS" if ok else "FAIL")
