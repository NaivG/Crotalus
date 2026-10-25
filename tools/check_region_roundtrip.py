"""Chunks survive a save/load round trip at negative coordinates.

The bug this covers: `RegionManager.load_chunk` handed the region-local
coordinates to `RegionFile.load_chunk`, which built the Chunk from them. The
two only agree inside the origin region, so world chunk (-1, 0) came back
positioned at (31, 0). The chunk manager filed it there, `get_chunk(-1, 0)`
answered None forever, and the generator's retry path re-read the same
mis-keyed chunk instead of generating -- one coordinate that stalled a whole
view behind it.

So the check is deliberately asymmetric around the origin: it only exercises
the coordinates where local and world disagree, and it checks both what came
back and what the manager filed it under. Asserting only on the origin region
would pass against the broken code.

Run: python tools/check_region_roundtrip.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from loguru import logger
logger.remove()

from world.chunk import ChunkPos
from world.dimension import DimensionType
from world.world import World

# Local and world coordinates diverge as soon as either axis is negative.
COORDS = [(-1, 0), (0, -1), (-1, -1), (-4, 3), (3, -4), (-33, 17), (17, -33)]


def fingerprint(chunk):
    """Enough of a chunk to tell two of them apart after a round trip."""
    return [
        (int(np.asarray(s.blocks).sum()), int(s._non_air_count))
        for s in chunk.sections
    ]


tmp = Path(tempfile.mkdtemp())
try:
    world = World(name="rt", seed=4242, world_dir=tmp / "rt")

    before = {}
    for cx, cz in COORDS:
        chunk = world.load_chunk(cx, cz, dimension=DimensionType.OVERWORLD)
        before[(cx, cz)] = fingerprint(chunk)
    world.save()
    print(f"saved {len(COORDS)} chunks to {world.world_dir}")

    # A second World on the same directory: fresh chunk manager, nothing
    # carried over in memory. This is what the server does on restart.
    reloaded = World(name="rt", world_dir=tmp / "rt")
    cm = reloaded.overworld.chunk_manager

    fails = 0
    for cx, cz in COORDS:
        chunk = reloaded.load_chunk(cx, cz, dimension=DimensionType.OVERWORLD)
        if chunk is None:
            print(f"  ({cx:4d},{cz:4d})  not found on disk at all")
            fails += 1
            continue
        # The chunk has to be *positioned* at the coordinate that was asked
        # for. The manager then files it under that position, which is the
        # lookup the streamer and the generator both depend on.
        if (chunk.position.x, chunk.position.z) != (cx, cz):
            print(f"  ({cx:4d},{cz:4d})  positioned at "
                  f"({chunk.position.x},{chunk.position.z}) -- filed under "
                  f"the wrong key")
            fails += 1
            continue
        if cm.get_chunk(ChunkPos(cx, cz)) is None:
            print(f"  ({cx:4d},{cz:4d})  unreachable by coordinate")
            fails += 1
            continue
        if fingerprint(chunk) != before[(cx, cz)]:
            print(f"  ({cx:4d},{cz:4d})  content changed across the round trip")
            fails += 1
            continue
        print(f"  ({cx:4d},{cz:4d})  ok")

    print("RESULT:", "PASS" if not fails else f"FAIL ({fails} chunk(s))")
    sys.exit(0 if not fails else 1)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
