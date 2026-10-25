"""Regression: a background generation request must actually produce a chunk.

The bug this guards against was silent -- a chunk request asked the chunk
manager for a chunk that was never loaded, got None back from what is a pure
lookup, and returned without raising. Every request "succeeded" while
generating nothing, and because the streamer sends strictly nearest-first
and stops at the first chunk that is not ready, one such chunk stalled the
entire view until the client gave up.

With generation in worker processes the same shape has new ways to bite: a
spawn child dying during bootstrap kills its whole pool silently, and a
request tagged with the wrong dimension generates terrain nobody will ever
see. Both are covered here.

The body runs under the __main__ guard: the pool spawns children that
re-import this file, and a module-level World would make every child try to
create its own world.

Run: python tools/check_background_generation.py
"""

import sys, tempfile, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
logger.remove()


def main() -> None:
    from world.chunk_gen_thread import chunk_generator
    from world.dimension import DimensionType
    from world.world import World

    tmp = Path(tempfile.mkdtemp())
    world = World(name="bg", seed=1234, world_dir=tmp / "bg")

    fails = 0

    # The generator is wired up; the pool starts lazily on the first request.
    print("completion sink:", world.insert_generated_chunk.__qualname__)
    print("pool running before first request:", chunk_generator.is_running)

    # A request tagged with a dimension must land in that dimension, and only
    # there. The generator's workers are shared across dimensions, so a
    # request that loses its dimension generates in the wrong world and
    # leaves the player's actual view empty.
    for dim_type, coord in [
        (DimensionType.OVERWORLD, (50, -30)),
        (DimensionType.NETHER, (-7, 3)),
        (DimensionType.END, (11, 11)),
    ]:
        dim = world.dimensions[dim_type]
        assert dim.get_chunk(*coord) is None, "precondition: chunk not loaded"

        queued = world.request_chunk(coord[0], coord[1], dim_type, priority=5)
        if not queued:
            fails += 1
            print(f"FAIL: request for {dim_type.value} was not queued")
            continue

        chunk = None
        deadline = time.time() + 120.0  # covers the pool's spawn + import
        while time.time() < deadline:
            chunk = dim.get_chunk(*coord)
            if chunk is not None and chunk.is_generated:
                break
            time.sleep(0.02)

        ok = chunk is not None and chunk.is_generated
        print(f"{dim_type.value:22} queued={queued} generated={ok}")
        if not ok:
            fails += 1

        leaked = [o.value for o, d in world.dimensions.items()
                  if o is not dim_type and d.get_chunk(*coord) is not None]
        if leaked:
            fails += 1
            print(f"FAIL: {dim_type.value} request leaked into {leaked}")
        else:
            print(f"{dim_type.value:22} leaked=[]")

    if not chunk_generator.is_running:
        fails += 1
        print("FAIL: generator pool did not start on the first request")
    else:
        print("pool running after requests:", True,
              f"({chunk_generator.max_workers} workers)")

    world.unload(save=False)

    print()
    if fails:
        print(f"RESULT: FAIL ({fails})")
        sys.exit(1)
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
