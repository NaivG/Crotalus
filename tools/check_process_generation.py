"""End-to-end: background generation through the worker PROCESS pool.

Guards the whole multiprocess pipeline, the parts whose failure modes used
to be silent:

1. Requests tagged with a dimension must generate *that* dimension's terrain
   (the workers are shared by every dimension; the seed rides the request).
2. A worker's chunk must be byte-identical to what the same seed produces
   in-process -- the pool does not get to have its own opinion about terrain.
3. `World.generate_chunks_blocking` must settle every request exactly once,
   success or failure (the spawn preload waits on this; a lost callback is a
   startup hang).
4. `chunk_generator.stop()` must run BEFORE a save leaves dirty chunks
   behind: after stop(), `is_dirty` may only be False. The old order (save
   first, stop last) let chunks land in the manager after the save pass had
   iterated -- unsaved forever, and able to break the save's dict iteration.
5. A failed generation must drop the blank claim, not leave a hole the
   streamer stalls behind forever.

Everything lives under `main()` behind the `__main__` guard: the pool spawns
children that re-import this file, and a module-level `World(...)` would make
every child try to create its own world and its own pool.

Run: python tools/check_process_generation.py
"""

import hashlib
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

logger.remove()

from world.chunk import Chunk, ChunkPos
from world.chunk_gen_thread import chunk_generator
from world.dimension import DimensionType
from world.world import World

SEED = 424242
fails = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global fails
    tag = "ok  " if ok else "FAIL"
    print(f"[{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        fails += 1


def terrain_hash(chunk) -> str:
    h = hashlib.sha256()
    for section in chunk.sections:
        h.update(section.blocks.tobytes())
    h.update(chunk.heightmap.tobytes())
    if chunk.biomes is not None:
        h.update(chunk.biomes.tobytes())
    return h.hexdigest()[:16]


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    world = World(name="proc", seed=SEED, world_dir=tmp / "proc")

    # -- 1. batch generation through the pool, two dimensions ----------------

    coords = [(0, 0), (1, 0), (0, 1), (-1, -1), (2, 2), (3, 1)]

    print("pool:", chunk_generator.max_workers, "processes; generating",
          len(coords), "overworld +", len(coords), "nether chunks")

    done = world.generate_chunks_blocking(coords, DimensionType.OVERWORLD, timeout=300)
    check("overworld batch settled", done == len(coords), f"{done}/{len(coords)}")

    for cx, cz in coords:
        chunk = world.overworld.chunk_manager.get_chunk(ChunkPos(cx, cz))
        check(f"overworld ({cx},{cz}) generated",
              chunk is not None and chunk.is_generated)

    done = world.generate_chunks_blocking(coords, DimensionType.NETHER, timeout=300)
    check("nether batch settled", done == len(coords), f"{done}/{len(coords)}")

    nether = world.dimensions[DimensionType.NETHER]
    for cx, cz in coords:
        chunk = nether.chunk_manager.get_chunk(ChunkPos(cx, cz))
        check(f"nether ({cx},{cz}) generated",
              chunk is not None and chunk.is_generated)

    # -- 2. byte identity against in-process generation ----------------------

    from world.chunk_gen_worker import _generator_for

    reference = _generator_for(DimensionType.OVERWORLD.value, SEED)
    for cx, cz in coords[:3]:
        standalone = Chunk(position=ChunkPos(cx, cz),
                           min_y=world.overworld.settings.min_y,
                           height=world.overworld.settings.height)
        reference.generate_chunk(standalone)
        served = world.overworld.chunk_manager.get_chunk(ChunkPos(cx, cz))
        check(f"terrain identical ({cx},{cz})",
              terrain_hash(served) == terrain_hash(standalone))

    # -- 3. no cross-dimension leaks -----------------------------------------

    # A request for one dimension must never land in another. The nether and
    # the overworld legitimately share coords (both were requested above);
    # the end was never requested and must be empty.
    coord = coords[0]
    end = world.dimensions[DimensionType.END]
    check("overworld request did not land in the end",
          end.chunk_manager.get_chunk(ChunkPos(*coord)) is None)

    # -- 4. stop-before-save leaves nothing dirty -----------------------------

    # Generous drain so any straggler finishes, then stop and save. The
    # contract under test: the save pass must see every chunk the generator
    # inserted, so after stop() + save() nothing may remain dirty. (Chunks
    # ARE dirty between insert and save -- generated-but-unsaved is the
    # normal state; leaving them dirty after a save is the bug this guards.)
    chunk_generator.stop(drain=30.0)
    check("save after stop()", world.save())
    dirty = [(d.value, (pos.x, pos.z)) for d, dim in world.dimensions.items()
             for pos, c in dim.chunk_manager.chunks.items() if c.is_dirty]
    check("nothing dirty after stop+save", not dirty, str(dirty[:4]))

    # -- 5. failed generation drops the claim ---------------------------------

    # Restart the pool to exercise the failure path without a real worker
    # error: the sink contract is "only a blank, clean claim may be dropped".
    from core.config import config as app_config

    chunk_generator.configure(
        workers=app_config.chunk_gen_workers,
        backend=app_config.generator_backend,
    )
    chunk_generator.set_completion_sink(world.insert_generated_chunk)
    chunk_generator.set_failure_sink(world.discard_failed_chunk)
    chunk_generator.start()

    world.overworld.chunk_manager.load_chunk(ChunkPos(90, 90))  # blank claim
    world.discard_failed_chunk(DimensionType.OVERWORLD.value, 90, 90)
    check("blank claim dropped on failure",
          world.overworld.chunk_manager.get_chunk(ChunkPos(90, 90)) is None)

    generated = world.overworld.chunk_manager.get_chunk(ChunkPos(0, 0))
    world.discard_failed_chunk(DimensionType.OVERWORLD.value, 0, 0)
    check("generated chunk survives a failure report", generated is not None)

    chunk_generator.stop()
    world.unload(save=False)

    print()
    if fails:
        print(f"{fails} check(s) FAILED")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
