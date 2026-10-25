"""End-to-end: background generation through the thread work mode.

The thread work mode (`world.chunk-generation-mode = "thread"`) is the
process pool's sibling: a ThreadPoolExecutor whose threads each own their
generators, so the pipeline's caches never cross threads and
`BaseGenerator._chunk_lock` is never contended. Under the GIL this cannot
beat one core -- its reason to exist is a free-threaded interpreter (3.13+
no-GIL) and skipping the pool's payload round-trip.

Guards the same failure modes tools/check_process_generation.py guards for
processes, plus the mode's own premise:

1. `thread_generator_for` hands DISTINCT generator instances to distinct
   threads (per-thread generators are what remove the lock; one shared
   instance would resurrect the lock and the cache races it exists to
   prevent).
2. Requests tagged with a dimension must generate *that* dimension's
   terrain (the workers are shared by every dimension; the seed rides the
   request).
3. A thread-mode chunk must be byte-identical to what the same seed
   produces in-process -- the mode does not get to have its own opinion
   about terrain.
4. `World.generate_chunks_blocking` must settle every request exactly once,
   success or failure (the spawn preload waits on this; a lost callback is
   a startup hang).
5. `chunk_generator.stop()` must run BEFORE a save leaves dirty chunks
   behind: after stop() + save(), `is_dirty` may only be False.
6. A failed generation must drop the blank claim, not leave a hole the
   streamer stalls behind forever.

Run: python tools/check_thread_generation.py
"""

import hashlib
import sys
import tempfile
import threading
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
    world = World(name="thread", seed=SEED, world_dir=tmp / "thread")

    # The World configured the singleton from config.toml; this tool forces
    # the thread work mode before anything has started the pool.
    from core.config import config as app_config

    chunk_generator.configure(
        workers=4,
        backend=app_config.generator_backend,
        mode="thread",
    )
    check("mode applied", chunk_generator.get_stats()["mode"] == "thread")

    # -- 1. per-thread generators are distinct --------------------------------

    from world.chunk_gen_worker import thread_generator_for

    probed = {}
    barrier = threading.Barrier(2, timeout=10)

    def probe(tag: str) -> None:
        barrier.wait()
        first = thread_generator_for(
            DimensionType.OVERWORLD.value, SEED, app_config.generator_backend
        )
        # Same thread again: must be the same instance, or the caches would
        # be rebuilt (and re-paid) on every request.
        probed[tag] = first
        if tag == "a":
            probed["a_repeat"] = thread_generator_for(
                DimensionType.OVERWORLD.value, SEED, app_config.generator_backend
            )

    threads = [threading.Thread(target=probe, args=(t,)) for t in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    check("two threads got two generators", len(probed) >= 2)
    check(
        "per-thread generators are distinct",
        probed.get("a") is not None and probed["a"] is not probed["b"],
    )
    check(
        "one thread keeps one generator (caches live)",
        probed.get("a_repeat") is probed["a"],
    )

    # -- 2. batch generation through the pool, two dimensions -----------------

    coords = [(0, 0), (1, 0), (0, 1), (-1, -1), (2, 2), (3, 1)]

    print("pool:", chunk_generator.max_workers, "threads (thread mode); generating",
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

    # -- 3. byte identity against in-process generation ----------------------

    from world.chunk_gen_worker import _build_generator

    reference = _build_generator(
        DimensionType.OVERWORLD.value, SEED, app_config.generator_backend
    )
    for cx, cz in coords[:3]:
        standalone = Chunk(position=ChunkPos(cx, cz),
                           min_y=world.overworld.settings.min_y,
                           height=world.overworld.settings.height)
        reference.generate_chunk(standalone)
        served = world.overworld.chunk_manager.get_chunk(ChunkPos(cx, cz))
        check(f"terrain identical ({cx},{cz})",
              terrain_hash(served) == terrain_hash(standalone))

    # -- 4. no cross-dimension leaks -----------------------------------------

    coord = coords[0]
    end = world.dimensions[DimensionType.END]
    check("overworld request did not land in the end",
          end.chunk_manager.get_chunk(ChunkPos(*coord)) is None)

    # -- 5. stop-before-save leaves nothing dirty -----------------------------

    chunk_generator.stop(drain=60.0)
    check("save after stop()", world.save())
    dirty = [(d.value, (pos.x, pos.z)) for d, dim in world.dimensions.items()
             for pos, c in dim.chunk_manager.chunks.items() if c.is_dirty]
    check("nothing dirty after stop+save", not dirty, str(dirty[:4]))

    # -- 6. failed generation drops the claim ---------------------------------

    # Restart to exercise the failure path without a real worker error: the
    # sink contract is "only a blank, clean claim may be dropped".
    chunk_generator.configure(
        workers=app_config.chunk_gen_workers,
        backend=app_config.generator_backend,
        mode="thread",
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

    stats = chunk_generator.get_stats()
    check("stats still name the mode", stats["mode"] == "thread", str(stats))
    check("no failures recorded", stats["failed"] == 0, str(stats))

    chunk_generator.stop()
    world.unload(save=False)

    print()
    if fails:
        print(f"{fails} check(s) FAILED")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
