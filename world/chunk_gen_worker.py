"""
Worker side of background chunk generation, in both work modes.

`ChunkGeneratorThread` (world/chunk_gen_thread.py) runs one of two work
modes, and this module is everything that runs *inside* their workers:

* **multiprocess** dispatches requests to a pool of interpreter processes;
  `init_pool` / `generate_payload` / `chunk_from_payload` are that mode's
  worker side.
* **thread** runs a ThreadPoolExecutor inside this process; `thread_generator_for`
  and `generate_chunk_direct` are its worker side, and give every pool thread
  generators of its own.

Both have to satisfy the same constraints:

* **Spawn-safe** (multiprocess). On Windows the pool children are fresh
  interpreters. Every function here is module-level (pickle resolves them by
  qualified name) and the pool initializer, not import time, does the
  per-child setup.
* **Self-contained per dimension.** Workers are shared by every dimension,
  so a request carries the dimension it is for, and each (dimension, seed)
  pair lazily builds its own generator -- constructed exactly the way
  `Dimension.__init__` builds one, so a worker's terrain is byte-identical
  to what a synchronous generation would produce (verified by
  tools/check_process_generation.py for processes and
  tools/check_thread_generation.py for threads).
* **One generator per worker.** The generator caches parsed trees that carry
  per-chunk state (`NoiseRouter`'s `FlatCache` is the sharpest case), so
  sharing one generator between concurrent workers would need the chunk
  lock and serialize everything. Instead every worker owns its generators:
  one set per PROCESS in multiprocess mode (module globals are per-child),
  one set per THREAD in thread mode (thread-locals). That is what removes
  the lock from the thread work mode -- not a deleted `threading.Lock`, but
  the absence of anything worth locking.
* **Array payloads** (multiprocess). A finished `Chunk` cannot cross a
  process boundary: it carries a `threading.Lock`. The worker ships a plain
  dict of numpy arrays instead, and the parent rebuilds the chunk with
  `chunk_from_payload`. The thread mode never leaves this process, so it
  hands back the chunk object itself and skips that round-trip.
* **Light is not shipped, and that is now deliberate for a different reason
  than it used to be.** Generation does not write light -- the light engine
  does, on the actor thread, once the chunk's nine neighbours are resident
  (`world/light_engine.py`). So the arrays the worker would send are the
  `ChunkSection` defaults, which the receiving side must not mistake for a
  light level; it is `Chunk.light_populated` that says whether they are real,
  and it stays False across this boundary. Sending them would be 4 KB of
  per-section defaults and an invitation to read them.
"""

import threading
import time
from typing import Any, Dict, Tuple

# (dimension value, seed) -> generator. Built lazily on first use so a
# worker only pays for the dimensions it actually serves. Module globals are
# per-process, which is exactly the sharing policy the multiprocess mode
# wants: one generator set per worker process, never shared across them.
_GENERATORS: Dict[tuple, "BaseGenerator"] = {}
_BACKEND: str = "vanilla"


class _ThreadGenerators(threading.local):
    """The calling thread's own generator store (thread work mode).

    A `threading.local` subclass gives every thread that touches
    `_THREAD_GENERATORS` a fresh instance, so each ThreadPoolExecutor worker
    owns its generator dict exclusively: no lock, and no two threads ever
    share the per-chunk cache state the generator holds. Executor threads
    are reused across tasks, so the caches live as long as the pool does.
    """

    def __init__(self):
        self.generators: Dict[tuple, "BaseGenerator"] = {}


_THREAD_GENERATORS = _ThreadGenerators()


def init_pool(backend: str) -> None:
    """Pool initializer. Runs once per worker process, before any request."""
    global _BACKEND
    _BACKEND = backend

    # The parent owns the log. A spawned child re-imports this module and
    # loguru with it, which would add a second stderr sink per child.
    from loguru import logger

    logger.remove()


def _build_generator(dimension: str, seed: int, backend: str):
    """A fresh generator for one dimension, built exactly as `Dimension` builds it.

    The single place both modes construct generators, so a worker's terrain
    cannot drift from a synchronous generation's.
    """
    from world.dimension import DimensionSettings, DimensionType
    from world.generator import GeneratorType, GeneratorSettings, create_generator

    dim_type = DimensionType(dimension)  # ValueError for an unknown dimension
    settings = DimensionSettings.for_dimension(dim_type)
    return create_generator(
        GeneratorSettings(
            generator_type=settings.generator_type or GeneratorType.NORMAL,
            seed=seed,
            preset=settings.preset,
        ),
        backend=backend,
    )


def _generator_for(dimension: str, seed: int):
    """This worker PROCESS's generator for one dimension (multiprocess mode)."""
    key = (dimension, seed)
    generator = _GENERATORS.get(key)
    if generator is None:
        generator = _build_generator(dimension, seed, _BACKEND)
        _GENERATORS[key] = generator
    return generator


def thread_generator_for(dimension: str, seed: int, backend: str):
    """The CALLING THREAD's generator for one dimension (thread work mode).

    Keyed by (dimension, seed, backend) in the thread's own store. Two
    threads asking for the same dimension get two generator instances --
    that isolation is the whole design: every cache a generator holds is
    per-chunk state, so per-thread generators are what let the pipeline run
    lock-free (see `BaseGenerator`'s docstring).
    """
    key = (dimension, seed, backend)
    generators = _THREAD_GENERATORS.generators
    generator = generators.get(key)
    if generator is None:
        generator = _build_generator(dimension, seed, backend)
        generators[key] = generator
    return generator


def generate_payload(chunk_x: int, chunk_z: int, dimension: str, seed: int) -> dict:
    """Generate one chunk standalone and return it as a picklable payload.

    Never touches any chunk manager: the chunk is created, filled and packed
    entirely inside this process. The parent reconstructs and inserts it.
    """
    from world.chunk import Chunk, ChunkPos
    from world.dimension import DimensionSettings, DimensionType

    generator = _generator_for(dimension, seed)
    window = DimensionSettings.for_dimension(DimensionType(dimension))
    chunk = Chunk(
        position=ChunkPos(chunk_x, chunk_z),
        min_y=window.min_y,
        height=window.height,
    )

    start = time.perf_counter()
    generator.generate_chunk(chunk)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    payload = chunk_payload(chunk)
    payload["dimension"] = dimension
    payload["ms"] = elapsed_ms
    return payload


def generate_chunk_direct(
    chunk_x: int, chunk_z: int, dimension: str, seed: int, backend: str
) -> Tuple[Any, float]:
    """Generate one chunk in the calling thread; return (chunk, elapsed_ms).

    The thread work mode's task. The chunk is created, filled and returned
    entirely inside this process -- no payload, no rebuild: the finished
    object goes straight to the completion sink. The generator is the
    calling thread's own (see `thread_generator_for`), so nothing here ever
    synchronizes with another worker.
    """
    from world.chunk import Chunk, ChunkPos
    from world.dimension import DimensionSettings, DimensionType

    generator = thread_generator_for(dimension, seed, backend)
    window = DimensionSettings.for_dimension(DimensionType(dimension))
    chunk = Chunk(
        position=ChunkPos(chunk_x, chunk_z),
        min_y=window.min_y,
        height=window.height,
    )

    start = time.perf_counter()
    generator.generate_chunk(chunk)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return chunk, elapsed_ms


def chunk_payload(chunk) -> dict:
    """Pack a finished chunk into arrays that survive a process boundary.

    `Chunk` itself carries a `threading.Lock`, which pickle refuses. The
    arrays are the whole state a parent needs: section block data with the
    maintained non-air counts, the heightmap and the biome volume. Light is
    not packed -- generation never writes it (see module docstring).
    """
    return {
        "x": chunk.position.x,
        "z": chunk.position.z,
        "min_y": chunk.min_y,
        "height": chunk.height,
        "ys": [section.y for section in chunk.sections],
        "blocks": [section.blocks for section in chunk.sections],
        "non_air": [section._non_air_count for section in chunk.sections],
        "heightmap": chunk.heightmap,
        "biomes": chunk.biomes,
    }


def chunk_from_payload(payload: dict, *, mark_generated: bool = True):
    """Rebuild a `Chunk` from `chunk_payload` output on the parent side."""
    from world.chunk import Chunk, ChunkPos, ChunkSection

    chunk = Chunk(
        position=ChunkPos(payload["x"], payload["z"]),
        min_y=payload["min_y"],
        height=payload["height"],
    )
    # `Chunk.__post_init__` created the section list for this window; the
    # payload's sections are in the same order, so this is an in-place
    # replacement of the defaults, not a resize.
    for index, (y, blocks, non_air) in enumerate(
        zip(payload["ys"], payload["blocks"], payload["non_air"])
    ):
        chunk.sections[index] = ChunkSection(
            y=y, blocks=blocks, _non_air_count=non_air
        )
    chunk.heightmap[:] = payload["heightmap"]
    if payload["biomes"] is not None:
        chunk.biomes = payload["biomes"]
    if mark_generated:
        # Fresh from the generator: it exists, and it owes the world a save.
        chunk.is_generated = True
        chunk.is_dirty = True
    return chunk
