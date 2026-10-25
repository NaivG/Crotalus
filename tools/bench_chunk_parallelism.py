"""Benchmark: would multiprocessing actually speed up chunk generation?

`ChunkGeneratorThread` runs 4 worker threads, but two things serialise the
work no matter how many threads there are:

1. `BaseGenerator._chunk_lock` -- one generator object per dimension, and the
   parsed trees it caches carry per-chunk state, so the whole pipeline runs
   under one lock (world/generator.py, "One lock, and it is load-bearing").
2. The GIL -- the vanilla biome and surface passes are pure Python, and pure
   Python does not release the GIL.

Together that is why 4 workers measured ~1.18x. What no measurement answers
yet is what the same real pipeline does under true process parallelism, and
what that costs. This script measures the vanilla overworld pipeline
(terrain + biomes + surface + carvers + features) under four worker models:

  serial          one worker, one generator            (baseline)
  threads-shared  4 threads on one shared generator     (what production does)
  threads-forked  4 threads, one generator per thread   (lock gone, GIL left)
  processes       4 processes, one generator each, the finished block arrays
                  pickled back to the parent            (true parallelism+IPC)

Every model generates the same chunks with the same seed, and the generated
block data is hashed across models: the worker model must not change the
terrain, byte for byte.

Run: python tools/bench_chunk_parallelism.py [chunks_per_config] [workers]
"""

import hashlib
import itertools
import os
import pickle
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

logger.remove()

SEED = 0
WORKERS = 4
# Warm-up chunk, outside every timed coordinate set. The first chunk of a
# session pays the `FlatCache` population (~600-900 ms) and, per process, one
# numba JIT for the carvers -- none of that belongs in the steady-state numbers.
WARM = (-9999, -9999)


def _make_generator():
    """A generator exactly like a dimension builds one, pinned to vanilla."""
    from world.dimension import DimensionSettings, DimensionType
    from world.generator import GeneratorType, GeneratorSettings, create_generator

    ds = DimensionSettings.for_dimension(DimensionType.OVERWORLD)
    settings = GeneratorSettings(
        generator_type=ds.generator_type or GeneratorType.NORMAL,
        seed=SEED,
        preset=ds.preset,
    )
    return create_generator(settings, backend="vanilla")


def _make_chunk(cx: int, cz: int):
    """A chunk spanning the overworld's real vertical window (-64..319)."""
    from world.chunk import Chunk, ChunkPos
    from world.dimension import DimensionSettings, DimensionType

    ds = DimensionSettings.for_dimension(DimensionType.OVERWORLD)
    return Chunk(position=ChunkPos(cx, cz), min_y=ds.min_y, height=ds.height)


def chunk_payload(chunk) -> dict:
    """What a worker process would ship back to the parent.

    `Chunk` itself cannot cross the process boundary -- it carries a
    `threading.Lock`, which pickle refuses. The arrays can, and they are the
    whole payload a parent needs to rebuild the chunk.
    """
    return {
        "ys": [s.y for s in chunk.sections],
        "blocks": [s.blocks for s in chunk.sections],
        "sky_light": [s.sky_light for s in chunk.sections],
        "non_air": [s._non_air_count for s in chunk.sections],
        "heightmap": chunk.heightmap,
        "biomes": chunk.biomes,
    }


def payload_hash(payload: dict) -> str:
    """Byte identity of the terrain, so worker models can be compared."""
    h = hashlib.sha256()
    for blocks in payload["blocks"]:
        h.update(blocks.tobytes())
    h.update(payload["heightmap"].tobytes())
    if payload["biomes"] is not None:
        h.update(payload["biomes"].tobytes())
    return h.hexdigest()[:16]


def _stats(per_chunk):
    if not per_chunk:
        return "n/a"
    avg = sum(per_chunk) / len(per_chunk)
    return f"avg {avg * 1000:7.1f} ms  min {min(per_chunk) * 1000:7.1f}  max {max(per_chunk) * 1000:7.1f}"


# -- 1. serial ---------------------------------------------------------------

def run_serial(coords):
    gen = _make_generator()
    gen.generate_chunk(_make_chunk(*WARM))

    per_chunk, hashes = [], {}
    sample = None
    t0 = time.perf_counter()
    for cx, cz in coords:
        chunk = _make_chunk(cx, cz)
        t1 = time.perf_counter()
        gen.generate_chunk(chunk)
        per_chunk.append(time.perf_counter() - t1)
        payload = chunk_payload(chunk)
        hashes[(cx, cz)] = payload_hash(payload)
        if sample is None:
            sample = payload
    return time.perf_counter() - t0, per_chunk, hashes, len(pickle.dumps(sample))


# -- 2. threads on one shared generator (what ChunkGeneratorThread does) -----

def run_threads_shared(coords, workers: int = WORKERS):
    gen = _make_generator()
    gen.generate_chunk(_make_chunk(*WARM))

    work: queue.Queue = queue.Queue()
    for c in coords:
        work.put(c)
    per_chunk, hashes = [], {}
    lock = threading.Lock()

    def worker():
        while True:
            try:
                cx, cz = work.get_nowait()
            except queue.Empty:
                return
            chunk = _make_chunk(cx, cz)
            t1 = time.perf_counter()
            gen.generate_chunk(chunk)
            dt = time.perf_counter() - t1
            payload = chunk_payload(chunk)
            with lock:
                per_chunk.append(dt)
                hashes[(cx, cz)] = payload_hash(payload)

    threads = [threading.Thread(target=worker, name=f"bench-{i}") for i in range(workers)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return time.perf_counter() - t0, sorted(per_chunk), hashes


# -- 3. threads, one generator per thread (lock gone, GIL still there) -------

def run_threads_forked(coords, workers: int = WORKERS):
    gens = [_make_generator() for _ in range(workers)]
    for gen in gens:
        gen.generate_chunk(_make_chunk(*WARM))

    per_chunk, hashes = [], {}
    lock = threading.Lock()
    counter = itertools.count()
    total = len(coords)

    def worker(gen):
        while True:
            i = next(counter)
            if i >= total:
                return
            cx, cz = coords[i]
            chunk = _make_chunk(cx, cz)
            t1 = time.perf_counter()
            gen.generate_chunk(chunk)
            dt = time.perf_counter() - t1
            payload = chunk_payload(chunk)
            with lock:
                per_chunk.append(dt)
                hashes[(cx, cz)] = payload_hash(payload)

    threads = [threading.Thread(target=worker, args=(gen,)) for gen in gens]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return time.perf_counter() - t0, sorted(per_chunk), hashes


# -- 4. processes, one generator each, arrays pickled back -------------------

_PROC_GEN = None


def _proc_init():
    global _PROC_GEN
    logger.remove()  # children re-import this module; keep their stderr quiet
    _PROC_GEN = _make_generator()
    _PROC_GEN.generate_chunk(_make_chunk(*WARM))  # FlatCache + numba JIT, untimed


def _proc_generate(coord):
    cx, cz = coord
    chunk = _make_chunk(cx, cz)
    _PROC_GEN.generate_chunk(chunk)
    return chunk_payload(chunk)


def run_processes(coords, workers: int = WORKERS):
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    t_setup0 = time.perf_counter()
    pool = ctx.Pool(processes=workers, initializer=_proc_init)
    try:
        # First result proves one worker is fully up; the probes then fan work
        # out so stragglers finish their own init+warm before the timed phase.
        pool.apply(_proc_generate, (WARM,))
        probes = [pool.apply_async(_proc_generate, ((-9000 - i, -9000 - i),))
                  for i in range(workers)]
        for p in probes:
            p.get()
        t_setup = time.perf_counter() - t_setup0

        t0 = time.perf_counter()
        payloads = pool.map(_proc_generate, list(coords))
        for payload in payloads:  # the parent hashes what it received
            payload_hash(payload)
        wall = time.perf_counter() - t0
        hashes = {(cx, cz): payload_hash(p) for (cx, cz), p in zip(coords, payloads)}
        return wall, t_setup, hashes
    finally:
        pool.terminate()
        pool.join()


# ----------------------------------------------------------------------------

def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else WORKERS
    coords = [(i, 0) for i in range(n)]

    print(f"python {sys.version.split()[0]}  |  {os.cpu_count()} logical cpus")
    try:
        import psutil
        print(f"physical cores: {psutil.cpu_count(logical=False)}")
    except Exception:
        pass
    print(f"backend: vanilla overworld, seed {SEED}, {n} chunks per model, {workers} workers\n")

    print(f"[1/4] serial, 1 generator ...", flush=True)
    wall_s, per_s, hash_s, payload_size = run_serial(coords)
    print(f"      wall {wall_s:6.2f}s   {_stats(per_s)}")
    print(f"      result payload per chunk: ~{payload_size / 1024:.0f} KB (pickled)")

    print(f"[2/4] threads x{workers}, one SHARED generator (= production) ...", flush=True)
    wall_ts, per_ts, hash_ts = run_threads_shared(coords, workers)
    print(f"      wall {wall_ts:6.2f}s   {_stats(per_ts)}   speedup {wall_s / wall_ts:.2f}x")

    print(f"[3/4] threads x{workers}, one generator PER THREAD (no lock) ...", flush=True)
    wall_tf, per_tf, hash_tf = run_threads_forked(coords, workers)
    print(f"      wall {wall_tf:6.2f}s   {_stats(per_tf)}   speedup {wall_s / wall_tf:.2f}x")

    print(f"[4/4] processes x{workers}, one generator each, arrays shipped back ...", flush=True)
    wall_p, t_setup, hash_p = run_processes(coords, workers)
    print(f"      setup {t_setup:6.2f}s (spawn + import + per-worker warm, one-time)")
    print(f"      wall {wall_p:6.2f}s   avg {wall_p / n * 1000:7.1f} ms   speedup {wall_s / wall_p:.2f}x")

    print()
    same = (hash_s == hash_ts == hash_tf == hash_p)
    if same:
        print("terrain identity: all four models produced byte-identical chunks")
    else:
        print("TERRAIN MISMATCH:")
        for cx, cz in coords:
            hs = {(k, v) for k, v in [("serial", hash_s[(cx, cz)]),
                                      ("shared", hash_ts[(cx, cz)]),
                                      ("forked", hash_tf[(cx, cz)]),
                                      ("procs", hash_p[(cx, cz)])]}
            if len(hs) > 1:
                print(f"  chunk ({cx},{cz}): {hs}")


if __name__ == "__main__":
    main()
