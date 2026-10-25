"""Benchmark: the two chunk-generation work modes, through the real pipeline.

Runs the actual `chunk_generator` singleton (priority queue, dispatcher,
completion sink) in each work mode and measures settled chunks per second,
so numbers from a GIL build and a free-threaded Python 3.13+ build compare
directly:

    python tools/bench_gen_modes.py [chunks_per_mode] [workers] [mode]

mode is "thread", "multiprocess" or "all" (default). Per mode the pool
starts fresh, one warm-up chunk pays the FlatCache population and the
carver JIT (untimed), then `chunks` chunks settle through the real request
path. The block data is hashed as each chunk lands, and the hashes are
compared across modes afterwards: the work mode must not change the
terrain, byte for byte.

Expectations: on a GIL build "multiprocess" wins by roughly the core count
and "thread" sits near 1.0x (its per-thread generators remove the generator
lock, not the GIL). On a free-threaded 3.13+ build "thread" should scale
with cores while keeping its edge of no payload serialization.

Run: python tools/bench_gen_modes.py
"""

import hashlib
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

logger.remove()  # the bench prints its own results; keep the log quiet

from world.chunk_gen_thread import chunk_generator

SEED = 0
DIMENSION = "minecraft:overworld"
# Warm-up chunk, outside every timed coordinate set: the first chunk of a
# fresh worker pays FlatCache population and the carver JIT.
WARM = (-9999, -9999)


def _gil_state() -> str:
    probe = getattr(sys, "_is_gil_enabled", None)
    if probe is not None:
        try:
            return "GIL enabled" if probe() else "GIL disabled (free-threaded)"
        except Exception:
            pass
    return "GIL state unknown (assumed enabled)"


def terrain_hash(chunk) -> str:
    h = hashlib.sha256()
    for section in chunk.sections:
        h.update(section.blocks.tobytes())
    h.update(chunk.heightmap.tobytes())
    if chunk.biomes is not None:
        h.update(chunk.biomes.tobytes())
    return h.hexdigest()[:16]


def run_mode(mode: str, coords, workers: int, backend: str = "vanilla") -> dict:
    """Generate `coords` through the real singleton in one work mode."""
    settled = threading.Condition()
    state = {"pending": 0}
    failures: list = []
    hashes: dict = {}
    hash_lock = threading.Lock()

    def on_settle() -> None:
        with settled:
            state["pending"] -= 1
            settled.notify_all()

    def sink(dimension: str, chunk) -> None:
        with hash_lock:
            hashes[(chunk.position.x, chunk.position.z)] = terrain_hash(chunk)

    def fail(dimension: str, x: int, z: int) -> None:
        failures.append((dimension, x, z))
        on_settle()

    chunk_generator.configure(workers=workers, backend=backend, mode=mode)
    chunk_generator.set_completion_sink(sink)
    chunk_generator.set_failure_sink(fail)
    chunk_generator.start()

    def batch(coords_, timeout: float) -> float:
        """Request and wait; returns wall seconds (raises if the wait stalls)."""
        with settled:
            state["pending"] = len(coords_)
        t0 = time.perf_counter()
        for x, z in coords_:
            if not chunk_generator.request_chunk(
                x, z, DIMENSION, priority=0, callback=on_settle, seed=SEED
            ):
                failures.append((DIMENSION, x, z))
                on_settle()  # refused: settled without work
        with settled:
            ok = settled.wait_for(lambda: state["pending"] <= 0, timeout=timeout)
        if not ok:
            raise TimeoutError(
                f"{mode}: {state['pending']} chunk(s) never settled"
            )
        return time.perf_counter() - t0

    try:
        batch([WARM], timeout=300.0)  # warm-up, untimed
        # One probe per worker, untimed: a single warm-up chunk only warms
        # the one worker that took it. The others pay their one-time setup
        # (registry load, generator build, FlatCache population) inside the
        # timed phase otherwise, burying the steady-state numbers -- the
        # same reason tools/bench_chunk_parallelism.py fans out probes.
        probes = [(-9000 - i, -9000 - i) for i in range(workers)]
        batch(probes, timeout=300.0)
        failures.clear()

        wall = batch(coords, timeout=60.0 + 10.0 * len(coords))
        stats = dict(chunk_generator.get_stats())
        failed = len(failures)
    finally:
        chunk_generator.stop()

    return {
        "mode": mode,
        "wall": wall,
        "chunks_per_s": len(coords) / wall if wall > 0 else 0.0,
        "failed": failed,
        "hashes": hashes,
        "workers": stats.get("workers", workers),
    }


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    mode_arg = (sys.argv[3] if len(sys.argv) > 3 else "all").strip().lower()

    if mode_arg not in ("all", "thread", "multiprocess"):
        print(f"unknown mode {mode_arg!r}; use 'thread', 'multiprocess' or 'all'")
        sys.exit(2)
    modes = ["thread", "multiprocess"] if mode_arg == "all" else [mode_arg]

    if workers <= 0:
        import os

        workers = max(1, min(os.cpu_count() or 4, 8))

    coords = [(i, 0) for i in range(n)]
    print(f"python {sys.version.split()[0]}  |  {_gil_state()}  |  "
          f"{workers} workers  |  {n} chunks per mode\n")

    results = []
    for index, mode in enumerate(modes, start=1):
        print(f"[{index}/{len(modes)}] mode={mode} ...", flush=True)
        try:
            result = run_mode(mode, coords, workers)
        except TimeoutError as e:
            print(f"      FAILED: {e}")
            chunk_generator.stop()
            sys.exit(1)
        print(
            f"      wall {result['wall']:6.2f}s   "
            f"{result['chunks_per_s']:5.2f} chunks/s   failed {result['failed']}"
        )
        results.append(result)

    if len(results) == 2:
        fast, slow = sorted(results, key=lambda r: r["wall"])
        print(f"\nspeedup {fast['mode']}/{slow['mode']}: "
              f"{slow['wall'] / fast['wall']:.2f}x")

    same = len({frozenset(r["hashes"].items()) for r in results}) == 1
    if same:
        print("terrain identity: all modes produced byte-identical chunks")
    else:
        print("TERRAIN MISMATCH:")
        for cx, cz in coords:
            variants = {(r["mode"], r["hashes"].get((cx, cz))) for r in results}
            if len(variants) > 1:
                print(f"  chunk ({cx},{cz}): {sorted(variants)}")


if __name__ == "__main__":
    main()
