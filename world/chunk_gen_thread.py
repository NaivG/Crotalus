"""
Background chunk generation: a priority queue in this process, workers in
one of two interchangeable work modes.

The pipeline's real cost is the vanilla biome and surface passes -- pure
Python, which never releases the GIL -- so how workers are organized decides
what throughput is possible:

* **multiprocess** (default): one dispatcher thread pops requests and
  submits them to a pool of worker PROCESSES (world/chunk_gen_worker.py).
  True parallelism on any build; measured ~2.5x (4 workers) to ~3.7x (8
  workers) on a 4-core/8-thread box, see tools/bench_chunk_parallelism.py.
  A finished chunk crosses the process boundary as a dict of numpy arrays
  and is rebuilt here by `chunk_from_payload`.
* **thread**: a ThreadPoolExecutor whose worker threads each build their
  own generators (thread-locals, world/chunk_gen_worker.py). No generator
  is ever shared between threads, so the caches' per-chunk state cannot
  race and `BaseGenerator._chunk_lock` is never contended -- the same shape
  bench_chunk_parallelism.py measures as "threads-forked", the ceiling
  under the GIL (~1.0x: threads cannot split pure-Python work). The mode
  exists for free-threaded Python 3.13+ builds, where the same structure
  scales with cores, and it skips the payload round-trip: the chunk object
  goes straight to the completion sink.

Select the mode with `world.chunk-generation-mode` in config.toml
(`thread` or `multiprocess`); `World` passes it to `configure()` before
`start()`. The seed is not part of the pool: it travels with each request,
so workers stay world-agnostic and build their generators lazily, per
(dimension, seed).

Both modes keep the old surface: the PriorityQueue with its semantics
(priority first, arrival sequence as FIFO tie-break) and its old API
(`request_chunk`, ...), one dispatcher thread, and a completion path that
hands the finished chunk to the completion sink. `World` installs a sink
that inserts into the chunk manager; `core.server` replaces it with one
that posts a `ChunkReadyMsg` to the owning `DimensionActor`'s inbox, so
the insertion happens on the thread that owns that state.

**Shutdown ordering is load-bearing.** `stop()` must run BEFORE any world
save: a chunk landing in the chunk manager after the save pass has iterated
would stay dirty forever (its region write never happens), and
`World._save_chunks` iterates the manager's dict directly. `stop()`
therefore stops dispatching, cancels what has not started yet (thread
mode), waits (bounded) for in-flight results to be handed off, then kills
the pool. Whatever is still queued or mid-flight is discarded --
generation is deterministic, so the next boot regenerates exactly the same
chunks and nothing is lost.
"""

import itertools
import multiprocessing
import queue
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Optional, Tuple

from loguru import logger

from world.chunk_gen_worker import (
    chunk_from_payload,
    generate_chunk_direct,
    generate_payload,
    init_pool,
)


def _normalize_mode(mode: Optional[str]) -> str:
    """Reduce a configured mode to "thread" or "multiprocess".

    Unknown values fall back to the multiprocess default with a warning,
    here rather than in the config reader: config is a dumb reader, and the
    error belongs where the legal values are known. None keeps whatever is
    already set (call sites that do not care about the mode).
    """
    if mode is None:
        return ""
    value = str(mode).strip().lower()
    if value in ("thread", "threads"):
        return "thread"
    if value in ("multiprocess", "multi-process", "process", "processes", "mp"):
        return "multiprocess"
    logger.warning(
        f"Unknown chunk generation mode {mode!r}; using 'multiprocess' "
        f"(legal values: 'thread', 'multiprocess')"
    )
    return "multiprocess"


def _gil_is_enabled() -> bool:
    """Best effort: does this interpreter currently run with a GIL?

    Python 3.13+ free-threaded builds expose `sys._is_gil_enabled()` (a
    free-threaded build with the GIL forced back on reports True). Anything
    that cannot tell counts as GIL-on, which is the safe assumption.
    """
    probe = getattr(sys, "_is_gil_enabled", None)
    if probe is None:
        return False # by default, Python 3.12 and earlier sys did not have attrs 
    else:
        try:
            return bool(probe())
        except Exception:
            pass
    try:
        import sysconfig

        if sysconfig.get_config_var("Py_GIL_DISABLED"):
            return False
    except Exception:
        pass
    return True


@dataclass
class ChunkRequest:
    """Request for chunk generation.

    Not orderable: a heap key must be able to break every tie, and comparing
    these by coordinate (or worse, by `callback`) would be arbitrary at best.
    Queue keys use a monotonic sequence for that reason — see `request_chunk`.
    """
    chunk_x: int
    chunk_z: int
    # A DimensionType value, e.g. "minecraft:overworld". Kept as a plain
    # string because this is a queue entry; the worker converts it back, so a
    # stale one here fails loudly instead of silently generating in the
    # wrong dimension.
    dimension: str
    # The world seed, travelling with the request: worker processes are
    # world-agnostic and key their generators by (dimension, seed).
    seed: int = 0
    priority: int = 0  # Higher priority requests are processed first
    # Called once the request SETTLES: the finished chunk has been handed to
    # the completion sink, or generation failed. Not "the chunk exists" --
    # a caller that needs the chunk checks the chunk manager. Runs on a
    # worker-side thread (the process pool's result thread, or a thread-mode
    # worker), never the caller's thread.
    callback: Optional[Callable[[], None]] = field(default=None, repr=False)


@dataclass(frozen=True)
class ChunkReadyMsg:
    """A finished, generated chunk posted to the owning dimension's actor.

    Inserting it is the dimension's own job -- on the actor thread, which
    owns the chunk manager. That is the handoff shape the streaming design
    always called for (see world/DESIGN-chunk-streaming.md): workers never
    touch dimension state.
    """
    chunk: Any  # world.chunk.Chunk (quoted: this module stays import-light)


@dataclass(frozen=True)
class ChunkFailedMsg:
    """Generation failed for a claimed chunk; the actor drops the placeholder.

    Without this the blank claim would sit ungenerated forever and the
    streamer's strict ordering would stall the whole view behind it. The
    next pump re-requests the coord from scratch.
    """
    chunk_x: int
    chunk_z: int


class ChunkGeneratorThread:
    """
    Background chunk generation: priority queue + dispatcher + worker pool.

    The class keeps its historical name (call sites and the global
    `chunk_generator` below predate the pool); the only thread it runs
    itself is the dispatcher. `_mode` picks the worker pool behind it:
    "multiprocess" (spawn Pool, array payloads) or "thread"
    (ThreadPoolExecutor, per-thread generators, chunk objects direct).
    """

    def __init__(
        self,
        max_workers: int = 4,
        queue_size: int = 1000,
        mode: str = "multiprocess",
    ):
        """
        Initialize the chunk generator (not yet running).

        Args:
            max_workers: Workers (processes in multiprocess mode, threads in
                thread mode); `World` overwrites this from
                `world.chunk-generation-workers` via `configure()`.
            queue_size: Maximum size of the generation queue
            mode: Work mode, "thread" or "multiprocess"; `World` overwrites
                this from `world.chunk-generation-mode` via `configure()`.
        """
        self.max_workers = max_workers
        self._mode = _normalize_mode(mode) or "multiprocess"
        self._max_queue_size = queue_size

        # Generation queue (priority queue)
        self._queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=queue_size)

        # Tie-break counter for queue keys. Produced strictly under
        # `_seq_lock` together with the enqueue (see `request_chunk`): under
        # the GIL itertools.count was atomic on its own, a free-threaded
        # interpreter makes no such promise, and one collision would have
        # the heap compare ChunkRequest objects -- which are not orderable
        # and raise TypeError.
        self._sequence = itertools.count()
        self._seq_lock = threading.Lock()

        # Worker config. `configure()` applies world settings before
        # `start()`; the spawn context is named explicitly so the process
        # pool behaves identically on Windows and POSIX.
        self._backend = "vanilla"
        self._mp_context = multiprocessing.get_context("spawn")
        self._pool = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._dispatcher: Optional[threading.Thread] = None

        # Completion wiring. `World` installs these; `core.server` swaps in
        # actor-inbox variants while the tick pipeline is up. Sinks are plain
        # attribute reads, safe to swap while running.
        self._completion_sink: Optional[Callable[[str, Any], None]] = None
        self._failure_sink: Optional[Callable[[str, int, int], None]] = None

        # Shutdown control
        self._running = False
        # Results may reach the completion sink. False once stop() has passed
        # the drain: thread-mode tasks cannot be killed the way
        # pool.terminate() reaps processes, so a straggler that finishes
        # generation after the drain must not deliver into a chunk manager a
        # save may already be iterating. See `_complete`.
        self._accepting_results = True
        # True once a pool has actually been built. stop() on a generator
        # that never started is a no-op -- it must not arm the refusal below.
        self._ever_started = False
        # Set by stop(): afterwards bare requests are refused instead of
        # lazily resurrecting the pool during shutdown. An explicit start()
        # clears it.
        self._ever_stopped = False
        # Guards start()/stop() against racing callers: the lazy start runs
        # on whatever thread fires the first request.
        self._lifecycle_lock = threading.Lock()

        # In-flight bookkeeping. The semaphore bounds how many requests live
        # in the pool's internal FIFO (submit + result): two rounds per
        # worker keeps every worker fed while leaving priority meaningful --
        # a burst of low-priority requests cannot bury a high-priority one
        # that arrives a moment later, because the dispatcher only pulls
        # from the priority queue when a slot is free.
        self._slots = threading.BoundedSemaphore(max_workers * 2)

        # Statistics
        self._generated_count = 0
        self._failed_count = 0
        self._total_generation_time = 0.0
        self._in_flight = 0
        self._stats_lock = threading.Lock()

        logger.info(
            f"ChunkGeneratorThread initialized (pool of {max_workers} "
            f"workers, mode {self._mode!r})"
        )

    # -- wiring --------------------------------------------------------------

    def configure(
        self,
        workers: int,
        backend: str = "vanilla",
        mode: Optional[str] = None,
    ) -> None:
        """Apply world settings. Must run before `start()`.

        The generator is a module-global singleton; the world claims it here
        with its own worker count, backend and work mode. The seed is
        deliberately not part of this -- it travels on each request, so a
        second world could share the pool without rebuilding it. `mode=None`
        keeps the current mode (call sites that do not care).
        """
        if self._running:
            # Someone's World is already live. Sharing is correct: requests
            # carry their own seed and dimension, nothing here is per-world.
            logger.debug("Chunk generator already running; keeping its pool")
            return
        self.max_workers = max(1, int(workers))
        self._backend = backend
        normalized = _normalize_mode(mode)
        if normalized:
            self._mode = normalized
        self._slots = threading.BoundedSemaphore(self.max_workers * 2)
        logger.info(
            f"ChunkGeneratorThread configured: {self.max_workers} workers, "
            f"mode {self._mode!r}, backend {backend!r}"
        )

    def set_completion_sink(self, sink: Optional[Callable[[str, Any], None]]) -> None:
        """Install where finished chunks go: `sink(dimension, chunk)`.

        Swappable while running (a plain attribute read on the callback
        thread). The server routes by dimension into the actor inboxes;
        when the tick pipeline is down the world's direct insert takes over.
        """
        self._completion_sink = sink

    def set_failure_sink(self, sink: Optional[Callable[[str, int, int], None]]) -> None:
        """Install where failed requests are reported: `sink(dimension, x, z)`.

        The receiver drops the blank claim (if it is still blank) so the
        streamer re-requests the chunk instead of stalling behind it forever.
        """
        self._failure_sink = sink

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Start the dispatcher and spin up the workers for the chosen mode.

        Also called lazily by the first request: a pool of interpreters is a
        real cost (spawn + registry import per worker), and the callers who
        never use background generation -- most tools, tests -- should not
        pay it. Whoever generates, pays. Thread mode's executor is cheaper
        to build (thread-local generators fill lazily per thread), but it
        goes through the same lazy start for one behavior, not two.
        """
        with self._lifecycle_lock:
            if self._running:
                return
            self._ever_started = True
            self._ever_stopped = False
            self._running = True
            self._accepting_results = True
            if self._mode == "thread":
                self._executor = ThreadPoolExecutor(
                    max_workers=self.max_workers,
                    thread_name_prefix="ChunkGen-worker",
                )
            else:
                self._pool = self._mp_context.Pool(
                    processes=self.max_workers,
                    initializer=init_pool,
                    initargs=(self._backend,),
                )
            self._dispatcher = threading.Thread(
                target=self._dispatch_loop,
                name="ChunkGen-dispatch",
                daemon=True,
            )
            self._dispatcher.start()

        noun = "threads" if self._mode == "thread" else "processes"
        logger.info(
            f"ChunkGeneratorThread started: dispatching to {self.max_workers} "
            f"{noun} ({self._mode} mode)"
        )
        if self._mode == "thread" and _gil_is_enabled():
            logger.warning(
                "Thread work mode is running with the GIL enabled: "
                "per-thread generators remove the generator lock, but the "
                "pure-Python passes still serialize, so expect ~1.0x "
                "throughput. 'multiprocess' parallelizes on this build; "
                "'thread' is meant for free-threaded Python 3.13+."
            )

    def stop(self, timeout: float = 10.0, drain: float = 5.0) -> None:
        """Stop the generator. MUST run before any world save.

        Order inside this method is the whole point: stop dispatching, give
        in-flight generations a bounded window to land in the completion
        sink (they are finished work -- wasting them would be silly), then
        kill the pool. Anything still queued or still generating is
        discarded; the next boot regenerates identical chunks from the seed.

        Args:
            timeout: How long to wait for the dispatcher thread to exit.
            drain: How long to wait for in-flight generations to complete.
                Thread mode additionally cancels its not-yet-started futures
                up front, so the drain only covers genuinely running tasks.
        """
        if not self._ever_started:
            # Never started: nothing to drain, nothing to refuse later.
            return
        if not self._running and self._ever_stopped:
            return

        with self._lifecycle_lock:
            self._running = False
            self._ever_stopped = True

        # 0) Thread mode: cancel everything the dispatcher has not started
        #    yet. Each cancelled future settles immediately (see
        #    `_on_thread_settled`), releasing its slot and in-flight count
        #    right here, so the drain below only waits for generations that
        #    are actually running. Running tasks keep their threads -- a
        #    thread cannot be killed -- and the drain bounds the wait for
        #    their results. The executor reference stays until the
        #    dispatcher is joined; a straggler dispatch reading it must not
        #    find None.
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)

        # The completion callbacks keep running on the pool's result-handler
        # thread (multiprocess) or the worker threads (thread) until the
        # pool is torn down below, so the drain is real.

        # 1) Bounded drain of in-flight work.
        deadline = time.time() + max(0.0, drain)
        while True:
            with self._stats_lock:
                if self._in_flight == 0:
                    break
                remaining = deadline - time.time()
            if remaining <= 0:
                break
            time.sleep(min(0.02, remaining))
        with self._stats_lock:
            left_in_flight = self._in_flight
        if left_in_flight:
            logger.warning(
                f"Chunk generator stopped with {left_in_flight} generation(s) "
                f"in flight; discarded (deterministic: regenerated next boot)"
            )

        # Past the drain, stop refusing results. This is the thread-mode
        # substitute for pool.terminate()'s hard guarantee: multiprocess
        # reaps mid-chunk workers, threads cannot be killed -- so from here
        # a straggler's finished chunk is discarded instead of delivered
        # (`_complete` drops it), and once this method returns nothing lands
        # in the chunk manager anymore. That is what makes stop-before-save
        # safe in both modes.
        self._accepting_results = False

        # 2) Kill the pool, then the dispatcher. Workers may be mid-chunk;
        #    terminate() reaps them without waiting for the chunk to finish.
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None
        if self._dispatcher is not None:
            self._dispatcher.join(timeout=timeout)
            self._dispatcher = None
        self._executor = None

        # 3) Anything still queued was never dispatched. Drop it (and keep
        #    the unfinished-task count honest).
        discarded = self.clear_queue()

        logger.info(
            f"ChunkGeneratorThread stopped (generated: {self._generated_count}, "
            f"failed: {self._failed_count}, discarded queued: {discarded})"
        )

    # -- requests ------------------------------------------------------------

    def request_chunk(
        self,
        chunk_x: int,
        chunk_z: int,
        dimension: str = "minecraft:overworld",
        priority: int = 0,
        callback: Optional[Callable[[], None]] = None,
        seed: int = 0,
    ) -> bool:
        """
        Request a chunk to be generated.

        Args:
            chunk_x: Chunk X coordinate
            chunk_z: Chunk Z coordinate
            dimension: Dimension name
            seed: World seed the chunk is generated with
            priority: Request priority (higher = more urgent)
            callback: Called on a worker-side thread once the request
                settles (chunk delivered, or generation failed)

        Returns:
            True if request was queued, False if queue is full
        """
        if not self._running:
            if self._ever_stopped:
                # Stopped for real (shutdown): refuse instead of resurrecting
                # the pool for a straggler request.
                return False
            # Lazy start: the first request pays for the pool.
            self.start()

        try:
            request = ChunkRequest(
                chunk_x=chunk_x,
                chunk_z=chunk_z,
                dimension=dimension,
                seed=seed,
                priority=priority,
                callback=callback,
            )
            # Heap key: (negated priority, arrival sequence). The sequence
            # makes every key unique, so the heap never has to fall through to
            # comparing ChunkRequest objects — which are not orderable and
            # would raise TypeError whenever two requests shared a priority and
            # a timestamp. It also gives strict FIFO within a priority, which
            # wall-clock time could not guarantee (clock steps reorder it).
            # Sequence assignment and enqueue share one lock: the GIL made
            # `next()` atomic on its own, a free-threaded interpreter does
            # not, and a duplicated sequence would order requests by
            # comparing ChunkRequest objects (TypeError).
            with self._seq_lock:
                self._queue.put((-priority, next(self._sequence), request), block=False)
            return True
        except queue.Full:
            logger.warning(f"Chunk generation queue is full, rejecting request for ({chunk_x}, {chunk_z})")
            return False

    def request_chunks(
        self,
        chunks: list[Tuple[int, int]],
        dimension: str = "overworld",
        priority: int = 0,
        seed: int = 0,
    ) -> int:
        """
        Request multiple chunks to be generated.

        Args:
            chunks: List of (chunk_x, chunk_z) tuples
            dimension: Dimension name
            priority: Generation priority
            seed: World seed the chunks are generated with

        Returns:
            Number of requests successfully queued
        """
        count = 0
        for chunk_x, chunk_z in chunks:
            if self.request_chunk(chunk_x, chunk_z, dimension, priority, seed=seed):
                count += 1
        return count

    # -- dispatcher ------------------------------------------------------------

    def _dispatch_loop(self) -> None:
        """Dispatcher thread: priority queue -> worker pool (either mode).

        The get() is isolated from everything after it. Once an item is off
        the queue, exactly one task_done() must follow for it -- skipping
        that leaks the unfinished-task count, calling it twice underflows it
        and makes every later task_done() raise.
        """
        while True:
            try:
                _, _, request = self._queue.get(timeout=0.1)
            except queue.Empty:
                if not self._running:
                    break
                continue
            except Exception as e:
                logger.exception(f"Unexpected error dequeuing chunk request: {e}")
                continue

            try:
                if not self._running:
                    # Stopped with this item still undispatched: discard it.
                    # Whoever wants the chunk re-requests next session.
                    logger.debug(
                        f"Discarded queued chunk request ({request.chunk_x}, "
                        f"{request.chunk_z}) during shutdown"
                    )
                    continue
                self._dispatch(request)
            except Exception as e:
                # A dispatcher that dies here stops draining the queue, so
                # dispatch failures are logged and the loop continues.
                logger.exception(f"Unexpected error dispatching chunk request: {e}")
                with self._stats_lock:
                    self._failed_count += 1
                self._report_failure(request)
            finally:
                self._queue.task_done()

        logger.debug("Chunk generator dispatcher stopped")

    def _dispatch(self, request: ChunkRequest) -> None:
        """Submit one request to the workers. Never raises for pool errors."""
        # Wait for a free in-flight slot (bounded pool FIFO, see __init__).
        # Checks _running before and after the acquire: a stop() flipping
        # _running mid-acquire must not leave a slot held for a submit that
        # will never happen.
        while True:
            if not self._running:
                return  # discarded; the caller does the task_done()
            if self._slots.acquire(timeout=0.2):
                break
        if not self._running:
            self._slots.release()
            return

        # Counted before the submit: a fast task (or a fast failure) could
        # otherwise settle and decrement before this increment ran, dipping
        # the count below zero and lying to stop()'s drain.
        with self._stats_lock:
            self._in_flight += 1
        try:
            if self._mode == "thread":
                future = self._executor.submit(self._thread_task, request)
                # The settled callback owns the bookkeeping for BOTH endings:
                # a finished task, and a future `stop()` cancelled before it
                # ever started -- a cancelled task never runs, so this hook
                # is the only place its slot and count are released from.
                future.add_done_callback(partial(self._on_thread_settled, request))
            else:
                self._pool.apply_async(
                    generate_payload,
                    (request.chunk_x, request.chunk_z, request.dimension, request.seed),
                    callback=partial(self._on_result, request),
                    error_callback=partial(self._on_error, request),
                )
        except Exception as e:
            with self._stats_lock:
                self._in_flight -= 1
            self._slots.release()
            with self._stats_lock:
                self._failed_count += 1
            self._report_failure(request)
            logger.error(
                f"Chunk generation dispatch failed for "
                f"({request.chunk_x}, {request.chunk_z}): {e!r}"
            )
            return

    def _thread_task(self, request: ChunkRequest) -> None:
        """One generation in a pool thread, on that thread's own generator.

        Per-thread generators are the point of this work mode: every cache
        the pipeline owns (FlatCache, parsed surface rules, biome trees...)
        is per-thread state, so workers never synchronize through the
        generator and `BaseGenerator._chunk_lock` is never contended. The
        chunk object itself goes to the completion sink -- same process, no
        payload round-trip.
        """
        try:
            chunk, elapsed_ms = generate_chunk_direct(
                request.chunk_x,
                request.chunk_z,
                request.dimension,
                request.seed,
                self._backend,
            )
        except Exception as e:
            logger.error(
                f"Error generating chunk ({request.chunk_x}, {request.chunk_z}) in "
                f"{request.dimension}: {e!r}"
            )
            with self._stats_lock:
                self._failed_count += 1
            self._report_failure(request)
            # Settled-semantics: a caller waiting on the callback (the spawn
            # preload does) must hear about failures too, or it waits forever.
            self._settle_callback(request)
            return
        self._complete(request, chunk, elapsed_ms)

    def _on_thread_settled(self, request: ChunkRequest, future: Future) -> None:
        """Done-callback for a thread-mode task: release what the request held.

        Runs on the worker thread that finished the task -- or inline on the
        thread that cancelled it, for futures `stop()` cancelled before they
        started (a cancelled task never runs, so this hook is the only place
        its slot and in-flight count can be released from).
        """
        if future.cancelled():
            # Never started: nothing was generated, so there is no chunk and
            # no real failure -- but the request must still settle. Failure
            # semantics fit: the failure sink drops the blank claim (if it
            # is one), and the callback releases a waiting preload.
            self._report_failure(request)
            self._settle_callback(request)
        with self._stats_lock:
            self._in_flight -= 1
        self._slots.release()

    def _on_result(self, request: ChunkRequest, payload: dict) -> None:
        """Pool callback: a chunk finished. Runs on the pool's result thread."""
        try:
            try:
                chunk = chunk_from_payload(payload)
            except Exception as e:
                # A malformed payload would otherwise escape into the pool's
                # result-handler thread and take the callback path down with
                # it. Counted as a failure; the claim (if any) is dropped so
                # the streamer can re-request.
                logger.exception(
                    f"Failed to rebuild generated chunk "
                    f"({request.chunk_x}, {request.chunk_z}): {e}"
                )
                with self._stats_lock:
                    self._failed_count += 1
                self._report_failure(request)
                self._settle_callback(request)
                return

            self._complete(request, chunk, payload.get("ms", 0.0))
        finally:
            with self._stats_lock:
                self._in_flight -= 1
            self._slots.release()

    def _complete(self, request: ChunkRequest, chunk, elapsed_ms: float) -> None:
        """Hand a finished chunk to the completion sink.

        Shared by both modes: the pool's result thread arrives here from
        `_on_result` (after rebuilding the chunk from its payload), a
        thread-mode worker arrives straight from `_thread_task` with the
        chunk it just built.
        """
        if not self._accepting_results:
            # stop() has passed the drain: this result is too late. The next
            # boot regenerates the chunk from the seed, so nothing is lost --
            # the same fate a mid-chunk worker meets under pool.terminate().
            logger.debug(
                f"Discarded late chunk result ({request.chunk_x}, "
                f"{request.chunk_z}) after generator stop"
            )
            return
        sink = self._completion_sink
        if sink is None:
            logger.error(
                f"Generated chunk ({request.chunk_x}, {request.chunk_z}) has "
                f"nowhere to go: no completion sink installed"
            )
            with self._stats_lock:
                self._failed_count += 1
            self._report_failure(request)
            self._settle_callback(request)
            return
        try:
            sink(request.dimension, chunk)
        except Exception as e:
            logger.exception(
                f"Failed to hand off generated chunk "
                f"({request.chunk_x}, {request.chunk_z}): {e}"
            )
            with self._stats_lock:
                self._failed_count += 1
            # A sink that raised leaves the claim blank: report it, or the
            # streamer stalls behind that hole forever.
            self._report_failure(request)
            self._settle_callback(request)
            return

        with self._stats_lock:
            self._generated_count += 1
            self._total_generation_time += elapsed_ms / 1000.0

        self._settle_callback(request)

    def _settle_callback(self, request: ChunkRequest) -> None:
        """Run the request's own callback, if any. Every settle path calls it."""
        if request.callback:
            try:
                request.callback()
            except Exception as e:
                logger.exception(f"Error in chunk generation callback: {e}")

    def _on_error(self, request: ChunkRequest, exception: BaseException) -> None:
        """Pool callback: the worker raised. Runs on the pool's result thread."""
        logger.error(
            f"Error generating chunk ({request.chunk_x}, {request.chunk_z}) in "
            f"{request.dimension}: {exception!r}"
        )
        with self._stats_lock:
            self._failed_count += 1
        self._report_failure(request)
        # Settled-semantics: a caller waiting on the callback (the spawn
        # preload does) must hear about failures too, or it waits forever.
        self._settle_callback(request)
        with self._stats_lock:
            self._in_flight -= 1
        self._slots.release()

    def _report_failure(self, request: ChunkRequest) -> None:
        """Tell the failure sink (if any) so a blank claim can be dropped."""
        sink = self._failure_sink
        if sink is None:
            return
        try:
            sink(request.dimension, request.chunk_x, request.chunk_z)
        except Exception as e:
            logger.exception(f"Error in chunk generation failure sink: {e}")

    # -- introspection ---------------------------------------------------------

    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()

    @property
    def is_running(self) -> bool:
        """Check if generator is running."""
        return self._running

    def get_stats(self) -> dict:
        """Get generation statistics."""
        with self._stats_lock:
            avg_time = self._total_generation_time / max(1, self._generated_count)
            return {
                "mode": self._mode,
                "generated": self._generated_count,
                "failed": self._failed_count,
                "queue_size": self.queue_size,
                "in_flight": self._in_flight,
                "workers": self.max_workers,
                "avg_generation_time": avg_time,
            }

    def clear_queue(self) -> int:
        """Clear all pending requests."""
        count = 0
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
            # Each dequeued item still owes a task_done(), or the unfinished
            # count stays elevated and join() would never return.
            self._queue.task_done()
            count += 1
        return count


# Global chunk generator instance. Worker count, backend and work mode are
# applied by World via configure() before start(); the seed never lives here
# because it travels with each request.
chunk_generator = ChunkGeneratorThread(max_workers=4)
