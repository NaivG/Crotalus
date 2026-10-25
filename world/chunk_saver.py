"""
Streaming chunk save.

A chunk that leaves memory has to reach its region file before its object is
dropped, and it has to do so *without* stalling the dimension actor. The actor
budgets 50 ms per tick and already spends ~18 ms of it encoding chunk packets;
`AnvilChunkData.serialize` plus a region write is milliseconds more, and at
radius 10 a player crossing one chunk border makes ~25 chunks leave the view.
Saving those synchronously turns a step into a lag spike, so the tick path must
not do the I/O.

The save is therefore split in two. The actor thread only *hands over* the
`Chunk` (a list append plus a condition notify); a small pool of saver threads
serializes it and writes it. That handover is a transfer of ownership, not a
share: the caller must have removed the chunk from its manager first and must
not touch it afterwards. The rule is what makes this safe without a lock on the
chunk -- nothing but the saver references an unloaded chunk.

Ordering
--------
Entries are written in submission order and are never coalesced. Two entries for
one coordinate are two different `Chunk` objects representing two different
points in time, so dropping the older one would drop whatever edits it held.
Bounded depth plus an inline save on overflow keeps "a submitted chunk is
always written" true without letting the queue grow without limit.

That inline save is the backpressure: it costs the caller a disk write instead
of costing the process memory. It should be rare (the queue holds a few hundred
chunks), and it is logged so a queue that is permanently behind is visible.

Read-after-write
----------------
The hazard this design creates is a chunk being *re-loaded* before its own save
landed: the load would read the stale copy from disk, and the newer in-memory
copy would later write that staleness back. `flush_key` closes it -- the load
path waits for that coordinate's pending saves before reading the region file.
The wait is one chunk's work and only happens when the coordinate really is in
flight.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple

from loguru import logger

from world.chunk import Chunk
from world.dimension import DimensionType


def should_persist(chunk: Chunk) -> bool:
    """Whether an unloaded chunk is worth writing at all.

    Four cases, and the third is why `Chunk.is_persisted` exists:

    * blocks dirty -- the player changed something; the disk copy is wrong.
    * light dirty -- the flood-fill engine changed the light without changing
      a block. Kept as its own flag rather than folded into `is_dirty` because
      a chunk that scrolls out of view is lit on the way in and would otherwise
      look modified on every visit, rewriting its whole view to disk. See
      `Chunk.is_light_dirty`.
    * generated but never persisted -- the terrain exists only in memory, and
      regenerating it costs seconds per chunk with the vanilla backend while
      writing it costs milliseconds.
    * generated and unchanged since it was read from disk -- writing it back
      would rewrite bytes that are already there. Skipping this is what keeps
      "walk around the world" from turning into a continuous region rewrite.

    The policy lives here rather than in the lifecycle because both paths that
    can drop a chunk (the lifecycle's unload pass and the manager's LRU safety
    valve) have to answer it identically.
    """
    return (
        chunk.is_dirty
        or chunk.is_light_dirty
        or (chunk.is_generated and not chunk.is_persisted)
    )


def _warn_once(store: Dict[str, float], key: str, message: str, interval: float = 30.0) -> None:
    """Log `message` at most once per `interval` seconds for `key`."""
    now = time.monotonic()
    last = store.get(key)
    if last is None or now - last >= interval:
        store[key] = now
        logger.warning(message)


@dataclass
class SaveEntry:
    """One chunk on its way to disk. Owns the `Chunk` until the write finishes."""

    dimension: DimensionType
    chunk: Chunk
    submitted_at: float
    #: Set once the entry's bytes are in the region file (or the write failed).
    #: `flush_key` waits on it; every path must set it, including the failure
    #: path, or a loader would wait out its whole timeout.
    done: threading.Event = field(default_factory=threading.Event)
    error: Optional[BaseException] = None
    #: Set by `_finish_batch` so the in-flight accounting happens exactly once
    #: even if the batch path unwinds early. An entry whose `done` never fires
    #: would strand every loader waiting on its coordinate.
    finished: bool = False

    @property
    def key(self) -> Tuple[str, int, int]:
        pos = self.chunk.position
        return (self.dimension.value, pos.x, pos.z)


class ChunkSaver:
    """Writes unloaded chunks to their region files on its own threads.

    Constructed once per `World` (region files are per dimension, and the
    manager for each dimension is `World`'s). Started lazily on the first
    submit, so a world that never unloads a chunk never pays for the threads.
    """

    def __init__(
        self,
        region_manager_for: Callable[[DimensionType], Optional[object]],
        *,
        workers: int = 1,
        max_pending: int = 512,
        batch_size: int = 32,
        sync_writes: bool = False,
        name: str = "ChunkSaver",
    ):
        """
        Args:
            region_manager_for: dimension -> `RegionManager` (or None).
            workers: saver threads. Serialization is pure Python and holds the
                GIL, so one worker is the honest default; a second only helps
                once the write itself (I/O release) dominates.
            max_pending: queue depth before `submit` falls back to an inline
                save on the caller's thread.
            batch_size: how many ready entries a worker stages before asking
                the region layer to flush. Staging several chunks of one region
                and flushing once is what keeps a region file from being
                rewritten once per chunk.
            sync_writes: flush after every entry instead of batching
                (`server.sync-chunk-writes` in config.toml). Durable per chunk,
                slower.
            name: thread name prefix.
        """
        self._region_manager_for = region_manager_for
        self.workers = max(1, int(workers))
        self.max_pending = max(1, int(max_pending))
        self.batch_size = 1 if sync_writes else max(1, int(batch_size))
        self.sync_writes = sync_writes
        self._name = name

        self._cond = threading.Condition()
        self._queue: Deque[SaveEntry] = deque()
        #: key -> entries not yet written, including the ones a worker has
        #: already popped. `flush_key` has to see both: an entry is most
        #: likely to be *in flight* exactly when a loader asks about it, so
        #: looking only at the queue would answer "nothing pending" for the one
        #: case the wait exists for.
        self._pending: Dict[Tuple[str, int, int], List[SaveEntry]] = {}
        self._threads: List[threading.Thread] = []
        self._running = False
        self._in_flight = 0

        # Stats. `counters` is read by stats() only; every mutation happens
        # under `_cond` so the numbers are consistent with the queue.
        self._counters = {
            "submitted": 0,
            "saved": 0,
            "failed": 0,
            "inline": 0,
            "skipped_clean": 0,
            "flushes": 0,
        }
        self._bytes_written = 0
        self._warned: Dict[str, float] = {}
        self._closed = False

    # ---------------- lifecycle ----------------

    def start(self) -> None:
        """Start the saver threads. Idempotent."""
        with self._cond:
            if self._running or self._closed:
                return
            self._running = True
            for index in range(self.workers):
                thread = threading.Thread(
                    target=self._run,
                    name=f"{self._name}-{index}",
                    daemon=True,
                )
                self._threads.append(thread)
                thread.start()
        logger.info(f"{self._name} started ({self.workers} worker(s))")

    def close(self, timeout: float = 30.0) -> int:
        """Flush everything and stop the workers. Returns chunks still pending."""
        self.flush(timeout=timeout)
        with self._cond:
            self._closed = True
            self._running = False
            self._cond.notify_all()
        threads, self._threads = self._threads, []
        for thread in threads:
            thread.join(timeout=5.0)
        stats = self.stats()
        if threads:
            logger.info(
                f"{self._name} stopped ({stats['saved']} saved, {stats['failed']} failed)"
            )
        return stats["pending"]

    # ---------------- submit ----------------

    def submit(self, dimension: DimensionType, chunk: Chunk, *, force: bool = False) -> bool:
        """Hand a chunk to the saver.

        Returns True when the chunk is written or queued for writing, False
        only when it could not be written at all (no region manager for the
        dimension). `force=False` skips a chunk with nothing to write.

        The caller must have removed `chunk` from its `ChunkManager` (or
        otherwise own it exclusively) before calling: from here on the saver
        thread owns it.
        """
        if not force and not chunk.is_dirty:
            with self._cond:
                self._counters["skipped_clean"] += 1
            return True

        entry = SaveEntry(dimension=dimension, chunk=chunk, submitted_at=time.monotonic())
        if not self._closed:
            self.start()

        with self._cond:
            self._counters["submitted"] += 1
            # A closed saver has no workers left, so queueing would strand the
            # chunk; writing it here is the only way to keep the promise that a
            # submitted chunk is written. Same for a full queue: pay the disk
            # write on the caller rather than grow memory without limit.
            if self._closed:
                inline = True
                _warn_once(
                    self._warned,
                    "closed",
                    f"{self._name} is closed; saving on the caller thread",
                )
            elif len(self._queue) >= self.max_pending:
                inline = True
                _warn_once(
                    self._warned,
                    "inline",
                    f"{self._name} queue is full ({self.max_pending}); "
                    f"saving on the caller thread",
                )
            else:
                self._queue.append(entry)
                self._pending.setdefault(entry.key, []).append(entry)
                self._cond.notify()
                inline = False
            if inline:
                self._counters["inline"] += 1
                # Counted as in flight so `flush` still waits for it, and so
                # `_finish_batch` cannot drive the counter negative.
                self._in_flight += 1

        if inline:
            self._write_batch([entry])
        return True

    # ---------------- waiting / flushing ----------------

    def is_pending(self, dimension: DimensionType, x: int, z: int) -> bool:
        """Whether a save for this coordinate is queued or in flight."""
        key = (dimension.value, x, z)
        with self._cond:
            return bool(self._pending.get(key))

    def flush_key(
        self, dimension: DimensionType, x: int, z: int, timeout: float = 15.0
    ) -> bool:
        """Wait until this coordinate's pending saves have reached the file.

        Called by the load path before it reads a region file, so a chunk
        cannot be re-loaded from a copy the saver has not overwritten yet.
        Returns False on timeout -- the caller should log and read anyway
        rather than deadlock the dimension.
        """
        key = (dimension.value, x, z)
        with self._cond:
            events = [entry.done for entry in self._pending.get(key, ())]
        if not events:
            return True
        deadline = time.monotonic() + timeout
        for event in events:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not event.wait(remaining):
                logger.error(
                    f"{self._name}: timed out waiting for the save of "
                    f"({x}, {z}) in {dimension.value}"
                )
                return False
        return True

    def flush(self, timeout: Optional[float] = None) -> int:
        """Wait for the queue to drain. Returns the number left, 0 on success."""
        if self._closed:
            return self.pending
        with self._cond:
            if not self._queue and self._in_flight <= 0:
                return 0
        self.start()
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while self._queue or self._in_flight > 0:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    break
                self._cond.wait(0.1 if remaining is None else min(0.1, remaining))
            left = len(self._queue) + max(0, self._in_flight)
            self._counters["flushes"] += 1
        if left:
            logger.warning(f"{self._name}: flush timed out with {left} entries pending")
        return left

    # ---------------- worker ----------------

    def _run(self) -> None:
        while True:
            batch = self._next_batch()
            if batch is None:
                return
            try:
                self._write_batch(batch)
            except BaseException:  # pragma: no cover - defensive
                # Nothing may be left waiting on a batch that died: a saver
                # thread that unwinds without setting `done` would strand every
                # loader that asks about one of its coordinates.
                logger.exception(f"{self._name}: batch write failed")
                self._finish_batch(batch)

    def _next_batch(self) -> Optional[List[SaveEntry]]:
        """Block for work; return None when the saver is being closed."""
        with self._cond:
            while self._running and not self._queue:
                self._cond.wait(0.2)
            if not self._running and not self._queue:
                return None
            batch: List[SaveEntry] = []
            while self._queue and len(batch) < self.batch_size:
                batch.append(self._queue.popleft())
            self._in_flight += len(batch)
            return batch

    def _write_batch(self, batch: List[SaveEntry]) -> None:
        """Serialize and write a batch, then release everyone waiting on it."""
        try:
            self._stage_batch(batch)
        finally:
            self._finish_batch(batch)

    def _stage_batch(self, batch: List[SaveEntry]) -> None:
        grouped: Dict[DimensionType, List[SaveEntry]] = {}
        for entry in batch:
            grouped.setdefault(entry.dimension, []).append(entry)

        for dimension, entries in grouped.items():
            region_manager = None
            try:
                region_manager = self._region_manager_for(dimension)
            except Exception as error:  # pragma: no cover - defensive
                logger.exception(f"{self._name}: region lookup failed: {error}")
            if region_manager is None:
                for entry in entries:
                    entry.error = RuntimeError(
                        f"no region manager for {dimension.value}"
                    )
                continue

            # Serialize first: a failure here leaves the region file untouched,
            # which is the state to fail in.
            staged: List[SaveEntry] = []
            for entry in entries:
                try:
                    region_manager.save_chunk(entry.chunk)
                    staged.append(entry)
                except Exception as error:
                    entry.error = error
                    logger.exception(
                        f"{self._name}: failed to serialize chunk "
                        f"{entry.chunk.position} in {dimension.value}: {error}"
                    )

            if staged:
                try:
                    # One flush per dimension per batch. After the in-place
                    # region writer this only writes the chunks staged above.
                    region_manager.save_all()
                except Exception as error:
                    for entry in staged:
                        entry.error = error
                    logger.exception(
                        f"{self._name}: failed to write {dimension.value} region: {error}"
                    )

        saved = failed = 0
        for entry in batch:
            if entry.error is None:
                saved += 1
            else:
                failed += 1
        with self._cond:
            self._counters["saved"] += saved
            self._counters["failed"] += failed

    def _finish_batch(self, batch: List[SaveEntry]) -> None:
        """Release the batch exactly once: events, pending keys, counters."""
        with self._cond:
            for entry in batch:
                if entry.finished:
                    continue
                entry.finished = True
                entry.done.set()
                self._in_flight -= 1
                entries = self._pending.get(entry.key)
                if entries is not None:
                    try:
                        entries.remove(entry)
                    except ValueError:  # pragma: no cover - defensive
                        pass
                    if not entries:
                        del self._pending[entry.key]
            self._cond.notify_all()

    # ---------------- introspection ----------------

    @property
    def pending(self) -> int:
        with self._cond:
            return len(self._queue) + self._in_flight

    def stats(self) -> Dict[str, int]:
        with self._cond:
            out = dict(self._counters)
            out["pending"] = len(self._queue)
            out["in_flight"] = self._in_flight
            out["workers"] = self.workers
            out["max_pending"] = self.max_pending
        return out
