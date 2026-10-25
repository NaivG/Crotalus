"""The blocks the kernel may read: resident chunk sections, as flat arrays.

    view = WorldView(min_y=-64, height=384)
    view.add(chunk)                      # on load
    view.remove(chunk.position.x, chunk.position.z)   # on unload
    keys, slots = view.index_arrays()    # what the kernel binary-searches

`WorldView` holds the **live** `ChunkSection.blocks` arrays, not copies. That is
the whole point: the physics kernel runs every tick over the same chunks, and a
128 KB copy per chunk per tick is the difference between 0.6 ms and 300 ms.
`Chunk.raw_sections()` is the sanctioned way to get at them, and it documents
the two rules that come with the zero copy (read-only, owning thread only).

Why an index at all
-------------------

The kernel has to answer "which section array holds block (x, y, z)" without
touching a Python object, so it gets two parallel `int64`/`int32` arrays sorted
by chunk key and binary-searches them. Sorting happens lazily -- a chunk load
or unload sets a dirty flag and the next `index_arrays()` call rebuilds with
`np.argsort`, which for a few thousand chunks is tens of microseconds and
happens only when the resident set actually changed. Rebuilding every tick
would be work nobody asked for; rebuilding never is a stale world.

* **The container is `numba.typed.List`, not a Python `list`.** Passing a
  Python list of 15000 section arrays to an `njit` function costs **222 ms per
  call** -- numba walks the whole list on every call, whatever the function
  does with it. The same list as a `typed.List` costs 5 µs. Measured with
  `tools/bench_physics.py`; it is the difference between 0.7 ms and 700 ms per
  tick, and it is why this class exists rather than the solver handing over
  `[section.blocks for ...]`.

Missing chunks
--------------

A chunk that is not resident has **no entry** in the index, and the kernel
treats that as "cannot answer": the body is skipped for the tick and counted
(`deferred_unloaded`), keeping its position and velocity. It is deliberately
not air and deliberately not solid -- the same stance `entity/spawn_placement.py`
takes ("unloaded chunks reject, they do not read as air").
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from numba import types
from numba.typed import List as TypedList

__all__ = ["WorldView", "chunk_key", "unpack_chunk_key"]


def chunk_key(chunk_x: int, chunk_z: int) -> int:
    """A single `int64` naming a chunk column, ordered by `(x, z)`.

    `(x << 32) | z` with `z` masked to 32 bits: for `x1 < x2` every key of `x1`
    is below every key of `x2` (the low 32 bits never carry), so sorting by key
    sorts by chunk X then chunk Z and the kernel can binary-search. Two's
    complement makes the same expression work for negative coordinates in both
    Python and `njit`.

    The key is an `int64` and this is the one place it is defined, so the space
    it names is **|chunk_x|, |chunk_z| < 2^31** (about 3.4e10 blocks). Beyond
    that -- and for a non-finite coordinate, which casts to `INT64_MIN` -- the
    pack wraps and two different chunks can share a key. That is inherited by
    every user of it (the kernel's `_key`, `WorldView`'s index, the physics
    load-level gate): at such a coordinate they are all wrong *the same way*,
    rather than disagreeing with each other. Vanilla's world border is at
    +/-3e7 blocks (1.9e6 chunks), a thousand times inside the limit.
    """
    return (int(chunk_x) << 32) | (int(chunk_z) & 0xFFFFFFFF)


def unpack_chunk_key(key: int) -> Tuple[int, int]:
    """`(chunk_x, chunk_z)` from a key built by `chunk_key`.

    The inverse of the packing above, and the only place it is taken apart:
    `>> 32` is an arithmetic shift, so negative chunk X comes back negative,
    and the low word is re-signed because `chunk_key` masks `z` to unsigned.
    """
    value = int(key)
    chunk_x = value >> 32
    chunk_z = value & 0xFFFFFFFF
    if chunk_z >= 0x80000000:
        chunk_z -= 0x100000000
    return chunk_x, chunk_z


class WorldView:
    """Resident chunks as a flat list of section arrays plus a sorted index."""

    __slots__ = ("min_y", "height", "min_section_y", "sections_per_chunk",
                 "_sections", "_air", "capacity", "_count", "_dirty",
                 "_key_of_slot", "_alive", "_free", "_slot_of",
                 "_index_keys", "_index_slots", "_index_count")

    def __init__(self, min_y: int = -64, height: int = 384,
                 capacity: int = 512) -> None:
        if height <= 0 or height % 16:
            raise ValueError(f"height must be a positive multiple of 16, got {height}")
        self.min_y = int(min_y)
        self.height = int(height)
        self.min_section_y = self.min_y // 16
        self.sections_per_chunk = self.height // 16
        self.capacity = max(int(capacity), 1)
        #: One shared air array for slots that hold no chunk. The kernel can
        #: read it safely (it reads as air) but never should: the residency
        #: check refuses a tick whose chunks are not all present.
        self._air = np.zeros(4096, dtype=np.uint16)
        #: Flat `slot * sections_per_chunk + section_index` -> the live
        #: `uint16[4096]` array. A `typed.List` because a Python list of this
        #: size costs 222 ms to pass into `njit`; see the module docstring.
        self._sections: TypedList = TypedList.empty_list(types.uint16[::1])
        self._count = 0
        self._dirty = True
        self._key_of_slot = np.zeros(self.capacity, dtype=np.int64)
        self._alive = np.zeros(self.capacity, dtype=bool)
        self._free: List[int] = []
        self._slot_of: Dict[int, int] = {}
        self._index_keys = np.zeros(self.capacity, dtype=np.int64)
        self._index_slots = np.zeros(self.capacity, dtype=np.int32)
        self._index_count = 0

    # ----- membership -----

    def _grow(self, needed: int) -> None:
        capacity = self.capacity
        while capacity < needed:
            capacity *= 2
        for name, dtype in (("_key_of_slot", np.int64), ("_alive", bool),
                            ("_index_keys", np.int64), ("_index_slots", np.int32)):
            grown = np.zeros(capacity, dtype=dtype)
            grown[:self.capacity] = getattr(self, name)[:self.capacity]
            setattr(self, name, grown)
        self.capacity = capacity

    def _write_slot(self, slot: int, arrays) -> None:
        """Put a chunk's section arrays (or air) into a slot's flat range.

        The one place the `typed.List` is written, so the slot layout has a
        single definition. The dtype and shape check is not decoration: a
        section array of the wrong type would give the kernel a second
        specialization per call, and numba would recompile the whole kernel
        every tick rather than once.
        """
        per_chunk = self.sections_per_chunk
        base = slot * per_chunk
        while len(self._sections) < base + per_chunk:
            self._sections.append(self._air)
        for index, array in enumerate(arrays):
            if array.dtype != np.uint16 or array.shape != (4096,):
                raise TypeError(
                    f"section array must be uint16[4096], got {array.dtype}"
                    f"{array.shape}; chunk slot {slot}, section {index}")
            self._sections[base + index] = array

    def add(self, chunk) -> int:
        """Register a chunk (anything with `position` and `raw_sections()`).

        Returns its slot. Re-adding a resident chunk refreshes its section
        arrays -- a chunk object replaced in place (a reload) must not leave the
        view pointing at the old one's arrays.
        """
        key = chunk_key(chunk.position.x, chunk.position.z)
        slot = self._slot_of.get(key)
        if slot is None:
            if self._free:
                slot = self._free.pop()
            else:
                if self._count == self.capacity:
                    self._grow(self._count + 1)
                slot = self._count
                self._count += 1
            self._slot_of[key] = slot
            self._key_of_slot[slot] = key
            self._alive[slot] = True
            self._dirty = True
        arrays = chunk.raw_sections()
        if len(arrays) != self.sections_per_chunk:
            raise ValueError(
                f"chunk {chunk.position} has {len(arrays)} sections, "
                f"the view was built for {self.sections_per_chunk}")
        self._write_slot(slot, arrays)
        return slot

    def remove(self, chunk_x: int, chunk_z: int) -> bool:
        """Forget a chunk. False when it was not registered.

        The slot's section arrays go back to the shared air array rather than
        being deleted: a `typed.List` cannot shrink, and the slot is handed to
        the next chunk that loads.
        """
        key = chunk_key(chunk_x, chunk_z)
        slot = self._slot_of.pop(key, None)
        if slot is None:
            return False
        self._write_slot(slot, [self._air] * self.sections_per_chunk)
        self._key_of_slot[slot] = 0
        self._alive[slot] = False
        self._free.append(slot)
        self._dirty = True
        return True

    def has(self, chunk_x: int, chunk_z: int) -> bool:
        return chunk_key(chunk_x, chunk_z) in self._slot_of

    def slot_of(self, chunk_x: int, chunk_z: int) -> Optional[int]:
        return self._slot_of.get(chunk_key(chunk_x, chunk_z))

    def rebuild(self, chunks: Iterable) -> int:
        """Replace the whole view from an iterable of chunks. Returns the count.

        For tools and for the first tick after a dimension loads: a full rebuild
        is O(resident), and an incremental `add`/`remove` is what the streaming
        path should use.
        """
        self._slot_of.clear()
        self._alive[:self._count] = False
        self._key_of_slot[:self._count] = 0
        self._sections = TypedList.empty_list(types.uint16[::1])
        self._count = 0
        self._free = []
        self._dirty = True
        count = 0
        for chunk in chunks:
            self.add(chunk)
            count += 1
        return count

    def sync(self, chunks: Iterable, ready=None,
             max_adds: Optional[int] = None) -> Tuple[int, int]:
        """Make the view match `chunks` exactly. Returns `(added, removed)`.

        A diff rather than a pair of hooks on the chunk manager, and that is a
        deliberate trade: `ChunkManager` gains and loses chunks from at least
        four paths (a claim, a worker's finished chunk, the lifecycle's unload
        pass, the LRU valve) and a missed hook is a mob standing on a chunk that
        is gone, or falling through one that has not arrived. One pass over the
        resident set cannot miss a path, because it does not know about paths.

        It is also not a rebuild: nothing is re-sorted and no array is rewritten
        unless a chunk actually appeared or disappeared, so a quiet tick pays
        one dict lookup per resident chunk -- about 0.1 us each, measured
        against `physics.tick`'s 1.9 ms at 500 bodies.

        `ready` filters which chunks count as loaded. The actor passes
        `Chunk.is_generated`, so a claim whose terrain is still being generated
        is absent rather than read as air.

        `max_adds` is a storm valve, not a correctness knob. Every add costs
        ~0.22 ms of `typed.List` boxing (measured; see the module docstring for
        why the container is what it is), so the batch a spawn preload or a
        region load delivers between two ticks is a multi-hundred-ms tick when
        absorbed at once. With a cap, the batch drains over as many ticks as
        it takes -- the chunks not yet absorbed are simply absent, and absent
        is the view's designed safe answer ("freeze the body"), the same one a
        not-yet-generated claim gets. `None` (the tools' default) absorbs
        unbounded; removals are never capped, so unloading always completes
        within the pass that noticed it.
        """
        seen = set()
        added = 0
        present = 0
        for chunk in chunks:
            key = chunk_key(chunk.position.x, chunk.position.z)
            seen.add(key)
            if key in self._slot_of:
                present += 1
                continue
            if ready is not None and not ready(chunk):
                continue
            if max_adds is not None and added >= max_adds:
                # Over budget: absorb nothing this pass. `seen` still fills
                # (the removal check below needs it) and the skipped chunks
                # stay absent, so the next sync resumes the drain.
                continue
            self.add(chunk)
            added += 1
            present += 1
        removed = 0
        # `present` counts the chunks the view has *and* the manager still has.
        # If that is every slot, there is nothing to drop and the walk below is
        # skipped; anything else means at least one coordinate left memory.
        if present != len(self._slot_of):
            for key, slot in list(self._slot_of.items()):
                if key in seen:
                    continue
                self.remove(*unpack_chunk_key(key))
                removed += 1
        return added, removed

    # ----- the kernel's view -----

    def index_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """`(keys, slots)` sorted ascending by key, as the kernel wants them.

        Rebuilt only when a chunk was added or removed since the last call. The
        two arrays are views into this object's buffers -- valid until the next
        membership change, which is why the kernel call happens between them.
        """
        if self._dirty:
            live = np.flatnonzero(self._alive[:self._count])
            order = np.argsort(self._key_of_slot[live], kind="stable")
            self._index_slots[:live.size] = live[order]
            self._index_keys[:live.size] = self._key_of_slot[live[order]]
            self._index_count = int(live.size)
            self._dirty = False
        return (self._index_keys[:self._index_count],
                self._index_slots[:self._index_count])

    def section_arrays(self):
        """The flat `typed.List` the kernel indexes as `slot * per_chunk + j`.

        Hand this straight to `physics.kernel.move_batch`. Handing it a Python
        list of the same arrays instead costs 222 ms per call -- numba reflects
        the whole list every call. See the module docstring.
        """
        return self._sections

    def block_at(self, x: int, y: int, z: int) -> int:
        """Read one block the slow way, for tools and for the residency check.

        Out of the vertical window is air (vanilla has no blocks there), and an
        unloaded chunk is air **here** -- callers that must not treat unloaded
        as air check `has()` first, as the solver does.

        Each call boxes an array out of the `typed.List`, so this is a
        diagnostic path, not a per-tick one. A tool that needs a whole volume
        should read the chunk's own `raw_sections()`.
        """
        slot = self._slot_of.get(chunk_key(x >> 4, z >> 4))
        if slot is None:
            return 0
        section = y >> 4
        if section < self.min_section_y or section >= self.min_section_y + self.sections_per_chunk:
            return 0
        array = self._sections[slot * self.sections_per_chunk
                               + (section - self.min_section_y)]
        return int(array[(x & 15) + (z & 15) * 16 + (y & 15) * 256])

    def read_stats(self) -> Dict[str, int]:
        """Kept for callers that report the view's size; the counters are gone.

        Nothing counts block reads any more: the zero-motion fast path is proved
        by an empty view refusing to answer (see `check_physics_solver.py`)
        rather than by instrumentation inside the kernel, which cannot allocate
        a counter without becoming the thing it measures.
        """
        return {"chunks": len(self._slot_of), "sections": len(self._sections)}

    def __len__(self) -> int:
        return len(self._slot_of)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (f"<WorldView {len(self._slot_of)} chunks "
                f"y={self.min_y}..{self.min_y + self.height - 1}>")
