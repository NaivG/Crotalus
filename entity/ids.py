"""
Entity id allocation.

Every entity on the wire is named by a plain integer, and vanilla's rules for
that integer are stricter than they look: ids must be unique among *live*
entities, they must not be handed out twice even after the previous holder
went away, and the id the server uses internally is the *same* id the client
sees - there is no internal-to-network remapping anywhere in the protocol.
`add_entity` (0x01), `CLogin` and every `move_entity_*` packet all speak the
same number.

Pumpkin (the reference) keeps one process-wide `AtomicI32` starting at 1 and
never resets or reuses it:

    static CURRENT_ID: AtomicI32 = AtomicI32::new(1);   // entity/mod.rs:827

This module is that counter. Two properties matter enough to spell out:

* **Start at 1, not 0.** The fetch-add idiom Pumpkin comments on - "have that
  1 and not 0 because fetch_add returns previous value and 0 would be
  invalid" - exists because some code paths treat 0 as "no entity". Starting
  at 1 keeps 0 permanently out of circulation.
* **Monotonic, never reused.** A client that still holds a stale `add_entity`
  for id 7 must never meet a *new* entity that is also 7. The cheap way to
  guarantee that is to never wind the counter back, which is why `observe()`
  exists: if a future save-format ever restores entity ids from disk, the
  counter is raised past them instead of reset.

`Player` used to derive its entity id from an MD5 of its uuid, which is
stable across restarts but collides in principle and, worse, is a *different
scheme* from every other entity - the two families of ids could overlap. One
allocator for everything ends that.
"""

from __future__ import annotations

import threading

#ids start at 1: 0 is treated as invalid/no entity by client code paths
FIRST_ENTITY_ID = 1


class EntityIdAllocator:
    """The monotonic entity-id counter.

    Thread-safe by lock rather than by GIL-splinted increments: the event
    loop, the console thread and (in tests) several dimension threads all
    create entities, and an allocator that could hand out the same id twice
    under contention is not an allocator.
    """

    def __init__(self, start: int = FIRST_ENTITY_ID) -> None:
        self._next = start
        self._lock = threading.Lock()

    def allocate(self) -> int:
        """Claim the next id. Each id is returned exactly once, ever."""
        with self._lock:
            entity_id = self._next
            self._next += 1
            return entity_id

    def observe(self, entity_id: int) -> int:
        """Raise the counter past `entity_id` so it is never handed out again.

        For restoring persisted entities: if a save file contains id 400 and
        the counter is at 12, the next 388 allocations must skip over 400.
        Returns the observed high-water mark for logging.
        """
        with self._lock:
            if entity_id >= self._next:
                self._next = entity_id + 1
            return self._next

    def peek_next(self) -> int:
        """The id `allocate()` would return next. For diagnostics only."""
        with self._lock:
            return self._next

    def reset(self, start: int = FIRST_ENTITY_ID) -> None:
        """Wind the counter back. Test-only: a live server must never reset,
        because in-flight clients still remember the ids they were sent."""
        with self._lock:
            self._next = start


# The process-wide allocator. Pumpkin's is a static; this is the same thing
# with a name. Code should call `get_entity_id_allocator()`, so a test can
# swap the instance without reaching into module globals.
_DEFAULT_ALLOCATOR = EntityIdAllocator()


def get_entity_id_allocator() -> EntityIdAllocator:
    """The global allocator every entity id comes from."""
    return _DEFAULT_ALLOCATOR


def allocate_entity_id() -> int:
    """Claim one id from the global allocator (the common one-liner)."""
    return _DEFAULT_ALLOCATOR.allocate()
