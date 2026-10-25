"""The bodies the physics engine moves, as a structure of arrays.

    store = BodyStore()
    slot = store.add(entity_id=42, x=0.5, y=100.0, z=0.5, width=0.9, height=1.4)
    store.x[slot], store.vy[slot] = 0.0, -0.08

Why not a `Body` dataclass per entity
-------------------------------------

The kernel is `njit`, so the per-tick state has to arrive as arrays: a list of
Python objects cannot cross into `nopython` mode, and even if it could, the
attribute lookups would cost more than the collision sweep. `BodyStore` is that
array-of-fields layout, and nothing else. It is not "the entity" -- `Entity`
stays the identity/metadata object in `entity/`, and the solver copies the
result back into `EntityPosition` at the end of the tick.

The invariant that matters
--------------------------

**A slot is only valid between `add` and `remove`.** Removal marks the body
`REMOVED` rather than compacting immediately, because compacting renumbers
slots and any Python code holding one would silently start moving a different
mob; `compact()` does the renumbering once, in one place, at a defined moment
(the top of the solver's tick). `slot_of(entity_id)` is the only sanctioned way
to go from an entity to a body.

Deliberate omissions (see `PHYSICS_PLAN.md` §七): bodies have no `Entity`
reference, no health, no AI state and no persistence. Nothing here survives a
restart, and a mob that unloads and respawns gets a fresh body at its spawn
position.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

__all__ = ["BodyStore", "ON_GROUND", "SLEEPING", "REMOVED", "MOVED"]

#: Flag bits in `BodyStore.flags`, as `uint8`.
ON_GROUND = 1
SLEEPING = 2
REMOVED = 4
#: Set by the solver when a body's position actually changed this tick. Only
#: these get pushed to `EntityTracker` -- a body that did not move must not
#: send a packet, and the sleeping bodies never do.
MOVED = 8

#: Bodies whose speed is below this are candidates for sleeping. One 1e-7
#: blocks/tick is far below anything a client could render, and far above the
#: float noise a resting sweep can leave behind.
REST_SPEED = 1e-7

#: `flags & _NOT_SLEEPING` clears the sleeping bit, as a `uint8` array operand.
_NOT_SLEEPING = np.uint8(0xFF ^ SLEEPING)


class BodyStore:
    """Fixed-stride arrays of body state, plus the entity<->slot index.

    Every array is public and indexed by slot: the kernel takes them directly
    (`physics.kernel.move_batch`), so wrapping them in accessors would only add
    a call. Treat them as read-only from outside the physics package.
    """

    __slots__ = ("capacity", "count", "x", "y", "z", "vx", "vy", "vz",
                 "width", "height", "flags", "entity_id",
                 "_slot_of", "_live")

    def __init__(self, capacity: int = 256) -> None:
        self.capacity = max(int(capacity), 1)
        self.count = 0
        self.x = np.zeros(self.capacity, dtype=np.float64)
        self.y = np.zeros(self.capacity, dtype=np.float64)
        self.z = np.zeros(self.capacity, dtype=np.float64)
        self.vx = np.zeros(self.capacity, dtype=np.float64)
        self.vy = np.zeros(self.capacity, dtype=np.float64)
        self.vz = np.zeros(self.capacity, dtype=np.float64)
        self.width = np.zeros(self.capacity, dtype=np.float64)
        self.height = np.zeros(self.capacity, dtype=np.float64)
        self.flags = np.zeros(self.capacity, dtype=np.uint8)
        self.entity_id = np.zeros(self.capacity, dtype=np.int64)
        self._slot_of: Dict[int, int] = {}
        self._live = 0

    # ----- capacity -----

    def _grow(self, needed: int) -> None:
        capacity = self.capacity
        while capacity < needed:
            capacity *= 2
        for name in ("x", "y", "z", "vx", "vy", "vz", "width", "height"):
            grown = np.zeros(capacity, dtype=np.float64)
            grown[:self.count] = getattr(self, name)[:self.count]
            setattr(self, name, grown)
        for name, dtype in (("flags", np.uint8), ("entity_id", np.int64)):
            grown = np.zeros(capacity, dtype=dtype)
            grown[:self.count] = getattr(self, name)[:self.count]
            setattr(self, name, grown)
        self.capacity = capacity

    # ----- membership -----

    def add(self, entity_id: int, x: float, y: float, z: float,
            width: float, height: float, on_ground: bool = False) -> int:
        """Add (or re-position) a body and return its slot.

        Adding an entity that already has a body moves that body rather than
        creating a second one: a respawn path that calls `add` twice would
        otherwise leave a ghost the solver keeps stepping forever. A re-added
        body also wakes up -- it has just been placed somewhere new, and a
        sleeping body at the old position would never notice.
        """
        slot = self._slot_of.get(entity_id)
        if slot is None or (self.flags[slot] & REMOVED):
            if slot is None:
                if self.count == self.capacity:
                    self._grow(self.count + 1)
                slot = self.count
                self.count += 1
                self._slot_of[entity_id] = slot
                self._live += 1
            else:
                # A removed slot handed back by `add` before `compact` ran.
                self._slot_of[entity_id] = slot
                self._live += 1
            self.entity_id[slot] = entity_id
        self.x[slot] = x
        self.y[slot] = y
        self.z[slot] = z
        self.vx[slot] = 0.0
        self.vy[slot] = 0.0
        self.vz[slot] = 0.0
        self.width[slot] = width
        self.height[slot] = height
        self.flags[slot] = ON_GROUND if on_ground else 0
        return slot

    def remove(self, entity_id: int) -> bool:
        """Mark a body removed. The slot stays valid until the next `compact`.

        Returns False when the entity had no body -- an entity that was never
        physical (a player on another path, a mob that never spawned) is not an
        error, and the caller counting removals needs to know the difference.
        """
        slot = self._slot_of.pop(entity_id, None)
        if slot is None:
            return False
        self.flags[slot] = REMOVED
        self._live -= 1
        return True

    def compact(self) -> int:
        """Drop removed bodies, renumbering slots. Returns how many went.

        Slot numbers are only stable within a tick, which is why the solver
        compacts once at the top of its own tick and at no other time. A caller
        holding a slot across a `compact()` call is holding a bug.
        """
        dead = self.count - self._live
        if dead == 0:
            return 0
        keep = np.flatnonzero((self.flags[:self.count] & REMOVED) == 0)
        survivors = int(keep.size)
        for name in ("x", "y", "z", "vx", "vy", "vz", "width", "height",
                     "flags", "entity_id"):
            array = getattr(self, name)
            array[:survivors] = array[keep]
            array[survivors:self.count] = 0
        self.count = survivors
        self._slot_of = {int(self.entity_id[slot]): slot for slot in range(survivors)}
        return int(dead)

    # ----- lookup -----

    def slot_of(self, entity_id: int) -> Optional[int]:
        """The slot for an entity, or None when it has no body."""
        slot = self._slot_of.get(entity_id)
        if slot is None or (self.flags[slot] & REMOVED):
            return None
        return slot

    def entity_ids(self) -> Iterator[int]:
        """Every live body's entity id, in slot order."""
        for slot in range(self.count):
            if not (self.flags[slot] & REMOVED):
                yield int(self.entity_id[slot])

    # ----- sleeping -----

    def sleeping_mask(self) -> np.ndarray:
        """`bool[count]`, True where the body is asleep."""
        return (self.flags[:self.count] & SLEEPING) != 0

    def awake_indices(self) -> np.ndarray:
        """Slots the solver must integrate this tick, as `int32[?]`."""
        flags = self.flags[:self.count]
        return np.flatnonzero((flags & (SLEEPING | REMOVED)) == 0).astype(np.int32)

    def resting_indices(self) -> np.ndarray:
        """Sleeping slots, which get a support probe instead of a full step."""
        flags = self.flags[:self.count]
        return np.flatnonzero((flags & (SLEEPING | REMOVED)) == SLEEPING).astype(np.int32)

    def wake(self, slot: int) -> None:
        self.flags[slot] &= _NOT_SLEEPING

    def sleep(self, slot: int) -> None:
        self.flags[slot] |= SLEEPING
        self.vx[slot] = 0.0
        self.vy[slot] = 0.0
        self.vz[slot] = 0.0

    def sleep_indices(self, slots: np.ndarray) -> np.ndarray:
        """Mark every listed slot asleep. `slots` is an `int32`/`int64` array."""
        self.flags[slots] |= SLEEPING
        self.vx[slots] = 0.0
        self.vy[slots] = 0.0
        self.vz[slots] = 0.0
        return slots

    def wake_indices(self, slots: np.ndarray) -> np.ndarray:
        self.flags[slots] &= _NOT_SLEEPING
        return slots

    def speed_squared(self) -> np.ndarray:
        """`vx² + vy² + vz²` per slot, over the live range."""
        return (self.vx[:self.count] ** 2 + self.vy[:self.count] ** 2
                + self.vz[:self.count] ** 2)

    def position_of(self, entity_id: int) -> Optional[Tuple[float, float, float]]:
        slot = self.slot_of(entity_id)
        if slot is None:
            return None
        return (float(self.x[slot]), float(self.y[slot]), float(self.z[slot]))

    def __len__(self) -> int:
        return self._live

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"<BodyStore live={self._live} slots={self.count} cap={self.capacity}>"
