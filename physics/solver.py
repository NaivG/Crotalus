"""The per-tick physics step: integrate, sweep, sleep, and write back.

    solver = PhysicsSolver(view, block_shapes(), min_y=dimension.settings.min_y)
    solver.add(entity)                 # spawn
    solver.tick()                      # once per actor tick, before the tracker
    solver.stats()                     # -> StatsMsg["physics"]

Order inside one tick
---------------------

1. `compact()` removed bodies -- slot renumbering happens here and nowhere else.
2. Bodies below the world floor are removed through the caller's path. First,
   because the three gauges below then describe the bodies that are still in the
   store at the end of the tick -- see `tick()`.
3. **Awake** bodies get gravity, then one batched sweep, then drag: vertical
   air drag on all bodies, horizontal friction split by ground state.
4. **Sleeping** bodies get a support probe: the same sweep with a 1e-3 downward
   nudge. One that no longer has anything under it wakes up and falls on the
   next tick.
5. Bodies the *world* could not answer for (an unloaded chunk in the swept span)
   keep their position **and** their velocity, and are counted.
6. Positions go back into `EntityPosition`, and every body that actually moved
   is pushed into `EntityTracker` -- see below.

Gravity, drag, friction, and what is deliberately missing
---------------------------------------------------------

Per tick, vanilla's `LivingEntity.travel` is: add `-0.08` to the vertical
velocity, move, then multiply the result by `0.98`. That is what makes terminal
velocity `-3.92` blocks/tick rather than unbounded, and it is implemented here
because it is *air* drag.

**Horizontal friction is the ground/air split vanilla uses**, applied after the
sweep: a grounded body's horizontal velocity is multiplied by
`slipperiness * 0.91` and an airborne one by bare `0.91`, on top of nothing
else -- the `0.98` is vertical-only, exactly as in vanilla. The split is what
gives a walking mob a *steady state* at all: the AI adds acceleration every
tick (`entity/ai/__init__.py`'s moveRelative), and a body with only `0.98`
horizontal decay would converge at `accel / (1 - 0.98)` -- a cow at 2
blocks/tick. Grounded at `0.546` it converges at `accel / 0.454`, which is the
speed the attribute tables are written against.

The slipperiness is **the grass calibration, not a table** (0.6, the value of
every common block). The per-block columns exist in `physics/block_shapes.py`
for the round that reads them; until then ice, slime and everything slippery
decelerate at grass speed. That is written down rather than papered over --
`PHYSICS_PLAN.md` §七 tracks the table as the remaining gap, not friction
itself.

Load levels: a body in a chunk that is not ticking is paused
------------------------------------------------------------

`Chunk.mdx` splits loaded chunks into four load types, and only level <= 31
("Entity Ticking") runs entities at all: at level 32 ("Block Ticking") and 33
("Border") entities are still *accessible* -- readable, teleportable, saved and
sent to clients -- but they are not processed, and at 34 and above the chunk is
not in memory. The ticket model is `world/chunk_ticket.py`, the spawner already
asks it (`NaturalSpawner._candidate_chunks` gates on `ticks_entities`), and this
solver was the hole: a mob standing in a lazy chunk still got gravity, drag, a
full sweep and the sleep/wake bookkeeping. That is how a "frozen" chunk ends up
holding a mob that has visibly walked -- or fallen -- out of it.

The gate is `ticks_entities(chunk_x, chunk_z)`, asked once per tick for the
chunk each body stands in, and it is all-or-nothing for that body: a paused body
keeps its position, its velocity and its flags bit for bit; it gets no gravity,
no drag, no collision sweep, no support probe, and it is not removed for being
below the world. Everything "accessible" means is kept: the body stays in the
store, `set_position` / `set_velocity` / `remove_body` still work on it, and an
external write is published straight to `EntityPosition`/`EntityTracker`
instead of waiting for a tick that is paused.

"Move it from outside physics" means through one of those calls, and that is
true of every body, not just a paused one: `BodyStore` is the state and
`EntityPosition` is a copy of it, so writing `entity.position.x` directly is
overwritten the next time the body runs (`_sync_entities` copies the store out).
A paused body makes the difference visible -- it has no tick to hide behind, and
the next ticking tick snaps the entity back to wherever the store says it is.

Two consequences worth stating, because they are deliberate:

* **The gate is evaluated from the position at the top of the tick.** A body
  that crosses out of the ticking area during its own move keeps that tick's
  displacement and is paused from the next tick on. That is the same one-tick
  lag vanilla's `EntityTickList` has (its membership is refreshed by the chunk
  scheduler, not in the middle of a move), and it is what `Chunk.mdx` describes
  as "an entity that attempts to move into a block ticking chunk ... becomes
  suspended as soon as it leaves the entity ticking chunk".
* **No gate means no ticket model, not "everything is lazy".** A solver built
  without `ticks_entities` runs every body, which is what the synthetic worlds in
  `tools/check_physics_solver.py` and `tools/bench_physics.py` want. The actor
  always passes its dimension's query (`world/dimension_actor.py`).

Sleeping is a real skip, not "run anyway and get the same answer"
-----------------------------------------------------------------

The world is mostly full of mobs that are standing still, and the spawner can put
hundreds of them there. A sleeping body runs no integration, no gravity and no
full sweep -- only the support probe, which touches a handful of air blocks.
The invariant is that **a sleeping body's position and velocity are bit-for-bit
unchanged**; `tools/check_physics_solver.py` asserts that over 100 ticks rather
than trusting it.

Writing results to the wire
---------------------------

`Dimension.entities` is the truth for mobs, and `EntityTracker.update_entity`
is the ready-made way to get a position onto the wire (`entity_tracker.py:352`).
Only bodies that moved are pushed: a body that did not move produces no packet,
and neither does a sleeping one -- which lines up with the tracker's own
`update_interval` throttling instead of fighting it. This is the step that is
easy to forget and produces the classic "the server is moving it, the client
isn't" bug.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from physics.bodies import REST_SPEED, REMOVED, BodyStore
from physics.block_shapes import BlockShapes
from physics.kernel import (MAX_CANDIDATES, STATUS_DEFERRED, STATUS_MOVED,
                            STATUS_ON_GROUND, STATUS_OVERFLOW, move_batch,
                            warm_up)
from physics.world_view import WorldView, unpack_chunk_key

__all__ = ["PhysicsSolver", "AIR_DRAG", "AIR_FRICTION", "GRAVITY",
           "GROUND_FRICTION", "STEP_HEIGHT", "REST_PROBE", "VOID_MARGIN"]

#: Vanilla's `LivingEntity.travel` gravity, in blocks per tick squared.
GRAVITY = 0.08

#: Vanilla's air drag, applied to the clipped motion after every move. It is
#: what bounds a fall at `0.98 * 0.08 / (1 - 0.98) = 3.92` blocks per tick.
AIR_DRAG = 0.98

#: Vanilla's grounded horizontal friction: the block's slipperiness times 0.91.
#: Grass -- the calibration every block gets until the per-block slipperiness
#: table lands -- has slipperiness 0.6, so 0.6 * 0.91 == 0.546. Applied after
#: the sweep, on top of nothing else: the `AIR_DRAG` above is vertical-only.
GROUND_FRICTION = 0.6 * 0.91

#: Vanilla's airborne horizontal friction, bare `0.91`. Higher than the
#: grounded value and much lower than the vertical `AIR_DRAG`, which is why a
#: mob that jumps carries its speed through the arc and lands a little further
#: than it could have walked in the same ticks -- and why the airborne steady
#: state `accel / (1 - 0.91)` stays under half a block per tick instead of
#: diverging.
AIR_FRICTION = 0.91

#: Vanilla's default `Attributes.STEP_HEIGHT`. There is no attribute system yet,
#: so it is a named constant rather than a per-entity value (`PHYSICS_PLAN.md`
#: §八.4); when attributes land, this becomes the default and mobs read theirs.
STEP_HEIGHT = 0.6

#: How far a sleeping body is nudged downwards to ask "is anything under me".
#: Small enough to be invisible, large enough to produce a collision time
#: against a surface the body is resting exactly on.
REST_PROBE = 1e-3

#: A body this far below the world floor is gone, not falling. Vanilla's void
#: damage starts at `min_y - 64`; there is nothing to damage yet, so the body is
#: removed through the same path the spawner uses.
VOID_MARGIN = 64

#: `status` bit tests, as arrays.
_MOVED = np.uint8(STATUS_MOVED)
_ON_GROUND = np.uint8(STATUS_ON_GROUND)
#: `BodyStore.flags` bit for "standing on something", and its inverse.
_GROUND_FLAG = np.uint8(1)
_NOT_ON_GROUND = np.uint8(0xFF ^ 1)


class PhysicsSolver:
    """Owns the bodies, the world view's arrays and the per-tick step.

    Everything here belongs to one dimension actor thread. There are no locks
    on purpose: the same single-ownership rule the chunks and entities follow.
    """

    __slots__ = ("view", "shapes", "bodies", "min_y", "height",
                 "entities", "tracker", "on_removed", "step_height",
                 "gravity", "air_drag", "ground_friction", "air_friction",
                 "max_step", "ticks_entities",
                 "_candidates", "_motion", "_probe", "_status",
                 "_stats", "_last_ms", "_warmed")

    def __init__(self, view: WorldView, shapes: BlockShapes, min_y: int,
                 height: int, entities=None, tracker=None,
                 on_removed: Optional[Callable[[List[int]], None]] = None,
                 capacity: int = 256, step_height: float = STEP_HEIGHT,
                 gravity: float = GRAVITY, air_drag: float = AIR_DRAG,
                 ground_friction: float = GROUND_FRICTION,
                 air_friction: float = AIR_FRICTION,
                 ticks_entities: Optional[Callable[[int, int], bool]] = None
                 ) -> None:
        self.view = view
        self.shapes = shapes
        self.bodies = BodyStore(capacity)
        self.min_y = int(min_y)
        self.height = int(height)
        #: `entity_id -> Entity`. A dict, a `Dimension.entities`, anything with
        #: `.get`; the solver never holds an `Entity` reference of its own.
        self.entities = entities
        #: `EntityTracker.update_entity(entity_id, x, y, z, yaw, pitch, on_ground)`
        self.tracker = tracker
        self.on_removed = on_removed
        self.step_height = float(step_height)
        self.gravity = float(gravity)
        self.air_drag = float(air_drag)
        self.ground_friction = float(ground_friction)
        self.air_friction = float(air_friction)
        #: The load-level gate: `ticks_entities(chunk_x, chunk_z) -> bool`, as
        #: `Dimension.ticks_entities` spells it. A body whose chunk answers
        #: False is paused for the tick (see the module docstring); `None`
        #: means "there is no ticket model here", which runs every body and is
        #: what a synthetic world wants.
        #:
        #: Note where that puts a body whose chunk is not in memory at all: a
        #: chunk with no ticket is level 44 and answers False, so it is counted
        #: in `paused_not_ticking` rather than `deferred_unloaded` (which is now
        #: what the *kernel* says about a ticking body whose swept span reaches
        #: a chunk the view cannot read). Both mean "not moved this tick"; the
        #: level is simply the first question asked, and in a live server the
        #: difference lasts about one tick -- `NaturalSpawner._reconcile_residency`
        #: runs every tick and takes mobs out of chunks that are gone.
        self.ticks_entities = ticks_entities
        self.max_step = 0.0
        self._candidates = np.zeros((MAX_CANDIDATES, 6), dtype=np.float64)
        self._motion = np.zeros((0, 3), dtype=np.float64)
        self._probe = np.zeros(0, dtype=np.uint8)
        self._status = np.zeros(0, dtype=np.uint8)
        self._stats: Dict[str, int] = {
            "bodies": 0, "awake": 0, "sleeping": 0, "moved": 0,
            "deferred_unloaded": 0, "paused_not_ticking": 0, "overflow": 0,
            "removed_void": 0, "skipped_no_type": 0, "woken_by_probe": 0,
            "orphan_bodies": 0, "ticks": 0, "ms": 0.0,
        }
        self._last_ms = 0.0
        self._warmed = False

    def warm_up(self) -> None:
        """Compile the kernel now, so the first tick is not the first call.

        Called from the actor's setup path (a stall there is invisible) and
        idempotent. `cache=True` in the kernel means a restart only pays the
        LLVM check, not the compile.
        """
        warm_up()
        self._warmed = True

    # ----- membership -----

    def add_body(self, entity_id: int, x: float, y: float, z: float,
                 width: float, height: float, on_ground: bool = False) -> int:
        """Give an entity a body at a known position. Returns its slot."""
        return self.bodies.add(entity_id, x, y, z, width, height, on_ground)

    def add_entity(self, entity) -> Optional[int]:
        """Give an `Entity` a body from its own position and collision box.

        A `None` entity type means the entity has no collision box and physics
        has nothing to move it with. That is counted (`skipped_no_type`) rather
        than guessed at with a default box, because a wrong box is a mob stuck
        in the floor or floating above it.
        """
        entity_type = getattr(entity, "entity_type", None)
        if entity_type is None:
            self._stats["skipped_no_type"] += 1
            return None
        position = entity.position
        return self.add_body(
            int(entity.entity_id), position.x, position.y, position.z,
            float(entity_type.width), float(entity_type.height),
            bool(position.on_ground))

    def remove_body(self, entity_id: int) -> bool:
        return self.bodies.remove(entity_id)

    def has_body(self, entity_id: int) -> bool:
        return self.bodies.slot_of(entity_id) is not None

    def wake(self, entity_id: int) -> bool:
        """Wake a sleeping body, e.g. because something pushed it."""
        slot = self.bodies.slot_of(entity_id)
        if slot is None:
            return False
        self.bodies.wake(slot)
        return True

    def set_velocity(self, entity_id: int, vx: float, vy: float, vz: float) -> bool:
        """Set a body's velocity from outside physics, and wake it.

        This is how anything that is not the solver gets a body moving -- mob AI
        in the next round, a knockback today. Writing the velocity is only half
        of it: a sleeping body that is given velocity and *not* woken would sit
        there with the velocity of a moving body and never use it, which is the
        "one cow that never moves again" failure the plan calls out.

        Note that `EntityPosition.velocity_*` is an *output*: the solver writes
        the post-move velocity there for the wire, and reading it back as input
        would feed a stale value. `BodyStore` is the state.

        The write is gated by nothing at all -- not the load level, not
        sleeping. A paused body must still be movable from outside, so this
        publishes the new velocity onto `EntityPosition` and leaves the rest of
        the pose alone (`_publish_velocity`); the position reaches the wire when
        the body ticks, or immediately through `set_position`.
        """
        slot = self.bodies.slot_of(entity_id)
        if slot is None:
            return False
        self.bodies.vx[slot] = vx
        self.bodies.vy[slot] = vy
        self.bodies.vz[slot] = vz
        self.bodies.wake(slot)
        self._publish_velocity(slot)
        return True

    def set_position(self, entity_id: int, x: float, y: float, z: float,
                     on_ground: bool = False) -> bool:
        """Teleport a body from outside physics, and publish it.

        Same reasoning as `set_velocity`: a body moved from under the solver
        without being woken would keep resting on a block that is no longer
        there. It is also gated by nothing -- a teleport is exactly the "still
        accessible" half of a lazy chunk, so the pause does not block it.

        The new pose goes onto the entity *in the same call* rather than waiting
        for the next tick, for the same reason `_notify_tracker` exists at all:
        a body whose chunk is not ticking has no tick that would carry the move
        to the wire, and "accessible" would quietly mean "readable but
        invisible". `set_velocity` publishes its half the same way.
        """
        slot = self.bodies.slot_of(entity_id)
        if slot is None:
            return False
        store = self.bodies
        store.x[slot] = x
        store.y[slot] = y
        store.z[slot] = z
        store.vx[slot] = 0.0
        store.vy[slot] = 0.0
        store.vz[slot] = 0.0
        if on_ground:
            store.flags[slot] |= ON_GROUND
        else:
            store.flags[slot] &= _NOT_ON_GROUND
        store.wake(slot)
        slots = np.array([slot], dtype=np.int32)
        status = np.array([STATUS_ON_GROUND if on_ground else 0], dtype=np.uint8)
        self._sync_entities(slots, status)
        self._notify_tracker(slots)
        return True

    # ----- the tick -----

    def tick(self) -> None:
        """One 20 Hz step. Must run before `entity_tracker.pump()`.

        The load-level gate is resolved once, here, and handed to every step
        below: the ticket model cannot change inside a tick, and asking it four
        times would be four answers to one question.

        Removals run **before** the three state gauges (`awake`, `sleeping`,
        `paused_not_ticking`) are read, and a removal re-derives the mask because
        `compact()` renumbers slots. That ordering is what makes
        `bodies == awake + sleeping + paused_not_ticking` exact on every tick
        instead of "exact except on the tick something fell into the void"
        (`tools/check_physics_solver.py` §14 pins it on a removal tick).
        """
        started = time.perf_counter()
        self.bodies.compact()

        ticking = self._ticking_mask()
        if self._collect_removed(ticking):
            # Something was dropped, so the slot array shrank under the mask.
            # `compact()` is a no-op when there is nothing to renumber.
            self.bodies.compact()
            ticking = self._ticking_mask()

        self._stats["paused_not_ticking"] = (
            0 if ticking is None
            else int(self.bodies.count - np.count_nonzero(ticking)))

        deferred = self._step_awake(ticking)
        woken = self._probe_resting(ticking)

        self._last_ms = (time.perf_counter() - started) * 1000.0
        if self._last_ms > self.max_step:
            self.max_step = self._last_ms
        self._stats["ticks"] += 1
        self._stats["deferred_unloaded"] += deferred
        self._stats["woken_by_probe"] += woken
        self._refresh_stats()

    def _ticking_mask(self) -> Optional[np.ndarray]:
        """`bool[count]`: whether each body's own chunk ticks entities.

        `None` when there is no gate to ask (or no bodies), which every step
        below reads as "all of them tick" and skips the filtering for.

        The question is asked once per *chunk*, not once per body: bodies share
        chunks, and the ticket query walks the ticket table on every call, so a
        few hundred mobs standing in a dozen chunks would otherwise pay the same
        walk a few hundred times. The distinct chunks are found by sorting their
        packed keys in C (`np.unique`) and the Python loop below then runs once
        per distinct chunk -- the body count only decides how long the vector
        work is, never how many ticket queries happen.

        A packed 1-D key, not the `(x, z)` pairs themselves: `np.unique` over a
        `(count, 2)` array costs 270 us at 500 bodies against 40 us for the same
        data packed into one `int64` column (measured), and the packing is the
        same `chunk_key` the kernel binary-searches with.
        """
        gate = self.ticks_entities
        if gate is None:
            return None
        store = self.bodies
        count = store.count
        if count == 0:
            return None

        # `floor(x / 16)`, the same arithmetic `EntityPosition.chunk` does, so a
        # body and its entity can never disagree about which chunk it stands in.
        # Floor first, and never truncate toward zero: `int(-0.5)` is 0, so every
        # body west of zero would read its eastern neighbour's level -- a
        # one-chunk shift that only shows up where the two chunks differ, which
        # is exactly the ticking border.
        chunk_x = np.floor(store.x[:count] / 16.0).astype(np.int64)
        chunk_z = np.floor(store.z[:count] / 16.0).astype(np.int64)
        keys = (chunk_x << np.int64(32)) | (chunk_z & np.int64(0xFFFFFFFF))

        unique, inverse = np.unique(keys, return_inverse=True)
        flags = np.empty(unique.shape[0], dtype=bool)
        for index in range(unique.shape[0]):
            flags[index] = bool(gate(*unpack_chunk_key(int(unique[index]))))
        return flags[inverse]

    def _arrays(self) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, int]:
        sections = self.view.section_arrays()
        keys, slots = self.view.index_arrays()
        return sections, keys, slots, int(keys.shape[0])

    def _call(self, indices: np.ndarray, motion: np.ndarray,
              probe: np.ndarray) -> np.ndarray:
        """One `move_batch` call. Returns the status array."""
        count = indices.shape[0]
        if self._status.shape[0] < count:
            self._status = np.zeros(count, dtype=np.uint8)
        status = self._status[:count]
        sections, keys, slots, key_count = self._arrays()
        store = self.bodies
        move_batch(store.x, store.y, store.z, store.vx, store.vy, store.vz,
                   store.width, store.height, indices, motion, probe, status,
                   sections, self.view.sections_per_chunk,
                   self.view.min_section_y, keys, slots, key_count,
                   self.shapes.offsets, self.shapes.ids,
                   self.shapes.min_xyz, self.shapes.max_xyz,
                   self._candidates, self.step_height)
        return status

    def _step_awake(self, ticking: Optional[np.ndarray]) -> int:
        """Gravity, one sweep, drag, friction, ground bookkeeping and sleep.

        `ticking` drops the bodies whose chunk is not entity-ticking before any
        of that: a paused body is not integrated, not dragged, not given
        gravity, and never consulted for sleeping (so a paused body cannot fall
        asleep or be woken by this step either).
        """
        store = self.bodies
        awake = store.awake_indices()
        if ticking is not None and awake.size:
            keep = ticking[awake]
            if not keep.all():
                awake = awake[keep]
        self._stats["awake"] = int(awake.size)
        if awake.size == 0:
            return 0

        motion = self._motion
        if motion.shape[0] < awake.size:
            motion = np.zeros((awake.size, 3), dtype=np.float64)
            self._motion = motion
        motion = motion[:awake.size]
        motion[:, 0] = store.vx[awake]
        # Gravity goes into the *motion*, not into `store.vy`: a body the world
        # cannot answer for must come out of the tick with the velocity it went
        # in with, and writing gravity first would leak it into deferred bodies.
        motion[:, 1] = store.vy[awake] - self.gravity
        motion[:, 2] = store.vz[awake]

        probe = self._probe
        if probe.shape[0] < awake.size:
            probe = np.zeros(awake.size, dtype=np.uint8)
            self._probe = probe
        probe = probe[:awake.size]
        probe[:] = 0

        status = self._call(awake, motion, probe)
        answered = (status & np.uint8(STATUS_DEFERRED | STATUS_OVERFLOW)) == 0
        answered_slots = awake[answered]

        if answered_slots.size:
            store.vy[answered_slots] *= self.air_drag

        # Ground state, and whether it *changed*. A landing tick has no
        # displacement at all (the fall was clipped to exactly zero), so
        # "moved" alone would never tell a client that a mob is now standing --
        # it would render every landed mob as still falling.
        was_grounded = (store.flags[answered_slots] & _GROUND_FLAG) != 0
        is_grounded = (status[answered] & _ON_GROUND) != 0

        # Horizontal friction uses the *post-move* ground state: a body that
        # just landed decelerates like a walker from this tick on, and one
        # that just left the ground keeps its air friction through the arc.
        # The vertical `AIR_DRAG` above is vanilla's and stays on everything.
        if answered_slots.size:
            grounded_slots = answered_slots[is_grounded]
            store.vx[grounded_slots] *= self.ground_friction
            store.vz[grounded_slots] *= self.ground_friction
            airborne_slots = answered_slots[~is_grounded]
            if airborne_slots.size:
                store.vx[airborne_slots] *= self.air_friction
                store.vz[airborne_slots] *= self.air_friction

        store.flags[answered_slots] &= _NOT_ON_GROUND
        store.flags[answered_slots[is_grounded]] |= _GROUND_FLAG
        flipped = answered_slots[was_grounded != is_grounded]

        moved = awake[(status & _MOVED) != 0]
        # Every body the world answered for gets its `EntityPosition` refreshed,
        # because velocity and `on_ground` change on ticks with no displacement.
        self._sync_entities(answered_slots, status[answered])
        self._notify_tracker(np.union1d(moved, flipped))
        self._stats["moved"] = int(moved.size)
        self._stats["overflow"] += int(np.count_nonzero(
            (status & np.uint8(STATUS_OVERFLOW)) != 0))

        # Sleep: on the ground and going nowhere. `speed_squared` is squared, so
        # the threshold is squared too.
        grounded = awake[(status & _ON_GROUND) != 0]
        if grounded.size:
            slow = grounded[store.speed_squared()[grounded] <= REST_SPEED ** 2]
            if slow.size:
                store.sleep_indices(slow)
        return int(np.count_nonzero((status & np.uint8(STATUS_DEFERRED)) != 0))

    def _probe_resting(self, ticking: Optional[np.ndarray]) -> int:
        """Ask every sleeping body whether it still has support under it.

        The probe is the same sweep with a 1e-3 downward nudge: a supported body
        clips it to exactly zero and stays asleep with its position bit-for-bit
        unchanged, and an unsupported one reports `on_ground == False` and wakes.
        That is the whole "a block under a sleeping mob is removed" story -- the
        alternative, force-waking every N ticks, was rejected because it leaves
        the mob visibly hanging in the air for up to N ticks.

        Sleeping bodies in non-ticking chunks are skipped along with everything
        else: the probe is a block read and a sweep, and a lazy chunk runs
        nothing. The floor under such a mob can be removed freely; it notices on
        the first tick after its chunk starts ticking again, because the probe
        runs before any of the movement that would otherwise bury it.
        """
        store = self.bodies
        resting = store.resting_indices()
        if ticking is not None and resting.size:
            keep = ticking[resting]
            if not keep.all():
                resting = resting[keep]
        self._stats["sleeping"] = int(resting.size)
        if resting.size == 0:
            return 0

        motion = self._motion
        if motion.shape[0] < resting.size:
            motion = np.zeros((resting.size, 3), dtype=np.float64)
            self._motion = motion
        motion = motion[:resting.size]
        motion[:, 0] = 0.0
        motion[:, 1] = -REST_PROBE
        motion[:, 2] = 0.0

        probe = self._probe
        if probe.shape[0] < resting.size:
            probe = np.zeros(resting.size, dtype=np.uint8)
            self._probe = probe
        probe = probe[:resting.size]
        probe[:] = 1

        status = self._call(resting, motion, probe)
        unsupported = resting[(status & _ON_GROUND) == 0]
        if unsupported.size:
            store.wake_indices(unsupported)
        self._stats["overflow"] += int(np.count_nonzero(
            (status & np.uint8(STATUS_OVERFLOW)) != 0))
        return int(unsupported.size)

    def _sync_entities(self, slots: np.ndarray, status: np.ndarray) -> None:
        """Copy the solver's position, velocity and ground state onto the entities.

        Split from `_notify_tracker` on purpose. A body that lands has not moved
        on that tick, but its velocity just went to zero and `on_ground` just
        became true -- both of which the client needs and neither of which is a
        position packet.
        """
        if self.entities is None:
            return
        store = self.bodies
        for position in range(slots.shape[0]):
            slot = int(slots[position])
            entity = self.entities.get(int(store.entity_id[slot]))
            if entity is None:
                # The entity is gone but its body is not. Nothing here drops it:
                # removal is the remover's job (`DimensionActor`'s void path and
                # the spawner both call `remove_body`, and `compact()` takes the
                # slot on the next tick), and guessing from a missing entity
                # would race a dimension that is mid-handoff. Counting it is how
                # a missed `remove_body` shows up instead of leaking silently.
                self._stats["orphan_bodies"] += 1
                continue
            pose = entity.position
            pose.x = float(store.x[slot])
            pose.y = float(store.y[slot])
            pose.z = float(store.z[slot])
            pose.on_ground = bool(int(status[position]) & STATUS_ON_GROUND)
            pose.velocity_x = float(store.vx[slot])
            pose.velocity_y = float(store.vy[slot])
            pose.velocity_z = float(store.vz[slot])

    def _publish_velocity(self, slot: int) -> None:
        """Copy one body's velocity onto its `EntityPosition`, and nothing else.

        The external-write counterpart of the velocity half of `_sync_entities`,
        for `set_velocity`: a caller that gives a paused (or deferred) body
        velocity has to see it on the entity, because there may be no tick
        coming that would copy it. Position and ground state are deliberately
        untouched -- this call did not move the body, and claiming a new pose
        here would be `set_position`'s job.
        """
        if self.entities is None:
            return
        store = self.bodies
        entity = self.entities.get(int(store.entity_id[slot]))
        if entity is None:
            return
        pose = entity.position
        pose.velocity_x = float(store.vx[slot])
        pose.velocity_y = float(store.vy[slot])
        pose.velocity_z = float(store.vz[slot])

    def _notify_tracker(self, slots: np.ndarray) -> None:
        """Tell `EntityTracker` about the bodies that moved or landed.

        Only those: a body that did neither produces no packet, which is what
        keeps a world full of standing mobs from spending its tick budget on
        deltas the client would discard anyway. Call it after `_sync_entities`,
        which is what puts the current pose on the entity.
        """
        if self.tracker is None or self.entities is None:
            return
        store = self.bodies
        for position in range(slots.shape[0]):
            slot = int(slots[position])
            entity = self.entities.get(int(store.entity_id[slot]))
            if entity is None:
                continue
            pose = entity.position
            self.tracker.update_entity(int(store.entity_id[slot]),
                                       pose.x, pose.y, pose.z,
                                       pose.yaw, pose.pitch,
                                       bool(store.flags[slot] & _GROUND_FLAG))

    def _collect_removed(self, ticking: Optional[np.ndarray]) -> int:
        """Remove bodies that fell out of the world, through the caller's path.

        Returns how many went, which is what `tick()` uses to decide whether the
        slot array shrank under the tick's mask.

        A body in a non-ticking chunk is left alone even down here. Void removal
        deletes the entity from the world (`on_removed`), and "frozen but
        accessible" would be a lie if the engine could still destroy a mob in a
        lazy chunk -- the only way such a body is below the world is that
        something put it there from outside, and that something is the caller
        this pass would be reporting to.
        """
        store = self.bodies
        if store.count == 0:
            return 0
        below = np.flatnonzero(store.y[:store.count] < self.min_y - VOID_MARGIN)
        if below.size and ticking is not None:
            below = below[ticking[below]]
        if below.size == 0:
            return 0
        gone = [int(store.entity_id[slot]) for slot in below
                if not (store.flags[slot] & REMOVED)]
        if not gone:
            return 0
        for entity_id in gone:
            store.remove(entity_id)
        self._stats["removed_void"] += len(gone)
        if self.on_removed is not None:
            self.on_removed(gone)
        return len(gone)

    # ----- reporting -----

    def _refresh_stats(self) -> None:
        self._stats["bodies"] = len(self.bodies)
        self._stats["ms"] = round(self._last_ms, 4)

    def stats(self) -> Dict[str, int]:
        """The counters `StatsMsg["physics"]` reports. Copies, so callers cannot
        mutate the live counters by accident.

        `bodies`, `awake`, `sleeping` and `paused_not_ticking` describe the last
        completed tick (and add up: `bodies == awake + sleeping +
        paused_not_ticking`); everything else is a running total since boot or
        the last `reset_stats()`.
        """
        return dict(self._stats)

    def reset_stats(self) -> None:
        for key in self._stats:
            self._stats[key] = 0
        self.max_step = 0.0

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"<PhysicsSolver {self.bodies!r} view={self.view!r}>"
