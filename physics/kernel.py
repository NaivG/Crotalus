"""The movement kernel: swept-AABB collision, batched, in `njit`.

    from physics.kernel import move_batch, warm_up
    warm_up()                      # compile before the first tick, never inside it
    move_batch(...)                # one call per tick; arrays and scalars only

Ported from `Pumpkin/crates/pumpkin-core/src/entity/mod.rs:1397
adjust_movement_for_collisions`, `pumpkin-util/src/math/boundingbox.rs:260
calculate_collision_time` and `world/mod.rs:2326 get_block_collisions`, which are
themselves ports of vanilla's `Entity.move` / `AABB.collide`. The structure is
kept step for step and the names are kept findable, because the next person to
touch this will have the Java open next to it.

The four things that have to be right, and what breaks without each
------------------------------------------------------------------

1. **Y first, and `on_ground` only when moving down.** Vanilla's `Entity.move`
   is `this.onGround = vec3.y != movement.y && movement.y < 0.0`, which is what
   makes a body land and stay put instead of sinking a little every tick.
2. **A collision behind the direction of travel is not a collision.** The
   `collision_time < 0` early return is all of it: a body pressed against a wall
   must be free to move away from it, and a body on a floor must be free to move
   sideways. (`time == -0.0` is deliberately *not* `< 0`: a body resting exactly
   on a surface computes -0.0 against it, and a `<= 0.0` here is how a body
   falls through the floor it is standing on.)
3. **Candidates come from the swept box, not the starting box.** Collecting
   from the starting box only is how a fast-falling entity passes through the
   ground it should have hit. Vanilla's block range is `floor(min) - 1 ..
   floor(max) + 1` on all six sides; the extra block is what catches the 43
   vanilla shapes whose collision box leaves its own cell (a fence post reaches
   y = 1.5).
4. **A blocked horizontal move gets a step-up retry.** Without it a mob walks
   into a slab and stops instead of stepping onto it.

Deliberate deviations, with the reason
--------------------------------------

* **No 0.05 sub-stepping.** The plan reserved it, but it is redundant here:
  candidates come from the *swept* box, and each axis is then solved exactly
  (`collision_time` computes the crossing directly rather than sampling). A 1-D
  sweep against every block in the swept range cannot tunnel, at any speed.
  `tools/check_physics_solver.py` asserts that at terminal velocity rather than
  taking it on trust.
* **Air drag is not here.** The kernel moves a body by the motion it is handed;
  `physics/solver.py` decides what that motion is.
* **No entity-to-entity collision, no pistons, no `push_out_of_blocks`.** A body
  a block change has trapped stays trapped until a later sweep pushes it out
  along an axis -- the gap `PHYSICS_PLAN.md` §七 records.
* **`step_height` is an argument, not an attribute.** There is no attribute
  system yet (`PHYSICS_PLAN.md` §八.4); the solver passes vanilla's default 0.6.

The data plane
--------------

Every argument is an array or a scalar. The one exception is `sections`, a flat
`numba.typed.List` of the live `uint16[4096]` section arrays
(`arr[x + z*16 + y*256]`), indexed `slot * sections_per_chunk + section_index`.
**It must be a `typed.List` and not a Python `list`**: passing a Python list of
15000 section arrays costs 222 ms *per call* -- numba walks the whole list every
time, whatever the function then does with it -- against 5 us for the same list
as a `typed.List`. Measured with `tools/bench_physics.py`; it was the difference
between 0.7 ms and 700 ms per tick, and it is the single most expensive mistake
available in this signature. Numba unboxes an element per access, so the loops
hoist each array once per chunk rather than once per block. `candidates` is a
caller-owned `float64[MAX_CANDIDATES, 6]` scratch buffer, so nothing here
allocates.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:  # pragma: no cover - numba is a declared dependency
    from numba import njit

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator if not args else decorator(args[0])

__all__ = [
    "move_batch", "warm_up", "MAX_CANDIDATES",
    "STATUS_ON_GROUND", "STATUS_HORIZONTAL_COLLISION", "STATUS_DEFERRED",
    "STATUS_MOVED", "STATUS_OVERFLOW",
]

#: Bits written into `move_batch`'s `status` output, one entry per active body.
STATUS_ON_GROUND = 1
STATUS_HORIZONTAL_COLLISION = 2
#: A chunk in the swept span is not resident. The body was not moved at all.
STATUS_DEFERRED = 4
#: The body's position changed by more than `MOVE_EPSILON`.
STATUS_MOVED = 8
#: The candidate buffer filled up. The body was left where it was and the solver
#: counts this as an error -- never as "nothing to collide with".
STATUS_OVERFLOW = 16

#: For the `MOVED` flag only. `on_ground` uses vanilla's exact `!=`, because a
#: body clipped to exactly zero *did* collide and has to report it.
MOVE_EPSILON = 1e-9

#: Vanilla's `BlockCollisions` floors with `- 1e-7` / `+ 1e-7` and then expands
#: by one block, so a box resting exactly on a boundary does not claim the block
#: on the far side of it.
BLOCK_EPSILON = 1e-7

#: Scratch room for one body's candidate shapes. A 0.9x1.4 body falling at
#: terminal velocity sweeps ~7x8x7 blocks and the densest vanilla state carries 5
#: boxes (`cobblestone_wall`), so this is two orders of magnitude of headroom: an
#: overflow means the caller's box is wrong, not that the world is dense.
MAX_CANDIDATES = 4096


if _HAS_NUMBA:

    @njit(cache=True, inline="always")
    def _key(chunk_x, chunk_z):
        """`world_view.chunk_key`, kept identical to the Python one."""
        return (chunk_x << 32) | (chunk_z & 0xFFFFFFFF)

    @njit(cache=True, inline="always")
    def _find_slot(keys, slots, count, key):
        """Binary search the key-sorted index; -1 when the chunk is not resident."""
        low = 0
        high = count - 1
        while low <= high:
            mid = (low + high) >> 1
            value = keys[mid]
            if value == key:
                return slots[mid]
            if value < key:
                low = mid + 1
            else:
                high = mid - 1
        return -1

    @njit(cache=True, inline="always")
    def _clip_axis(bx0, by0, bz0, bx1, by1, bz1,
                   sx0, sy0, sz0, sx1, sy1, sz1,
                   mx, my, mz, axis, best):
        """One axis of vanilla's `AABB.collideX` / `collideY` / `collideZ`.

        Returns the smallest movement along `axis` that stops the box against a
        face of the shape, or `best` when the shape cannot stop it. `best` starts
        at the full movement, so "nothing in the way" needs no sentinel.

        **`delta` is a coordinate difference, not `movement * time`.** That is
        vanilla's own formula (`d = this.minX - box.maxX`) and it is load
        bearing: the time-fraction spelling lands one ulp *below* the surface,
        and the next tick reads a box whose feet are 2e-16 inside the floor as
        already penetrating, refuses to clip it, and the body falls through the
        world. Rounding a position onto a plane has to be done in the plane's own
        coordinates.

        The arrival point (`box + motion * delta / movement`) is only used for
        the "do the other two axes overlap there" test, where an ulp is noise.
        """
        movement = mx if axis == 0 else (my if axis == 1 else mz)
        if movement == 0.0:
            return best

        if movement > 0.0:
            self_plane = bx1 if axis == 0 else (by1 if axis == 1 else bz1)
            other_plane = sx0 if axis == 0 else (sy0 if axis == 1 else sz0)
        else:
            self_plane = bx0 if axis == 0 else (by0 if axis == 1 else bz0)
            other_plane = sx1 if axis == 0 else (sy1 if axis == 1 else sz1)

        delta = other_plane - self_plane
        # Wrong sign: the face is behind the direction of travel, so it cannot
        # stop this move. `delta == 0.0` is a box resting exactly against the
        # face and *is* a collision -- clipping to zero is what keeps a body on
        # the floor it is standing on.
        if movement > 0.0:
            if delta < 0.0 or delta >= best:
                return best
        else:
            if delta > 0.0 or delta <= best:
                return best

        time = delta / movement
        moved_x0 = bx0 + mx * time
        moved_y0 = by0 + my * time
        moved_z0 = bz0 + mz * time
        moved_x1 = bx1 + mx * time
        moved_y1 = by1 + my * time
        moved_z1 = bz1 + mz * time

        if axis != 0 and not (moved_x0 < sx1 and moved_x1 > sx0):
            return best
        if axis != 1 and not (moved_y0 < sy1 and moved_y1 > sy0):
            return best
        if axis != 2 and not (moved_z0 < sz1 and moved_z1 > sz0):
            return best
        return delta

    @njit(cache=True, inline="always")
    def _block_range(bx0, by0, bz0, bx1, by1, bz1):
        """Vanilla's `BlockCollisions` range: one block out on all six sides."""
        return (int(np.floor(bx0 - BLOCK_EPSILON)) - 1,
                int(np.floor(by0 - BLOCK_EPSILON)) - 1,
                int(np.floor(bz0 - BLOCK_EPSILON)) - 1,
                int(np.floor(bx1 + BLOCK_EPSILON)) + 1,
                int(np.floor(by1 + BLOCK_EPSILON)) + 1,
                int(np.floor(bz1 + BLOCK_EPSILON)) + 1)

    @njit(cache=True, inline="always")
    def _stretch(bx0, by0, bz0, bx1, by1, bz1, mx, my, mz):
        """The box swept by the motion -- the volume the body will occupy."""
        return (bx0 + (mx if mx < 0.0 else 0.0),
                by0 + (my if my < 0.0 else 0.0),
                bz0 + (mz if mz < 0.0 else 0.0),
                bx1 + (mx if mx > 0.0 else 0.0),
                by1 + (my if my > 0.0 else 0.0),
                bz1 + (mz if mz > 0.0 else 0.0))

    @njit(cache=True)
    def _all_resident(keys, slots, key_count, min_x, min_y, min_z, max_x, max_y, max_z):
        """Whether every chunk the block range touches is loaded.

        An unloaded chunk must not read as air (the body would fall through the
        world) and must not read as solid (it would be stuck in nothing), so the
        body is deferred instead -- `PHYSICS_PLAN.md` §4.7.
        """
        chunk_x = min_x >> 4
        while chunk_x <= (max_x >> 4):
            chunk_z = min_z >> 4
            while chunk_z <= (max_z >> 4):
                if _find_slot(keys, slots, key_count, _key(chunk_x, chunk_z)) < 0:
                    return False
                chunk_z += 1
            chunk_x += 1
        return True

    @njit(cache=True)
    def _collect(sections, sections_per_chunk, min_section_y,
                 keys, slots, key_count,
                 min_x, min_y, min_z, max_x, max_y, max_z,
                 bx0, by0, bz0, bx1, by1, bz1,
                 offsets, ids, shape_min, shape_max, candidates):
        """Shapes overlapping the box, into the caller's scratch buffer.

        Returns the count, or -1 when the buffer is too small. Blocks are walked
        chunk by chunk so the index is searched once per chunk touched rather
        than once per block.
        """
        count = 0
        block_y = min_y
        while block_y <= max_y:
            section = block_y >> 4
            in_window = (section >= min_section_y
                         and section < min_section_y + sections_per_chunk)
            block_z = min_z
            while block_z <= max_z:
                chunk_z = block_z >> 4
                block_x = min_x
                while block_x <= max_x:
                    chunk_x = block_x >> 4
                    end_x = (chunk_x << 4) + 15
                    if end_x > max_x:
                        end_x = max_x
                    if in_window:
                        slot = _find_slot(keys, slots, key_count,
                                          _key(chunk_x, chunk_z))
                        if slot >= 0:
                            array = sections[slot * sections_per_chunk
                                             + (section - min_section_y)]
                            x = block_x
                            while x <= end_x:
                                state = array[(x & 15) + (block_z & 15) * 16
                                              + (block_y & 15) * 256]
                                if state != 0:
                                    first = int(offsets[state])
                                    last = int(offsets[state + 1])
                                    for index in range(first, last):
                                        shape = int(ids[index])
                                        sx0 = shape_min[shape, 0] + x
                                        sy0 = shape_min[shape, 1] + block_y
                                        sz0 = shape_min[shape, 2] + block_z
                                        sx1 = shape_max[shape, 0] + x
                                        sy1 = shape_max[shape, 1] + block_y
                                        sz1 = shape_max[shape, 2] + block_z
                                        if (sx0 < bx1 and sx1 > bx0 and sy0 < by1
                                                and sy1 > by0 and sz0 < bz1
                                                and sz1 > bz0):
                                            if count >= candidates.shape[0]:
                                                return -1
                                            candidates[count, 0] = sx0
                                            candidates[count, 1] = sy0
                                            candidates[count, 2] = sz0
                                            candidates[count, 3] = sx1
                                            candidates[count, 4] = sy1
                                            candidates[count, 5] = sz1
                                            count += 1
                                x += 1
                    block_x = end_x + 1
                block_z += 1
            block_y += 1
        return count

    @njit(cache=True)
    def _sweep(bx0, by0, bz0, bx1, by1, bz1, mx, my, mz,
               candidates, cand_count):
        """Y, then X, then Z, moving the box between axes.

        Returning the clipped motion as a 3-tuple.

        Pumpkin resolves all three axes from the box the body *started* at. That
        is fine for gravity and for walls, but it cannot express a step-up: the
        whole point of the retry is that the body rises first and the slab that
        blocked it is no longer beside it. So the box is carried from axis to
        axis here, which is what vanilla's `vec32.add(collide(..., aabb.move(vec32)))`
        depends on. The two agree everywhere else -- for a body on the ground
        the Y clip is exactly zero, so the later axes start from the same box.
        """
        if my != 0.0:
            best = my
            for index in range(cand_count):
                best = _clip_axis(
                    bx0, by0, bz0, bx1, by1, bz1,
                    candidates[index, 0], candidates[index, 1], candidates[index, 2],
                    candidates[index, 3], candidates[index, 4], candidates[index, 5],
                    mx, my, mz, 1, best)
            my = best
            by0 = by0 + my
            by1 = by1 + my

        if mx != 0.0:
            best = mx
            for index in range(cand_count):
                best = _clip_axis(
                    bx0, by0, bz0, bx1, by1, bz1,
                    candidates[index, 0], candidates[index, 1], candidates[index, 2],
                    candidates[index, 3], candidates[index, 4], candidates[index, 5],
                    mx, 0.0, mz, 0, best)
            mx = best
            bx0 = bx0 + mx
            bx1 = bx1 + mx

        if mz != 0.0:
            best = mz
            for index in range(cand_count):
                best = _clip_axis(
                    bx0, by0, bz0, bx1, by1, bz1,
                    candidates[index, 0], candidates[index, 1], candidates[index, 2],
                    candidates[index, 3], candidates[index, 4], candidates[index, 5],
                    0.0, 0.0, mz, 2, best)
            mz = best
        return mx, my, mz

    @njit(cache=True)
    def _step_up(sections, sections_per_chunk, min_section_y,
                 keys, slots, key_count,
                 offsets, ids, shape_min, shape_max, candidates,
                 bx0, by0, bz0, bx1, by1, bz1, mx, my, mz, step_height):
        """Vanilla's step-up retry, as `Entity.collide` spells it.

        Two candidate moves are tried -- the same horizontal movement raised by
        `step_height`, and the same movement with no vertical component -- and
        whichever gets further horizontally wins. The winner is then dropped
        back onto whatever is under it, so a body that stepped onto a slab does
        not float above it.

        Returns `(x, y, z, ok)`; `ok` is False when a scratch buffer overflowed,
        and the caller then keeps the un-stepped result *and* reports the
        overflow, rather than pretending the step-up found nothing.
        """
        best_x = 0.0
        best_y = 0.0
        best_z = 0.0
        best_horizontal = -1.0
        for attempt in range(2):
            try_my = step_height if attempt == 0 else 0.0
            sx0, sy0, sz0, sx1, sy1, sz1 = _stretch(
                bx0, by0, bz0, bx1, by1, bz1, mx, try_my, mz)
            min_x, min_y, min_z, max_x, max_y, max_z = _block_range(
                sx0, sy0, sz0, sx1, sy1, sz1)
            count = _collect(
                sections, sections_per_chunk, min_section_y,
                keys, slots, key_count,
                min_x, min_y, min_z, max_x, max_y, max_z,
                sx0, sy0, sz0, sx1, sy1, sz1,
                offsets, ids, shape_min, shape_max, candidates)
            if count < 0:
                return 0.0, 0.0, 0.0, False
            up_x, up_y, up_z = _sweep(bx0, by0, bz0, bx1, by1, bz1,
                                      mx, try_my, mz, candidates, count)
            horizontal = up_x * up_x + up_z * up_z
            if horizontal > best_horizontal:
                best_horizontal = horizontal
                best_x, best_y, best_z = up_x, up_y, up_z

        if best_horizontal <= 0.0:
            return 0.0, 0.0, 0.0, True

        # Drop back onto the surface: from the moved box, move by whatever is
        # left of the original vertical motion (vanilla's `-vec32.y + movement.y`).
        drop = -best_y + my
        moved_x0 = bx0 + best_x
        moved_y0 = by0 + best_y
        moved_z0 = bz0 + best_z
        moved_x1 = bx1 + best_x
        moved_y1 = by1 + best_y
        moved_z1 = bz1 + best_z
        sx0, sy0, sz0, sx1, sy1, sz1 = _stretch(
            moved_x0, moved_y0, moved_z0, moved_x1, moved_y1, moved_z1, 0.0, drop, 0.0)
        min_x, min_y, min_z, max_x, max_y, max_z = _block_range(
            sx0, sy0, sz0, sx1, sy1, sz1)
        count = _collect(
            sections, sections_per_chunk, min_section_y,
            keys, slots, key_count,
            min_x, min_y, min_z, max_x, max_y, max_z,
            sx0, sy0, sz0, sx1, sy1, sz1,
            offsets, ids, shape_min, shape_max, candidates)
        if count < 0:
            return 0.0, 0.0, 0.0, False
        drop_x, drop_y, drop_z = _sweep(moved_x0, moved_y0, moved_z0,
                                        moved_x1, moved_y1, moved_z1,
                                        0.0, drop, 0.0, candidates, count)
        return best_x + drop_x, best_y + drop_y, best_z + drop_z, True

    @njit(cache=True, inline="always")
    def _store(x, y, z, vx, vy, vz, index, mx, my, mz):
        """Vanilla's `setPos(pos + vec3)` followed by `setDeltaMovement(vec3)`."""
        x[index] = x[index] + mx
        y[index] = y[index] + my
        z[index] = z[index] + mz
        vx[index] = mx
        vy[index] = my
        vz[index] = mz

    @njit(cache=True, inline="always")
    def _flags(mx, my, mz, out_x, out_y, out_z):
        """`on_ground`, `horizontal_collision` and `moved`.

        `out_*` is the motion that actually happened (the clipped motion), so
        `moved` tests it directly -- testing `out - motion` would call a body
        whose fall was clipped to exactly zero "moved", and that is a position
        packet every tick for a mob standing still.
        """
        value = 0
        # `Entity.move`: onGround = vec3.y != movement.y && movement.y < 0.0
        if out_y != my and my < 0.0:
            value |= STATUS_ON_GROUND
        if out_x != mx or out_z != mz:
            value |= STATUS_HORIZONTAL_COLLISION
        ax = out_x if out_x > 0.0 else -out_x
        ay = out_y if out_y > 0.0 else -out_y
        az = out_z if out_z > 0.0 else -out_z
        if ax > MOVE_EPSILON or ay > MOVE_EPSILON or az > MOVE_EPSILON:
            value |= STATUS_MOVED
        return value

    @njit(cache=True, inline="always")
    def _report(mx, my, mz, out_x, out_y, out_z, is_probe):
        """`_flags`, minus `MOVED` for a probe body: a probe moves nothing."""
        value = _flags(mx, my, mz, out_x, out_y, out_z)
        if is_probe:
            value &= ~STATUS_MOVED
        return value

    @njit(cache=True)
    def move_batch(x, y, z, vx, vy, vz, width, height,
                   active, motion, probe, status,
                   sections, sections_per_chunk, min_section_y,
                   keys, slots, key_count,
                   offsets, ids, shape_min, shape_max, candidates,
                   step_height):
        """Move every body in `active` by its motion, in place.

        `active` holds slot indices and `motion[k]` is the `(x, y, z)` movement
        for `active[k]`, in blocks per tick. `probe[k] != 0` means "answer only":
        position and velocity are left untouched and only `status[k]` is written,
        which is how a sleeping body is asked whether it still has anything under
        it without paying for a full step.

        Writes `x`/`y`/`z` and `vx`/`vy`/`vz` for every non-probe body -- the
        clipped motion *is* the new velocity, which is vanilla's
        `setDeltaMovement(vec3)` after `move`.
        """
        for k in range(active.shape[0]):
            index = active[k]
            status[k] = 0

            is_probe = probe[k] != 0
            mx = motion[k, 0]
            my = motion[k, 1]
            mz = motion[k, 2]

            if mx == 0.0 and my == 0.0 and mz == 0.0:
                # A body with nothing to do must not pay for a block lookup.
                continue

            half = width[index] * 0.5
            bx0 = x[index] - half
            by0 = y[index]
            bz0 = z[index] - half
            bx1 = x[index] + half
            by1 = y[index] + height[index]
            bz1 = z[index] + half

            sx0, sy0, sz0, sx1, sy1, sz1 = _stretch(
                bx0, by0, bz0, bx1, by1, bz1, mx, my, mz)
            min_x, min_y, min_z, max_x, max_y, max_z = _block_range(
                sx0, sy0, sz0, sx1, sy1, sz1)
            if not _all_resident(keys, slots, key_count,
                                 min_x, min_y, min_z, max_x, max_y, max_z):
                status[k] = STATUS_DEFERRED
                continue

            cand_count = _collect(
                sections, sections_per_chunk, min_section_y,
                keys, slots, key_count,
                min_x, min_y, min_z, max_x, max_y, max_z,
                sx0, sy0, sz0, sx1, sy1, sz1,
                offsets, ids, shape_min, shape_max, candidates)
            if cand_count < 0:
                status[k] = STATUS_OVERFLOW
                continue

            if cand_count == 0:
                # Nothing anywhere in the swept box: the whole motion lands.
                if not is_probe:
                    _store(x, y, z, vx, vy, vz, index, mx, my, mz)
                status[k] = _report(mx, my, mz, mx, my, mz, is_probe)
                continue

            out_x, out_y, out_z = _sweep(bx0, by0, bz0, bx1, by1, bz1,
                                         mx, my, mz, candidates, cand_count)

            # Step-up, but only for a body the ground is holding up: a body in
            # mid-air that clips a wall must not climb it. `on_ground` here is
            # vanilla's test on *this* sweep's result, before any retry.
            overflow = False
            grounded = out_y != my and my < 0.0
            if step_height > 0.0 and grounded and (out_x != mx or out_z != mz):
                up_x, up_y, up_z, ok = _step_up(
                    sections, sections_per_chunk, min_section_y,
                    keys, slots, key_count,
                    offsets, ids, shape_min, shape_max, candidates,
                    bx0, by0, bz0, bx1, by1, bz1,
                    mx, my, mz, step_height)
                if not ok:
                    # The scratch buffer filled up twice in one tick. Keep the
                    # safe (un-stepped) result and report it -- never silently
                    # treat a full buffer as "nothing there".
                    overflow = True
                elif (up_x * up_x + up_z * up_z) > (out_x * out_x + out_z * out_z):
                    out_x, out_y, out_z = up_x, up_y, up_z

            if not is_probe:
                _store(x, y, z, vx, vy, vz, index, out_x, out_y, out_z)
            status[k] = _report(mx, my, mz, out_x, out_y, out_z, is_probe)
            if overflow:
                status[k] |= STATUS_OVERFLOW

    _WARMED = [False]

    def warm_up() -> None:
        """Compile `move_batch` (and everything it calls) before any tick.

        `cache=True` makes the *second* process cheap, not the first: the first
        call after a source change pays the LLVM compile, hundreds of
        milliseconds, which is a dozen ticks of stall if it happens mid-tick.
        The solver calls this from the actor's setup path, where a stall is
        invisible; `tools/bench_jax_numba.py:205` warms JIT the same way.

        The warm world is synthetic on purpose -- a full cube and a half slab in
        a one-chunk world built from a two-entry shape table -- because then
        this needs no registry, and it still exercises every path including the
        step-up retry (the body walks into the slab). The body sits at the
        middle of the chunk rather than at a corner, so the block range's ±1
        expansion stays inside the one resident chunk.
        """
        if not _HAS_NUMBA or _WARMED[0]:
            return
        from numba import types
        from numba.typed import List as TypedList

        # State 1 is a full cube, state 2 a bottom slab; the tables are the only
        # thing that makes them mean anything to the kernel.
        offsets = np.array([0, 0, 1, 2], dtype=np.int64)
        ids = np.array([0, 1], dtype=np.int32)
        shape_min = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        shape_max = np.array([[1.0, 1.0, 1.0], [1.0, 0.5, 1.0]])
        section = np.zeros(4096, dtype=np.uint16)
        section[0:256] = 1                       # the y = 0 layer is the floor
        section[9 + 0 * 16 + 1 * 256] = 2        # a slab on it, at x = 9
        # A `typed.List`, because that is what the solver passes: warming the
        # kernel on a Python list would compile one specialization and leave the
        # real one to be compiled inside a tick.
        sections = TypedList.empty_list(types.uint16[::1])
        sections.append(section)
        keys = np.array([0], dtype=np.int64)     # chunk (0, 0)
        slots = np.array([0], dtype=np.int32)
        active = np.array([0], dtype=np.int32)
        motion = np.zeros((1, 3))
        probe = np.zeros(1, dtype=np.uint8)
        status = np.zeros(1, dtype=np.uint8)
        x = np.array([8.5])
        y = np.array([2.0])
        z = np.array([8.5])
        vx = np.zeros(1)
        vy = np.zeros(1)
        vz = np.zeros(1)
        width = np.array([0.6])
        height = np.array([1.8])
        candidates = np.zeros((MAX_CANDIDATES, 6))
        for _ in range(40):
            motion[0, 0] = 0.3
            motion[0, 1] = -0.08
            motion[0, 2] = 0.0
            move_batch(x, y, z, vx, vy, vz, width, height, active, motion, probe,
                       status, sections, 24, 0, keys, slots, 1,
                       offsets, ids, shape_min, shape_max, candidates, 0.6)
            if int(status[0]) & STATUS_MOVED == 0:
                break
        _WARMED[0] = True
