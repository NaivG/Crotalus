"""
Connected cave carving.

Caves are carved the way vanilla carves them: a few random start points per
chunk, each walking a wandering tunnel and boring a chain of overlapping
ellipsoids along the way. Branch points spawn child tunnels from wherever the
parent happens to be, and the children do the same, so a single start can
grow into a small network rather than a single corridor.

The earlier implementation here thresholded one 3D noise field
(`cave_noise > 0.6`) and that can only ever produce isolated blobs. The
superlevel set of a scalar field has whatever connectivity the field has, and
a 32-block-scale fBm field has none -- so no threshold produces tunnels. The
random walk gets its connectivity from a different source: consecutive
ellipsoids overlap along the path, so the carved set inherits the path's
continuity regardless of what the terrain noise looks like.

Tunnels are generated for a `radius`-chunk neighbourhood around the chunk
being built, not just for the chunk itself. A tunnel that starts inside the
chunk can walk out of it, so a chunk-local carver would chop every cave at the
16-block boundary. The walk is clipped to the target chunk when it writes, so
the neighbourhood costs some work and buys seam-free caves.
"""

import math
from typing import Optional

import numpy as np
from numba import njit

from world.legacy.noise import octave_noise_3d_array


# ---------------------------------------------------------------------------
# RNG
#
# A 32-bit LCG carried explicitly through the walk. Java's Random is a 48-bit
# LCG; matching it exactly buys nothing here (the terrain is hand-rolled, not
# vanilla, so the caves will never line up with a real world's anyway) and its
# multiplier overflows int64. Numerical Recipes' constants stay well inside
# int64 at every step, and taking bits 8..32 for the value avoids the
# low-bit correlation an LCG's low bits are notorious for.
# ---------------------------------------------------------------------------

_LCG_A = 1664525
_LCG_C = 1013904223
_U32 = 0xFFFFFFFF


@njit(inline="always")
def _rand_unit(state):
    """One LCG step. Returns (new_state, uniform in [0, 1))."""
    state = (state * _LCG_A + _LCG_C) & _U32
    return state, (state >> 8) * (1.0 / 16777216.0)


@njit(inline="always")
def _rand_below(state, bound):
    """One LCG step. Returns (new_state, integer in [0, bound))."""
    state = (state * _LCG_A + _LCG_C) & _U32
    return state, (state >> 8) % (bound if bound > 0 else 1)


@njit(inline="always")
def _mix32(value):
    """Murmur-style avalanche, so nearby inputs give unrelated seeds.

    Every step is masked back to 32 bits and the largest intermediate is
    2^32 * 0x45D9F3B < 2^59, so this never overflows int64.
    """
    value = (value ^ (value >> 16)) & _U32
    value = (value * 0x45D9F3B) & _U32
    value = (value ^ (value >> 16)) & _U32
    value = (value * 0x45D9F3B) & _U32
    return value ^ (value >> 16)


@njit(inline="always")
def _carver_seed(world_seed, cx, cz, index):
    """Seed for the index'th cave start in carver chunk (cx, cz).

    Deterministic in its arguments alone, never in generation order, which is
    what lets a tunnel owned by a neighbouring carver chunk come out
    identical no matter which chunk asks for it. Two chunks carving the same
    overlapping cell therefore agree instead of fighting.
    """
    h = _mix32(world_seed)
    h = _mix32(h ^ _mix32(cx * 0x1F1F1F1F + 0x01234567))
    h = _mix32(h ^ _mix32(cz * 0x1F1F1F1F + 0x89ABCDEF))
    h = _mix32(h ^ _mix32(index * 0x2545F491 + 0x9E3779B9))
    # A zero seed is an absorbing state for an LCG -- hand back a live one.
    return h if h != 0 else 0x5DEECE66D


# ---------------------------------------------------------------------------
# Carving primitives
# ---------------------------------------------------------------------------


@njit(inline="always")
def _can_reach(mcx, mcz, x, z, step, dist, thickness):
    """Can the tunnel still reach the target chunk with steps left?

    The walk is nearly straight, so the reachable disc grows as a circle of
    radius `dist - step` around the current position; the target chunk is
    approximated by its centre plus its own extent. Tunnels that start out of
    range are abandoned on their first step instead of walking the full
    length -- that early-out is most of the reason carving is affordable.
    """
    xd = x - (mcx * 16 + 8.0)
    zd = z - (mcz * 16 + 8.0)
    remaining = dist - step
    rr = thickness + 18.0
    return (xd * xd + zd * zd) - remaining * remaining <= rr * rr


@njit(inline="always")
def _carve_ellipsoid(mask, min_y, max_y, mcx, mcz, x, y, z, hr, vr):
    """Bore one ellipsoid, clipped to the target chunk's columns.

    Iterates y outermost so the rows that fall outside the ellipsoid are
    rejected once per row rather than once per column. That is worth the
    reordering: consecutive ellipsoids overlap heavily along a tunnel, so
    most columns are already air and the early-out skips them.
    """
    if hr <= 0.0 or vr <= 0.0:
        return

    center_x = mcx * 16 + 8.0
    center_z = mcz * 16 + 8.0
    reach = 16.0 + hr * 2.0
    if abs(x - center_x) > reach or abs(z - center_z) > reach:
        return

    base_x = mcx * 16
    base_z = mcz * 16

    y_lo = max(min_y, int(math.floor(y - vr)))
    y_hi = min(max_y, int(math.ceil(y + vr)))
    if y_lo > y_hi:
        return

    lo_x = int(max(0.0, math.floor(x - hr) - base_x - 1.0))
    hi_x = int(min(15.0, math.floor(x + hr) - base_x + 1.0))
    lo_z = int(max(0.0, math.floor(z - hr) - base_z - 1.0))
    hi_z = int(min(15.0, math.floor(z + hr) - base_z + 1.0))
    if lo_x > hi_x or lo_z > hi_z:
        return

    for wy in range(y_lo, y_hi + 1):
        yd = (wy - y) / vr
        # Horizontal disc still available at this height; <= 0 means the row
        # is entirely outside the ellipsoid.
        limit = 1.0 - yd * yd
        if limit <= 0.0:
            continue
        row = wy - min_y
        for xi in range(lo_x, hi_x + 1):
            xd = (base_x + xi + 0.5 - x) / hr
            xd2 = xd * xd
            if xd2 >= limit:
                continue
            for zi in range(lo_z, hi_z + 1):
                zd = (base_z + zi + 0.5 - z) / hr
                if xd2 + zd * zd < limit:
                    mask[row, zi, xi] = 1


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------

# A tunnel branches once and its children can branch again, so the live set
# is bounded at 2 + 4. Eight leaves headroom without unbounded growth.
_STACK_CAP = 8


@njit(cache=True)
def _carve_cave_start(mask, min_y, max_y, mcx, mcz, state, x, y, z,
                      dist, thickness, h_rot, v_rot, hr_mult, vr_mult, v_scale):
    """Walk one cave start, boring ellipsoids and branching on the way.

    The rotation update is the whole trick. `v_rot` is damped toward zero
    (x0.7, or x0.92 for the rare steep tunnel) so the tunnel flattens out
    into roughly horizontal corridors, while `h_rot` accumulates `y_rota` and
    swings the heading around. Damped-and-drifting is what produces corridors
    that run for a hundred blocks and then turn, rather than a random walk
    that wanders in circles.
    """
    s_x = np.empty(_STACK_CAP, np.float64)
    s_y = np.empty(_STACK_CAP, np.float64)
    s_z = np.empty(_STACK_CAP, np.float64)
    s_t = np.empty(_STACK_CAP, np.float64)
    s_hr = np.empty(_STACK_CAP, np.float64)
    s_vr = np.empty(_STACK_CAP, np.float64)
    s_step = np.empty(_STACK_CAP, np.int64)
    s_scale = np.empty(_STACK_CAP, np.float64)

    s_x[0] = x
    s_y[0] = y
    s_z[0] = z
    s_t[0] = thickness
    s_hr[0] = h_rot
    s_vr[0] = v_rot
    s_step[0] = 0
    s_scale[0] = v_scale
    sp = 1

    while sp > 0:
        sp -= 1
        x = s_x[sp]
        y = s_y[sp]
        z = s_z[sp]
        thickness = s_t[sp]
        h_rot = s_hr[sp]
        v_rot = s_vr[sp]
        step = s_step[sp]
        v_scale = s_scale[sp]

        state, r = _rand_below(state, dist // 2)
        split_point = r + dist // 4
        state, r = _rand_below(state, 6)
        steep = r == 0

        x_rota = 0.0
        y_rota = 0.0

        while step < dist:
            progress = math.pi * step / dist
            hr = 1.5 + math.sin(progress) * thickness
            vr = hr * v_scale
            cos_x = math.cos(v_rot)
            x += math.cos(h_rot) * cos_x
            y += math.sin(v_rot)
            z += math.sin(h_rot) * cos_x

            if steep:
                v_rot = v_rot * 0.92 + x_rota * 0.1
            else:
                v_rot = v_rot * 0.7 + x_rota * 0.1
            h_rot += y_rota * 0.1

            state, a = _rand_unit(state)
            state, b = _rand_unit(state)
            state, c = _rand_unit(state)
            x_rota = x_rota * 0.9 + (a - b) * c * 2.0
            state, a = _rand_unit(state)
            state, b = _rand_unit(state)
            state, c = _rand_unit(state)
            y_rota = y_rota * 0.75 + (a - b) * c * 4.0

            if step == split_point and thickness > 1.0:
                # Two thinner tunnels peel off at right angles to each other,
                # both from the parent's exact position. The right angle is
                # what makes branches read as branches -- spawn them with the
                # same heading and they overlap into one thick trunk.
                if sp + 2 <= _STACK_CAP:
                    for k in range(2):
                        state, t = _rand_unit(state)
                        s_x[sp + k] = x
                        s_y[sp + k] = y
                        s_z[sp + k] = z
                        s_t[sp + k] = t * 0.5 + 0.5
                        s_hr[sp + k] = h_rot + (math.pi * 0.5 if k == 1 else -math.pi * 0.5)
                        s_vr[sp + k] = v_rot / 3.0
                        s_step[sp + k] = step
                        s_scale[sp + k] = 1.0
                    sp += 2
                break

            state, r = _rand_below(state, 4)
            if r != 0:
                if not _can_reach(mcx, mcz, x, z, step, dist, thickness):
                    break
                _carve_ellipsoid(mask, min_y, max_y, mcx, mcz, x, y, z,
                                 hr * hr_mult, vr * vr_mult)

            step += 1


@njit(cache=True)
def _carve_chunk_caves(mask, min_y, max_y, mcx, mcz, world_seed, radius,
                       cave_count, max_length, y_min, y_max, hr_mult, vr_mult,
                       v_scale, room_chance):
    """Carve every cave whose start point falls in the radius neighbourhood."""
    half = max_length // 4
    for dx in range(-radius, radius + 1):
        for dz in range(-radius, radius + 1):
            ccx = mcx + dx
            ccz = mcz + dz
            for i in range(cave_count):
                state = _carver_seed(world_seed, ccx, ccz, i)
                state, ox = _rand_below(state, 16)
                state, oz = _rand_below(state, 16)
                x = ccx * 16 + ox
                z = ccz * 16 + oz

                # y is biased toward the bottom. The exponent is the dial:
                # start density per unit height goes as f**(1-exponent), so
                # 1.8 is a 3.2x falloff from the deepest to the shallowest
                # layer and leaves the middle of the crust sparsely peopled,
                # while 1.35 is only 1.8x and keeps tunnels running through
                # the whole depth. Caves still read as bottom-heavy, which is
                # what vanilla does, without a dead band in the middle.
                state, f = _rand_unit(state)
                y = y_min + int((f ** 1.35) * (y_max - y_min))

                # Per-cave multipliers on vanilla's tunnel shape. The drawn
                # ranges are the vanilla ones; the caller's multipliers scale
                # them, so 1.0 everywhere reproduces the reference and a
                # single number can widen or flatten every tunnel at once.
                state, a = _rand_unit(state)
                hr_mult_local = (0.75 + a * 0.5) * hr_mult
                state, a = _rand_unit(state)
                vr_mult_local = (0.5 + a * 0.25) * vr_mult
                state, a = _rand_unit(state)
                v_scale_local = (0.75 + a * 0.5) * v_scale
                state, a = _rand_unit(state)
                thickness = 0.5 + a * 0.5

                state, a = _rand_below(state, max_length - half)
                dist = max_length - a

                state, r = _rand_below(state, room_chance)
                tunnels = 1
                if r == 0:
                    # A wide, *flat* room at the start point, then a small
                    # bundle of tunnels out of it. This is what gives cave
                    # systems the open junctions you can stand in.
                    #
                    # The flattening matters more than it looks: a room is up
                    # to 17 blocks across, and scaled isotropically that is a
                    # 17-cube sphere -- a cavern big enough to swallow a
                    # chunk's worth of air on its own. Vanilla flattens the
                    # room to 0.5-0.75 of its width, which is what keeps it
                    # reading as a low chamber the tunnels feed into.
                    state, a = _rand_unit(state)
                    room_hr = 1.5 + (1.0 + a * 6.0)
                    state, a = _rand_unit(state)
                    _carve_ellipsoid(mask, min_y, max_y, mcx, mcz, x, y, z,
                                     room_hr, room_hr * (0.5 + a * 0.25))
                    state, r = _rand_below(state, 4)
                    tunnels = 1 + r

                for _ in range(tunnels):
                    state, a = _rand_unit(state)
                    h_rot = a * math.pi * 2.0
                    state, a = _rand_unit(state)
                    v_rot = (a - 0.5) / 4.0
                    state, a = _rand_unit(state)
                    _carve_cave_start(mask, min_y, max_y, mcx, mcz, state,
                                      x, y, z, dist, thickness, h_rot, v_rot,
                                      hr_mult_local, vr_mult_local, v_scale_local)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def carve_chunk_caves(mask: np.ndarray, min_y: int, max_y: int,
                      chunk_x: int, chunk_z: int, seed: int,
                      radius: int = 6, cave_count: int = 12,
                      max_length: int = 112, y_min: int = 8, y_max: int = 120,
                      hr_mult: float = 1.0, vr_mult: float = 1.0,
                      v_scale: float = 1.0, room_chance: int = 4) -> None:
    """Carve connected tunnels into `mask`, in place.

    `mask` is uint8 of shape (max_y - min_y + 1, 16, 16), indexed [y - min_y,
    z, x] -- the same [y, z, x] order the block arrays use. Existing non-zero
    cells are left alone, so this composes with any other cave pass.
    """
    _carve_chunk_caves(mask, min_y, max_y, chunk_x, chunk_z, seed & _U32,
                       radius, cave_count, max_length, y_min, y_max,
                       hr_mult, vr_mult, v_scale, room_chance)


_PERM_CACHE: dict = {}


def _local_perm(seed: int) -> np.ndarray:
    """Build (and cache) a permutation table without touching the global one.

    `init_perlin` writes straight to `world.legacy.noise._perm`, which every noise
    object in the process reads through. Constructing another noise object to
    get a cave field would therefore change the terrain too, so this builds
    the table locally and hands it to the sampler instead. Cached because
    rebuilding it costs a shuffle, and the cave field is sampled per chunk.
    """
    key = seed & 0x7FFFFFFF
    perm = _PERM_CACHE.get(key)
    if perm is None:
        rng = np.random.default_rng(key)
        p = np.arange(256, dtype=np.int64)
        rng.shuffle(p)
        perm = np.concatenate([p, p]).astype(np.int64)
        _PERM_CACHE[key] = perm
    return perm


def carve_cheese(mask: np.ndarray, min_y: int, max_y: int,
                 chunk_x: int, chunk_z: int, seed: int, scale: float = 96.0,
                 octaves: int = 2, threshold: float = 0.42) -> None:
    """Carve open chambers, in place, from a large-scale 3D noise field.

    This is the "cheese" half of vanilla's caves and it is meant to be
    blobby -- the connectedness in a cave system comes from the tunnels, and
    a noise field can only add the room-shaped volume they connect. Low
    frequency is what makes them read as chambers rather than more blobs:
    at the old 32-block scale every feature was a ball a player could walk
    around, at 96 the features are wider than the tunnels that feed them.
    """
    ys = np.arange(min_y, max_y + 1, dtype=np.float64)
    grid = (len(ys), 16, 16)
    wx = chunk_x * 16 + np.arange(16, dtype=np.float64)
    wz = chunk_z * 16 + np.arange(16, dtype=np.float64)

    field = octave_noise_3d_array(
        np.broadcast_to(wx[None, None, :], grid).ravel(),
        np.broadcast_to(ys[:, None, None], grid).ravel(),
        np.broadcast_to(wz[None, :, None], grid).ravel(),
        octaves, 0.5, 2.0, scale, _local_perm(seed),
    ).reshape(grid)

    mask[field > threshold] = 1
