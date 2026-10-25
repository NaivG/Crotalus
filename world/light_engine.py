"""LunarLight

The light engine for Crotalus: three numba kernels, and the per-dimension scheduler.

    light = DimensionLight(dimension.get_chunk, has_skylight=settings.has_skylight)
    light.queue_chunk(cx, cz)                        # a chunk entered memory
    light.block_changed(x, y, z, old_state, new_state)  # Dimension.set_block
    light.tick()                                     # actor thread, after the spawner

The two halves are independent on purpose: `relight_region`,
`update_block_light` and `update_sky_column` are `njit` kernels that take numpy
arrays and nothing else -- the same rule `physics/kernel.py` follows, and for
the same reason (a Python object per cell is the entire cost of a flood fill) --
while `DimensionLight` owns the queues, the 3x3 regions, the clock and the
policy. Algorithms live in the kernel and nowhere else; the scheduler never
computes a light level itself.

Why a 3x3 patch, and why it is idempotent
-----------------------------------------

Lighting a chunk means reading the **nine** chunks around it and writing back
only the middle one (`LIGHT_PLAN.md` §4.5, `world/light_region.py`). A light
level at a chunk border depends on the blocks on both sides of it, so a
single-chunk pass would have to guess, and the guess would show up as a seam
that a later packet would have to correct. With the 3x3 in hand the border is
computed once, from the real neighbours, and there is no second packet.

That is also what makes it idempotent rather than merely stable: `relight_region`
*discards* the light in all nine chunks, recomputes it from `blocks` alone, and
the centre chunk's nine neighbours will produce byte-identical values on their
own patches -- same blocks, same rules, same answer. Two adjacent chunks
computing the same border cell agree, so patches cannot fight, and running the
same patch twice is a no-op. (Cells are therefore computed up to nine times
over, which is the cost of never needing a repair pass.)

The kernel does not consume a neighbour's *light*, only its *blocks*. That is a
deliberate strengthening of `LIGHT_PLAN.md` §4.1's argument:

    > Eight neighbours' light may be stale (their own patch has not run yet).
    > That does not affect correctness: sky propagation is seeded entirely from
    > block data and column heights, never from a neighbour's light values; the
    > BFS reading a neighbour's light only decides "should this direction be
    > written", and a stale value costs at most one write, which the neighbour's
    > own patch necessarily performs.

Reading no neighbour light at all is stronger than "a stale value is harmless":
`ChunkSection.sky_light` defaults to `np.full(2048, 15)`, which unpacks to a
*checkerboard* of 15 and 0 (`world/light_layout.py:41`), and a fresh chunk is
`block_light` 0 everywhere -- treating either as real would light the centre
chunk's underground from a chunk that has never been lit. So the region's light
buffers are outputs, and a chunk's own patch is the only writer of its light.

Why opacity has exactly three values
------------------------------------

`opacity[state]` is 0, 1 or 15 for every one of the 29,671 states in
`assets/blocks.json` (16,240 / 10,872 / 2,559) and that column *is* vanilla's
`getLightBlock()` (`registry/blocks.py:847`). One rule covers all three:

    entering a neighbour: level - max(opacity, 1)      # 0 damps by 1, 15 by 15

`max(op, 1)` is vanilla's `getLightDampeningInto`; the 15 case is the "this face
is blocked" case (`level - 15 <= 0`, nothing is written). The three tiers are
what make water and leaves dim light by one per block instead of either passing
it through or stopping it, which is the whole difference between the old
heightmap approximation and a real light field. **Do not key occlusion off
`state_flags`**: no state in this build sets Pumpkin's `CAN_OCCLUDE` /
`IS_SOLID_RENDER` bits, so a "correction" that used them would make all 2,559
light-blocking states transparent (`LIGHT_PLAN.md` §1.1).

Sky light has one extra rule, in the BFS and in the decrease, and **not** in the
per-column fill (§4.2 / §8.2):

    level == 15 and the step is downward and the target's opacity == 0 -> 15

Without it a vertical shaft would dim by one per block on the way down; with it
applied to the column fill instead, water would not dim at all.

Partial convergence
-------------------

A tick spends at most `light-budget-ms` (default 4.0) on light, lights at most
`light-chunks-per-tick` (default 4) chunks and drains at most
`light-updates-per-tick` (default 4096) queue items. Light is **committed as it
is computed** -- written straight into `ChunkSection.sky_light` /
`.block_light` -- and the queues only mean "somewhere has not been reached yet".
So any light value a reader can see is self-consistent with the rules; what a
partial drain costs is *brightness*, never a wrong value: the decrease phase is
run to completion (vanilla's order: all decreases, then all increases), and a
cut-short increase phase leaves cells dimmer than they should be until a later
tick re-issues the work. Vanilla's `LevelLightEngine` spends a fixed budget per
tick the same way.

What it costs, measured (`tools/bench_light.py`, real generated chunks, and
§5.5 of the plan lists the 3x3 figure as untested): a full **3x3 patch is about
4.7 ms best / 4.9-5.4 ms p50** end to end -- ~0.8-1.6 ms copying nine chunks'
blocks in, ~2.4-2.8 ms in the kernel, ~0.5-0.6 ms packing the centre chunk back
-- against the plan's §4.8 estimate of 1.4 ms, which was a per-chunk number and
did not include reading the eight neighbours a patch needs. A single block change
is ~0.13-0.36 ms of kernel plus ~0.5 ms of write-back, and a tick that changed no
light writes nothing back at all. **The default budget therefore lights about
0.7-0.8 chunks per tick**, not four: the queue keeps the rest, and a player's
first view fills in over several ticks instead of one long stall.
`light-chunks-per-tick` stays the ceiling and the clock is the limit, so raising
`light-budget-ms` is the knob that matters.

Three things were measured out of that path since, and none of them is a
lighting idea -- which is the point:

* **The BFS is not the cost.** On a flat overworld patch the entire sky BFS is
  1,412 expansions, ~2% of the kernel. What the kernel spends its time on is two
  linear passes over all 884,736 cells of the region: the column-top/emitter
  scan and the per-column sky fill.
* **The BFS buffers are allocated once** (`LightQueues`). They used to be
  `np.empty`d inside every kernel call: three arrays of 3.5 MB per block edit,
  on the actor thread, inside the budget. numba's allocator cannot cache an
  array that size, so a patch was mmap/munmap-ing 3.5 MB. Removing it took the
  kernel from 3.2/4.5 ms to 2.4-2.7/2.5-2.8 ms (best/p50).
* **Unpacking is a numba loop, not numpy** (`light_region._unpack_section`).
  `LightRegion.build` expands 432 nibble arrays; the strided numpy form measured
  ~10 us each (~6.5 ms, more than the whole kernel) against ~0.5 us for the
  loop, so a build with `read_light=True` went from 7.6 ms to 1.5-2.2 ms p50.
  `world/light_layout.py`'s numpy version stays as the reference implementation
  and `tools/check_light_layout.py` pins the two together byte for byte.

Starlight's light-data management, which those first two measurements pointed at
-------------------------------------------------------------------------------

That is now the third figure, and it is the one the two linear passes were
waiting for. On `tools/bench_light.py`'s nine real chunks the region holds **138
all-air sections of 216** -- 64% of the 884,736 cells -- and on the centre chunk
alone **19 of 24 sections are uniform** (4 solid, 15 air), so 20,480 mixed cells
carry the whole 98,304-cell answer. Scanning the other 78,208 finds nothing, and
that was 81% of the kernel.

`LightRegion.air_sections` is the metadata the passes were missing
(`ChunkSection.non_air_count`, `world/chunk.py:152`), and `relight_region` now
takes it. Two things fall out of it, and they are separate wins:

* a flagged section is skipped by the column-top/emitter scan outright -- air has
  no opacity and emits nothing -- which is the whole of pass 1's saving;
* its sky is *seeded* to 15 rather than written cell by cell, and the fill's
  above-the-top loop skips it. That second half is not free: it is only sound
  where the section really is above every column top, which `is_empty` alone does
  not say. A block placed high in a cave makes an all-air section per-cell
  between two patches, so `relight_region` decides it from pass 1's own `tops`
  grid -- 256 comparisons per section, **2 us for all 216** -- and falls back to
  the per-cell fill where the predicate fails. Getting *that* wrong is what
  `tools/check_light_engine_incremental.py` caught: 16 cells left at 15 where the
  answer was 14.

Measured on the same fixture, same machine, best of 40: **2.29 ms -> 1.33 ms**
for a full 3x3 patch, **-42%**, with both layers identical byte for byte to the
unmasked kernel across all 9 chunks (15.9 M cells, `tools/check_light_batch.py`).
The remaining cost is the BFS, the 78 sections that really do hold blocks, and
the column fill inside them.

The table is optional on purpose. `NO_SECTION_TABLE` (`(0, 0)`: numba has no
optional arrays and a scalar has no `.shape` to type) means "no table, scan
everything", and it is what every caller outside `DimensionLight._light_chunk`
passes -- the tools, `warm_up`. The oracle comparison in
`tools/check_light_engine.py` therefore still exercises the general kernel, and
the masked path is checked by comparing it against that kernel rather than by
trusting it.

The work queues are **per dimension**, not per chunk, and that is why unloading
has an answer: a queue item whose chunk is gone is dropped and counted
(`dropped_unloaded`). There is no cross-chunk re-light task like vanilla's
`checkBlock`, so a chunk that loads next to one that is already lit can be up to
one level dark on the shared border until its own patch runs (`LIGHT_PLAN.md`
§4.7 -- the one known gap this round accepts).

Block edits queued inside one tick are merged per cell and drained oldest first
(`DimensionLight`'s class docstring has the measurement): without that, placing
and breaking the same block inside one tick leaves the block's light behind,
because the break is queued as a decrease and would run *before* the placement's
increase. Vanilla's decrease-before-increase still holds inside every kernel
call, and one edit per tick behaves exactly as it always did.
"""

from __future__ import annotations

from loguru import logger
import time
from collections import deque
from typing import Callable, Optional, Set, Tuple, List

import numpy as np

from world.chunk import Chunk
from world.light_layout import CELL_COUNT, MAX_LIGHT_LEVEL, SECTION_SIZE
from world.light_region import (
    CENTER_CHUNK_INDEX,
    REGION_CHUNKS,
    REGION_EDGE,
    SECTION_AIR,
    SECTION_MIXED,
    SKY_SEEDED,
    LightRegion,
)

# logger = logging.getLogger(__name__)

try:  # pragma: no cover - numba is a declared dependency
    from numba import njit

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False
    logger.warning("[LunarLight] numba not installed, falling back to Python")

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator if not args else decorator(args[0])


__all__ = [
    "DimensionLight",
    "LightQueues",
    "relight_region",
    "update_block_light",
    "update_sky_column",
    "warm_up",
]



#: Region-wide horizontal edge, in blocks: three chunks of 16.
REGION_COLUMNS = REGION_EDGE * SECTION_SIZE

#: State ids are `uint16` on disk, so this is the size of the flat tables
#: (`registry.blocks.light_opacity_table`).
STATE_ID_SPACE = 65536

#: A level of 1 would propagate 0 into every neighbour, so only cells above this
#: are worth a queue slot (`LIGHT_PLAN.md` §4.3: "新值 > 1 才继续入队").
PROPAGATE_ABOVE = 1

#: The "no section table" argument for `relight_region`. A `(0, 0)` array rather
#: than `None` or `-1`: numba has no optional arrays, and a scalar would make the
#: kernel's `air_sections.shape` unresolvable at type-inference time (both arms
#: of a branch are typed, whatever the runtime value is). `tools/*` and
#: `warm_up` pass this and get the scan-everything kernel.
NO_SECTION_TABLE = np.zeros((0, 0), dtype=np.uint8)

#: Queue entries hold region coordinates in one int32: 6 bits `wx`, 6 bits `wz`,
#: the rest `ry` (a region-relative world Y). 47 | 47 << 6 | 383 << 12 < 2**21.
_QUEUE_COLUMN_BITS = 6
_QUEUE_COLUMN_MASK = (1 << _QUEUE_COLUMN_BITS) - 1
_QUEUE_Y_SHIFT = 2 * _QUEUE_COLUMN_BITS


# ---------------------------------------------------------------------------
# Coordinates: the kernels work in *region* space, not chunk space
# ---------------------------------------------------------------------------
#
# Everything below uses `(wx, wz, ry)`: `wx`/`wz` are 0..47 across the 3x3 patch
# and `ry` is 0..height-1 *relative to the chunk's `min_y`* (so section `s` of
# `blocks` owns `s*16 <= ry < s*16 + 16`). A caller holding world coordinates
# converts once: `wx = 16 + (x & 15)`, `ry = y - chunk.min_y`. The centre chunk
# therefore always sits at `16 <= wx < 32`, `16 <= wz < 32`, which is what lets
# a kernel take "the chunk-local x" and still touch its neighbours.


@njit(cache=True, inline="always")
def _region_chunk(wx, wz):
    """`chunk_idx` of the region column, matching `light_region.chunk_index`."""
    return ((wz >> 4) * REGION_EDGE) + (wx >> 4)


@njit(cache=True, inline="always")
def _region_cell(wx, wz, ry):
    """Flat cell index for region coordinates.

    Spelled out rather than imported because it is the innermost expression of
    every loop here; it is exactly `light_layout.cell_index(wx & 15, ry & 15,
    wz & 15)` = `x + z*16 + y_local*256`, the layout `ChunkSection.blocks` uses.
    """
    return (wx & 15) + ((wz & 15) << 4) + ((ry & 15) << 8)


@njit(cache=True, inline="always")
def _pack_position(wx, wz, ry):
    """One queue entry: the three coordinates, packed into an int32."""
    return wx | (wz << _QUEUE_COLUMN_BITS) | (ry << _QUEUE_Y_SHIFT)


@njit(cache=True)
def _column_top(blocks, opacity, wx, wz, height):
    """Highest `ry` in this column whose state is not fully transparent, or -1.

    **The light engine's own top, not `Chunk.heightmap`.** `heightmap` is
    `WORLD_SURFACE` -- the highest *non-air* block, so leaves, tall grass and a
    torch count -- and only the overworld generator maintains it; sky light
    needs the highest block with `opacity > 0`. Using the heightmap would start
    the column fill above the leaves and light a whole forest floor from the
    wrong height, which is a bug that looks like a rendering quirk rather than a
    wrong number. `LIGHT_PLAN.md` §4.3 and the §4.5 review both call for this
    column scan; it is 384 array reads on the cold path and stops at the first
    hit.
    """
    c = _region_chunk(wx, wz)
    base = (wx & 15) + ((wz & 15) << 4)
    for ry in range(height - 1, -1, -1):
        if opacity[blocks[c, ry >> 4, base + ((ry & 15) << 8)]] > 0:
            return ry
    return -1


@njit(cache=True, inline="always")
def _seed_sky_column(sky, queue, tail, capacity, wx, wz, top,
                     north_top, south_top, west_top, east_top):
    """Stage 2 of §4.3: the lateral seed cells of one column.

    A cell is a seed when it holds light *and* is either the column's own top
    block or lies below some neighbouring column's top -- both mean "the sky
    reaches here and the side is open". This pass is the **only** source of sky
    light inside a cave: skip it and the result is the old heightmap picture
    (bright above the surface, black below), which is exactly what the engine
    exists to replace.

    The scan runs from the tallest of the five tops **all the way down** and
    stops only at a dark cell at or below this column's own top -- Pumpkin's
    `SkyLightPropagator::convert_light` (`engine.rs:726-749`), which checks the
    light first and the position second. Scanning only down to `top` looks
    equivalent and is not: the column fill subtracts `opacity` and opacity 0
    subtracts nothing, so a column whose top is water, glass or a flower and
    whose inside is air stays at 14 all the way down. Those cells are *lit*,
    they can light a neighbour, and the short scan never queued one of them. On
    a nine-chunk scene that lost 153 cells and left the kernel's own light field
    with 21 cells that its own rules could still raise.

    Do not "simplify" `range(highest, -1, -1)` back to `range(highest, top - 1,
    -1)`: `LIGHT_PLAN.md` §4.3 said the latter, and saying it is how this bug
    was written in the first place.
    """
    highest = top
    if north_top > highest:
        highest = north_top
    if south_top > highest:
        highest = south_top
    if west_top > highest:
        highest = west_top
    if east_top > highest:
        highest = east_top

    c = _region_chunk(wx, wz)
    base = (wx & 15) + ((wz & 15) << 4)
    for ry in range(highest, -1, -1):
        cell = base + ((ry & 15) << 8)
        level = int(sky[c, ry >> 4, cell])
        if level == 0:
            # A dark cell at or below our own top means the fill has already
            # run out: everything under it is dark too, so there is nothing
            # left to seed and the scan can stop.
            if ry <= top:
                break
            continue
        if (ry == top or ry < north_top or ry < south_top
                or ry < west_top or ry < east_top):
            if level > PROPAGATE_ABOVE and tail < capacity:
                queue[tail] = _pack_position(wx, wz, ry)
                tail += 1
    return tail


@njit(cache=True, inline="always")
def _spread_into(light, blocks, opacity, queue, tail, capacity,
                 wx, wz, ry, level, downward, is_sky):
    """One direction of the increase BFS: write `level - damp` into a neighbour.

    Returns the (possibly advanced) queue tail. All arithmetic is on int64 --
    `level - op` on a `uint8` would wrap to 255 and that was the first bug the
    plan's investigation hit (§9.1.1) -- and the write is guarded by
    `new > current`, so a value that cannot improve the cell is never stored and
    never enqueued.
    """
    c = _region_chunk(wx, wz)
    s = ry >> 4
    cell = _region_cell(wx, wz, ry)
    op = int(opacity[blocks[c, s, cell]])
    if is_sky and downward and level == MAX_LIGHT_LEVEL and op == 0:
        # §4.2: full sky light does not dim going down through open air, which
        # is what makes a shaft carry 15 all the way to the cave mouth.
        new = MAX_LIGHT_LEVEL
    else:
        damp = op
        if damp < 1:
            damp = 1
        new = int(level) - damp
    if new <= int(light[c, s, cell]):
        return tail
    light[c, s, cell] = new
    if new > PROPAGATE_ABOVE and tail < capacity:
        queue[tail] = _pack_position(wx, wz, ry)
        tail += 1
    return tail


@njit(cache=True, inline="always")
def _spread_six(light, blocks, opacity, queue, tail, capacity,
                wx, wz, ry, level, is_sky, height):
    """The six neighbours of one cell, in the increase BFS."""
    if wx > 0:
        tail = _spread_into(light, blocks, opacity, queue, tail, capacity,
                            wx - 1, wz, ry, level, False, is_sky)
    if wx < REGION_COLUMNS - 1:
        tail = _spread_into(light, blocks, opacity, queue, tail, capacity,
                            wx + 1, wz, ry, level, False, is_sky)
    if wz > 0:
        tail = _spread_into(light, blocks, opacity, queue, tail, capacity,
                            wx, wz - 1, ry, level, False, is_sky)
    if wz < REGION_COLUMNS - 1:
        tail = _spread_into(light, blocks, opacity, queue, tail, capacity,
                            wx, wz + 1, ry, level, False, is_sky)
    if ry > 0:
        tail = _spread_into(light, blocks, opacity, queue, tail, capacity,
                            wx, wz, ry - 1, level, True, is_sky)
    if ry < height - 1:
        tail = _spread_into(light, blocks, opacity, queue, tail, capacity,
                            wx, wz, ry + 1, level, False, is_sky)
    return tail


@njit(cache=True)
def _bfs_increase(light, blocks, opacity, queue, head, tail, capacity,
                  is_sky, height, budget):
    """Drain the increase queue. Returns the number of cells expanded.

    A cell is enqueued only when its stored value strictly increased, so this
    terminates and every value it writes is a legal propagation. `budget > 0`
    stops early (the caller may re-issue the same update; the increase side is
    monotone, so re-running it continues rather than restarts).
    """
    pops = 0
    while head < tail:
        if budget > 0 and pops >= budget:
            break
        key = queue[head]
        head += 1
        pops += 1
        wx = key & _QUEUE_COLUMN_MASK
        wz = (key >> _QUEUE_COLUMN_BITS) & _QUEUE_COLUMN_MASK
        ry = key >> _QUEUE_Y_SHIFT
        level = light[_region_chunk(wx, wz), ry >> 4, _region_cell(wx, wz, ry)]
        tail = _spread_six(light, blocks, opacity, queue, tail, capacity,
                           wx, wz, ry, level, is_sky, height)
    return pops


@njit(cache=True, inline="always")
def _decrease_into(light, blocks, opacity, dq, dq_levels, dq_tail, capacity,
                   iq, iq_tail, iq_capacity, wx, wz, ry, old, downward, is_sky):
    """One direction of the decrease BFS. Returns `(dq_tail, iq_tail)`.

    Vanilla's rule: a neighbour that could have been lit through the cell being
    cleared is cleared too and the walk continues; a neighbour that is *brighter
    than the cell being cleared* cannot have come from it, so it is an
    independent source and goes to the increase queue to re-fill what was just
    removed.

    The cell is written to 0 **before** it is enqueued (§9.1.2). Enqueue first
    and the next cell reads the old value, decides "that is not what I would
    have received" and stops the chain one cell in -- which is how the plan's
    first draft lost the 15 -> 14 case.
    """
    c = _region_chunk(wx, wz)
    s = ry >> 4
    cell = _region_cell(wx, wz, ry)
    level = int(light[c, s, cell])
    if level == 0:
        return dq_tail, iq_tail
    op = int(opacity[blocks[c, s, cell]])
    damp = op
    if damp < 1:
        damp = 1
    expected = int(old) - damp
    if is_sky and downward and old == MAX_LIGHT_LEVEL and op == 0:
        # §9.1.3: the value this direction *did* write was 15, not old - 1. Get
        # this wrong and removing a block over water clears the water column
        # from the wrong side of the rule.
        expected = MAX_LIGHT_LEVEL
    if level == expected or level < old:
        light[c, s, cell] = 0
        if dq_tail < capacity:
            dq[dq_tail] = _pack_position(wx, wz, ry)
            dq_levels[dq_tail] = level
            dq_tail += 1
    elif level > PROPAGATE_ABOVE and iq_tail < iq_capacity:
        iq[iq_tail] = _pack_position(wx, wz, ry)
        iq_tail += 1
    return dq_tail, iq_tail


@njit(cache=True)
def _bfs_decrease(light, blocks, opacity, dq, dq_levels, head, tail, capacity,
                  iq, iq_tail, iq_capacity, is_sky, height):
    """Drain the decrease queue, seeding the increase queue as it goes.

    Returns `(pops, iq_tail)`. Runs to completion for whatever the caller
    enqueued; `DimensionLight` never leaves this half-converged, because a
    too-bright light field is the one state a reader must not see.
    """
    pops = 0
    while head < tail:
        key = dq[head]
        old = dq_levels[head]
        head += 1
        pops += 1
        wx = key & _QUEUE_COLUMN_MASK
        wz = (key >> _QUEUE_COLUMN_BITS) & _QUEUE_COLUMN_MASK
        ry = key >> _QUEUE_Y_SHIFT
        if wx > 0:
            tail, iq_tail = _decrease_into(
                light, blocks, opacity, dq, dq_levels, tail, capacity,
                iq, iq_tail, iq_capacity, wx - 1, wz, ry, old, False, is_sky)
        if wx < REGION_COLUMNS - 1:
            tail, iq_tail = _decrease_into(
                light, blocks, opacity, dq, dq_levels, tail, capacity,
                iq, iq_tail, iq_capacity, wx + 1, wz, ry, old, False, is_sky)
        if wz > 0:
            tail, iq_tail = _decrease_into(
                light, blocks, opacity, dq, dq_levels, tail, capacity,
                iq, iq_tail, iq_capacity, wx, wz - 1, ry, old, False, is_sky)
        if wz < REGION_COLUMNS - 1:
            tail, iq_tail = _decrease_into(
                light, blocks, opacity, dq, dq_levels, tail, capacity,
                iq, iq_tail, iq_capacity, wx, wz + 1, ry, old, False, is_sky)
        if ry > 0:
            tail, iq_tail = _decrease_into(
                light, blocks, opacity, dq, dq_levels, tail, capacity,
                iq, iq_tail, iq_capacity, wx, wz, ry - 1, old, True, is_sky)
        if ry < height - 1:
            tail, iq_tail = _decrease_into(
                light, blocks, opacity, dq, dq_levels, tail, capacity,
                iq, iq_tail, iq_capacity, wx, wz, ry + 1, old, False, is_sky)
    return pops, iq_tail


# ---------------------------------------------------------------------------
# The kernels
# ---------------------------------------------------------------------------


@njit(cache=True)
def relight_region(blocks, opacity, emitter_ids, emitter_levels,
                   sky, block_light, do_sky, air_sections,
                   queue, dq_levels, iq):
    """Light a whole 3x3 patch from scratch. Returns `(sky, block)` expansions.

    * `blocks` / `sky` / `block_light` -- `(9, sections, 4096)` uint16 / uint8,
      region layout (`world/light_region.py`);
    * `opacity` -- `uint8[65536]` state id -> light opacity;
    * `emitter_ids` / `emitter_levels` -- `(E,)` int32 / uint8, the emitting
      states (`registry.blocks.light_emitters`);
    * `do_sky` -- False for a dimension without sky light (the nether and the
      end): the sky layer is zeroed and stages 1-3 are skipped entirely;
    * `air_sections` -- `(9, sections)` uint8 region layout of `SECTION_AIR` /
      `SECTION_MIXED` (`world/light_region.py`), or **`NO_SECTION_TABLE`** for
      "no table, scan everything". A section flagged `SECTION_AIR` holds nothing
      but air, so the column-top scan, the emitter scan and the sky fill have
      nothing to find in it; see below;
    * `queue` / `dq_levels` / `iq` -- the BFS buffers, see `LightQueues`. Only
      `queue` is used here; the other two are taken so that all three kernels
      take the same triple and a caller can hand them all the same buffers.

    Both light layers are **outputs**: nothing that was in them is read. That is
    what makes the patch independent of whether the neighbours have been lit yet
    (§4.1) and byte-for-byte repeatable.

    Order: one contiguous pass collects each column's opaque top and the
    emitter seeds, then the sky column fill, the lateral seeds and the BFS, then
    the block light BFS. The column tops and the emitter scan share a pass
    because both are a table lookup per cell and doing them twice was most of
    the patch's cost.

    Uniform sections, and why the seed is decided from the tops
    ----------------------------------------------------------

    The two linear passes are 81% of a real patch (pass 1 0.93 ms, the sky fill
    1.00 ms, of 2.37 ms measured on `tools/bench_light.py`'s nine generated
    chunks), and on that terrain **138 of the 216 chunk-sections hold nothing
    but air** -- 64% of the 884,736 cells, scanned twice, in which neither pass
    can find anything. `air_sections` is what lets them be skipped, and only
    `DimensionLight._light_chunk` passes a table: every other caller in the tree
    (the tools, `warm_up`) passes `NO_SECTION_TABLE` and gets exactly the old
    kernel, which is what keeps the oracle comparison in
    `tools/check_light_engine.py` a check of the *general* path.

    The scan half is unconditional: air has no opacity and emits nothing, so a
    flagged section cannot contribute a column top or an emitter seed.

    The fill half is not, and this is the part worth reading twice. Skipping the
    fill's above-the-top stores in a flagged section leaves the value the
    initialisation put there, and the fill would have written 15 -- but only if
    every column of that section really is *under* its own top. `is_empty` says
    nothing about tops, and a top moves without any block being added to the
    section: put one opaque block high in an air section and the all-air section
    below it is suddenly below a top, where the answer is per-cell (14, 0, ...)
    rather than 15. A seed that trusted `is_empty` left 16 cells at 15 there, and
    `tools/check_light_engine_incremental.py` caught it as "incremental != full".

    So the predicate is pass 1's own `tops`: a section is seeded exactly when
    every one of the 256 columns in its footprint has `tops < base_y`, which *is*
    the fill's rule (`ry > top` -> 15) applied to a whole section. It costs 2 us
    for all 216 sections and turns "the initialisation and the fill agree" from a
    hope into an identity, because it is the same array both of them read. The
    result is checked byte for byte against the unmasked kernel, over both layers
    and all nine chunks, in `tools/check_light_batch.py` -- which is the only
    reason to believe any of the above.
    """
    sections = blocks.shape[1]
    height = sections * SECTION_SIZE
    capacity = REGION_CHUNKS * sections * CELL_COUNT

    # State id -> luminance, so the emitter scan is an index instead of a
    # searchsorted per cell. `light_emitters` is 1,626 of the 29,671 states, and
    # a section usually holds none of them.
    luminance = np.zeros(STATE_ID_SPACE, dtype=np.uint8)
    for e in range(emitter_ids.shape[0]):
        state = emitter_ids[e]
        if 0 <= state < STATE_ID_SPACE:
            luminance[state] = emitter_levels[e]

    # Both layers are output-only, and both start as zeroes.
    use_air_sections = air_sections.shape[0] == REGION_CHUNKS
    sky[:, :, :] = 0
    block_light[:, :, :] = 0
    #: Whether the sky layer's 15s have been written into a section wholesale.
    #: Filled in below, once pass 1 has produced the tops it depends on.
    sky_seeded = np.zeros((REGION_CHUNKS, sections), dtype=np.uint8)

    tops = np.full((REGION_COLUMNS, REGION_COLUMNS), -1, dtype=np.int32)

    # One pass over the region: column tops + block-light seeds.
    tail = 0
    for c in range(REGION_CHUNKS):
        base_x = (c % REGION_EDGE) * SECTION_SIZE
        base_z = (c // REGION_EDGE) * SECTION_SIZE
        for s in range(sections):
            if use_air_sections and air_sections[c, s] == SECTION_AIR:
                # Air has no opacity and emits nothing: no top to record, no
                # seed to place. This is where the 884,736-cell scan loses the
                # cells of every all-air section -- 64% of them on real terrain.
                continue
            for i in range(CELL_COUNT):
                state = blocks[c, s, i]
                y_local = i >> 8
                if opacity[state] > 0:
                    wx = base_x + (i & 15)
                    wz = base_z + ((i >> 4) & 15)
                    top = (s << 4) + y_local
                    if top > tops[wz, wx]:
                        tops[wz, wx] = top
                level = luminance[state]
                if level > 0:
                    block_light[c, s, i] = level
                    if tail < capacity:
                        queue[tail] = _pack_position(
                            base_x + (i & 15),
                            base_z + ((i >> 4) & 15),
                            (s << 4) + y_local,
                        )
                        tail += 1
    block_expansions = _bfs_increase(block_light, blocks, opacity, queue,
                                     0, tail, capacity, False, height, 0)

    if not do_sky:
        return 0, block_expansions

    # The table's third value, and the one that makes the seeding safe: a
    # `SECTION_AIR` section holds nothing, so the fill's above-the-top branch
    # writes 15 into *every* cell of it -- but only when it really is above every
    # column top. A block placed high in a cave can make that false between two
    # patches (measured: a stone block at y=82 put section 9's top over all-air
    # section 8, and a blind seed left 16 cells at 15 instead of 14, which
    # `tools/check_light_engine_incremental.py` caught as "incremental != full").
    #
    # The test is the fill's own rule applied to a whole section: every cell of
    # it is above its column's top exactly when `base_y > tops[wz, wx]` for every
    # column of the section's 16x16 footprint in this chunk. `tops` is pass 1's
    # output, so this costs 256 comparisons per section and **2 us for all 216**
    # on the fixture -- against the 143 us that writing 15 into all 138 candidate
    # sections costs.
    if use_air_sections:
        for c in range(REGION_CHUNKS):
            base_x = (c % REGION_EDGE) * SECTION_SIZE
            base_z = (c // REGION_EDGE) * SECTION_SIZE
            for s in range(sections):
                if air_sections[c, s] != SECTION_AIR:
                    continue
                base_y = s << 4
                seeded = True
                for z in range(base_z, base_z + SECTION_SIZE):
                    for x in range(base_x, base_x + SECTION_SIZE):
                        if tops[z, x] >= base_y:
                            # A top reaches into this section, so part of it is
                            # under it and its light is per-cell, not uniform.
                            seeded = False
                            break
                    if not seeded:
                        break
                if seeded:
                    sky_seeded[c, s] = 1
                    sky[c, s, :] = MAX_LIGHT_LEVEL

    # Stage 1: per-column fill. Everything from `top` up is fully lit -- there is
    # no opaque block above it by definition -- and from `top` down the column
    # loses `opacity` per block and stops for good at 0. The downward
    # no-dampening rule of §4.2 is deliberately *not* here: this is the column's
    # own light, and vanilla's SkyLightSources subtracts opacity (so water dims
    # by one per block); the rule belongs to the BFS below.
    for wz in range(REGION_COLUMNS):
        for wx in range(REGION_COLUMNS):
            top = tops[wz, wx]
            c = _region_chunk(wx, wz)
            base = (wx & 15) + ((wz & 15) << 4)
            start = top + 1 if top >= 0 else 0
            for ry in range(start, height):
                s = ry >> 4
                if use_air_sections and sky_seeded[c, s]:
                    # Above every top in this section, so the seed's 15 is
                    # already what this store would have written.
                    continue
                sky[c, s, base + ((ry & 15) << 8)] = MAX_LIGHT_LEVEL
            if top >= 0:
                level = MAX_LIGHT_LEVEL
                for ry in range(top, -1, -1):
                    cell = base + ((ry & 15) << 8)
                    level = level - int(opacity[blocks[c, ry >> 4, cell]])
                    if level <= 0:
                        break
                    sky[c, ry >> 4, cell] = level

    # Stage 2: lateral seeds, from the same `tops` the fill used -- one top per
    # column, never two (§4.3).
    sky_tail = 0
    for wz in range(REGION_COLUMNS):
        for wx in range(REGION_COLUMNS):
            top = tops[wz, wx]
            north_top = tops[wz - 1, wx] if wz > 0 else -1
            south_top = tops[wz + 1, wx] if wz < REGION_COLUMNS - 1 else -1
            west_top = tops[wz, wx - 1] if wx > 0 else -1
            east_top = tops[wz, wx + 1] if wx < REGION_COLUMNS - 1 else -1
            sky_tail = _seed_sky_column(sky, queue, sky_tail, capacity,
                                        wx, wz, top,
                                        north_top, south_top, west_top, east_top)

    # Stage 3: BFS the seeds out into everything they can reach.
    sky_expansions = _bfs_increase(sky, blocks, opacity, queue,
                                   0, sky_tail, capacity, True, height, 0)
    return sky_expansions, block_expansions


@njit(cache=True)
def update_block_light(block_light, blocks, opacity, emitter_ids, emitter_levels,
                       x, y, z, old_level, new_level, dq, dq_levels, iq, budget=0):
    """Re-light after one block changed. Returns the number of expansions.

    `x`/`z` are chunk-local (0..15) and `y` is region-relative
    (`world_y - chunk.min_y`) -- the same frame `world/light_region.py` uses, and
    the block is in the **centre** chunk of the region. `old_level`/`new_level`
    are the old and new state's *luminance*; the opacity is re-read from
    `blocks`, so a change that only moves opacity (placing stone in a lit
    corridor) is handled by the same path. `dq` / `dq_levels` / `iq` are the BFS
    buffers, see `LightQueues`.

    Structure, in vanilla's order:

    1. if the cell is too bright for where it is now, clear it **first** and
       walk the decrease chain, seeding the increase queue with the cells that
       are brighter than the one being cleared;
    2. write the new emission and BFS it out;
    3. enqueue the six neighbours, which is what lets light *into* a cell that
       just became transparent (breaking a wall next to a torch has no decrease
       to carry it).

    `budget > 0` caps the increase side only; the decrease side always finishes,
    because stopping it halfway would leave a light value that cannot be
    explained by any source. Re-issuing the same call after a capped run is safe
    and continues the increase (it only ever raises values).
    """
    sections = blocks.shape[1]
    height = sections * SECTION_SIZE
    capacity = REGION_CHUNKS * sections * CELL_COUNT

    c = CENTER_CHUNK_INDEX
    s = y >> 4
    cell = x + (z << 4) + ((y & 15) << 8)
    wx0 = SECTION_SIZE + x
    wz0 = SECTION_SIZE + z
    current = int(block_light[c, s, cell])

    dq_tail = 0
    iq_tail = 0
    expansions = 0

    if current > 0 and (current > new_level or new_level < old_level):
        # Clear self before enqueueing (§9.1.2). `current > new_level` is the
        # general case (the cell holds more light than its own emission
        # explains, including the opacity-only change); `new_level < old_level`
        # is the emitter that just got dimmer.
        block_light[c, s, cell] = 0
        dq[0] = _pack_position(wx0, wz0, y)
        dq_levels[0] = current
        dq_tail = 1
        pops, iq_tail = _bfs_decrease(
            block_light, blocks, opacity, dq, dq_levels, 0, dq_tail, capacity,
            iq, iq_tail, capacity, False, height)
        expansions += pops

    if new_level > int(block_light[c, s, cell]):
        block_light[c, s, cell] = new_level
    level = int(block_light[c, s, cell])
    if level > PROPAGATE_ABOVE and iq_tail < capacity:
        iq[iq_tail] = _pack_position(wx0, wz0, y)
        iq_tail += 1

    # Step 3: neighbours that can now push light into this cell.
    if x > 0:
        iq_tail = _enqueue_lit(block_light, iq, iq_tail, capacity, wx0 - 1, wz0, y)
    if x < SECTION_SIZE - 1:
        iq_tail = _enqueue_lit(block_light, iq, iq_tail, capacity, wx0 + 1, wz0, y)
    if z > 0:
        iq_tail = _enqueue_lit(block_light, iq, iq_tail, capacity, wx0, wz0 - 1, y)
    if z < SECTION_SIZE - 1:
        iq_tail = _enqueue_lit(block_light, iq, iq_tail, capacity, wx0, wz0 + 1, y)
    if y > 0:
        iq_tail = _enqueue_lit(block_light, iq, iq_tail, capacity, wx0, wz0, y - 1)
    if y < height - 1:
        iq_tail = _enqueue_lit(block_light, iq, iq_tail, capacity, wx0, wz0, y + 1)

    expansions += _bfs_increase(block_light, blocks, opacity, iq, 0, iq_tail,
                                capacity, False, height, budget)
    return expansions


@njit(cache=True, inline="always")
def _enqueue_lit(light, queue, tail, capacity, wx, wz, ry):
    """Push a cell onto the increase queue if it holds propagating light."""
    if tail >= capacity:
        return tail
    if int(light[_region_chunk(wx, wz), ry >> 4, _region_cell(wx, wz, ry)]) > PROPAGATE_ABOVE:
        queue[tail] = _pack_position(wx, wz, ry)
        tail += 1
    return tail


@njit(cache=True)
def update_sky_column(sky, blocks, opacity, x, z, dq, dq_levels, iq, col_top=-1):
    """Recompute one sky column of the centre chunk. Returns expansions.

    `x`/`z` are chunk-local, as in `update_block_light`. The whole column is
    rescanned from the world top -- a change of more than 16 blocks moves the
    light of the entire column, so there is nothing cheaper that is also correct
    (`LIGHT_PLAN.md` §4.6).

    `col_top` is an optional hint: the region-relative Y of the column's
    highest non-transparent block, or -1 to compute it. The hint is *verified*
    before use (anything above it must be transparent), so a stale or
    wrong-frame value costs a scan rather than a wrong light field -- pass -1
    unless you just computed the top with the same `opacity` criterion.

    Every cell whose ideal column value differs is written and queued: brighter
    goes to the increase queue, dimmer is **cleared and then** enqueued with its
    old value (§9.1.2 again), and the lateral seeds of the 3x3 columns around
    this one are re-derived, because a block that changes a column's top changes
    which of its neighbours' cells the sky can reach sideways.
    """
    sections = blocks.shape[1]
    height = sections * SECTION_SIZE
    capacity = REGION_CHUNKS * sections * CELL_COUNT

    c = CENTER_CHUNK_INDEX
    wx0 = SECTION_SIZE + x
    wz0 = SECTION_SIZE + z

    top = col_top
    if top < 0 or top >= height:
        top = _column_top(blocks, opacity, wx0, wz0, height)
    else:
        # A hint is only usable if the cell it names is actually opaque and
        # nothing above it is; otherwise it is stale (or in the wrong
        # coordinate frame, e.g. a world Y where a region-relative one was
        # asked for) and the scan decides. Both checks together are O(height)
        # at worst, against the column scan they replace -- the hint is not
        # worth a wrong light field.
        probe = _region_chunk(wx0, wz0)
        base = (wx0 & 15) + ((wz0 & 15) << 4)
        if opacity[blocks[probe, top >> 4, base + ((top & 15) << 8)]] == 0:
            top = _column_top(blocks, opacity, wx0, wz0, height)
        else:
            for ry in range(height - 1, top, -1):
                if opacity[blocks[probe, ry >> 4, base + ((ry & 15) << 8)]] > 0:
                    top = _column_top(blocks, opacity, wx0, wz0, height)
                    break

    dq_tail = 0
    iq_tail = 0

    base = (wx0 & 15) + ((wz0 & 15) << 4)
    # Above the top the column's own light is full.
    for ry in range(height - 1, top, -1):
        cell = base + ((ry & 15) << 8)
        level = int(sky[c, ry >> 4, cell])
        if level < MAX_LIGHT_LEVEL:
            sky[c, ry >> 4, cell] = MAX_LIGHT_LEVEL
            iq_tail = _enqueue_position(iq, iq_tail, capacity, wx0, wz0, ry)
    # From the top down: the ideal value, compared against what is stored.
    if top >= 0:
        level = MAX_LIGHT_LEVEL
        for ry in range(top, -1, -1):
            cell = base + ((ry & 15) << 8)
            level = level - int(opacity[blocks[c, ry >> 4, cell]])
            if level < 0:
                level = 0
            current = int(sky[c, ry >> 4, cell])
            if level > current:
                sky[c, ry >> 4, cell] = level
                iq_tail = _enqueue_position(iq, iq_tail, capacity, wx0, wz0, ry)
            elif level < current:
                sky[c, ry >> 4, cell] = 0
                if dq_tail < capacity:
                    dq[dq_tail] = _pack_position(wx0, wz0, ry)
                    dq_levels[dq_tail] = current
                    dq_tail += 1
            if level == 0:
                # Everything below is in full shadow as far as this column is
                # concerned; a lateral source can still light it, and the
                # increase queue above is what carries that.
                for below in range(ry - 1, -1, -1):
                    cell_below = base + ((below & 15) << 8)
                    current = int(sky[c, below >> 4, cell_below])
                    if current == 0:
                        continue
                    sky[c, below >> 4, cell_below] = 0
                    if dq_tail < capacity:
                        dq[dq_tail] = _pack_position(wx0, wz0, below)
                        dq_levels[dq_tail] = current
                        dq_tail += 1
                break

    # Lateral seeds for this column and its eight neighbours: the changed
    # column's top moves which side cells the sky reaches (stage 2 of §4.3).
    for dz in (-1, 0, 1):
        for dx in (-1, 0, 1):
            nwx = wx0 + dx
            nwz = wz0 + dz
            if 0 <= nwx < REGION_COLUMNS and 0 <= nwz < REGION_COLUMNS:
                ntop = _column_top(blocks, opacity, nwx, nwz, height)
                north_top = (_column_top(blocks, opacity, nwx, nwz - 1, height)
                             if nwz > 0 else -1)
                south_top = (_column_top(blocks, opacity, nwx, nwz + 1, height)
                             if nwz < REGION_COLUMNS - 1 else -1)
                west_top = (_column_top(blocks, opacity, nwx - 1, nwz, height)
                            if nwx > 0 else -1)
                east_top = (_column_top(blocks, opacity, nwx + 1, nwz, height)
                            if nwx < REGION_COLUMNS - 1 else -1)
                iq_tail = _seed_sky_column(sky, iq, iq_tail, capacity,
                                           nwx, nwz, ntop,
                                           north_top, south_top, west_top, east_top)

    expansions = 0
    if dq_tail > 0:
        pops, iq_tail = _bfs_decrease(sky, blocks, opacity, dq, dq_levels,
                                      0, dq_tail, capacity,
                                      iq, iq_tail, capacity, True, height)
        expansions += pops
    expansions += _bfs_increase(sky, blocks, opacity, iq, 0, iq_tail,
                                capacity, True, height, 0)
    return expansions


@njit(cache=True, inline="always")
def _enqueue_position(queue, tail, capacity, wx, wz, ry):
    if tail < capacity:
        queue[tail] = _pack_position(wx, wz, ry)
        tail += 1
    return tail


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

#: `[world]` budget keys (`config.toml`). `core.config` is imported lazily: a
#: module-level import would drag `core/__init__.py` (and therefore the server)
#: into `import world` -- `world/generator.py:477` explains the import cycle.
DEFAULT_BUDGET_MS = 4.0
DEFAULT_CHUNKS_PER_TICK = 4
DEFAULT_UPDATES_PER_TICK = 4096


def _configured(name: str, default, cast):
    """One `[world]` key, clamped to something usable. Config or the default."""
    try:
        from core.config import config
    except ImportError:  # pragma: no cover - tools without the config module
        return default
    value = config.get(f"world.{name}", default)
    try:
        return cast(value)
    except (TypeError, ValueError):
        return default


class LightQueues:
    """The three BFS buffers the kernels work on, allocated once and reused.

    Every kernel used to `np.empty` its own queues on entry. At the overworld's
    `9 sections x 24 x 4096` region that is **3.5 MB per array, three arrays
    per call** -- 10.6 MB malloc'd and freed *per block edit* and per sky column
    rescan, on the actor thread, inside the light budget. numba's allocator
    cannot cache an array that size, so it was mmap/munmap traffic on the tick
    path.

    The bound itself did not change: `capacity` is still
    `REGION_CHUNKS * sections * CELL_COUNT`, exactly what the kernels derived
    internally, so no update that used to fit now does not. What changed is that
    the memory is allocated **once per `DimensionLight`** (about 10.6 MB, on
    first use) instead of once per call.

    Shrinking `capacity` is deliberately *not* done here. The obvious argument
    -- a cell is enqueued only when its stored value strictly increases, and
    values live in 0..15, so a cell is enqueued at most 15 times -- gives
    `15 x region cells`, which is *larger* than today's bound. Today's bound is
    a heuristic that holds because a BFS off one source writes each cell once
    (measured peak: 1,412 entries for a whole 3x3 patch). Making that rigorous,
    or sizing from a measured high-water mark with an overflow counter, is its
    own piece of work; until then the honest saving is "stop paying for it on
    every edit", not "allocate less".

    One instance serves all three kernels: `relight_region` uses `dq` alone,
    and the two incremental kernels use the triple. Numba takes the arrays by
    reference, so the caller must not resize them while a kernel is running --
    which, being actor-thread only, never happens.
    """

    __slots__ = ("capacity", "dq", "dq_levels", "iq", "sections")

    def __init__(self, sections: int) -> None:
        self.sections = int(sections)
        self.capacity = REGION_CHUNKS * self.sections * CELL_COUNT
        #: Position queue, and the level each cleared cell used to hold (the
        #: decrease BFS needs both, the increase BFS only the former).
        self.dq = np.empty(self.capacity, dtype=np.int32)
        self.dq_levels = np.empty(self.capacity, dtype=np.int32)
        self.iq = np.empty(self.capacity, dtype=np.int32)

    def args(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The triple, in the order every kernel takes it."""
        return self.dq, self.dq_levels, self.iq


class DimensionLight:
    """One dimension's light queues and budgets. Actor-thread only, zero locks.

    The same thread that owns the `Chunk` objects calls `tick()`; the light
    arrays it writes are the ones `world/chunk.py:412` hands out "on the owning
    thread". Nothing here is re-entrant or thread-safe, and nothing needs to be:
    `world/dimension_actor.py` runs `tick()` on its own thread, after the
    spawner and before the entity tracker, so a torch placed this tick is
    already visible to this tick's spawn check.

    Ordering inside one tick
    ------------------------

    Block edits are **merged per cell** and drained **oldest first**. Both
    halves of that matter, and both were measured on a real chunk:

    * Vanilla's "all decreases, then all increases" (`LIGHT_PLAN.md` §4.6) is
      the order *inside one block change* -- it assumes a change's old level is
      still the level the light was built from, because vanilla reads it out of
      the world when the change is queued. Two edits to the same cell queued in
      one tick break that assumption: place a glowstone and break it again and
      the break (a decrease) would run first, find a dark cell, do nothing, and
      then the placement's increase would flood 2,089 cells with the light of a
      block that no longer exists. Nothing repairs light that is too bright, so
      it is a wrong answer rather than a slow one.
    * Merging fixes it at the source: the cell's *first* state is the baseline
      and its *last* state the target, and a net no-op (place + break, same
      opacity and luminance) never reaches a kernel at all. What is left is the
      order, and the order has to be the order the edits happened in, so the two
      deques are walked by a sequence number rather than one after the other.

    A single edit, and the steady state of one edit per tick, behave exactly as
    they did before the merge existed: one item, no merge, its own
    decrease-then-increase inside the kernel. `tools/check_light_batch.py` pins
    all three cases (single, same-cell burst, multi-cell burst) against a cold
    patch.
    """

    def __init__(
        self,
        get_chunk,
        *,
        has_skylight: Optional[bool] = None,
        budget_ms: Optional[float] = None,
        chunks_per_tick: Optional[int] = None,
        updates_per_tick: Optional[int] = None,
        opacity: Optional[np.ndarray] = None,
        emitters: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        """`get_chunk` is `Dimension.get_chunk` or the `Dimension` itself.

        Passing the dimension is what the actor does, and then
        `has_skylight` defaults to `dimension.settings.has_skylight` -- the
        nether and the end must not get a sky layer (their `sky_light` is
        written as 0). The three budgets default to the `[world]` keys in
        `config.toml`; pass them explicitly to override (the tools do).

        `opacity`/`emitters` are the registry tables (`registry.blocks`). They
        are read on the first call after `GameRegistry.freeze()`; a caller that
        holds its own registry passes its tables here instead.
        """
        self._dimension = None
        if callable(get_chunk):
            self._get_chunk = get_chunk
        else:
            self._dimension = get_chunk
            self._get_chunk = get_chunk.get_chunk
        if has_skylight is None:
            settings = getattr(self._dimension, "settings", None)
            has_skylight = bool(getattr(settings, "has_skylight", True))
        #: False for the nether and the end (`DimensionSettings.has_skylight`):
        #: their sky layer is written as 0 and never touched again.
        self.has_skylight = bool(has_skylight)

        if budget_ms is None:
            budget_ms = _configured("light-budget-ms", DEFAULT_BUDGET_MS, float)
        if chunks_per_tick is None:
            chunks_per_tick = _configured("light-chunks-per-tick",
                                          DEFAULT_CHUNKS_PER_TICK, int)
        if updates_per_tick is None:
            updates_per_tick = _configured("light-updates-per-tick",
                                           DEFAULT_UPDATES_PER_TICK, int)
        self.budget_ms = max(0.0, float(budget_ms))
        self.chunks_per_tick = max(0, int(chunks_per_tick))
        self.updates_per_tick = max(0, int(updates_per_tick))

        if opacity is None or emitters is None:
            from registry.blocks import light_emitters, light_opacity_table

            if opacity is None:
                opacity = light_opacity_table()
            if emitters is None:
                emitters = light_emitters()
        self._opacity = np.ascontiguousarray(opacity, dtype=np.uint8)
        emitter_ids, emitter_levels = emitters
        self._emitter_ids = np.ascontiguousarray(emitter_ids, dtype=np.int32)
        self._emitter_levels = np.ascontiguousarray(emitter_levels, dtype=np.uint8)

        #: State id -> luminance, for the Python-side decision in
        #: `block_changed` (the kernels build their own from the two arrays).
        self._luminance = np.zeros(STATE_ID_SPACE, dtype=np.uint8)
        self._luminance[self._emitter_ids] = self._emitter_levels

        #: Chunks waiting for a full 3x3 patch, in request order. `_pending_set`
        #: is the dedup: a chunk asked for twice is lit once.
        self._pending_chunks: "deque[Tuple[int, int]]" = deque()
        self._pending_set: Set[Tuple[int, int]] = set()

        #: The four incremental queues of §4.6. Block edits carry the old and
        #: new state; sky items are columns (the sky rule does not depend on
        #: luminance, only on opacity, so a column is re-derived wholesale).
        #:
        #: Block edits are also indexed by cell (`_edit_index`) so that a cell
        #: edited twice inside one tick is *merged* rather than queued twice, and
        #: `_drain_blocks` walks the two deques in **insertion order** (the
        #: sequence number in slot 0 of each item) instead of "all decreases,
        #: then all increases". Which deque an item sits in is its classification
        #: at first insertion; the drain order does not depend on it, so that
        #: choice only shows up in `stats()`.
        self._block_decrease: "deque[List]" = deque()
        self._block_increase: "deque[List]" = deque()
        self._sky_decrease: "deque[Tuple[int, int, int, int]]" = deque()
        self._sky_increase: "deque[Tuple[int, int, int, int]]" = deque()
        #: `(chunk_x, chunk_z, cell)` -> the queued edit item, for merging.
        self._edit_index: dict = {}
        self._edit_seq = 0
        #: Columns queued for a rescan, so a second edit to the same column in
        #: one tick does not queue it twice.
        self._dirty_columns: Set[Tuple[int, int, int, int]] = set()

        #: The 3x3 region currently open, and for which chunk. One at a time:
        #: patches and edits for the same chunk reuse it across ticks (opening
        #: one costs 9 ms), and it is dropped whenever the work moves elsewhere.
        self._region: Optional[LightRegion] = None
        self._region_key: Optional[Tuple[int, int]] = None
        #: Whether the open region holds light that has not been written back.
        self._region_dirty: bool = False
        #: The BFS buffers, built on the first kernel call and reused for the
        #: life of this dimension (~10.6 MB, once). Keyed on the section count
        #: so a dimension whose window changed cannot hand a kernel a short
        #: queue -- the kernels derive `capacity` from the region's shape and
        #: would then write past the end of these.
        self._queues: Optional[LightQueues] = None

        #: Chunks whose light was written since the last `take_dirty()`. The
        #: streamer drains it and sends `CLightUpdate`; `is_light_dirty` on the
        #: chunk is the same fact for the saver.
        self._dirty_chunks: Set[Tuple[int, int]] = set()

        #: Counters, for `stats()` and for the tools that check them.
        self.dropped_unloaded = 0
        self.deferred_chunks = 0
        self.fallback_patches = 0
        self.chunks_lit = 0
        self.sections_written = 0
        self.expansions = 0
        self.ticks = 0
        self.errors = 0
        self.last_error: Optional[str] = None

    # ---------------- requests (called by the lifecycle / streamer) --------

    def request_chunk(self, pos) -> None:
        """Ask for a full patch of this chunk. Idempotent while it is queued.

        `pos` is a `ChunkPos` or an `(x, z)` pair. `ChunkLifecycle` calls this
        when a chunk enters memory and the streamer calls it when a chunk is
        about to be sent: light has to exist before either reader looks at it.

        The chunk stays queued while any of its eight neighbours is missing
        (`LIGHT_PLAN.md` §4.5) -- a patch is never computed against air standing
        in for an unloaded chunk.
        """
        key = (pos.x, pos.z) if hasattr(pos, "x") else (pos[0], pos[1])
        self.queue_chunk(key[0], key[1])

    def queue_chunk(self, chunk_x: int, chunk_z: int) -> None:
        """`request_chunk` for a bare coordinate pair."""
        key = (chunk_x, chunk_z)
        if key in self._pending_set:
            return
        self._pending_set.add(key)
        self._pending_chunks.append(key)

    def block_changed(self, x: int, y: int, z: int,
                      old_state: int, new_state: int) -> bool:
        """Tell the engine that a block changed at world `(x, y, z)`.

        Called by `Dimension.set_block` **after** the block array and the
        heightmap are committed, on the actor thread. Returns False when nothing
        about light can have changed (same opacity *and* same luminance, §4.6
        step 2), or when the chunk is not loaded -- the latter counted in
        `dropped_unloaded`, the §4.7 answer for a queue item with no chunk.

        Only queues; the work happens in `tick()`, inside the budget.
        """
        old_op = int(self._opacity[old_state])
        new_op = int(self._opacity[new_state])
        old_lum = int(self._luminance[old_state])
        new_lum = int(self._luminance[new_state])
        if old_op == new_op and old_lum == new_lum:
            return False

        chunk_x = x >> 4
        chunk_z = z >> 4
        chunk = self._get_chunk(chunk_x, chunk_z)
        if chunk is None:
            self.dropped_unloaded += 1
            return False
        if not getattr(chunk, "light_populated", False):
            # Never lit: its light arrays are defaults, so there is no field to
            # update. The full patch reads `blocks` (which the caller has
            # already committed), so the change is not lost.
            self.fallback_patches += 1
            self.queue_chunk(chunk_x, chunk_z)
            return False

        local_y = y - chunk.min_y
        local_x = x & 15
        local_z = z & 15
        cell = local_x + (local_z << 4) + ((local_y & 15) << 8)
        key = (chunk_x, chunk_z, cell)
        item = self._edit_index.get(key)
        if item is None:
            # `[seq, chunk_x, chunk_z, local_x, local_y, local_z, cell,
            #   old_state, old_lum, new_state, new_lum]`, and mutable so a
            # second edit to the same cell can update it in place.
            item = [self._edit_seq, chunk_x, chunk_z, local_x, local_y, local_z, cell,
                    int(old_state), old_lum, int(new_state), new_lum]
            self._edit_seq += 1
            self._edit_index[key] = item
            if new_lum >= old_lum:
                self._block_increase.append(item)
            else:
                self._block_decrease.append(item)
        else:
            # The same cell, again, before any tick drained it. The *first*
            # state stays the baseline and the last one is the target: the
            # light field only ever needed to get from one to the other, and a
            # queue that kept both would apply them in an order that is not the
            # order they happened in (see the class docstring).
            item[9] = int(new_state)
            item[10] = new_lum

        # Mirror the newest state into the open region now, if it is this
        # chunk's. `_drain_sky` re-derives a column's top from the region's
        # blocks, so a cached region that still held the old block would make
        # the sky recompute see no change at all. A region built later reads the
        # chunk, which the caller has already committed.
        if self._region is not None and self._region_key == (chunk_x, chunk_z):
            self._region.blocks[CENTER_CHUNK_INDEX, local_y >> 4, cell] = new_state

        # Sky depends on opacity alone, so a luminance-only change cannot move
        # it (the column fill is a function of the opacity column and nothing
        # else) -- no rescan for a torch placed in the open.
        if new_op != old_op:
            column = (chunk_x, chunk_z, x & 15, z & 15)
            if column not in self._dirty_columns:
                self._dirty_columns.add(column)
                if new_op > old_op:
                    self._sky_decrease.append(column)
                else:
                    self._sky_increase.append(column)
        return True

    # ---------------- the tick ----------------

    def tick(self) -> None:
        """Drain what the budget allows. Never raises; never blocks.

        Order is vanilla's: every decrease before any increase. A budget of 0
        (or a deadline already past) does nothing at all, which is what
        `tools/check_light_engine.py` pins: the same world must reach the same
        light field whether it converged in one tick or in fifty.

        **This runs on the actor's tick path**, so an exception here would take
        the whole tick down with it. Anything a kernel or a half-built chunk
        throws is caught, counted in `errors` and logged; the light is left
        where it was, which is a dimmer chunk rather than a dead dimension.
        """
        self.ticks += 1
        if self.budget_ms <= 0.0:
            return
        try:
            self._run_tick()
        except Exception as error:  # noqa: BLE001 - see the docstring
            self.errors += 1
            self.last_error = f"{type(error).__name__}: {error}"
            logger.exception("light tick failed; leaving the light where it is")

    def _run_tick(self) -> None:
        deadline = time.perf_counter() + self.budget_ms / 1000.0
        processed = 0
        # Block edits run in the order they were queued (each one doing its own
        # decrease-then-increase inside the kernel); sky columns are independent
        # of that order because a rescan re-derives the whole column from the
        # blocks, so they stay split by direction.
        processed += self._drain_blocks(deadline, processed)
        processed += self._drain_sky(self._sky_decrease, deadline, processed)
        processed += self._drain_sky(self._sky_increase, deadline, processed)
        self._drain_chunks(deadline)
        self._flush()

    def _over_budget(self, deadline: float, processed: int) -> bool:
        return processed >= self.updates_per_tick or time.perf_counter() >= deadline

    def _pop_edit(self):
        """The oldest queued block edit, whichever of the two deques holds it."""
        decrease = self._block_decrease
        increase = self._block_increase
        if not decrease:
            return increase.popleft() if increase else None
        if not increase:
            return decrease.popleft()
        return (decrease if decrease[0][0] < increase[0][0] else increase).popleft()

    def _drain_blocks(self, deadline: float, processed: int) -> int:
        """Drain block edits oldest-first, merging nothing at this point.

        The items are already merged per cell (see `block_changed`); what is
        left here is the *order*. A cell edited and edited back inside one tick
        is a net no-op -- same opacity, same luminance -- and is dropped without
        a kernel call, which is what keeps "place a glowstone, break it again"
        from leaving the glowstone's light in the world.
        """
        done = 0
        while self._block_decrease or self._block_increase:
            if self._over_budget(deadline, processed + done):
                break
            item = self._pop_edit()
            if item is None:
                break
            del self._edit_index[(item[1], item[2], item[6])]
            done += 1
            if item[8] == item[10] and self._opacity[item[7]] == self._opacity[item[9]]:
                continue
            region = self._open_region(item[1], item[2])
            if region is None:
                continue
            region.blocks[CENTER_CHUNK_INDEX, item[4] >> 4, item[6]] = item[9]
            self._region_dirty = True
            self.expansions += update_block_light(
                region.block_light, region.blocks, self._opacity,
                self._emitter_ids, self._emitter_levels,
                item[3], item[4], item[5], item[8], item[10],
                *self._queues_for(region.sections).args(),
            )
        return done

    def _drain_sky(self, queue, deadline: float, processed: int) -> int:
        done = 0
        while queue:
            if self._over_budget(deadline, processed + done):
                break
            chunk_x, chunk_z, x, z = queue.popleft()
            self._dirty_columns.discard((chunk_x, chunk_z, x, z))
            if not self.has_skylight:
                done += 1
                continue
            region = self._open_region(chunk_x, chunk_z)
            if region is None:
                continue
            self._region_dirty = True
            self.expansions += update_sky_column(
                region.sky, region.blocks, self._opacity, x, z,
                *self._queues_for(region.sections).args(),
            )
            done += 1
        return done

    def _drain_chunks(self, deadline: float) -> int:
        attempts = 0
        lit = 0
        while self._pending_chunks and attempts < self.chunks_per_tick:
            if time.perf_counter() >= deadline:
                break
            chunk_x, chunk_z = self._pending_chunks.popleft()
            self._pending_set.discard((chunk_x, chunk_z))
            attempts += 1
            if self._light_chunk(chunk_x, chunk_z):
                lit += 1
            else:
                # A neighbour is missing: stay in the queue (§4.5), and do not
                # retry it this tick -- the other queued chunks get their turn.
                self._pending_set.add((chunk_x, chunk_z))
                self._pending_chunks.append((chunk_x, chunk_z))
        return lit

    # ---------------- region plumbing ----------------

    def _queues_for(self, sections: int) -> LightQueues:
        """The BFS buffers for a region of `sections` sections, allocated once.

        Actor-thread only, like everything else here, so a plain lazy
        initialisation is enough: the first kernel call in a dimension pays
        ~10.6 MB of allocation and every call after it pays nothing.
        """
        queues = self._queues
        if queues is None or queues.sections != sections:
            queues = LightQueues(sections)
            self._queues = queues
        return queues

    def _open_region(self, chunk_x: int, chunk_z: int) -> Optional[LightRegion]:
        """The 3x3 region for a chunk, building it if it is not the open one.

        Returns None when the incremental kernels must not run: no chunk, an
        incomplete 3x3, or a 3x3 in which some chunk has never been lit. The
        last case sends a **full patch** of this chunk instead -- the
        incremental kernels read their neighbours' light, and an unlit
        neighbour's arrays are defaults (`sky_light` is `np.full(2048, 15)`),
        so using them would write light into this chunk that no source explains
        and that nothing would ever take back out. The queue item is dropped
        because that patch recomputes this chunk from its blocks, which already
        contain the change.
        """
        if self._region is not None and self._region_key == (chunk_x, chunk_z):
            return self._region
        self._flush()
        self._region = None
        self._region_key = None
        chunk = self._get_chunk(chunk_x, chunk_z)
        if chunk is None:
            self.dropped_unloaded += 1
            return None
        region = LightRegion.build(chunk, self._get_chunk)
        if not region.ready:
            self.deferred_chunks += 1
            return None
        if not region.all_lit:
            self.fallback_patches += 1
            self.queue_chunk(chunk_x, chunk_z)
            return None
        self._region = region
        self._region_key = (chunk_x, chunk_z)
        self._region_dirty = False
        return region

    def _flush(self) -> int:
        """Commit the open region's centre chunk, if anything moved in it.

        Called at the end of every tick and whenever the open chunk changes, so
        what `ChunkSection.sky_light` holds is always a complete patch result --
        "light is committed as it is computed" (§4.7). The queues, not the light
        arrays, are what is allowed to be behind.

        The region itself stays open across ticks: reopening one costs a build
        (216 block copies plus, for the incremental path, 432 `unpack_light`
        calls -- 9 ms measured), and a player placing one block per tick would
        pay that every tick. It is dropped whenever the work moves to another
        chunk or a full patch runs, both of which go through `_flush` first.

        A tick that changed no light writes nothing back: packing 48 arrays to
        discover they are identical is ~0.5 ms of pure waste, and on a server
        with no players editing blocks this is the per-tick cost.
        """
        if self._region is None:
            return 0
        if not self._region_dirty:
            return 0
        region = self._region
        self._region_dirty = False
        return self._commit(region)

    def _commit(self, region: LightRegion) -> int:
        """Write a patch result back and book-keep what it means.

        Only `is_light_dirty`, never `is_dirty`: a freshly lit chunk in a
        player's view is not a modified chunk, and marking it modified would
        rewrite the whole view to disk (`world/README.md:139`). `light_populated`
        is set by the write-back itself, because from here on the codec reads
        these bytes instead of deriving light from the heightmap.
        """
        changed = region.write_back()
        self.sections_written += changed
        if changed:
            pos = region.center.position
            region.center.is_light_dirty = True
            self._dirty_chunks.add((pos.x, pos.z))
        return changed

    def take_dirty(self) -> Set[Tuple[int, int]]:
        """Return and clear the chunks whose light changed since the last call.

        The streamer drains this once per tick and sends `CLightUpdate` (0x2F)
        to whoever can see those chunks, then lets `mark_saved`/the streamer
        clear `is_light_dirty`. Convergence is not required: vanilla ships
        intermediate light and so does this (`LIGHT_PLAN.md` §4.7) -- the set
        means "the bytes differ from the last time anyone was told", no more.
        """
        dirty = self._dirty_chunks
        self._dirty_chunks = set()
        return dirty

    def _light_chunk(self, chunk_x: int, chunk_z: int) -> bool:
        """One full 3x3 patch. True when it ran and was written back."""
        chunk = self._get_chunk(chunk_x, chunk_z)
        if chunk is None:
            self.dropped_unloaded += 1
            return True  # nothing to keep waiting for
        # `read_light=False`: `relight_region` writes both layers from `blocks`,
        # so unpacking 18 light arrays here would be 8 ms spent to throw the
        # result away.
        region = LightRegion.build(chunk, self._get_chunk, read_light=False)
        if not region.ready:
            # A missing neighbour. Nothing has been touched: the region another
            # chunk's edits are using stays open, so a queue full of chunks
            # waiting for their neighbours does not cost an edit its cache.
            self.deferred_chunks += 1
            return False
        # Past this point the patch runs, and it rewrites the light of a chunk
        # that may be a neighbour of the open region -- so commit that region
        # and drop it (`LIGHT_PLAN.md` §4.7's stale-neighbour gap, avoided).
        self._flush()
        self._region = None
        self._region_key = None
        sky_expansions, block_expansions = relight_region(
            region.blocks, self._opacity, self._emitter_ids, self._emitter_levels,
            region.sky, region.block_light, self.has_skylight,
            region.air_sections,
            *self._queues_for(region.sections).args(),
        )
        self.expansions += sky_expansions + block_expansions
        self._commit(region)
        self.chunks_lit += 1
        return True

    # ---------------- inspection ----------------

    @property
    def pending(self) -> int:
        """Items waiting: chunks, block edits, sky columns."""
        return (len(self._pending_chunks) + len(self._block_decrease)
                + len(self._block_increase) + len(self._sky_decrease)
                + len(self._sky_increase))

    @property
    def pending_chunks(self) -> int:
        """Chunks waiting for a full patch (a missing neighbour keeps them here)."""
        return len(self._pending_chunks)

    def stats(self) -> dict:
        """The counters `dimension_actor`'s stats message and the tools read."""
        return {
            "dim_light_chunks_lit": self.chunks_lit,
            "dim_light_sections": self.sections_written,
            "dim_light_expansions": self.expansions,
            "dim_light_dropped": self.dropped_unloaded,
            "dim_light_deferred": self.deferred_chunks,
            "dim_light_fallbacks": self.fallback_patches,
            "dim_light_pending": self.pending,
            "dim_light_pending_chunks": self.pending_chunks,
            "dim_light_block_decrease": len(self._block_decrease),
            "dim_light_block_increase": len(self._block_increase),
            "dim_light_sky_decrease": len(self._sky_decrease),
            "dim_light_sky_increase": len(self._sky_increase),
            "dim_light_ticks": self.ticks,
            "dim_light_errors": self.errors,
        }


def warm_up() -> None:
    """Compile the three kernels before the first tick.

    `cache=True` makes the *second* process cheap, not the first: the first call
    after a source change pays LLVM, seconds for kernels this size
    (`LIGHT_PLAN.md` §5.1: ~4.8 s for the sky kernel, reused across processes
    afterwards, ~0.8 s total). Landing that inside a tick is a multi-second
    stall, so the server should call this during startup -- the same shape
    `physics.kernel.warm_up` has. The world is synthetic, and deliberately two
    sections tall: one with a floor, a wall to walk the decrease through and two
    emitters, one all air, so the table's skip branch (`air_sections`, the extra
    branch `relight_region` grew) is compiled here rather than inside the first
    tick that happens to meet an empty section.
    """
    if not _HAS_NUMBA:
        return
    logger.info("[LunarLight] Compiling light kernels, this will take a while on the first load...")
    sections = 2
    blocks = np.zeros((REGION_CHUNKS, sections, CELL_COUNT), dtype=np.uint16)
    opacity = np.zeros(STATE_ID_SPACE, dtype=np.uint8)
    opacity[1] = 15
    opacity[2] = 1
    # A floor, a wall to walk the decrease through, and two emitters.
    for c in range(REGION_CHUNKS):
        blocks[c, 0, 0:256] = 1
        blocks[c, 0, 100] = 1
        blocks[c, 0, 200] = 2
        blocks[c, 0, 300] = 3
    emitter_ids = np.array([3], dtype=np.int32)
    emitter_levels = np.array([14], dtype=np.uint8)
    # Section 1 is air in every chunk: `SECTION_AIR`, so the scan, the sky fill
    # and the block-light clear all take their "nothing here" branch.
    air_sections = np.full((REGION_CHUNKS, sections), SECTION_MIXED, dtype=np.uint8)
    air_sections[:, 1] = SECTION_AIR
    sky = np.zeros((REGION_CHUNKS, sections, CELL_COUNT), dtype=np.uint8)
    block_light = np.zeros((REGION_CHUNKS, sections, CELL_COUNT), dtype=np.uint8)
    queues = LightQueues(sections).args()

    relight_region(blocks, opacity, emitter_ids, emitter_levels,
                   sky, block_light, True, air_sections, *queues)
    update_block_light(block_light, blocks, opacity, emitter_ids, emitter_levels,
                       0, 0, 0, 0, 14, *queues)
    update_sky_column(sky, blocks, opacity, 0, 0, *queues)
    logger.info("[LunarLight] Successfully compiled light kernels.")
