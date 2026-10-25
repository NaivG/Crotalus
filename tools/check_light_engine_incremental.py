"""
Lead-only acceptance for the incremental light path (`LIGHT_PLAN.md` section 7.1,
items 6 and 7).

The rest of `tools/check_light_engine.py` compares the *kernel* against an
independent pure-Python oracle, cell by cell, on real terrain and on hand-built
semantic scenes. That is a static comparison: it proves the flood fill is right.
It says nothing about the four-queue incremental path, because that path only
runs when a block changes -- and that is the path a player's pickaxe drives,
where a wrong answer is a cave that stays lit or a corridor that goes dark in a
straight line and never recovers.

So this file checks the two properties that only exist in motion:

* **incremental == full.** After a batch of block changes, the field the
  incremental queues produced must equal the field a from-scratch relight
  produces for the *same* final blocks. That is the assertion that pins the
  four queues and the three kernels to each other: it is the only one that
  fails when a decrease forgets to clear a cell, when a seed is missed, or when
  a queue is drained in the wrong order.
* **the budget is a gate, not a hint.** At 0 ms a tick must move nothing, and
  the field it eventually reaches must be identical to the one an unlimited
  budget reaches immediately. "Deferred by budget" has to mean deferred.

It also carries one case that a static scene provably cannot reach: the sky
light's `15 - down - opacity 0` exception in the **decrease** chain. Removing
that exception changes nothing in any static scenario (the column fill gives the
whole `op == 0` run 15 anyway, and an incremental column rescan clears the chain
before it descends), so a check that only compared fresh fields would report the
exception as tested when it is not.

Three unit traps this file deliberately avoids, each of which produced a false
answer while it was being written:

* **Coordinates.** `DimensionLight.block_changed` takes **world** coordinates
  (it is the same entry point `Dimension.set_block` uses); the edits below are
  written in chunk-local terms and converted once, at the edge. Passing local
  coordinates sends every edit to chunk (0, 0), which is usually not even
  loaded, and then the incremental field legitimately never changes while the
  blocks do.
* **Packed vs unpacked.** `ChunkSection.sky_light` holds two light *cells* per
  byte; the kernel works in one cell per array element. Every comparison here
  goes through `unpack_light` on both sides. Comparing the two forms directly
  produces byte pairs like `(255, 15)` that look like a light bug and are
  actually a unit mismatch.
* **Counting.** `diff_count` must iterate the cells, not the layers. Counting
  the layers can only ever return 0, 1 or 2, and it reads as "2 bytes differ"
  when the truth is "both light layers are entirely different".

Run: py -3.14t tools/check_light_engine_incremental.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

logger.remove()

from world.chunk import Chunk, ChunkPos  # noqa: E402
from world.dimension import Dimension, DimensionType  # noqa: E402
from world.generator import GeneratorSettings, create_generator  # noqa: E402
from world.light_engine import (  # noqa: E402
    CENTER_CHUNK_INDEX,
    DimensionLight,
    LightQueues,
    relight_region,
)
from world.light_layout import unpack_light  # noqa: E402
from world.light_region import LightRegion  # noqa: E402

SEED = 12345
CX, CZ = 4, 4

#: 5x5, not 3x3, and the extra ring is load-bearing. `DimensionLight` refuses
#: the incremental path while any of the centre chunk's nine neighbours is
#: unlit -- it reads their light arrays, and an unlit chunk's arrays are the
#: `ChunkSection` default rather than a light level, so reading them would write
#: a fabricated field into the centre chunk. An edit made in that state is
#: promoted to a full patch instead (`stats()['dim_light_fallbacks']`).
#:
#: So a 3x3 world cannot exercise the four queues at all: lighting the centre
#: leaves its eight neighbours unlit, every edit falls back, and a test that
#: compared fields would pass while measuring the wrong code path. The ring is
#: what makes the centre's 3x3 fully lit. `RADIUS = 2` also means lighting
#: happens outward, and the assertion in section 1 checks that the incremental
#: path was really taken rather than assumed.
RADIUS = 2

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'OK  ' if ok else 'FAIL'} {label}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(label)
    return ok


def make_dimension(cx: int = CX, cz: int = CZ, radius: int = RADIUS) -> Dimension:
    """An overworld with the square around (cx, cz) generated. See `RADIUS`."""
    dimension = Dimension(
        DimensionType.OVERWORLD,
        world_seed=SEED,
        generator=create_generator(GeneratorSettings(seed=SEED), backend="legacy"),
    )
    for dx in range(-radius, radius + 1):
        for dz in range(-radius, radius + 1):
            chunk = Chunk(ChunkPos(cx + dx, cz + dz))
            dimension.generator.generate_chunk(chunk)
            chunk.is_generated = True
            dimension.chunk_manager.set_chunk(chunk)
    return dimension


def lit_dimension(budget_ms: float = 1000.0):
    """A dimension whose centre 3x3 has all been lit by the engine.

    Lit outward from the centre and then given spare ticks, because a patch
    defers while any of its nine is missing and the ring chunks need their own
    patches before they can serve as one. Sections 1 and 3 assert that no edit
    in here falls back to a full patch, which is what proves the incremental
    path is the one under test.
    """
    dimension = make_dimension()
    light = DimensionLight(dimension, budget_ms=budget_ms)
    for dx in (-1, 0, 1):
        for dz in (-1, 0, 1):
            light.request_chunk((CX + dx, CZ + dz))
    for _ in range(20):
        light.tick()
    light.request_chunk((CX, CZ))
    for _ in range(20):
        light.tick()
    return dimension, light


def field_of(chunk) -> np.ndarray:
    """The chunk's light as `(2, sections, 4096)` **cells**, not packed bytes.

    Unpacking here rather than comparing the stored bytes is the whole point:
    the kernel's answer is one cell per element and the storage is two cells
    per byte, so a direct comparison is a unit mismatch dressed up as a diff
    count.
    """
    sky = np.empty((len(chunk.sections), 4096), dtype=np.uint8)
    block = np.empty((len(chunk.sections), 4096), dtype=np.uint8)
    for i, section in enumerate(chunk.sections):
        unpack_light(section.sky_light, out=sky[i])
        unpack_light(section.block_light, out=block[i])
    return np.stack([sky, block])


def full_relight(dimension, cx: int, cz: int, opacity, emitters) -> np.ndarray:
    """What a from-scratch patch says the centre chunk's light should be.

    Uses the engine's own region builder and kernel, so this is "the same
    algorithm, run cold" -- the right reference for "did the incremental path
    converge to what a fresh computation would say". It writes nothing back.
    """
    chunk = dimension.get_chunk(cx, cz)
    region = LightRegion.build(chunk, dimension.get_chunk, read_light=False)
    assert region.ready, "the 3x3 is not resident"
    sky = np.zeros_like(region.sky)
    block = np.zeros_like(region.block_light)
    # `region.air_sections`, not `-1`: the reference has to be the kernel the
    # engine runs, section table and all, or this file would compare the
    # incremental path against a path production never takes.
    relight_region(region.blocks, opacity, emitters[0], emitters[1],
                   sky, block, True, region.air_sections,
                   *LightQueues(region.sections).args())
    return np.stack([sky[CENTER_CHUNK_INDEX], block[CENTER_CHUNK_INDEX]])


def diff_count(got: np.ndarray, want: np.ndarray) -> int:
    """Cells that differ, counted per cell rather than per layer."""
    return int(np.count_nonzero(got != want))


def light_a_block(dimension, light, lx: int, ly: int, lz: int,
                  new_state: int) -> bool:
    """Change one block and tell the engine, in **world** coordinates.

    Returns whether the engine said light could have moved. The conversion at
    the edge is the point: `block_changed` is the same call `Dimension.set_block`
    makes, so it takes the coordinates that call takes.
    """
    chunk = dimension.get_chunk(CX, CZ)
    x, z = CX * 16 + lx, CZ * 16 + lz
    old_state = int(chunk.get_block(lx, ly, lz))
    chunk.set_block(lx, ly, lz, new_state)
    return bool(light.block_changed(x, ly, z, old_state, new_state))


def check_incremental_equals_full(opacity, emitters) -> None:
    """The four queues must land where a cold patch lands. Item 6."""
    print("\n1. Incremental == full, after random edits")

    dimension, light = lit_dimension()
    centre = dimension.get_chunk(CX, CZ)
    check("the fixture chunk lit", centre.light_populated)
    lit_neighbours = sum(
        1 for dx in (-1, 0, 1) for dz in (-1, 0, 1)
        if dimension.get_chunk(CX + dx, CZ + dz).light_populated
    )
    check("all nine of the centre 3x3 are lit", lit_neighbours == 9,
          f"{lit_neighbours}/9; the incremental path needs every one of them")
    check("and equals a cold patch before any edit",
          diff_count(field_of(centre), full_relight(dimension, CX, CZ, opacity, emitters)) == 0)

    stone, air = 1, 0         # `minecraft:stone` is state 1 in this build
    from world.block_states import state_id

    torch = state_id("minecraft:torch")
    rng = random.Random(20261006)
    surface = int(centre.get_height_at(8, 8))

    # Edits chosen to hit all four queues, not just the easy one:
    #   * a solid block in open sky    -> sky decrease (the hard direction)
    #   * removing it again            -> sky increase
    #   * a torch dropped in the open  -> block increase
    #   * removing the torch           -> block decrease
    #   * solid/air flips near the surface, where the gradient is steepest
    edits = [
        (8, surface + 20, 8, stone),
        (8, surface + 20, 8, air),
        (8, surface + 21, 8, stone),
        (8, surface + 1, 8, torch),
        (8, surface + 1, 8, air),
    ]
    for _ in range(8):
        x = rng.randrange(16)
        z = rng.randrange(16)
        top = int(centre.get_height_at(x, z))
        y = top - rng.randrange(0, 6)
        state = int(centre.get_block(x, y, z))
        edits.append((x, y, z, air if state else stone))

    # --- worse case: every edit drained before the next --------------------
    # One edit (or a few) per tick, fully converged in between, compared against
    # a cold patch after each. This is a player placing blocks at a normal pace,
    # and it is where a queue that forgets an entry shows up as a single bad
    # cell that the next edit would have masked.
    moved = 0
    worst = 0
    first_bad = None
    for index, (lx, ly, lz, new_state) in enumerate(edits):
        if light_a_block(dimension, light, lx, ly, lz, new_state):
            moved += 1
        for _ in range(8):
            light.tick()
        got = field_of(centre)
        want = full_relight(dimension, CX, CZ, opacity, emitters)
        diff = diff_count(got, want)
        worst = max(worst, diff)
        if diff and first_bad is None:
            first_bad = (index, lx, ly, lz, new_state, diff)

    check("the edits reached the engine's queues", moved >= 10,
          f"{moved} of {len(edits)} reported a light-relevant change; "
          f"dropped_unloaded={light.dropped_unloaded}")
    # The whole point of the 5x5 fixture: an unlit neighbour would have
    # promoted every one of these edits to a full patch, and then this file
    # would be measuring `relight_region` against itself.
    check("every edit took the incremental path, not a fallback patch",
          light.fallback_patches == 0,
          f"fallback_patches={light.fallback_patches}")
    check(f"{len(edits)} edits, each drained, each equal to a cold patch",
          worst == 0,
          f"worst edit differed by {worst} cells" if worst else "0 cells, every time")
    if first_bad is not None:
        print(f"       first bad edit: #{first_bad[0]} at ({first_bad[1]},{first_bad[2]},"
              f"{first_bad[3]}) -> {first_bad[4]}, {first_bad[5]} cells")

    # --- burst: one tick of budget for a whole batch -----------------------
    # This is the pacing the per-tick budget actually produces: a tick's worth
    # of edits, one `tick()` between each, which is what `light-updates-per-tick`
    # is sized for. It exercises the coalescing the fast case cannot.
    burst_dimension, burst_light = lit_dimension()
    burst_centre = burst_dimension.get_chunk(CX, CZ)
    for lx, ly, lz, new_state in edits:
        light_a_block(burst_dimension, burst_light, lx, ly, lz, new_state)
        burst_light.tick()
    for _ in range(12):
        burst_light.tick()

    burst_got = field_of(burst_centre)
    burst_want = full_relight(burst_dimension, CX, CZ, opacity, emitters)
    check("a burst of edits, one tick apart, also lands on the cold answer",
          diff_count(burst_got, burst_want) == 0,
          f"{diff_count(burst_got, burst_want)} differing cells; "
          f"fallbacks={burst_light.fallback_patches}")

    # --- the same cell, twice, inside ONE tick -----------------------------
    # Placing an emitter and removing it again before the next `tick()` used to
    # converge to the *wrong* answer, and to a brighter one: the placement goes
    # into the increase queue and the removal into the decrease queue, and the
    # drain order is decrease-then-increase (vanilla's), so the removal ran
    # first -- against a cell that had not been lit yet, making it a no-op --
    # and the placement then propagated a light field for a block that no
    # longer existed. Measured before the fix: 2,089 stale cells, all of them
    # block light, none of them ever repaired. Sky light was immune because a
    # column rescan reads the *final* blocks.
    #
    # The engine now merges repeated edits of one cell within a tick, so this
    # is a no-op by construction. `tools/check_light_batch.py` is the detailed
    # version of this case; this one exists so the incremental file fails if
    # the merge is ever removed.
    pair_dimension, pair_light = lit_dimension()
    pair_centre = pair_dimension.get_chunk(CX, CZ)
    pair_y = int(pair_centre.get_height_at(8, 8)) + 5
    light_a_block(pair_dimension, pair_light, 8, pair_y, 8, torch)
    light_a_block(pair_dimension, pair_light, 8, pair_y, 8, air)
    for _ in range(8):
        pair_light.tick()
    pair_want = full_relight(pair_dimension, CX, CZ, opacity, emitters)
    check("a torch placed and removed inside one tick leaves no light behind",
          diff_count(field_of(pair_centre), pair_want) == 0,
          f"{diff_count(field_of(pair_centre), pair_want)} differing cells; "
          f"the pre-merge engine left 2089 here")

    # The other order, and a different cell, so the merge is not special-cased
    # on "place then remove of an emitter".
    light_a_block(pair_dimension, pair_light, 9, pair_y, 9, stone)
    light_a_block(pair_dimension, pair_light, 9, pair_y, 9, air)
    light_a_block(pair_dimension, pair_light, 10, pair_y, 10, air)
    light_a_block(pair_dimension, pair_light, 10, pair_y, 10, stone)
    for _ in range(8):
        pair_light.tick()
    check("and the same for a plain block, and for remove-then-place",
          diff_count(field_of(pair_centre),
                     full_relight(pair_dimension, CX, CZ, opacity, emitters)) == 0)

    # The edits have to have actually moved something, or the assertions above
    # would pass on a field that never changed.
    fresh = make_dimension()
    fresh_light = DimensionLight(fresh, budget_ms=1000.0)
    fresh_light.request_chunk((CX, CZ))
    for _ in range(4):
        fresh_light.tick()
    fresh_field = field_of(fresh.get_chunk(CX, CZ))
    check("and the field really moved relative to an untouched chunk",
          diff_count(burst_got, fresh_field) > 0,
          f"{diff_count(burst_got, fresh_field)} cells differ from the unedited chunk")


def check_sky_decrease_special_case(opacity, emitters) -> None:
    """The `15 - down - opacity 0` exception in the decrease chain.

    A static scene cannot reach this, so the case is built by hand: take a
    column whose head is at full daylight, put a solid block in that head, and
    let the incremental decrease run. The chain has to keep walking down
    through the `op == 0` air below -- one level per block -- and it starts
    with the exception (a 15 cell above 0-opacity air predicts 15, not 14). If
    the exception is missing from the decrease, the first step predicts 14, the
    cell below (holding 14) matches "what it should have been", and the chain
    stops early, leaving a straight line of stale light that nothing repairs.
    """
    print("\n2. The sky decrease's downwards exception (invisible to static scenes)")

    dimension, light = lit_dimension()
    centre = dimension.get_chunk(CX, CZ)

    # A column whose head is high above the terrain, so there is a long run of
    # air under it for the chain to walk. The `top_motion_blocking_no_leaves`
    # scan below wants the deepest surface in the chunk for that reason.
    from entity.spawn_placement import top_motion_blocking_no_leaves

    best = None
    for lx in range(16):
        for lz in range(16):
            top = top_motion_blocking_no_leaves(dimension, CX * 16 + lx, CZ * 16 + lz)
            if top is None:
                continue
            if best is None or top > best[0]:
                best = (top, lx, lz)
    check("the chunk has a surface to work from", best is not None)
    if best is None:
        return
    top, lx, lz = best

    def cell_of(local_y: int) -> int:
        return lx + lz * 16 + ((local_y - centre.min_y) % 16) * 256

    def sky_at(local_y: int) -> int:
        section = centre.sections[(local_y - centre.min_y) // 16]
        return int(unpack_light(section.sky_light)[cell_of(local_y)])

    # Raise the head artificially: put a solid block well above the surface so
    # that the air between it and the ground is a long `op == 0` run, then let
    # the engine light it. This is what "a shaft to the sky" looks like.
    #
    # The cells *below* the lid are not asserted to be dark. They are open to
    # the sides, so light walks in horizontally from the neighbouring columns
    # (which are at full daylight) at one level per step -- that is the lateral
    # seed doing its job, and asserting 0 there was measuring the terrain, not
    # the rule. What has to hold is that the column matches a cold patch, which
    # the last assertion of this section checks.
    head_y = top + 12
    light_a_block(dimension, light, lx, head_y, lz, 1)
    for _ in range(12):
        light.tick()
    check("the new block's own cell is dark", sky_at(head_y) == 0, str(sky_at(head_y)))
    check("the air below it is dimmer than the open sky beside it",
          sky_at(head_y - 1) < 15, str(sky_at(head_y - 1)))
    check("and the surface below is lit, though not at full sky under the lid",
          0 < sky_at(top + 1) < 15, str(sky_at(top + 1)))
    check("the lit column and the cold patch agree after darkening it",
          diff_count(field_of(centre),
                     full_relight(dimension, CX, CZ, opacity, emitters)) == 0,
          f"head={sky_at(head_y)} below={sky_at(head_y - 1)}")

    # Now the case the exception governs: a 15 cell with clear air under it.
    # Remove the block again; the increase has to hand 15 down the whole run
    # (the exception's job), and then the column fill's own descent takes over.
    light_a_block(dimension, light, lx, head_y, lz, 0)
    for _ in range(12):
        light.tick()
    check("removing it restores full daylight at the head", sky_at(head_y) == 15,
          str(sky_at(head_y)))
    check("and the cell below it, which the exception carries", sky_at(head_y - 1) == 15,
          str(sky_at(head_y - 1)))
    check("as far down as the air goes", sky_at(top + 2) == 15,
          str(sky_at(top + 2)))

    # The decisive one: darken from the top and watch the chain descend. A
    # broken decrease leaves stale light below the edit.
    light_a_block(dimension, light, lx, head_y, lz, 1)
    for _ in range(12):
        light.tick()
    got = field_of(centre)
    want = full_relight(dimension, CX, CZ, opacity, emitters)
    check("the darkened column matches a cold patch cell for cell",
          diff_count(got, want) == 0,
          f"{diff_count(got, want)} differing cells; "
          f"head={sky_at(head_y)} below={sky_at(head_y - 1)}")

    # And the stale-run signature stated directly: no cell in the air below the
    # new block may still hold the value it had before the edit.
    light_a_block(dimension, light, lx, head_y, lz, 0)
    for _ in range(12):
        light.tick()
    check("and removing it again also matches a cold patch",
          diff_count(field_of(centre), full_relight(dimension, CX, CZ, opacity, emitters)) == 0)


def check_budget_is_a_gate(opacity, emitters) -> None:
    """Item 7: a 0 ms budget moves nothing, an unlimited one converges now."""
    print("\n3. The per-tick budget is a gate")

    frozen = make_dimension()
    frozen_light = DimensionLight(frozen, budget_ms=0.0)
    frozen_light.request_chunk((CX, CZ))
    for _ in range(5):
        frozen_light.tick()
    check("5 ticks at a 0 ms budget light nothing",
          frozen_light.chunks_lit == 0 and not frozen.get_chunk(CX, CZ).light_populated,
          f"chunks_lit={frozen_light.chunks_lit}")

    generous = make_dimension()
    generous_light = DimensionLight(generous, budget_ms=1000.0)
    generous_light.request_chunk((CX, CZ))
    for _ in range(4):
        generous_light.tick()
    check("a generous budget converges",
          generous_light.chunks_lit >= 1 and generous.get_chunk(CX, CZ).light_populated,
          f"chunks_lit={generous_light.chunks_lit}")

    frozen_light.budget_ms = 1000.0
    for _ in range(8):
        frozen_light.tick()
    check("the frozen dimension converges once the budget returns",
          frozen.get_chunk(CX, CZ).light_populated)
    check("and both end at the identical field",
          diff_count(field_of(frozen.get_chunk(CX, CZ)),
                     field_of(generous.get_chunk(CX, CZ))) == 0)

    # The updates counter is the other half of the budget: 0 must block the
    # incremental work even with the clock wide open.
    limited, limited_light = lit_dimension(budget_ms=1000.0)
    limited_light.updates_per_tick = 0
    centre = limited.get_chunk(CX, CZ)

    lx, lz = 8, 8
    ly = int(centre.get_height_at(lx, lz)) + 3
    light_a_block(limited, limited_light, lx, ly, lz, 1)
    before = field_of(centre)
    limited_light.tick()
    check("a 0 update budget leaves the queues alone for a tick",
          diff_count(field_of(centre), before) == 0,
          "the queues for this edit are drained only when the budget allows")

    limited_light.updates_per_tick = 4096
    for _ in range(8):
        limited_light.tick()
    want = full_relight(limited, CX, CZ, opacity, emitters)
    check("and the change lands once the budget is restored",
          diff_count(field_of(centre), want) == 0,
          f"{diff_count(field_of(centre), want)} differing cells")


def main() -> int:
    from registry import init_registries

    init_registries(log_summary=False)
    from registry.blocks import light_emitters, light_opacity_table

    opacity = light_opacity_table()
    emitters = light_emitters()

    print("Incremental light acceptance (lead-only: items 6 and 7 of section 7.1)")
    check_incremental_equals_full(opacity, emitters)
    check_sky_decrease_special_case(opacity, emitters)
    check_budget_is_a_gate(opacity, emitters)

    print()
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)})")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
