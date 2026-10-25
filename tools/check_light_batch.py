"""Batched edits must land where a cold patch lands.

    py -3.14t tools/check_light_batch.py

`world/light_engine.py`'s incremental queues freeze each edit's level at the
moment it is queued, while vanilla reads it out of the world when the change is
applied. One edit per tick cannot tell the two apart. Two edits to the *same
cell* inside one tick can, and the failure is silent and bright:

* place a glowstone and break it again, both before the next `tick()`;
* the break is queued as a decrease and the placement as an increase, and the
  scheduler used to drain every decrease before any increase -- so the break ran
  first, found a dark cell, did nothing, and the placement then flooded ~2,089
  cells with the light of a block that no longer exists;
* nothing repairs light that is too bright, so the world keeps it.

The fix is in `DimensionLight.block_changed`: edits to one cell inside one tick
are merged (first state is the baseline, last state the target) and the two
deques are drained oldest-first. This tool is the guard for all of it, and it
compares against a cold `relight_region` -- the same kernel, run from scratch --
so it fails if either half regresses.

Section 1 is the sensitive one: run against the pre-fix scheduler (decreases
first, no merge) it reports **2,089** stale lit cells where it now reports 0.
Section 2 is a broader net -- a mixed batch of place/remove/sky edits across
cells -- and passes under both schedulers, because separate cells commute: the
bug needs two edits to the *same* cell inside one tick. Section 3 pins that one
edit per tick is unchanged, and section 4 pins the sky decrease's clear-self
step (§9.1.2: a block in open sky takes the cell below it from 15 to 14).

**Not** "a tick should never see two edits": it should, and it does -- a console
command, a plugin, a future `/fill` or two packets can all land in one tick. The
contract is that the batched answer equals the one-at-a-time answer.

Same style as the other `tools/check_*.py`: every failure is collected and named
rather than raised, `main()` returns the exit code.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from registry.blocks import light_emitters, light_opacity_table  # noqa: E402
from tools.bench_light import build_chunks, load_fixture  # noqa: E402
from world.block_states import state_id  # noqa: E402
from world.light_engine import (  # noqa: E402
    NO_SECTION_TABLE,
    CENTER_CHUNK_INDEX,
    DimensionLight,
    LightQueues,
    relight_region,
    warm_up,
)
from world.light_region import REGION_CHUNKS, SECTION_AIR, LightRegion  # noqa: E402
from world.light_layout import unpack_light  # noqa: E402

FAILURES = []
SEED = 12345
MIN_Y, HEIGHT = -64, 384


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'OK  ' if ok else 'FAIL'} {label}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(label)
    return ok


# ---------------------------------------------------------------------------
# the world: nine lit chunks in the middle of a real 5x5 grid
# ---------------------------------------------------------------------------


def lit_world():
    """`(chunks, light)`: real terrain, centre chunk and its 8 neighbours lit."""
    chunks = build_chunks(load_fixture(SEED))
    light = DimensionLight(lambda x, z: chunks.get((x, z)), budget_ms=1000.0)
    for dz in (-1, 0, 1):
        for dx in (-1, 0, 1):
            light.queue_chunk(dx, dz)
    for _ in range(40):
        if light.pending_chunks == 0:
            break
        light.tick()
    return chunks, light


def cold_patch(chunks, cx: int = 0, cz: int = 0, masked: bool = True):
    """What a from-scratch patch says the centre chunk's two layers should be.

    `masked=False` passes `NO_SECTION_TABLE` and gets the scan-everything kernel;
    section 5 compares the two, so the table is checked against the kernel it
    replaces rather than against itself.
    """
    region = LightRegion.build(chunks[(cx, cz)], lambda x, z: chunks.get((x, z)),
                               read_light=False)
    sky = np.zeros_like(region.sky)
    block = np.zeros_like(region.block_light)
    table = region.air_sections if masked else NO_SECTION_TABLE
    relight_region(region.blocks, light_opacity_table(), light_emitters()[0],
                   light_emitters()[1], sky, block, True, table,
                   *LightQueues(len(region.center.sections)).args())
    return sky[CENTER_CHUNK_INDEX], block[CENTER_CHUNK_INDEX]


def field(chunk):
    """The chunk's two layers as nibble values, in the kernel's own layout."""
    sky = np.stack([unpack_light(s.sky_light) for s in chunk.sections])
    block = np.stack([unpack_light(s.block_light) for s in chunk.sections])
    return sky, block


def diff(got, want) -> int:
    return int(sum(np.count_nonzero(a != b) for a, b in zip(got, want)))


def dark_air_cell(chunk, x: int = 8, z: int = 8):
    """The lowest air cell with air below it, so a light has room to spread."""
    for y in range(MIN_Y + 2, MIN_Y + HEIGHT - 2):
        if chunk.get_block(x, y, z) == 0 and chunk.get_block(x, y - 1, z) == 0:
            return x, y, z
    raise RuntimeError("no air pocket in column (8, 8)")


def place(chunk, light, x, y, z, state, tick=False):
    old = int(chunk.get_block(x, y, z))
    chunk.set_block(x, y, z, state)
    light.block_changed(x + chunk.position.x * 16, y, z + chunk.position.z * 16, old, state)
    if tick:
        light.tick()


# ---------------------------------------------------------------------------
# 1. the case that was broken
# ---------------------------------------------------------------------------


def check_same_cell_burst() -> None:
    print("\n1. One cell placed and broken inside one tick")
    chunks, light = lit_world()
    centre = chunks[(0, 0)]
    glowstone = state_id("minecraft:glowstone")
    x, y, z = dark_air_cell(centre)

    place(centre, light, x, y, z, glowstone)      # no tick between these two
    place(centre, light, x, y, z, 0)
    light.tick()

    got = field(centre)
    want = cold_patch(chunks)
    check("the incremental path ran (no fallback patch)", light.fallback_patches == 0,
          f"fallbacks={light.fallback_patches}")
    check("the block light goes back to what a cold patch says",
          np.array_equal(got[1], want[1]), f"{diff(got[1:], want[1:])} cells differ")
    check("no emitter light is left behind",
          int(np.count_nonzero(got[1])) == int(np.count_nonzero(want[1])),
          f"{int(np.count_nonzero(got[1]))} lit cells, cold {int(np.count_nonzero(want[1]))}")
    check("and the sky layer is untouched", np.array_equal(got[0], want[0]),
          f"{diff(got[:1], want[:1])} cells differ")


# ---------------------------------------------------------------------------
# 2. a batch across cells, both directions, inside one tick
# ---------------------------------------------------------------------------


def check_multi_cell_burst() -> None:
    print("\n2. A batch across cells inside one tick (place + remove + sky)")
    chunks, light = lit_world()
    centre = chunks[(0, 0)]
    glowstone = state_id("minecraft:glowstone")
    torch = state_id("minecraft:torch")
    stone = state_id("minecraft:stone")

    # Converged first: two emitters placed one tick apart, so the batch below has
    # something real to remove.
    x, y, z = dark_air_cell(centre, 8, 8)
    place(centre, light, x, y, z, glowstone, tick=True)
    bx, by, bz = dark_air_cell(centre, 4, 12)
    place(centre, light, bx, by, bz, glowstone, tick=True)

    surface = int(centre.get_height_at(2, 2))
    lid_y = surface + 20

    # One tick, four edits: a removal at (8,8) that cancels the emitter placed
    # earlier, a placement somewhere else, and a lid raised and dropped again.
    place(centre, light, x, y, z, 0)
    place(centre, light, bx, by, bz, torch)
    place(centre, light, 2, lid_y, 2, stone)
    place(centre, light, 2, lid_y, 2, 0)
    light.tick()

    got = field(centre)
    want = cold_patch(chunks)
    check("the incremental path ran (no fallback patch)", light.fallback_patches == 0,
          f"fallbacks={light.fallback_patches}")
    check("a burst of different edits lands on the cold answer",
          diff(got, want) == 0, f"{diff(got, want)} cells differ")


# ---------------------------------------------------------------------------
# 3. the single-edit behaviour the merge must not have moved
# ---------------------------------------------------------------------------


def check_single_edits_unchanged() -> None:
    print("\n3. One edit per tick, still exactly what it was")
    chunks, light = lit_world()
    centre = chunks[(0, 0)]
    glowstone = state_id("minecraft:glowstone")
    stone = state_id("minecraft:stone")

    x, y, z = dark_air_cell(centre, 8, 8)
    x2, y2, z2 = dark_air_cell(centre, 5, 11)
    surface = int(centre.get_height_at(7, 7))
    sequence = [
        (x, y, z, glowstone),
        (x2, y2, z2, stone),
        (7, surface + 6, 7, stone),
        (7, surface + 6, 7, 0),
        (x, y, z, 0),
    ]
    for ex, ey, ez, state in sequence:
        place(centre, light, ex, ey, ez, state, tick=True)

    got = field(centre)
    want = cold_patch(chunks)
    check("five edits, one per tick, land on the cold answer", diff(got, want) == 0,
          f"{diff(got, want)} cells differ")
    check("no fallback was needed", light.fallback_patches == 0,
          f"fallbacks={light.fallback_patches}")


# ---------------------------------------------------------------------------
# 4. the sky column recompute (§9.1.2, the decrease's clear-self step)
# ---------------------------------------------------------------------------


def check_sky_decrease() -> None:
    print("\n4. A block in open sky darkens the cell under it (15 -> 14)")
    chunks, light = lit_world()
    centre = chunks[(0, 0)]
    stone = state_id("minecraft:stone")

    surface = int(centre.get_height_at(9, 9))
    head = surface + 6
    while centre.get_block(9, head, 9) != 0 or centre.get_block(9, head - 1, 9) != 0:
        head += 1

    def sky_at(y):
        section = centre.sections[(y - MIN_Y) // 16]
        return int(unpack_light(section.sky_light)[9 + 9 * 16 + ((y - MIN_Y) % 16) * 256])

    before = sky_at(head - 1)
    place(centre, light, 9, head, 9, stone, tick=True)
    check("the column was at full daylight", before == 15, str(before))
    check("the cell under the new block went 15 -> 14", sky_at(head - 1) == 14,
          str(sky_at(head - 1)))
    check("the block itself is dark", sky_at(head) == 0, str(sky_at(head)))
    check("the cell above is still full daylight", sky_at(head + 1) == 15,
          str(sky_at(head + 1)))
    check("matches a cold patch", diff(field(centre), cold_patch(chunks)) == 0,
          f"{diff(field(centre), cold_patch(chunks))} cells differ")


# ---------------------------------------------------------------------------
# 5. the section table: metadata vs arrays, and masked vs unmasked
# ---------------------------------------------------------------------------
#
# `LightRegion.air_sections` lets `relight_region` skip the 884,736-cell scan of
# every all-air section, and skip the sky fill's above-the-top stores in the ones
# that are provably above their own column tops. Four separate claims have to hold
# for that to be an optimization rather than a bug, and each is checked here on
# real generated terrain:
#
#   (a) the table agrees with the blocks -- recomputed from the arrays, the
#       predicate `ChunkSection.is_empty` is standing in for;
#   (b) the masked kernel produces the *same bytes* as the unmasked one, over
#       both layers and all nine chunks, not just the centre chunk;
#   (c) a section whose top is pushed over it by a *later* block is not seeded
#       -- the case a blind seed gets wrong (`check_seed_under_a_new_top`);
#   (d) the patch that the engine actually runs -- `lit_world`
#       -> `DimensionLight._light_chunk` -- lands on that answer too.
#
# (b) and (c) are the assertions that matter. The table is derived from metadata
# the light kernel never sees, so "the light is still right" is the only thing
# worth pinning, and comparing against `NO_SECTION_TABLE` (the kernel with no
# table at all) makes it a check of the optimization rather than a restatement
# of it. (c) exists because (b) alone passed while the seed was blind: the
# divergence needs a block placed in an air section *above* another air section,
# which no static fixture has.


def check_section_table() -> None:
    chunks = build_chunks(load_fixture(SEED))
    centre = chunks[(0, 0)]
    region = LightRegion.build(centre, lambda x, z: chunks.get((x, z)),
                               read_light=False)
    assert region.ready
    table = region.air_sections
    total = int(table.size)

    # (a) the table against the arrays it claims to describe
    from_arrays = ~region.blocks.any(axis=2)
    flagged = table == SECTION_AIR
    disagreements = int((flagged != from_arrays).sum())
    check("every section the table flags really is all air",
          disagreements == 0,
          f"{int(flagged.sum())} of {total} flagged, {disagreements} disagree")
    check("and every section it does not flag really does hold blocks",
          int((from_arrays & ~flagged).sum()) == 0,
          f"{total - int(flagged.sum())} sections have blocks,"
          f" {int((from_arrays & ~flagged).sum())} of them are all air anyway")
    check("the table is not vacuously empty or full",
          0 < int(flagged.sum()) < total,
          f"{int(flagged.sum())} of {total}")

    # the same predicate straight off the chunk objects, which is where the
    # table comes from: if `is_empty` ever stops tracking `blocks`, this fails
    # here rather than showing up as unexplainable light
    meta_flagged = np.array(
        [[section.is_empty for section in chunk.sections] for chunk in region.chunks],
        dtype=bool,
    )
    check("`ChunkSection.is_empty` agrees with the block arrays cell for cell",
          int((meta_flagged != from_arrays).sum()) == 0,
          f"{total} sections, {int((meta_flagged != from_arrays).sum())} differ")

    # (b) masked == unmasked, every cell of both layers
    opacity = light_opacity_table()
    emitters = light_emitters()
    queues = LightQueues(len(centre.sections)).args()
    masked_sky = np.zeros_like(region.sky)
    masked_block = np.zeros_like(region.block_light)
    plain_sky = np.zeros_like(region.sky)
    plain_block = np.zeros_like(region.block_light)
    relight_region(region.blocks, opacity, emitters[0], emitters[1],
                   masked_sky, masked_block, True, table, *queues)
    relight_region(region.blocks, opacity, emitters[0], emitters[1],
                   plain_sky, plain_block, True, NO_SECTION_TABLE, *queues)
    cells = int(masked_sky.size)
    sky_diff = int((masked_sky != plain_sky).sum())
    block_diff = int((masked_block != plain_block).sum())
    check("the masked kernel writes the same sky light as the unmasked one",
          sky_diff == 0, f"{cells * 9:,} cells compared over 9 chunks,"
                         f" {sky_diff} differ")
    check("the masked kernel writes the same block light as the unmasked one",
          block_diff == 0, f"{cells * 9:,} cells compared over 9 chunks,"
                           f" {block_diff} differ")
    check("and the sky field is not empty or trivially uniform",
          int(masked_sky.sum()) > 0 and int((masked_sky == 0).sum()) > 0,
          f"sky sum {int(masked_sky.sum())}, dark cells"
          f" {int((masked_sky == 0).sum()):,} of {masked_sky.size:,}")
    # This terrain has no emitters (no natural glowstone/lava in the fixture), so
    # the block layer is 0 everywhere and that comparison is weak evidence on its
    # own. `check_block_light_under_the_table` below runs a torch through the same
    # two kernels so the layer is exercised, and `check_cave_under_a_top` covers
    # the flagged-but-dark section that this surface terrain cannot have.
    check("the block layer is empty here, as this terrain should be",
          int(masked_block.sum()) == 0,
          f"block sum {int(masked_block.sum())} (no emitters in the fixture)")

    # The seed, seen from the centre chunk: at least one flagged section really
    # is uniform 15 (so the skip is being taken). Its counterpart -- a flagged
    # section under a top, which must stay dark and per-cell -- is covered by
    # `check_cave_under_a_top`; 19 of 24 sections being uniform is the whole
    # premise of the optimization, and this is where that is asserted.
    layers = [masked_sky[4, s] for s in range(len(centre.sections))
              if from_arrays[4][s]]
    uniform_15 = [layer for layer in layers
                  if int(layer.min()) == 15 and int(layer.max()) == 15]
    check("flagged sections above their tops hold 15 and skip the fill",
          len(uniform_15) >= 10,
          f"{len(uniform_15)} of {len(layers)} flagged sections in the centre"
          f" chunk are uniform 15")

    # (c) the engine's own patch lands on the same answer. Before the mutations
    # below, which take copies: the fixture is shared with `lit_world`, and the
    # chunks the engine lit are the ones it returns.
    lit, _light = lit_world()
    got = field(lit[(0, 0)])
    want = cold_patch(lit)
    check("a patch through DimensionLight equals the unmasked kernel",
          diff(got, want) == 0, f"{diff(got, want)} cells differ")

    # (d) a flagged section that a *later* block puts under a top must not be
    # seeded. This is the case a blind seed gets wrong, and this terrain has no
    # static instance of it: one opaque block in an air section with all-air
    # sections under it.
    check_seed_under_a_new_top(dict(chunks))
    check_cave_under_a_top(dict(chunks))
    check_block_light_under_the_table(dict(chunks))


def check_seed_under_a_new_top(chunks) -> None:
    """A block in open sky over an all-air section: 14 below it, not 15.

    The engine's patch and the unmasked kernel are both run on the same edited
    world, so a seed that ignores the new top shows up as the two disagreeing --
    which is exactly how `tools/check_light_engine_incremental.py` caught it.
    """
    chunks = dict(chunks)
    centre = chunks[(0, 0)]
    surface = int(centre.get_height_at(8, 8))
    # Deliberately high above the terrain: the section under the new block has to
    # be *empty air*, not the ground, or there is no flagged section for the top
    # to reach into. `tools/check_light_engine_incremental.py`'s failing case was
    # a block at y=82 -- sixteen blocks over the surface -- with the top landing
    # in section 9 and all-air section 8 under it.
    head = surface + 20
    while centre.get_block(8, head, 8) != 0 or centre.get_block(8, head - 1, 8) != 0:
        head += 1

    before = {
        s: centre.sections[s].is_empty for s in range(len(centre.sections))
    }
    centre.set_block(8, head, 8, 1)
    after = {
        s: centre.sections[s].is_empty for s in range(len(centre.sections))
    }
    head_section = (head - centre.min_y) // 16
    # The section the new top reaches over: the nearest all-air one at or below
    # it. On this terrain the surface is high enough that there may be none --
    # then the block's own section is the only one affected and there is nothing
    # to check, which `check_cave_under_a_top` covers instead with a pocket that
    # is guaranteed to be under a top.
    empties = [s for s in range(head_section + 1) if before[s]]
    check("the edit lands in a section that was all air",
          bool(before[head_section]) and not after[head_section],
          f"section {head_section} was empty: {before[head_section]}"
          f" -> {after[head_section]}")
    if not empties:
        check("no all-air section under the new top (nothing else to compare)",
              True, f"the surface reaches section {head_section}")
    else:
        under_section = max(empties)
        check("and an all-air section sits below the new top",
              under_section < head_section or not after[head_section],
              f"section {under_section} is empty, section {head_section} has the"
              f" block (is_empty now {after[head_section]})")

    region = LightRegion.build(centre, lambda x, z: chunks.get((x, z)),
                               read_light=False)
    table = region.air_sections
    sky_masked = np.zeros_like(region.sky)
    sky_plain = np.zeros_like(region.sky)
    block = np.zeros_like(region.block_light)
    queues = LightQueues(len(centre.sections)).args()
    opacity = light_opacity_table()
    emitters = light_emitters()
    relight_region(region.blocks, opacity, emitters[0], emitters[1],
                   sky_masked, block, True, table, *queues)
    relight_region(region.blocks, opacity, emitters[0], emitters[1],
                   sky_plain, block, True, NO_SECTION_TABLE, *queues)
    differing = int((sky_masked != sky_plain).sum())
    check("a block over an air section: masked sky == unmasked sky",
          differing == 0,
          f"{int(sky_masked.size) * 9:,} cells compared, {differing} differ")

    section = (head - centre.min_y) // 16
    layer = sky_masked[4, section]
    under = int(layer.max())
    cell = (8 + (8 << 4) + (((head - centre.min_y) % 16) << 8))
    check("the block's own cell is dark", int(sky_masked[4, section, cell]) == 0,
          str(int(sky_masked[4, section, cell])))
    below = cell - 256
    check("the air under it is 14, not a seeded 15",
          int(sky_masked[4, section, below]) == 14,
          f"section {section} max {under}, cell below"
          f" {int(sky_masked[4, section, below])}")


def _two_kernels(centre, chunks):
    """`(masked_sky, plain_sky, masked_block, plain_block, region)`."""
    region = LightRegion.build(centre, lambda x, z: chunks.get((x, z)),
                               read_light=False)
    opacity = light_opacity_table()
    emitters = light_emitters()
    queues = LightQueues(len(centre.sections)).args()
    layers = []
    for table in (region.air_sections, NO_SECTION_TABLE):
        sky = np.zeros_like(region.sky)
        block = np.zeros_like(region.block_light)
        relight_region(region.blocks, opacity, emitters[0], emitters[1],
                       sky, block, True, table, *queues)
        layers.append((sky, block))
    (masked_sky, masked_block), (plain_sky, plain_block) = layers
    return masked_sky, plain_sky, masked_block, plain_block, region


def check_cave_under_a_top(chunks) -> None:
    """A flagged all-air section *under* a top is dark and per-cell, never seeded.

    This terrain has no cave (every all-air section is above the surface), so the
    case is built: the first all-air section above the surface, with a block put
    in the section over it. That leaves an all-air section whose own light is
    per-cell rather than 15 -- the second half of the `sky_seeded` predicate, and
    the one `SECTION_AIR` alone gets wrong. Built from the terrain's own surface
    rather than by carving, because a hand-carved pocket leaves the rest of the
    section solid and is never flagged.
    """
    chunks = dict(chunks)
    centre = chunks[(0, 0)]
    first_air = min(s for s, section in enumerate(centre.sections)
                    if section.is_empty)
    above = first_air + 1
    if above >= len(centre.sections):
        check("no section above the first all-air one to build the case in", False)
        return

    # one opaque block in the section above: now the top lands there and the
    # all-air section below it is under that top
    base = centre.min_y + above * 16
    centre.set_block(8, base + 8, 8, 1)
    check("the all-air section below the new top is still all air",
          centre.sections[first_air].is_empty,
          f"section {first_air}, non_air_count"
          f" {centre.sections[first_air].non_air_count}")

    masked_sky, plain_sky, _mb, _pb, region = _two_kernels(centre, chunks)
    differing = int((masked_sky != plain_sky).sum())
    check("an all-air section under a top: masked sky == unmasked sky",
          differing == 0,
          f"{masked_sky.size * 9:,} cells compared, {differing} differ")

    check("that section is flagged all-air",
          int(region.air_sections[4, first_air]) == SECTION_AIR,
          f"section {first_air} flag {int(region.air_sections[4, first_air])}")
    # Not "dark": one column now has a top in it, the other 255 still see sky, so
    # the section is per-cell light -- 15 in the open columns, 14 in the blocked
    # one (the fill subtracts the block's own opacity). That split is exactly what
    # a blind seed would flatten to 15, and the byte-for-byte comparison above is
    # what proves the split is right.
    layer = masked_sky[4, first_air]
    plain_layer = plain_sky[4, first_air]
    blocked_cell = 8 + (8 << 4) + (8 << 8)
    open_cells = int(np.count_nonzero(layer == 15))
    check("the blocked column is dimmed under the new top, not at 15",
          int(layer[blocked_cell]) == 14, str(int(layer[blocked_cell])))
    check("the other columns still see sky, so the section is per-cell",
          open_cells == 4096 - 16 and int(layer.max()) == 15,
          f"{open_cells} of 4096 cells at 15, max {int(layer.max())}")
    check("and the unmasked kernel wrote the same per-cell layer",
          int((layer != plain_layer).sum()) == 0,
          f"{int((layer != plain_layer).sum())} cells differ within the section")
    check("and the section above it holds the new top",
          int(masked_sky[4, above,
                         (8 + (8 << 4) + (8 << 8))]) == 0,
          f"section {above}, the block's own cell")


def check_block_light_under_the_table(chunks) -> None:
    """The block layer, through both kernels, with a real emitter.

    `check_section_table`'s block comparison is over a layer that is 0 everywhere
    (this terrain has no emitters), so it proves "both kernels write nothing".
    A torch makes it prove something: light seeded in one section and spread
    across the ones around it, which is where a skip that dropped an emitter --
    or a section the BFS needed to walk through -- would show up.
    """
    chunks = dict(chunks)
    centre = chunks[(0, 0)]
    torch = state_id("minecraft:torch")
    surface = int(centre.get_height_at(4, 4))
    y = surface + 1
    while centre.get_block(4, y, 4) != 0:
        y += 1
    centre.set_block(4, y, 4, torch)

    masked_sky, plain_sky, masked_block, plain_block, _region = _two_kernels(
        centre, chunks)
    sky_diff = int((masked_sky != plain_sky).sum())
    block_diff = int((masked_block != plain_block).sum())
    check("a torch: masked block light == unmasked block light",
          block_diff == 0,
          f"{masked_block.size * 9:,} cells compared, {block_diff} differ")
    check("a torch: masked sky light == unmasked sky light",
          sky_diff == 0, f"{sky_diff} cells differ")
    lit = int(np.count_nonzero(masked_block))
    check("and the torch actually lit something",
          lit > 100, f"{lit} lit cells")
    torch_section = (y - centre.min_y) // 16
    check("its own cell carries the emission",
          int(masked_block[4, torch_section,
                           (4 + (4 << 4) + (((y - centre.min_y) % 16) << 8))]) > 0,
          f"section {torch_section}")


def main() -> int:
    warm_up()
    check_same_cell_burst()
    check_multi_cell_burst()
    check_single_edits_unchanged()
    check_sky_decrease()
    check_section_table()
    print(f"\nRESULT: {'PASS' if not FAILURES else 'FAIL'} ({len(FAILURES)})")
    for label in FAILURES:
        print(f"  - {label}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
