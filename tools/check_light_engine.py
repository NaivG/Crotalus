"""Does the light kernel agree with the independent oracle, cell for cell?

`tools/light_oracle.py` is a pure-Python translation of the rules in
`LIGHT_PLAN.md` section 4 and knows nothing about `world/light_engine.py`. This
tool runs both over the same blocks and compares every cell of both light
layers. Four groups do that:

  2. a real generated 3x3 patch, no shafts;
  3. the same patch with section 5.2's two shafts in the centre chunk;
  4. the six hand-built semantic scenes, each dropped into the centre of a
     stone patch so the kernel and the oracle see identical inputs;
  5. the column-top rules: the kernel's own `_column_top` against the oracle's
     scan of the same rule (vanilla's `LIGHT_BLOCKING`, the highest cell with
     `opacity > 0`), and the relation between that rule and the non-air top
     section 4.3 names.

     Guardrail for whoever touches this next: **which** of the two definitions a
     kernel picks turns out not to matter for the final field -- the column fill
     subtracts `opacity` either way, so an opacity-0 block on top changes
     nothing below it. What does matter is that the same choice is used
     *consistently*: the fill and the stage-2 seeds must be scanning for the
     same top, or the seeds float above the fill and the field stops being a
     fixed point. That is why both are exported (`region_surfaces` for non-air,
     `region_blocking_tops` for `opacity > 0`) and why this section pins the
     kernel's to one of them instead of accepting "close enough".

Section 6 is the other half of the job and imports no kernel: the oracle's own
invariants (a sealed torch cave's `14 - d`, a water column's `15 - n`, glass and
opaque roofs, the column fill being monotone, the final field being a closed
fixed point of the propagation rules, and the BFS agreeing with a heap-based
max-field relaxation). A kernel and a validator written from one misreading
would agree with each other, so the rules have to be pinned somewhere that does
not mention the kernel at all.

Every comparison prints how many cells it compared. "0 differences" is not
evidence if the comparison was over an empty set, and a region comparison that
silently compared one chunk instead of nine would look the same.

**Not covered here, on purpose.** This file is the **static** half: the kernel
against an independent oracle over fixed blocks. That proves the flood fill is
right, and it says nothing about the four-queue incremental path, because that
path only runs when a block changes.

The **dynamic** half lives in `tools/check_light_engine_incremental.py` (the
lead's file, `LIGHT_PLAN.md` section 7.1 items 6 and 7): "incremental == full
relight" after a batch of block changes, and "the budget is a gate, not a hint".
Between the two files section 7.1 is covered; neither is complete alone.

That split is also why one rule is pinned only over there. The sky light's
`15 - down - opacity 0` exception changes nothing in *any* static scene -- the
column fill already gives the whole `op == 0` run 15, and an incremental column
rescan clears the chain before it descends -- so a fresh-field comparison would
report that exception as tested when it is not. `decrease_reference` and
`incremental_reference` are exported from `tools/light_oracle.py` for that file
to use; this one imports the kernel only to compare finished fields.

**Independence, stated plainly.** The same person wrote the oracle and this
comparison, which makes the four kernel-vs-oracle groups weak evidence on their
own: a misreading shared by both sides would pass. What compensates is section
6 (rules pinned with no kernel import), the hand-computed expectations in the
semantic scenes, `max_field_reference` (a second algorithm for the same rules),
and the cell counts printed everywhere. The strongest single assertion here is
not "it matches the oracle" but "**the kernel's own field is closed under its own
transfer rule**" (sections 2 and 3): that one needs no oracle to be right, and it
was what turned a 153-cell disagreement into a proven kernel defect.

Run: py -3.14t tools/check_light_engine.py
"""

from __future__ import annotations

import ast
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger

logger.remove()

import world.light_engine as engine
from registry.blocks import light_emitters, light_opacity_table
from world.block_states import BLOCKS
from tools.light_oracle import (
    CELLS_PER_SECTION, HEIGHT, MIN_Y, SECTIONS, SHAFT_SPECS, block_composition,
    build_base_volume, build_region_volumes, chunk_cell, chunk_index,
    column_fill_reference, light_statistics, make_shafts, max_field_reference,
    region_blocking_tops, region_surfaces, registry_emitter_tables,
    registry_opacity_table, relight_reference, relight_reference_region,
    scene_glass_roof, scene_sealed_torch_cave, scene_solid_roof,
    scene_water_column, semantic_scenes,
)

#: `LIGHT_PLAN.md` section 5.2's engine row. The two shafts' positions are not
#: recorded in the plan: `SHAFT_SPECS` is a sweep-based recovery that reproduces
#: the row to within three cells, so the plan cross-check below is a 1% band and
#: the tool prints the spec it used. It is a regression pin, not a target.
PLAN_ENGINE_LIT = 66_082
PLAN_ENGINE_MIDDLE = 694
PLAN_TOLERANCE = 0.01

FAILURES: list = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'OK  ' if ok else 'FAIL'} {label}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(label)
    return ok


# ==================== shared helpers ====================


def kernel_relight(blocks: np.ndarray, opacity: np.ndarray, emitter_ids: np.ndarray,
                   emitter_levels: np.ndarray, do_sky: bool = True):
    """Run the kernel over a `(9, sections, 4096)` region; returns both layers.

    Always with `NO_SECTION_TABLE`, the scan-everything kernel: the section table
    is a `ChunkSection.is_empty` product and this file has no chunks at all, and
    comparing the *general* kernel is what makes this file the reference. The
    table is checked against this same call, byte for byte, in
    `tools/check_light_batch.py`.
    """
    sky = np.zeros(blocks.shape, dtype=np.uint8)
    block_light = np.zeros(blocks.shape, dtype=np.uint8)
    engine.relight_region(blocks, opacity, emitter_ids, emitter_levels,
                          sky, block_light, do_sky, engine.NO_SECTION_TABLE,
                          *engine.LightQueues(blocks.shape[1]).args())
    return sky, block_light


def oracle_relight(blocks: np.ndarray, opacity: np.ndarray, emitter_ids: np.ndarray,
                   emitter_levels: np.ndarray, do_sky: bool = True):
    """The same call on the oracle, with the same region layout."""
    return relight_reference_region(blocks, opacity, emitter_ids, emitter_levels, do_sky)


def describe_first_diffs(kernel_layer: np.ndarray, oracle_layer: np.ndarray,
                         limit: int = 4) -> str:
    """Up to `limit` differing cells as `chunk (x, y, z) kernel vs oracle`."""
    differing = np.argwhere(kernel_layer != oracle_layer)
    parts = []
    for c, s, i in differing[:limit]:
        parts.append(
            f"chunk {int(c)} (x={int(i) % 16}, "
            f"y={MIN_Y + int(s) * 16 + (int(i) // 256) % 16}, "
            f"z={(int(i) // 16) % 16}) kernel {int(kernel_layer[c, s, i])} "
            f"vs oracle {int(oracle_layer[c, s, i])}")
    if len(differing) > limit:
        parts.append(f"... {len(differing) - limit} more")
    return "; ".join(parts)


def compare_layer(label: str, kernel_layer: np.ndarray, oracle_layer: np.ndarray) -> None:
    """One light layer, every cell, with the size of the comparison printed."""
    total = int(kernel_layer.size)
    differing = int((kernel_layer != oracle_layer).sum())
    detail = f"{total:,} cells compared, {differing:,} differ"
    if differing:
        detail += " -- " + describe_first_diffs(kernel_layer, oracle_layer)
    check(label, differing == 0, detail)


def state_at(region: np.ndarray, wx: int, wz: int, ry: int) -> int:
    """Block state id at region-local `(wx, wz)` and region-relative `ry`."""
    ci = engine._region_chunk(wx, wz)
    cell = (wx & 15) + ((wz & 15) << 4) + ((ry & 15) << 8)
    return int(region[ci, ry >> 4, cell])


def region_grid(layer: np.ndarray) -> np.ndarray:
    """A `(9, sections, 4096)` region layer as one `(height, 48, 48)` [y, z, x] grid.

    The closure rule is about face neighbours, and two cells in different chunks
    are neighbours in the world but not in a chunk's own array, so the region has
    to be put back together before the rule can be checked across a border.
    """
    cells = np.asarray(layer).reshape(9, SECTIONS, 16, 16, 16)
    grid = np.zeros((HEIGHT, 48, 48), dtype=np.asarray(layer).dtype)
    for dz in (-1, 0, 1):
        for dx in (-1, 0, 1):
            grid[:, 16 * (dz + 1):16 * (dz + 1) + 16,
                 16 * (dx + 1):16 * (dx + 1) + 16] = cells[chunk_index(dx, dz)].reshape(
                     HEIGHT, 16, 16)
    return grid


# ==================== 1. inputs ====================


def check_inputs() -> None:
    print("\n1. inputs: the engine's tables and the fixture")
    engine_table = light_opacity_table()
    oracle_table = registry_opacity_table()
    check("registry.blocks.light_opacity_table() == the oracle's own build",
          bool(np.array_equal(engine_table, oracle_table)),
          f"{int((engine_table != oracle_table).sum())} of {engine_table.size:,} "
          f"states differ")

    engine_ids, engine_levels = light_emitters()
    oracle_ids, oracle_levels = registry_emitter_tables()
    engine_pairs = sorted(zip(engine_ids.tolist(), engine_levels.tolist()))
    oracle_pairs = sorted(zip(oracle_ids.tolist(), oracle_levels.tolist()))
    check("registry.blocks.light_emitters() == the oracle's own build",
          engine_pairs == oracle_pairs, f"{len(engine_pairs)} emitters on each side")

    started = time.perf_counter()
    region = build_region_volumes()
    elapsed = time.perf_counter() - started
    check("the 3x3 fixture is nine real chunks of (24, 4096) uint16",
          region.shape == (9, SECTIONS, CELLS_PER_SECTION) and region.dtype == np.uint16,
          f"{region.shape}, built in {elapsed:.1f}s (cached after the first run)")
    check("its centre is the single-column fixture of section 5.4",
          bool(np.array_equal(region[chunk_index(0, 0)], build_base_volume())))

    composition = block_composition(region[chunk_index(0, 0)], engine_table,
                                    BLOCKS.WATER, BLOCKS.LAVA)
    check("the centre column is section 5.4's: air 67,499 / water 27 / lava 3",
          composition["air"] == 67_499 and composition["water"] == 27
          and composition["lava"] == 3, str(composition))


# ==================== 2/3. the real 3x3 patch ====================


def check_kernel_matches_oracle(name: str, centre: np.ndarray, opacity: np.ndarray,
                                emitter_ids: np.ndarray, emitter_levels: np.ndarray) -> None:
    """One region: the kernel and the oracle over every cell of both layers."""
    print(f"\n{name}")
    region = build_region_volumes().copy()
    region[chunk_index(0, 0)] = centre

    started = time.perf_counter()
    oracle_sky, oracle_block = oracle_relight(region, opacity, emitter_ids, emitter_levels)
    oracle_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    kernel_sky, kernel_block = kernel_relight(region, opacity, emitter_ids, emitter_levels)
    kernel_ms = (time.perf_counter() - started) * 1000

    label = name.split(".", 1)[0]
    compare_layer(f"{label} sky light, all nine chunks", kernel_sky, oracle_sky)
    compare_layer(f"{label} block light, all nine chunks", kernel_block, oracle_block)

    # The same closure rule section 6 asserts on the oracle, applied to the
    # kernel's own output. A field that fails this is inconsistent with its own
    # transfer rule -- a stronger statement than "it disagrees with the oracle",
    # and the one that says which side is wrong.
    kernel_sky_violations = closure_violations(
        region_grid(kernel_sky), region_grid(region), opacity, True)
    kernel_block_violations = closure_violations(
        region_grid(kernel_block), region_grid(region), opacity, False)
    check(f"{label} the kernel's own sky field is closed under its rules",
          kernel_sky_violations == 0,
          f"{6 * 48 * 48 * HEIGHT:,} transfer checks, "
          f"{kernel_sky_violations} cells a neighbour could still raise")
    check(f"{label} the kernel's own block field is closed under its rules",
          kernel_block_violations == 0,
          f"{6 * 48 * 48 * HEIGHT:,} transfer checks, "
          f"{kernel_block_violations} cells a neighbour could still raise")

    stats = light_statistics(kernel_sky[chunk_index(0, 0)], kernel_block[chunk_index(0, 0)])
    print(f"  centre chunk: sky_lit {stats['sky_lit']:,}, middle {stats['sky_middle']}, "
          f"block_lit {stats['block_lit']:,}, any_lit {stats['any_lit']:,} | "
          f"{9 * SECTIONS * CELLS_PER_SECTION:,} cells per layer | "
          f"oracle {oracle_ms:.0f} ms vs kernel {kernel_ms:.1f} ms per patch")


def check_plan_rows(opacity: np.ndarray, emitter_ids: np.ndarray,
                    emitter_levels: np.ndarray) -> None:
    """Section 5.2's rows on the single closed column, in a 1% band."""
    print("\n3b. section 5.2's rows on the closed single column (recovered spec)")
    base = build_base_volume()
    shafted, dug = make_shafts(base)
    print(f"  SHAFT_SPECS (recovered by sweep; the plan does not record them): "
          f"{SHAFT_SPECS} -> dug {dug}")
    for label, volume, want_lit, want_middle in (
            ("no shaft", base, None, None),
            ("two shafts", shafted, PLAN_ENGINE_LIT, PLAN_ENGINE_MIDDLE)):
        sky, _block = relight_reference(volume, opacity, emitter_ids, emitter_levels, True)
        stats = light_statistics(sky, np.zeros_like(sky))
        if want_lit is None:
            check(f"{label}: the closed column is {stats['sky_lit']:,} lit / "
                  f"{stats['sky_middle']} middle",
                  stats["sky_lit"] == 65_360 and stats["sky_middle"] == 28,
                  "pinned: the plan's 66,082/694 is the *shafted* row, not this one")
            continue
        lit_ok = abs(stats["sky_lit"] - want_lit) <= want_lit * PLAN_TOLERANCE
        middle_ok = abs(stats["sky_middle"] - want_middle) <= max(8, want_middle // 20)
        check(f"{label}: within {PLAN_TOLERANCE:.0%} of the plan's "
              f"{want_lit:,} / {want_middle}",
              lit_ok and middle_ok,
              f"got {stats['sky_lit']:,} / {stats['sky_middle']}")


# ==================== 4. the semantic scenes ====================


def check_kernel_on_scenes(opacity: np.ndarray, emitter_ids: np.ndarray,
                           emitter_levels: np.ndarray) -> None:
    print("\n4. kernel vs oracle on the six semantic scenes (centre of a stone patch)")
    stone = BLOCKS.STONE
    for name, (scene, _info) in semantic_scenes().items():
        region = np.full((9, SECTIONS, CELLS_PER_SECTION), stone, dtype=np.uint16)
        region[chunk_index(0, 0)] = scene
        oracle_sky, oracle_block = oracle_relight(region, opacity, emitter_ids,
                                                  emitter_levels)
        kernel_sky, kernel_block = kernel_relight(region, opacity, emitter_ids,
                                                  emitter_levels)
        compare_layer(f"scene {name}: sky light", kernel_sky, oracle_sky)
        compare_layer(f"scene {name}: block light", kernel_block, oracle_block)


# ==================== 5. the column-top rules ====================


def check_column_tops(opacity: np.ndarray) -> None:
    print("\n5. column tops: the kernel's rule vs the oracle's two")
    region = build_region_volumes()
    oracle_blocking = region_blocking_tops(region, opacity)

    kernel_tops = np.full((48, 48), -1, dtype=np.int32)
    for wz in range(48):
        for wx in range(48):
            kernel_tops[wz, wx] = engine._column_top(region, opacity, wx, wz, HEIGHT)
    check("engine._column_top == oracle region_blocking_tops (opacity > 0)",
          bool(np.array_equal(kernel_tops, oracle_blocking)),
          f"{48 * 48} columns compared, "
          f"{int((kernel_tops != oracle_blocking).sum())} differ "
          f"(the kernel's own helper is called deliberately: this pins the rule)")

    surfaces = region_surfaces(region)
    nonair_ry = (surfaces - MIN_Y).astype(np.int32)
    differs = nonair_ry != oracle_blocking
    count = int(differs.sum())
    transparent = [state_at(region, int(wx), int(wz), int(nonair_ry[wz, wx]))
                   for wz, wx in np.argwhere(differs)]
    check("the non-air top of section 4.3 is never below the blocking top",
          bool(np.all(nonair_ry >= oracle_blocking)),
          f"{48 * 48} columns compared, {count} differ (the non-air one is above)")
    check("each column where they differ has a transparent block on top",
          all(int(opacity[state]) == 0 for state in transparent),
          f"{count} columns checked, states "
          f"{sorted({int(opacity[state]) for state in transparent})} on top")
    check("they are two rules, not one implementation",
          count > 0,
          f"{count} of {48 * 48} columns: the oracle's stage-2 seeds use the non-air "
          f"top, the kernel the blocking one; sections 2-4 prove the fields agree anyway. "
          f"Which definition is picked does not matter for the field -- using one "
          f"consistently does (a fill and a seed scan that disagree stop the field "
          f"being a fixed point)")


# ==================== 6. oracle-only invariants ====================


def _transfer(source: np.ndarray, target_opacity: np.ndarray, downward: bool,
              is_sky: bool) -> np.ndarray:
    """The level `source` delivers into a cell whose opacity is `target_opacity`."""
    damp = np.where(target_opacity <= 1, 1, target_opacity)
    out = np.maximum(source - damp, 0)
    if is_sky:
        out = np.where(downward & (source == 15) & (target_opacity == 0), 15, out)
    return out


def closure_violations(light_grid: np.ndarray, blocks_grid: np.ndarray,
                       opacity_table: np.ndarray, is_sky: bool) -> int:
    """Cells a neighbour could still raise: 0 means the field is a fixed point.

    Both arguments are `(y, z, x)` grids -- one chunk's 16x16 columns or a
    region's 48x48 (`region_grid`). This is the general form of section 7.1 item
    2's "a column's sky light never rises going down". That statement is true of
    the *column fill* and false of the final field -- side light legitimately
    makes a lower cell brighter -- so what is asserted here is closure under the
    propagation rules, which is what "the BFS ran to exhaustion" means.
    """
    light = np.asarray(light_grid, dtype=np.int16)
    states = np.asarray(blocks_grid, dtype=np.uint16)
    if light.shape != states.shape:
        raise ValueError(f"light {light.shape} and blocks {states.shape} differ")
    op = np.asarray(opacity_table, dtype=np.int16)[states]
    violations = 0
    for axis, source_part, target_part, downward in (
            (0, slice(None, -1), slice(1, None), False),    # source below, target up
            (0, slice(1, None), slice(None, -1), True),     # source above, target down
            (1, slice(None, -1), slice(1, None), False),
            (1, slice(1, None), slice(None, -1), False),
            (2, slice(None, -1), slice(1, None), False),
            (2, slice(1, None), slice(None, -1), False)):
        source_index = [slice(None)] * 3
        target_index = [slice(None)] * 3
        source_index[axis] = source_part
        target_index[axis] = target_part
        source = light[tuple(source_index)]
        target = light[tuple(target_index)]
        target_op = op[tuple(target_index)]
        violations += int((_transfer(source, target_op, downward, is_sky) > target).sum())
    return violations


def check_oracle_invariants(opacity: np.ndarray, emitter_ids: np.ndarray,
                            emitter_levels: np.ndarray) -> None:
    print("\n6. oracle-only invariants (this section imports no kernel)")
    base = build_base_volume()
    shafted, _dug = make_shafts(base)

    volume, info = scene_sealed_torch_cave()
    sky, block = relight_reference(volume, opacity, emitter_ids, emitter_levels, True)
    flat = block.reshape(-1)
    shaft = [int(flat[chunk_cell(info["x"], info["torch_y"] + d, info["z"])])
             for d in range(info["depth"])]
    check("sealed cave: a torch decays 14 - d, distance 7 is 7 and 15 is 0",
          shaft == [max(14 - d, 0) for d in range(info["depth"])]
          and shaft[7] == 7 and shaft[15] == 0,
          f"{info['depth']} cells checked, distance 7 -> {shaft[7]}, "
          f"distance 15 -> {shaft[15]}")
    sky_column = [int(sky.reshape(-1)[chunk_cell(info["x"], y, info["z"])])
                  for y in range(info["torch_y"], info["top"])]
    check("sealed cave: no sky light anywhere in the shaft",
          set(sky_column) == {0}, f"{len(sky_column)} cells checked")

    volume, info = scene_water_column()
    sky, _block = relight_reference(volume, opacity, emitter_ids, emitter_levels, True)
    flat = sky.reshape(-1)
    got = [int(flat[chunk_cell(info["x"], info["top"] - (n - 1), info["z"])])
           for n in range(1, info["depth"] + 1)]
    check("water column: the cell n below the surface is 15 - n",
          got == [15 - n for n in range(1, info["depth"] + 1)],
          f"{len(got)} cells checked, got {got}")

    volume, info = scene_solid_roof()
    sky, _block = relight_reference(volume, opacity, emitter_ids, emitter_levels, True)
    below = int(sky.reshape(-1)[chunk_cell(info["x"], info["roof_y"] - 1, info["z"])])
    check("opaque roof: the cell directly below it has sky light 0", below == 0,
          f"got {below}")

    volume, info = scene_glass_roof()
    sky, _block = relight_reference(volume, opacity, emitter_ids, emitter_levels, True)
    flat = sky.reshape(-1)
    under = int(flat[chunk_cell(info["x"], info["roof_y"] - 1, info["z"])])
    deep = int(flat[chunk_cell(info["x"], info["floor_y"] + 1, info["z"])])
    check("glass roof (opacity 0) does not shade: 15 under it and at the floor",
          under == 15 and deep == 15, f"got {under}, {deep}")

    for label, volume in (("no shaft", base), ("two shafts", shafted)):
        fill = np.asarray(column_fill_reference(volume, opacity),
                          dtype=np.uint8).reshape(HEIGHT, 16, 16)
        rising = int((fill[1:] < fill[:-1]).sum())
        check(f"column fill ({label}) never rises going down",
              rising == 0, f"{HEIGHT * 256} cells checked, {rising} rises")

    sky, block = relight_reference(shafted, opacity, emitter_ids, emitter_levels, True)
    shafted_grid = np.asarray(shafted, dtype=np.uint16).reshape(HEIGHT, 16, 16)
    sky_violations = closure_violations(
        np.asarray(sky, dtype=np.uint8).reshape(HEIGHT, 16, 16), shafted_grid, opacity, True)
    block_violations = closure_violations(
        np.asarray(block, dtype=np.uint8).reshape(HEIGHT, 16, 16), shafted_grid,
        opacity, False)
    check("the final sky field is closed under the propagation rules",
          sky_violations == 0,
          f"{6 * HEIGHT * 256} transfer checks, {sky_violations} violations")
    check("the final block field is closed under the propagation rules",
          block_violations == 0,
          f"{6 * HEIGHT * 256} transfer checks, {block_violations} violations")

    final = np.asarray(sky, dtype=np.uint8).reshape(HEIGHT, 16, 16)
    rises = int((final[1:] < final[:-1]).sum())
    print(f"  note: the shafted field has {rises} cells brighter than the cell above "
          f"them, so section 7.1 item 2's \"a column never brightens going down\" is a "
          f"property of the column fill (asserted above), not of the final field")
    print("  note: the sky '15 down through opacity 0' exception is inert in every "
          "scene in\n  this file; check_light_engine_incremental.py pins it in the "
          "decrease chain, which\n  is the only place it decides anything")

    for label, volume in (("no shaft", base), ("two shafts", shafted)):
        sky, block = relight_reference(volume, opacity, emitter_ids, emitter_levels, True)
        relaxed_sky, relaxed_block = max_field_reference(
            volume, opacity, emitter_ids, emitter_levels, True)
        sky_diff = int((sky != relaxed_sky).sum())
        block_diff = int((block != relaxed_block).sum())
        check(f"BFS == heap-based max field ({label})",
              sky_diff == 0 and block_diff == 0,
              f"{2 * HEIGHT * 256} cells compared, sky {sky_diff} and block "
              f"{block_diff} differ")


# ==================== 7. dead code ====================


def check_no_serializer_importer() -> None:
    print("\n7. world/serializer.py has no importer (LIGHT_PLAN section 9.2)")
    target = ROOT / "world" / "serializer.py"
    check("world/serializer.py still exists", target.is_file())
    skip_dirs = {"__pycache__", "venv", ".venv", "node_modules", ".git", "Pumpkin"}
    offenders = []
    scanned = 0
    for path in sorted(ROOT.rglob("*.py")):
        if path == target or skip_dirs.intersection(path.parts):
            continue
        scanned += 1
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any("serializer" in alias.name.split(".") for alias in node.names):
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom):
                parts = (node.module or "").split(".") if node.module else []
                names = [alias.name for alias in node.names]
                if ("serializer" in parts
                        or (parts and parts[-1] == "world" and "serializer" in names)
                        or (node.level and "serializer" in names)):
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    check("nothing imports it (AST scan of every .py outside Pumpkin/)",
          not offenders,
          f"{scanned} files scanned, {len(offenders)} importers"
          + (f": {offenders[:5]}" if offenders else ""))


# ==================== 8. scope ====================


def print_scope_note() -> None:
    print("\n8. not covered by this file")
    print("  This is the *static* half of section 7.1: the kernel against the oracle over")
    print("  fixed blocks. The *dynamic* half -- item 6 'incremental == full relight' and")
    print("  item 7 'the budget is a gate, not a hint' -- is "
          "tools/check_light_engine_incremental.py,")
    print("  the lead's file. Neither covers section 7.1 alone.")
    print("  One rule is pinned only over there, and deliberately: the sky light's")
    print("  '15 - down - opacity 0' exception changes nothing in any static scene, so a")
    print("  fresh-field comparison reports it as tested when it is not. That file's")
    print("  decrease-chain case is the one that reaches it; the oracle side it uses")
    print("  (decrease_reference / incremental_reference) is exported from "
          "tools/light_oracle.py.")


def main() -> int:
    print("Light engine checks: kernel vs the independent oracle")
    started = time.perf_counter()
    warm = time.perf_counter()
    engine.warm_up()
    print(f"  (kernel warm-up {time.perf_counter() - warm:.1f}s; the numba cache "
          f"makes later runs cheap)")

    opacity = light_opacity_table()
    emitter_ids, emitter_levels = light_emitters()

    check_inputs()
    check_kernel_matches_oracle("2. kernel vs oracle: real 3x3 patch, no shafts",
                                build_base_volume(), opacity, emitter_ids, emitter_levels)
    shafted, _dug = make_shafts(build_base_volume())
    check_kernel_matches_oracle("3. kernel vs oracle: real 3x3 patch, two shafts",
                                shafted, opacity, emitter_ids, emitter_levels)
    check_plan_rows(opacity, emitter_ids, emitter_levels)
    check_kernel_on_scenes(opacity, emitter_ids, emitter_levels)
    check_column_tops(opacity)
    check_oracle_invariants(opacity, emitter_ids, emitter_levels)
    check_no_serializer_importer()
    print_scope_note()

    print()
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)}) in {time.perf_counter() - started:.1f}s")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print(f"RESULT: PASS in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
