"""
The ore kernel's acceptance: does `_ore_kernel` place exactly what the
interpreted reference places?

    python tools/check_ore_veins.py

`world/vanilla/features.py` runs `ore` through a numba kernel; the interpreted
`_place_ore_python` stays as the reference. The contract is `carver.py`'s: the
two paths must agree block for block, draw for draw -- and here also counter for
counter. This tool is what makes that claim checkable rather than hopeful.

The layers, strongest first
---------------------------
1. **The kernel RNG draws.** `kernel_rng` re-implements the draw methods for the
   compiled walks; every draw it can make is compared against the Python
   generator it was seeded from, for all three generator kinds, and `state_to`
   round-trips the state back into a live Python stream.

2. **The discard semantics.** `should_not_discard || !is_exposed_to_air`, pinned
   as a truth table with the draw counts: `discard = 1.0` places a buried ore
   and consumes no roll, `0 < discard < 1` rolls before the exposure scan, and
   the exposure scan short-circuits in `Direction.values()` order. This is the
   behaviour Pumpkin's `OreFeature::should_place` has, and the one a reading of
   "discard chance on air exposure" as "never place" gets wrong.

3. **Synthetic vein differential.** Every `ore` config in
   `assets/configured_features.json` plus edge-case configs, at positions that
   straddle the chunk border and the world floor/ceiling, on terrain fixtures
   with caves and water (so the exposure scan and the dropped-write counters get
   exercised). Python vs kernel: blocks, RNG end state, the four write counters
   and the return value must match exactly.

4. **Whole-pipeline differential.** Real chunks through the real `decorate`
   pipeline, compiled vs interpreted. Every block and every counter, including
   `reads_outside`, must match exactly.

5. **The speed.** The same workload timed on both paths; the kernel must be at
   least 4x the interpreted one or the whole arrangement is not worth its
   complexity.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from world.vanilla import features as features_module  # noqa: E402
from world.vanilla import kernel_rng  # noqa: E402
from world.vanilla.features import (  # noqa: E402
    FeatureContext,
    FeatureStats,
    RuleTest,
    Surface,
    _ore_should_place,
    block_facts,
    decorate,
    load_configured_features,
)
from world.vanilla.random_source import (  # noqa: E402
    LegacyRandom,
    WorldgenRandom,
    Xoroshiro,
)

SEED = 4242
MIN_Y = -64
HEIGHT = 384

failures: List[str] = []


def check(label: str, condition: bool, detail: str = "") -> bool:
    marker = "OK  " if condition else "FAIL"
    print(f"  {marker} {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(label)
    return condition


def report(label: str, detail: str) -> None:
    print(f"  ---- {label}: {detail}")


class _StubTable:
    """One biome, no feature lists: the ore paths never ask for either."""

    features = {"test:plains": ()}
    ids = {"test:plains": 1}
    names = {1: "test:plains"}


# ==========================================================================
# Layer 1: the kernel RNG draws
# ==========================================================================


def layer1_rng() -> None:
    print("Layer 1 - the kernel RNG draws")
    kinds = (
        ("worldgen (overworld)", kernel_rng.RNG_WORLDGEN,
         lambda: WorldgenRandom(Xoroshiro.from_seed(0xC0FFEE))),
        ("legacy (nether)", kernel_rng.RNG_LEGACY, lambda: LegacyRandom(0xC0FFEE)),
        ("raw xoroshiro", kernel_rng.RNG_RAW, lambda: Xoroshiro.from_seed(0xC0FFEE)),
    )
    for label, kind, factory in kinds:
        python_side = factory()
        state = kernel_rng.state_from(python_side)
        check(f"{label} state kind", int(state[0]) == kind, f"kind={int(state[0])}")
        python_side = factory()
        mismatches = 0
        first = None
        for i in range(3000):
            which = i % 5
            if which == 0:
                want = np.float32(python_side.next_f32())
                got = kernel_rng.next_f32(state)
                same = np.float32(want) == np.float32(got)
            elif which == 1:
                want = python_side.next_f64()
                got = kernel_rng.next_f64(state)
                same = want == got
            elif which == 2:
                want = python_side.next_bounded_i32(3)
                got = int(kernel_rng.next_bounded_i32(state, 3))
                same = want == got
            elif which == 3:
                want = python_side.next_bounded_i32(31)
                got = int(kernel_rng.next_bounded_i32(state, 31))
                same = want == got
            else:
                want = python_side.next_i64()
                got = int(kernel_rng.next_i64(state))
                same = want == got
            if not same:
                mismatches += 1
                if first is None:
                    first = (i, want, got)
        check(f"{label} draws match the Python generator", mismatches == 0,
              f"3000 mixed draws, {mismatches} mismatches"
              + (f", first {first}" if first else ""))

    # `state_to`: a kernel-advanced state written back into a Python generator
    # that has seen the same draws must keep that generator's stream going.
    reference = WorldgenRandom(Xoroshiro.from_seed(7))
    carried = WorldgenRandom(Xoroshiro.from_seed(7))
    state = kernel_rng.state_from(carried)
    for _ in range(50):
        kernel_rng.next_f64(state)
        int(kernel_rng.next_bounded_i32(state, 3))
    for _ in range(50):
        reference.next_f64()
        reference.next_bounded_i32(3)
    kernel_rng.state_to(carried, state)
    same = all(reference.next_f32() == carried.next_f32() for _ in range(20))
    check("state_to hands the stream back to Python intact", same, "")


# ==========================================================================
# Layer 2: the discard semantics
# ==========================================================================


def _discard_case(discard: float, exposed: bool, expect_place: bool,
                  expect_draw: bool) -> None:
    facts = block_facts()
    stone = facts.default_state("minecraft:stone")
    air = 0
    blocks = np.full((HEIGHT, 16, 16), stone, dtype=np.uint16)
    if exposed:
        blocks[100 - MIN_Y + 1, 8, 8] = air  # one air neighbour above the cell
    biomes = np.zeros((HEIGHT // 4, 4, 4), dtype=np.int32)
    surface = Surface(None)
    stats = FeatureStats()
    ctx = FeatureContext(0, 0, blocks, biomes, MIN_Y, HEIGHT, _StubTable(), stats,
                         surface)
    surface.bind(ctx)
    rule = RuleTest("tag_match", tag="stone_ore_replaceables")
    rng = WorldgenRandom(Xoroshiro.from_seed(1))
    before = (rng.source.lo, rng.source.hi)
    placed = _ore_should_place(ctx, rule, np.float32(discard), stone, rng,
                               (8, 100, 8))
    after = (rng.source.lo, rng.source.hi)
    drew = before != after
    ok = placed == expect_place and drew == expect_draw
    check(f"discard={discard}, exposed={exposed}",
          ok,
          f"placed={placed} (want {expect_place}), drew={drew} (want {expect_draw})")


def layer2_discard() -> None:
    print("Layer 2 - discard semantics and their draw counts")
    # `should_not_discard || !is_exposed_to_air`, pinned as a truth table.
    _discard_case(0.0, exposed=False, expect_place=True, expect_draw=False)
    _discard_case(0.0, exposed=True, expect_place=True, expect_draw=False)
    _discard_case(1.0, exposed=False, expect_place=True, expect_draw=False)
    _discard_case(1.0, exposed=True, expect_place=False, expect_draw=False)
    # 0 < discard < 1 rolls before the exposure scan, whatever the scan finds.
    for exposed in (False, True):
        placed_draws = 0
        for seed in range(400):
            facts = block_facts()
            stone = facts.default_state("minecraft:stone")
            blocks = np.full((HEIGHT, 16, 16), stone, dtype=np.uint16)
            if exposed:
                blocks[100 - MIN_Y + 1, 8, 8] = 0
            biomes = np.zeros((HEIGHT // 4, 4, 4), dtype=np.int32)
            surface = Surface(None)
            ctx = FeatureContext(0, 0, blocks, biomes, MIN_Y, HEIGHT, _StubTable(),
                                 FeatureStats(), surface)
            surface.bind(ctx)
            rule = RuleTest("tag_match", tag="stone_ore_replaceables")
            rng = WorldgenRandom(Xoroshiro.from_seed(seed))
            before = (rng.source.lo, rng.source.hi)
            placed = _ore_should_place(ctx, rule, np.float32(0.5), stone, rng,
                                       (8, 100, 8))
            drew = before != (rng.source.lo, rng.source.hi)
            if placed:
                placed_draws += 1
            if not drew:
                check(f"discard=0.5, exposed={exposed}: every match rolls",
                      False, f"seed {seed} drew nothing")
                return
        # Exposed: only the `draw >= 0.5` half places. Buried: every match does.
        want = "50-ish %" if exposed else "100%"
        rate = placed_draws / 400
        ok = (0.4 < rate < 0.6) if exposed else rate == 1.0
        check(f"discard=0.5, exposed={exposed}: placement rate", ok,
              f"{placed_draws}/400 rolls placed ({want})")


# ==========================================================================
# Layer 3: synthetic vein differential
# ==========================================================================


_FIXTURES = ("solid", "caves", "water", "island")


def _fixture(kind: str, facts) -> np.ndarray:
    blocks = np.zeros((HEIGHT, 16, 16), dtype=np.uint16)
    stone = facts.default_state("minecraft:stone")
    deepslate = facts.default_state("minecraft:deepslate")
    bedrock = facts.default_state("minecraft:bedrock")
    water = facts.default_state("minecraft:water")
    for row in range(HEIGHT):
        y = MIN_Y + row
        if y == MIN_Y:
            blocks[row, :, :] = bedrock
        elif y < 0:
            blocks[row, :, :] = deepslate
        elif y < 96:
            blocks[row, :, :] = stone
    if kind == "caves":
        rng = np.random.default_rng(11)
        for _ in range(24):
            cx, cy, cz = (int(rng.integers(0, 16)), int(rng.integers(-60, 96)),
                          int(rng.integers(0, 16)))
            r = int(rng.integers(1, 3))
            for x in range(max(0, cx - r), min(16, cx + r + 1)):
                for y in range(max(0, cy - r - MIN_Y), min(HEIGHT, cy + r + 1 - MIN_Y)):
                    for z in range(max(0, cz - r), min(16, cz + r + 1)):
                        blocks[y, z, x] = 0
    elif kind == "water":
        blocks[63 - MIN_Y:80 - MIN_Y] = water
    elif kind == "island":
        blocks[:96 - MIN_Y] = 0
        blocks[92 - MIN_Y:96 - MIN_Y] = stone
    return blocks


def _run_synthetic(config: dict, pos, blocks_template: np.ndarray,
                   biomes: np.ndarray, compiled: bool):
    blocks = blocks_template.copy()
    stats = FeatureStats()
    surface = Surface(None)
    ctx = FeatureContext(0, 0, blocks, biomes, MIN_Y, HEIGHT, _StubTable(), stats,
                         surface)
    surface.bind(ctx)
    rng = WorldgenRandom(Xoroshiro.from_seed(SEED))
    call = (features_module._place_ore_compiled if compiled
            else features_module._place_ore_python)
    placed = call(ctx, config, rng, pos)
    return blocks, stats, (rng.source.lo, rng.source.hi), placed


def _trial(config: dict, pos, fixture_kind: str, facts) -> Tuple[bool, str]:
    blocks_template = _fixture(fixture_kind, facts)
    biomes = np.zeros((HEIGHT // 4, 4, 4), dtype=np.int32)
    py_blocks, py_stats, py_rng, py_placed = _run_synthetic(
        config, pos, blocks_template, biomes, compiled=False)
    kn_blocks, kn_stats, kn_rng, kn_placed = _run_synthetic(
        config, pos, blocks_template, biomes, compiled=True)
    if not np.array_equal(py_blocks, kn_blocks):
        return False, f"blocks differ ({int((py_blocks != kn_blocks).sum())} cells)"
    if py_rng != kn_rng:
        return False, f"RNG end state {py_rng} vs {kn_rng}"
    if py_placed != kn_placed:
        return False, f"return {py_placed} vs {kn_placed}"
    for field in ("writes_dropped", "writes_outside_xz", "writes_outside_y",
                  "reads_outside"):
        if getattr(py_stats, field) != getattr(kn_stats, field):
            return False, (f"{field} {getattr(py_stats, field)} vs "
                           f"{getattr(kn_stats, field)}")
    return True, ""


def layer3_synthetic(configured) -> None:
    print("Layer 3 - synthetic vein differential (python vs kernel)")
    facts = block_facts()
    real = []
    for name, entry in sorted(configured.items()):
        if entry.type in ("ore", "scattered_ore"):
            real.append((name, entry.config))
    report("real ore configs", str(len(real)))
    edge = [
        ("edge:always_true_discard_1", {
            "size": 6, "discard_chance_on_air_exposure": 1.0,
            "targets": [{"target": {"predicate_type": "minecraft:always_true"},
                         "state": {"Name": "minecraft:gold_block"}}]}),
        ("edge:random_block_match", {
            "size": 9, "discard_chance_on_air_exposure": 0.3,
            "targets": [{"target": {"predicate_type": "minecraft:random_block_match",
                                    "block": "minecraft:stone",
                                    "probability": 0.4},
                         "state": {"Name": "minecraft:iron_ore"}}]}),
        ("edge:block_match_one_target", {
            "size": 12, "discard_chance_on_air_exposure": 0.0,
            "targets": [{"target": {"predicate_type": "minecraft:block_match",
                                    "block": "minecraft:deepslate"},
                         "state": {"Name": "minecraft:deepslate_gold_ore"}}]}),
        ("edge:size_1", {
            "size": 1, "discard_chance_on_air_exposure": 0.5,
            "targets": [{"target": {"predicate_type": "minecraft:tag_match",
                                    "tag": "#minecraft:stone_ore_replaceables"},
                         "state": {"Name": "minecraft:coal_ore"}}]}),
    ]
    corpus = [(name, config) for name, config in real] + edge
    positions = [(0, MIN_Y, 0), (15, MIN_Y + 4, 15), (7, 32, 8), (8, 64, 7),
                 (1, 96, 14), (14, 380, 1), (0, 200, 15), (15, -20, 0)]
    trials = 0
    first_fail = None
    for name, config in corpus:
        for pos in positions:
            for fixture_kind in _FIXTURES:
                ok, detail = _trial(config, pos, fixture_kind, facts)
                trials += 1
                if not ok and first_fail is None:
                    first_fail = (name, pos, fixture_kind, detail)
    check("every config/position/fixture matches on blocks, draws and counters",
          first_fail is None,
          f"{trials} trials"
          + (f"; first fail {first_fail}" if first_fail else ""))


# ==========================================================================
# Layer 4: whole-pipeline differential on real chunks
# ==========================================================================


def layer4_pipeline() -> None:
    print("Layer 4 - real chunks through decorate, compiled vs interpreted")
    if not features_module._HAS_NUMBA:
        check("numba is installed", False, "no compiled path to compare")
        return
    from world.chunk import Chunk, ChunkPos
    from world.generator import GeneratorSettings, create_generator

    placed = features_module.load_placed_features()
    configured = load_configured_features()
    gen = create_generator(GeneratorSettings(seed=SEED, preset="overworld"))
    gen.set_seed(SEED)
    chunks = [(0, 0), (7, 3), (-2, -5)]
    all_blocks_equal = True
    counter_notes = []
    for cx, cz in chunks:
        probe = Chunk(ChunkPos(cx, cz))
        gen._generate_terrain(probe)
        gen.layers.biomes.apply(probe)
        gen.layers.surface.apply(probe, gen.layers.table)
        gen.layers.carvers.apply(probe, gen.layers.table, sea_level=63)
        template = probe.blocks_view().copy()
        biomes = probe.biome_volume().copy()

        outcomes = {}
        for compiled in (False, True):
            features_module._USE_NUMBA = compiled
            blocks = template.copy()
            stats = decorate(cx, cz, blocks, biomes, preset="overworld", seed=SEED,
                             min_y=MIN_Y, height=HEIGHT,
                             biome_table=gen.layers.table,
                             placed_features=placed,
                             configured_features=configured)
            outcomes[compiled] = (blocks, stats)
        features_module._USE_NUMBA = features_module._HAS_NUMBA

        py_blocks, py_stats = outcomes[False]
        kn_blocks, kn_stats = outcomes[True]
        same = np.array_equal(py_blocks, kn_blocks)
        all_blocks_equal &= same
        if not same:
            report(f"chunk ({cx},{cz})",
                   f"{int((py_blocks != kn_blocks).sum())} cells differ")
        for field in ("features_attempted", "features_placed", "writes_dropped",
                      "writes_outside_xz", "writes_outside_y", "reads_outside"):
            if getattr(py_stats, field) != getattr(kn_stats, field):
                counter_notes.append(
                    f"({cx},{cz}) {field}: {getattr(py_stats, field)} vs "
                    f"{getattr(kn_stats, field)}")
        report(f"chunk ({cx},{cz})",
               f"placed {py_stats.features_placed}/{kn_stats.features_placed}, "
               f"ore writes dropped {py_stats.writes_dropped}/"
               f"{kn_stats.writes_dropped}, reads_outside "
               f"{py_stats.reads_outside}/{kn_stats.reads_outside} "
               f"(python/kernel)")
    check("every chunk is block-identical between the paths", all_blocks_equal, "")
    check("every counter matches exactly", not counter_notes,
          "; ".join(counter_notes[:4]))


# ==========================================================================
# Layer 5: the speed
# ==========================================================================


def layer5_speed(configured) -> None:
    print("Layer 5 - the speed")
    if not features_module._HAS_NUMBA:
        check("numba is installed", False, "nothing to time")
        return
    facts = block_facts()
    config = None
    for name, entry in configured.items():
        if name.endswith("ore_granite") or entry.type == "ore" and entry.config.get("size") == 33:
            config = entry.config
            break
    if config is None:
        config = next(e.config for e in configured.values() if e.type == "ore")
    blocks_template = _fixture("caves", facts)
    biomes = np.zeros((HEIGHT // 4, 4, 4), dtype=np.int32)
    positions = [(x % 16, -20 + (x % 7) * 40, (x * 7) % 16) for x in range(120)]

    def timed(compiled: bool) -> float:
        # One Surface for the whole batch, like a real chunk's decorate call
        # shares one across every ore feature in it -- building the frozen
        # `columns` grid is per-chunk work and re-paying it per vein would be
        # the thing measured, not the kernel.
        blocks = blocks_template.copy()
        surface = Surface(None)
        call = (features_module._place_ore_compiled if compiled
                else features_module._place_ore_python)
        t0 = time.perf_counter()
        for pos in positions:
            stats = FeatureStats()
            ctx = FeatureContext(0, 0, blocks, biomes, MIN_Y, HEIGHT, _StubTable(),
                                 stats, surface)
            surface.bind(ctx)
            rng = WorldgenRandom(Xoroshiro.from_seed(SEED))
            call(ctx, config, rng, pos)
        return time.perf_counter() - t0

    compiled = timed(True)
    python = timed(False)
    report("workload", f"{len(positions)} veins, size {config['size']}, one chunk")
    report("interpreted", f"{python * 1000:.0f} ms ({python / len(positions) * 1000:.2f} ms/vein)")
    report("compiled", f"{compiled * 1000:.0f} ms ({compiled / len(positions) * 1000:.2f} ms/vein)")
    check("the compiled path is at least 4x the interpreted one",
          compiled * 4.0 <= python,
          f"{python / max(compiled, 1e-9):.1f}x")


# ==========================================================================


def main() -> int:
    print("Ore-kernel acceptance")
    print(f"  seed {SEED}, overworld preset, {MIN_Y}..{MIN_Y + HEIGHT}")
    print(f"  numba {features_module._HAS_NUMBA}, "
          f"compiled path {'on' if features_module._USE_NUMBA else 'off'}")
    print()

    layer1_rng()
    print()
    layer2_discard()
    print()
    configured = load_configured_features()
    layer3_synthetic(configured)
    print()
    layer4_pipeline()
    print()
    layer5_speed(configured)

    print()
    if failures:
        print(f"RESULT: FAIL ({len(failures)} check(s))")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
