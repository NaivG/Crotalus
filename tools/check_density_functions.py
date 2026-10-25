"""
Density-function acceptance: does the interpreter still produce vanilla's numbers?

    python tools/check_density_functions.py            # check
    python tools/check_density_functions.py --record   # re-record the corpus baseline

What this guards
----------------
A density-function interpreter can be *plausible* and still wrong. The failure
mode is silent and severe: sign-flipped ridges, caves that never reach the
surface, an ocean floor two blocks off. Nothing raises. The only defence is
comparing against numbers that came from somewhere else, so this tool runs four
independent layers, from most to least trustworthy.

1. **The random sources.** Stafford mixing, Xoroshiro draws, the LCG. These
   come from vectors checked against Java directly. If this layer fails,
   everything above it is noise, and the remaining layers will fail in
   confusing ways -- so it runs first and short-circuits.

2. **The overworld noise router at seed 0.** Eleven slots against values
   Pumpkin's test suite verified against a real Minecraft. This is the
   strongest test in the file: reproducing all eleven simultaneously means the
   RNG, the Perlin construction, all three octave-normalisation factors, and
   every density-function node type are correct *at once*.

3. **The vein tables.** 125 positions each for `vein_toggle`, `vein_ridged` and
   `vein_gap`. Only `vein_toggle` is checked, and the reason is in
   `VEIN_SLOT_NOTES` below -- read it before re-enabling the other two.

4. **The corpus.** Every function in `assets/density_function_tests.json`,
   evaluated on a grid, against a recorded baseline. This is the only layer
   that exercises the full node surface (splines, `RangeChoice`,
   `WeirdScaledSampler`, `Interpolated`), so it is also the layer that is only
   as good as the recorded numbers. Re-record it only when layers 1-3 pass, and
   only when you know *why* the numbers changed.

A note on the two phases
------------------------
Layers 2 and 3 evaluate the *proto* router, where wrappers are pass-throughs;
that is how the reference values were produced. Layer 4 evaluates with wrappers
live, because that is what terrain generation actually does. Comparing a proto
golden value against a wrapper-live evaluation is a common way to conclude the
interpolator is broken when it is behaving exactly as designed.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402

logger.remove()

from world.vanilla.density import (  # noqa: E402
    Context,
    cell_size_for,
    collect_flat_caches,
    load_corpus,
    load_parameters,
    parse,
)
from world.vanilla.random_source import (  # noqa: E402
    LegacyRandom,
    Xoroshiro,
    mix_stafford_13,
)

#: The world seed the reference values were taken at. Pumpkin's suite uses 0.
SEED = 0

#: Relative tolerance. The reference is f32 and this is f32, so the floor is not
#: zero -- two implementations that round identically still differ if the
#: compiler reassociates a sum. Pumpkin's own `assert_close` uses 1e-5.
TOL = 1e-5

failures: List[str] = []


def check(label: str, condition: bool, detail: str = "") -> bool:
    marker = "OK  " if condition else "FAIL"
    print(f"  {marker} {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(label)
    return condition


# ==========================================================================
# Layer 1: the random sources
# ==========================================================================

#: (input, expected) for Stafford variant 13, from Java. The negative entries
#: are written as signed; the function is unsigned, so they are masked here.
STAFFORD_CASES: List[Tuple[int, int]] = [
    (0, 0), (1, 6238072747940578789), (64, -8456553050427055661),
    (4096, -1125827887270283392), (262144, -120227641678947436),
    (16777216, 6406066033425044679), (1073741824, 3143522559155490559),
    (16, -2773008118984693571), (1024, 8101005175654470197),
    (65536, -3551754741763842827),
]

#: Ten consecutive draws from `Xoroshiro::from_seed(0)`, against Java.
XORO_I32 = [-160476802, 781697906, 653572596, 1337520923, -505875771,
            -47281585, 342195906, 1417498593, -1478887443, 1560080270]
XORO_F64 = [0.16474369376959186, 0.7997457290026366, 0.2511961888876212,
            0.11712489470639631, 0.0997124786680137, 0.7566797430601416,
            0.7723285712021574, 0.9420469457586381, 0.48056202536813664,
            0.6099690583914598]
XORO_F32 = [0.16474366, 0.7997457, 0.25119615, 0.117124856, 0.09971243,
            0.7566797, 0.77232856, 0.94204694, 0.48056197, 0.609969]
XORO_BOUNDED_10 = [9, 1, 1, 3, 8, 9, 0, 3, 6, 3]
XORO_TRIANGULAR = [6.824989823834776, 10.670356470906125, 6.71516367803936,
                   9.151408127217596, 9.352964834883384, 8.291618967842293,
                   8.954549938640508, 11.833001837470519, 10.65851306020791,
                   11.684676364031647]
XORO_GAUSSIAN = [-0.48540690699780015, 0.43399227545320296, -0.3283265251019599,
                 -0.5052497078202575, -0.3772512828630807, 0.2419080215945433,
                 -0.42622066207565135, 2.411315261138953, -1.1419147030553274,
                 -0.05849758093810378]
LEGACY_TRIANGULAR = [124.52156858525856, 104.34902101162372, 113.2163439160276,
                     70.01738222704547, 96.89666691951828, 107.30284075808541,
                     106.16817675813144, 79.11264482608078, 73.96721613927062,
                     81.72419521080646]

U64 = 0xFFFFFFFFFFFFFFFF


def layer1_random_sources() -> bool:
    print("1. The random sources (vectors checked against Java)")

    bad = [(i, mix_stafford_13(i), e) for i, e in STAFFORD_CASES
           if mix_stafford_13(i) != (e & U64)]
    check("Stafford mix_13", not bad, f"{len(STAFFORD_CASES)} cases")

    r = Xoroshiro.from_seed(0)
    got = [r.next_i32() for _ in XORO_I32]
    check("Xoroshiro next_i32", got == [v & 0xFFFFFFFF for v in XORO_I32], f"{len(got)} draws")

    r = Xoroshiro.from_seed(0)
    got = [r.next_f64() for _ in XORO_F64]
    worst = max(abs(a - b) for a, b in zip(got, XORO_F64))
    check("Xoroshiro next_f64", worst < 1e-15, f"worst {worst:.2e}")

    r = Xoroshiro.from_seed(0)
    got = [r.next_f32() for _ in XORO_F32]
    worst = max(abs(a - b) for a, b in zip(got, XORO_F32))
    check("Xoroshiro next_f32", worst < 1e-7, f"worst {worst:.2e}")

    r = Xoroshiro.from_seed(0)
    got = [r.next_bounded_i32(10) for _ in XORO_BOUNDED_10]
    check("Xoroshiro next_bounded_i32(10)", got == XORO_BOUNDED_10, f"{got}")

    r = Xoroshiro.from_seed(0)
    got = [r.next_gaussian() for _ in XORO_GAUSSIAN]
    worst = max(abs(a - b) for a, b in zip(got, XORO_GAUSSIAN))
    check("Xoroshiro next_gaussian", worst < 1e-12, f"worst {worst:.2e}")

    r = Xoroshiro.from_seed(0)
    got = [r.next_triangular(10.0, 5.0) for _ in XORO_TRIANGULAR]
    worst = max(abs(a - b) for a, b in zip(got, XORO_TRIANGULAR))
    check("Xoroshiro next_triangular", worst < 1e-12, f"worst {worst:.2e}")

    r = LegacyRandom(0)
    got = [r.next_triangular(100.0, 50.0) for _ in LEGACY_TRIANGULAR]
    worst = max(abs(a - b) for a, b in zip(got, LEGACY_TRIANGULAR))
    check("LegacyRandom (LCG) next_triangular", worst < 1e-9, f"worst {worst:.2e}")

    # The canonical java.util.Random(0) stream -- the one every LCG port gets
    # wrong first.
    r = LegacyRandom(0)
    known = [-1155484576, -723955400, 1033096058, -1690734402, -1557280266]
    got = [r.next_i32() for _ in known]
    got = [v - (1 << 32) if v >= (1 << 31) else v for v in got]
    check("LegacyRandom matches java.util.Random(0)", got == known, f"{got[:3]}...")

    return not failures


# ==========================================================================
# Layer 2: the overworld noise router at seed 0
# ==========================================================================

#: Pumpkin's `normal_surface_noisified`, verified against a real Minecraft.
#: Keyed by the slot names `assets/density_function.json` uses.
ROUTER_SEED0: Dict[str, float] = {
    "barrierNoise": -0.54002273,
    "fluidLevelFloodednessNoise": -0.4709572,
    "fluidLevelSpreadNoise": -0.05726914,
    "lavaNoise": -0.16423604,
    "temperature": 0.11823799,
    "vegetation": -0.0013601681,
    "continents": -0.008171953,
    "erosion": -0.10391074,
    "depth": 0.4118821,
    "ridges": 0.011110324,
    "finalDensity": 0.15719144,
}

#: Slots the reference suite does not pin, with the reason. The surface
#: estimator needs `find_top_surface`, which this dump predates (see
#: `VEIN_SLOT_NOTES`); `veinGap` and `veinRidged` are the vein rework.
ROUTER_UNPINNED = ("initialDensityWithoutJaggedness", "veinToggle",
                   "veinRidged", "veinGap")


def layer2_router(ctx: Context, router: dict) -> None:
    print(f"2. Overworld noise router at seed {SEED} (Java-verified values)")
    worst_overall = 0.0
    for slot, want in ROUTER_SEED0.items():
        node = parse(router[slot], ctx, cell_size_for("overworld", slot),
                     pass_through_wrappers=True)
        got = float(node.eval(ctx, 0, 0, 0))
        rel = abs(got - want) / max(abs(want), 1.0)
        worst_overall = max(worst_overall, rel)
        check(f"{slot:28s} = {got: .8f}", rel <= TOL,
              f"want {want: .8f}, rel {rel:.2e}")
    print(f"       worst relative error across all slots: {worst_overall:.2e}")
    print(f"       not pinned by the reference suite: "
          f"{', '.join(ROUTER_UNPINNED)}")


# ==========================================================================
# Layer 3: the vein tables
# ==========================================================================

#: Why `vein_ridged` and `vein_gap` are not checked.
#:
#: `assets/density_function.json` predates 1.21.9 -- it has no
#: `find_top_surface` anywhere, which is what 25w31a added. The ore-vein mask
#: was reworked in the same era, so this dump carries the *old* definitions:
#:
#:   vein_gap     this dump:  Noise(ore_gap)
#:                vanilla:   sub(-0.3, Noise(ore_gap))
#:   vein_ridged  this dump:  MAX(ABS(a), ABS(b)) + (-0.08)
#:                vanilla:   range_choice(vein_toggle, [-0.4,0.4), -1.0,
#:                                       0.08 - MAX(ABS(a), ABS(b)))
#:
#: The reference values are the newer form, so they disagree by construction --
#: `vein_gap` by exactly `-mine - 0.3`, which is what a missing `sub` looks
#: like. `vein_toggle` still matches, and is checked below.
#:
#: To check these two, first replace the vein slots in
#: `assets/density_function.json` with the 1.21.9+ definitions from
#: `Pumpkin/assets/datapack/data/minecraft/worldgen/density_function/overworld/
#: ore_vein/`. Do not "fix" the interpreter to match them; the dump is the
#: input the rest of this project reads.
VEIN_SLOT_NOTES = {
    "vein_toggle": ("veinToggle", True),
    "vein_ridged": ("veinRidged", False),
    "vein_gap": ("veinGap", False),
}


def layer3_veins(ctx: Context, router: dict, goldens: dict) -> None:
    print("3. Vein tables (125 positions each)")
    for name, (slot, enabled) in VEIN_SLOT_NOTES.items():
        if not enabled:
            print(f"  SKIP {name:12s} {VEIN_SLOT_NOTES[name] and ''}"
                  f"{'dump predates 1.21.9; see VEIN_SLOT_NOTES'}")
            continue
        rows = goldens.get(name)
        if not rows:
            check(f"{name} golden data", False, "no rows")
            continue
        node = parse(router[slot], ctx, cell_size_for("overworld", slot),
                     pass_through_wrappers=True)
        worst, bad = 0.0, 0
        for x, y, z, want in rows:
            got = float(node.eval(ctx, x, y, z))
            rel = abs(got - want) / max(abs(want), 1.0)
            worst = max(worst, rel)
            bad += rel > TOL
        check(f"{name:12s} {len(rows)} positions", bad == 0,
              f"worst rel {worst:.2e}, {bad} outside {TOL:g}")


# ==========================================================================
# Layer 4: the corpus
# ==========================================================================

#: A grid chosen to cover the cases that matter: Y inside and outside every
#: gradient, and XZ deliberately landing on *and between* interpolation cell
#: corners, so a broken `Interpolated` blend cannot hide behind corners-only
#: agreement.
CORPUS_XZ = (-101, -100, -99, -50, -3, 0, 1, 47)
CORPUS_Y = (-64, -40, -10, 0, 20, 64, 120, 300)
CORPUS_POINTS: List[Tuple[int, int, int]] = [
    (x, y, z) for x in CORPUS_XZ for y in CORPUS_Y for z in CORPUS_XZ
]

BASELINE_PATH = Path(__file__).resolve().parent / "baselines" / "density_functions.json"


def _corpus_positions() -> List[Tuple[int, int, int]]:
    return CORPUS_POINTS


def evaluate_corpus(ctx: Context, corpus: dict) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for name, tree in corpus.items():
        node = parse(tree, ctx)
        caches = collect_flat_caches(node)
        values: List[float] = []
        for index, (x, y, z) in enumerate(_corpus_positions()):
            if index % 97 == 0 and caches:
                # FlatCache is per-chunk; drop it periodically the way a real
                # chunk boundary would, so the memo cannot paper over a bug.
                for cache in caches:
                    cache.begin_context(ctx.begin_chunk(x // 16, z // 16))
            values.append(float(node.eval(ctx, x, y, z)))
        out[name] = values
    return out


def layer4_corpus(ctx: Context, corpus: dict, record: bool) -> None:
    count = len(_corpus_positions())
    print(f"4. Corpus from density_function_tests.json "
          f"({len(corpus)} functions x {count} positions)")

    start = time.perf_counter()
    got = evaluate_corpus(ctx, corpus)
    elapsed = time.perf_counter() - start
    print(f"       evaluated in {elapsed:.1f}s")

    if record or not BASELINE_PATH.exists():
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "_comment": (
                "Recorded by tools/check_density_functions.py --record. These are "
                "outputs of the interpreter, not values from Minecraft; they exist "
                "to catch drift, so re-record only when layers 1-3 pass and you "
                "know why the numbers moved."
            ),
            "seed": SEED,
            "positions": _corpus_positions(),
            "values": got,
        }
        with open(BASELINE_PATH, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1)
        check("baseline recorded", True, str(BASELINE_PATH.relative_to(ROOT)))
        return

    with open(BASELINE_PATH, encoding="utf-8") as handle:
        base = json.load(handle)

    if base.get("positions") != [list(p) for p in _corpus_positions()]:
        check("baseline grid matches", False,
              "CORPUS_XZ/CORPUS_Y changed; re-record or revert the grid")
        return

    for name in sorted(corpus):
        want = base["values"].get(name)
        if want is None:
            check(f"{name} in baseline", False, "missing; re-record")
            continue
        worst, bad = 0.0, 0
        for value, expected in zip(got[name], want):
            rel = abs(value - expected) / max(abs(expected), 1.0)
            worst = max(worst, rel)
            bad += rel > 1e-6
        check(f"{name:42s}", bad == 0, f"worst rel {worst:.2e}")


# ==========================================================================


def main() -> int:
    record = "--record" in sys.argv

    parameters = load_parameters()
    ctx = Context(SEED, parameters)

    with open(ROOT / "assets" / "density_function.json", encoding="utf-8") as handle:
        routers = json.load(handle)
    overworld = routers["overworld"]

    corpus_path = ROOT / "assets" / "density_function_tests.json"
    with open(corpus_path, encoding="utf-8") as handle:
        corpus = json.load(handle)

    goldens_path = ROOT / "assets" / "vein_golden_values.json"
    goldens = {}
    if goldens_path.exists():
        with open(goldens_path, encoding="utf-8") as handle:
            goldens = json.load(handle)

    if not layer1_random_sources():
        print("\nRESULT: FAIL (random sources are wrong; later layers would be "
              "meaningless)")
        return 1

    layer2_router(ctx, overworld)
    layer3_veins(ctx, overworld, goldens)
    layer4_corpus(ctx, corpus, record)

    print()
    if failures:
        print(f"RESULT: FAIL ({len(failures)} check(s))")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
