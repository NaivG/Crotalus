"""
Scalar vs batched: does `eval_array` reproduce `eval` exactly?

    python tools/check_density_vectorized.py

What this guards
----------------
`world.vanilla` has two evaluation paths over the same node trees: `eval` for
one position and `eval_array` for a batch. The batch path exists purely for
speed -- a chunk is ~100k positions -- so it is not allowed to change a single
rounding. That claim is only worth something if it is checked, and the check
must be *bit* equality, not a tolerance: at f32, "close enough" per sample
accumulates into a surface that moves.

The comparison is against the scalar path itself (which
`tools/check_density_functions.py` pins to values verified against a real
Minecraft), across:

1. **The corpus** -- every function in `density_function_tests.json`, on the
   same grid the baseline uses, plus offsets that land on and off
   interpolation-cell corners and boundaries.
2. **The overworld router slots** -- all fifteen, both phases: proto
   (wrappers erased, the golden-value phase) and chunk (wrappers live, what
   terrain generation uses).
3. **The chunk conveniences** -- `fill_chunk` and `surface_heights` against
   per-point scalar calls, and `surface_height` against its own batched column.

Positions are deliberately awkward: negative coordinates (Python and numpy
disagree about `%` on negatives if you get it wrong), cell origins and cell
interiors, and y values on both sides of every `YClampedGradient` break.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402

logger.remove()

from world.vanilla.density import (  # noqa: E402
    cell_size_for,
    load_corpus,
    load_parameters,
    parse,
)
from world.vanilla.router import NoiseRouter  # noqa: E402

SEED = 0

failures: List[str] = []


def check(label: str, condition: bool, detail: str = "") -> bool:
    marker = "OK  " if condition else "FAIL"
    print(f"  {marker} {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(label)
    return condition


def compare(label: str, got: np.ndarray, want: np.ndarray) -> None:
    """Bit-exact comparison: NaN equals NaN, -0.0 equals +0.0."""
    same = (got == want) | (np.isnan(got) & np.isnan(want))
    bad = int((~same).sum())
    if bad:
        worst = int(np.flatnonzero(~same.ravel())[0])
        print(f"  FAIL {label}: {bad}/{got.size} mismatched; first at [{worst}]: "
              f"scalar {want.ravel()[worst]!r} vs batch {got.ravel()[worst]!r}")
        failures.append(label)
    else:
        print(f"  OK   {label}  ({got.size} points, bit-identical)")


#: A grid that lands on, inside, and across interpolation cells -- including
#: negatives, where `%` and floor-div must match Python's, not C's.
PROBE_XZ = (-103, -102, -101, -100, -99, -97, -50, -5, -4, -1, 0, 1, 3, 4, 7, 8, 47, 48)
PROBE_Y = (-64, -63, -41, -40, -33, -9, -8, -1, 0, 7, 8, 20, 63, 64, 120, 300, 319)


def probe_points() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = [(x, y, z) for x in PROBE_XZ for y in PROBE_Y for z in PROBE_XZ]
    xs = np.asarray([p[0] for p in pts], dtype=np.int64)
    ys = np.asarray([p[1] for p in pts], dtype=np.int64)
    zs = np.asarray([p[2] for p in pts], dtype=np.int64)
    return xs, ys, zs


def corpus_vectors(ctx, corpus: dict) -> None:
    print("1. Corpus functions: scalar vs batch")
    xs, ys, zs = probe_points()
    for name, tree in sorted(corpus.items()):
        node = parse(tree, ctx)
        want = np.asarray([float(node.eval(ctx, int(x), int(y), int(z)))
                           for x, y, z in zip(xs, ys, zs)], dtype=np.float32)
        got = node.eval_array(ctx, xs, ys, zs)
        compare(f"{name:42s}", got, want)


def router_vectors() -> None:
    print("2. Router slots: scalar vs batch (proto and chunk phase)")
    xs, ys, zs = probe_points()

    router = NoiseRouter.from_assets("overworld", seed=SEED)
    router.begin_chunk(-3, 5)
    ctx = router.context
    for slot in sorted(router):
        node = router.slot(slot)
        want = np.asarray([float(node.eval(ctx, int(x), int(y), int(z)))
                           for x, y, z in zip(xs, ys, zs)], dtype=np.float32)
        got = node.eval_array(ctx, xs, ys, zs)
        compare(f"chunk-phase {slot:28s}", got, want)

    # Proto phase: wrappers erased. The parse in `router` is chunk-phase, so
    # build proto trees against the same context.
    import json
    with open(ROOT / "assets" / "density_function.json", encoding="utf-8") as handle:
        trees = json.load(handle)["overworld"]
    for slot in sorted(trees):
        node = parse(trees[slot], ctx, cell_size_for("overworld", slot),
                     pass_through_wrappers=True)
        want = np.asarray([float(node.eval(ctx, int(x), int(y), int(z)))
                           for x, y, z in zip(xs, ys, zs)], dtype=np.float32)
        got = node.eval_array(ctx, xs, ys, zs)
        compare(f"proto-phase  {slot:28s}", got, want)


def chunk_conveniences() -> None:
    print("3. Chunk conveniences: fill_chunk / surface_heights / surface_height")

    router = NoiseRouter.from_assets("overworld", seed=SEED)
    chunk_x, chunk_z = -3, 5

    vol = router.fill_chunk(chunk_x, chunk_z, y_min=-64, y_max=320)
    check("fill_chunk shape", vol.shape == (384, 16, 16), f"{vol.shape}")

    # Spot-check the volume against scalar calls: corners of the chunk, a
    # band of interior points, and every y of one column (which crosses all
    # the interpolation cells in Y).
    spots = [(0, 0, 0), (15, 0, 15), (0, 383, 15), (15, 383, 0),
             (8, 100, 8), (3, 250, 12), (11, -64 + 300, 4)]
    bad = 0
    for zi, yi, xi in spots:
        x = chunk_x * 16 + xi
        z = chunk_z * 16 + zi
        y = -64 + yi
        got = float(vol[yi, zi, xi])
        want = float(router.final_density(x, y, z))
        if not ((got == want) or (np.isnan(got) and np.isnan(want))):
            bad += 1
            print(f"    mismatch at ({x}, {y}, {z}): scalar {want!r} vs volume {got!r}")
    for yi in range(384):
        got = float(vol[yi, 7, 9])
        want = float(router.final_density(chunk_x * 16 + 9, -64 + yi, chunk_z * 16 + 7))
        if not ((got == want) or (np.isnan(got) and np.isnan(want))):
            bad += 1
            print(f"    mismatch on column at y={-64 + yi}: scalar {want!r} vs volume {got!r}")
    check("fill_chunk vs scalar final_density", bad == 0, f"{bad} mismatches")

    heights = router.surface_heights(chunk_x, chunk_z, min_y=-64, max_y=320)
    check("surface_heights shape", heights.shape == (16, 16), f"{heights.shape}")
    bad = 0
    for zi, xi in ((0, 0), (15, 15), (7, 9), (3, 12)):
        x = chunk_x * 16 + xi
        z = chunk_z * 16 + zi
        want = router.surface_height(x, z, -64, 320)
        got = int(heights[zi, xi])
        want_v = -64 if want is None else want
        if got != want_v:
            bad += 1
            print(f"    mismatch at ({x}, {z}): scalar {want!r} vs batch {got}")
    check("surface_heights vs scalar surface_height", bad == 0, f"{bad} mismatches")


def main() -> int:
    parameters = load_parameters()
    from world.vanilla.density import Context
    ctx = Context(SEED, parameters)

    corpus = load_corpus()

    corpus_vectors(ctx, corpus)
    router_vectors()
    chunk_conveniences()

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
