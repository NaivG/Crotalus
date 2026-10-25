"""
Biome acceptance: does the multi-noise source pick the same biome vanilla does?

    python tools/check_biomes.py

What this guards
----------------
A biome is one integer per 4x4x4 cell, chosen by nearest-neighbour search in a
six-dimensional climate space. Three things make it easy to be wrong in a way
nothing notices:

* The six climate slots are sampled by the **proto** router - wrappers erased.
  Sampling the chunk-phase router instead runs every value through
  `Interpolated`, which turns a step into a ramp and moves biome borders.
* The sampler takes **quart** coordinates and shifts them left by two. Passing
  block coordinates raises nothing and spreads every biome four times as far.
* The tree keeps a **last-result cache**, so a tie goes to whichever leaf was
  found first *in that history*. Row order in a batch is therefore part of the
  answer, and a "fix" that reorders the batch changes biomes at borders.

The layers
----------
1. **The climate sampler**, against 24576 quantized values from a real
   Minecraft.
2. **A whole vanilla chunk's biome volume**, against
   `Pumpkin/assets/tests/biome_no_blend_no_beard_0.json`: chunk (5,5) at seed 0,
   all 1536 quart cells.
3. **The large corpus**, 724271 samples of `(x, y, z) -> biome` at seed 0. This
   is the only layer that covers a wide area, and it is also the only one with a
   known disagreement - see `CORPUS_TOLERANCE`.
4. **The batched path**, which must agree with the scalar one.
5. **Crafted ties**, pitting the compiled kernel against the vanilla walk
   (`world.vanilla.biome._search_walk`, the answer's definition) and against an
   independent spec, on trees where several leaves sit at exactly the same
   distance. Real climate points almost never tie (0 in 220 probes), so the
   golden data above cannot see a tie-break regression - this layer is the only
   thing that can. The rules it pins: a `lastResult` candidate survives unless
   a leaf strictly beats it, and a tie between leaves goes to the one the
   traversal reaches first.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402

logger.remove()

from world.vanilla import biome as B  # noqa: E402

PUMPKIN = ROOT.parent if (ROOT.parent / "Pumpkin").exists() else ROOT
PUMPKIN_TESTS = ROOT / "Pumpkin" / "assets" / "tests"
PUMPKIN_ASSETS = ROOT / "Pumpkin" / "assets"

#: The corpus is 0.15% wrong and the disagreement is not noise. Every residual
#: sample expects `minecraft:sunflower_plains` in a two-cell-thick slab where the
#: climate values (temperature 336, humidity 1093, continentalness 545, depth
#: ~5052, weirdness -6962) are nowhere near any of that biome's 76 parameter
#: boxes, all of which require humidity <= -3500. Three axes disagree at once, so
#: no coordinate reinterpretation can rescue it: the corpus's parameter list
#: differs from the one in `assets/multi_noise_biome_tree.json` for that biome.
#: The chunk dump in layer 2 is the tie-breaker, and it is exact.
#:
#: Do not raise this without doing the same arithmetic on the new misses.
CORPUS_TOLERANCE = 0.0025

failures: List[str] = []


def check(label: str, condition: bool, detail: str = "") -> bool:
    marker = "OK  " if condition else "FAIL"
    print(f"  {marker} {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(label)
    return condition


def check_climate() -> None:
    print("1. climate sampler")
    path = PUMPKIN_ASSETS / "multi_noise_sample_no_blend_no_beard_0_0_0.json"
    if not path.exists():
        print(f"  SKIP  {path} is not present")
        return
    golden = json.load(open(path, encoding="utf-8"))
    source = B.MultiNoiseBiomeSource.from_assets("overworld", 0)
    sample = golden[:6000]
    xs = np.asarray([row[0] for row in sample], dtype=np.int64)
    ys = np.asarray([row[1] for row in sample], dtype=np.int64)
    zs = np.asarray([row[2] for row in sample], dtype=np.int64)
    want = np.asarray([[row[3], row[4], row[5], row[6], row[7], row[8]] for row in sample],
                      dtype=np.int64)
    got = source.sampler.sample_many(xs, ys, zs)
    exact = int((got == want).all(axis=1).sum())
    check("quantized climate values match vanilla", exact == len(sample),
          f"{exact}/{len(sample)} rows, {len(sample) * 6} values")

    # Scalar and batch disagreeing would mean the shift to block coordinates is
    # applied in one path and not the other.
    scalar = np.asarray([source.sampler.sample(int(x), int(y), int(z))
                         for x, y, z in zip(xs[:256], ys[:256], zs[:256])], dtype=np.int64)
    check("scalar and batch agree", np.array_equal(scalar, got[:256]))


def check_chunk_dump() -> None:
    print()
    print("2. a vanilla chunk's biome volume")
    path = PUMPKIN_TESTS / "biome_no_blend_no_beard_0.json"
    ids_path = Path(os.environ.get("PUMPKIN_BIOME_IDS", ""))
    if not path.exists():
        print(f"  SKIP  {path} is not present")
        return
    golden = json.load(open(path, encoding="utf-8"))[0]
    table = B.biome_table()
    source = B.MultiNoiseBiomeSource.from_assets("overworld", 0)

    # The dump is in Pumpkin's 26.3 id space, which has two biomes Crotalus's
    # 1.21.11 data does not (`dappled_forest`, `sulfur_caves`), so ids shift by
    # one from index 8. Translate through names, which are version-independent.
    pumpkin_ids = _pumpkin_biome_ids()
    if pumpkin_ids is None:
        print("  SKIP  Pumpkin's biome id table is not readable")
        return

    cx, cz = golden["x"], golden["z"]
    wrong = 0
    for bx, by, bz, dump_id in golden["data"]:
        name = pumpkin_ids.get(int(dump_id))
        if name is None:
            wrong += 1
            continue
        want = table.id_of(name)
        got = source.biome_id_at_quart(cx * 4 + bx, by, cz * 4 + bz)
        wrong += got != want
    check(f"chunk ({cx},{cz}) at seed 0, every quart cell",
          wrong == 0, f"{wrong}/{len(golden['data'])} wrong")

    # And the batched path must reproduce the same volume.
    volume = B.MultiNoiseBiomeSource.from_assets("overworld", 0).fill_volume(cx, cz, -64, 384)
    wrong = 0
    for bx, by, bz, dump_id in golden["data"]:
        want = table.id_of(pumpkin_ids[int(dump_id)])
        wrong += int(volume[by + 16, bz, bx]) != want
    check("fill_volume reproduces the same chunk", wrong == 0,
          f"{wrong}/{len(golden['data'])} wrong")


def _leaf(name, boxes):
    return {"_type": "leaf", "biome": name,
            "parameters": [{"min": lo, "max": hi} for lo, hi in boxes]}


def _branch(kids, boxes):
    return {"_type": "branch", "subTree": kids,
            "parameters": [{"min": lo, "max": hi} for lo, hi in boxes]}


def _brute(tree, point, cand):
    """Independent spec: min (distance, traversal order) over leaves, candidate
    rule. The builder numbers nodes in DFS preorder, so the leaf the traversal
    reaches first is the one with the smallest index - `check_tiebreak` asserts
    that numbering before trusting this.
    """
    best = None
    for node in range(tree.node_count):
        if tree.is_leaf[node] == 0:
            continue
        d = int(B._node_distance_kernel(tree.params, node, point))
        key = (d, node)
        if best is None or key < best[0]:
            best = (key, node)
    if cand >= 0:
        d_cand = int(B._node_distance_kernel(tree.params, cand, point))
        if d_cand <= best[0][0]:
            return cand
    return best[1]


def _numbers_nodes_in_traversal_order(tree):
    """Does node index order equal DFS preorder over the built arrays?

    The spec's tie-break ("the leaf the traversal reaches first") is the
    smallest leaf index only while this holds - the builder numbers nodes in
    preorder today, and this check fails loudly if that ever stops being true.
    """
    order = []
    stack = [tree.root]
    while stack:
        node = stack.pop()
        order.append(node)
        start = int(tree.child_off[node])
        for i in range(int(tree.child_count[node]) - 1, -1, -1):
            stack.append(int(tree.children[start + i]))
    return order == list(range(tree.node_count))


def check_tiebreak() -> None:
    print()
    print("5. kernel vs walk on crafted ties")
    ids = {"minecraft:plains": 1, "minecraft:desert": 2, "minecraft:badlands": 3}
    wide = [(0, 10)] * 6 + [(0, 0)]        # six climate ranges + offset axis
    far = [(200, 200)] * 6 + [(0, 0)]

    # Two leaves with *identical* boxes tie everywhere; a third leaf is far off
    # so the interesting candidates are "the other twin" and "something worse".
    twins = _branch(
        [_branch([_leaf("minecraft:plains", wide), _leaf("minecraft:desert", wide)], wide),
         _leaf("minecraft:badlands", far)],
        [(0, 200)] * 6 + [(0, 0)])
    # Two point-boxes 10 apart on the temperature axis: x=5 is exactly
    # equidistant (the tie), x=6 is uniquely the right box.
    apart = _branch(
        [_leaf("minecraft:plains", [(0, 0)] + [(0, 10)] * 5 + [(0, 0)]),
         _leaf("minecraft:desert", [(10, 10)] + [(0, 10)] * 5 + [(0, 0)]),
         _leaf("minecraft:badlands", far)],
        [(0, 200)] * 6 + [(0, 0)])
    # A branch whose own box contains the point while no leaf under it does:
    # the walk hands the tie to a candidate *branch* node, and so must the kernel.
    hollow = _branch([_leaf("minecraft:plains", [(30, 30)] + [(0, 0)] * 5 + [(0, 0)])],
                     [(0, 0)] + [(0, 0)] * 5 + [(0, 0)])

    def leaf_of(tree, name):
        return next(i for i in range(tree.node_count)
                    if tree.is_leaf[i] == 1 and tree.names[i] == name)

    def check_case(label, tree, point, cand, expect):
        assert _numbers_nodes_in_traversal_order(tree), "index order is not traversal order"
        point = np.append(np.asarray(point, dtype=np.int64), 0)  # pad the offset axis
        got_walk = B._search_walk(tree.params, tree.is_leaf, tree.child_off,
                                  tree.child_count, tree.children, point, tree.root, cand)
        got = tree.get_index(point, cand)
        ok = got == expect and got_walk == expect
        check(f"{label}", ok,
              f"want {expect}, walk {got_walk}, kernel {got}" if not ok else "")

    tree = B.BiomeTree(twins, ids)
    plains, desert, badlands = (leaf_of(tree, f"minecraft:{n}")
                                for n in ("plains", "desert", "badlands"))
    check_case("identical boxes, cold", tree, (5,) * 6, -1, plains)
    check_case("identical boxes, candidate is the twin", tree, (5,) * 6, desert, desert)
    check_case("identical boxes, candidate is worse", tree, (5,) * 6, badlands, plains)

    tree = B.BiomeTree(apart, ids)
    plains, desert, badlands = (leaf_of(tree, f"minecraft:{n}")
                                for n in ("plains", "desert", "badlands"))
    check_case("equidistant point, cold", tree, (5, 5, 5, 5, 5, 5), -1, plains)
    check_case("equidistant point, candidate wins the tie",
               tree, (5, 5, 5, 5, 5, 5), desert, desert)
    check_case("equidistant point, worse candidate loses",
               tree, (5, 5, 5, 5, 5, 5), badlands, plains)
    check_case("past the midpoint is uniquely the right box",
               tree, (6, 5, 5, 5, 5, 5), -1, desert)

    tree = B.BiomeTree(hollow, ids)
    hollow_root = tree.root
    check_case("candidate branch beats every leaf", tree, (0,) * 6, hollow_root, hollow_root)

    # Fuzz: random nested trees with duplicated boxes, random points and
    # candidates (including branch nodes). Walk, kernel and the independent
    # spec must agree on every row - this is where an order-dependent shortcut
    # would show up as a disagreement.
    if not B._HAS_NUMBA:
        print("  SKIP  fuzz (numba kernel not installed)")
        return
    rng = np.random.default_rng(20261005)
    for trial in range(40):
        n_leaves = int(rng.integers(4, 10))
        pool = [tuple(int(v) for v in rng.integers(-30, 30, size=12)) for _ in range(max(2, n_leaves // 2))]
        leaves = []
        for i in range(n_leaves):
            raw = pool[int(rng.integers(len(pool)))]  # duplicates force exact ties
            boxes = [(raw[2 * j], raw[2 * j] + int(rng.integers(0, 5))) for j in range(6)] + [(0, 0)]
            leaves.append(_leaf(f"minecraft:{['plains', 'desert', 'badlands'][i % 3]}", boxes))

        def union_of(nodes):
            out = []
            for a in range(7):
                out.append((min(n["parameters"][a]["min"] for n in nodes),
                            max(n["parameters"][a]["max"] for n in nodes)))
            return out

        kids = [leaves[0], leaves[1], _branch(leaves[2:], union_of(leaves[2:]))]
        payload = _branch(kids, union_of(kids))
        tree = B.BiomeTree(payload, ids)
        if not _numbers_nodes_in_traversal_order(tree):
            check("fuzz: index order is traversal order", False, f"trial {trial}")
            return
        pts = rng.integers(-40, 40, size=(25, 6)).astype(np.int64)
        for p in range(pts.shape[0]):
            point = np.ascontiguousarray(np.append(pts[p], 0))
            for cand in (-1, int(rng.integers(tree.node_count))):
                got = tree.get_index(point, cand)
                want = _brute(tree, point, cand)
                if got != want:
                    check(f"fuzz trial {trial} point {p} cand {cand}",
                          False, f"kernel {tree.names[got]!r} spec {tree.names[want]!r}")
                    return
        # and the batched chained path against the walk, row by row
        rows = np.ascontiguousarray(np.hstack([pts, np.zeros((pts.shape[0], 1), dtype=np.int64)]))
        got = tree.get_many(rows, -1)
        last = -1
        for p in range(rows.shape[0]):
            last = B._search_walk(tree.params, tree.is_leaf, tree.child_off,
                                  tree.child_count, tree.children, rows[p], tree.root, last)
            if int(got[p]) != last:
                check(f"fuzz trial {trial} batch row {p}", False,
                      f"kernel {tree.names[int(got[p])]!r} walk {tree.names[last]!r}")
                return
    check("fuzz: kernel == walk == spec on random tied trees", True,
          "40 trees x 25 points x 2 candidates + chained batches")


def _pumpkin_biome_ids():
    """Pumpkin's 26.3 biome id -> name, parsed out of its generated table."""
    import re

    path = ROOT / "Pumpkin" / "crates" / "pumpkin-data" / "src" / "generated" / "biome.rs"
    if not path.exists():
        return None
    source = open(path, encoding="utf-8").read()
    pairs = re.findall(r'id:\s*(\d+),\s*\n\s*registry_id:\s*"([a-z_]+)"', source)
    if not pairs:
        return None
    return {int(a): f"minecraft:{b}" for a, b in pairs}


def check_corpus() -> None:
    print()
    print("3. the large corpus")
    path = PUMPKIN_ASSETS / "multi_noise_biome_source_test.json"
    if not path.exists():
        print(f"  SKIP  {path} is not present")
        return
    big = json.load(open(path, encoding="utf-8"))
    table = B.biome_table()
    source = B.MultiNoiseBiomeSource.from_assets("overworld", 0)
    step = 12
    sample = big[::step]
    xs = np.asarray([e[0] for e in sample], dtype=np.int64)
    ys = np.asarray([e[1] for e in sample], dtype=np.int64)
    zs = np.asarray([e[2] for e in sample], dtype=np.int64)
    expected = np.asarray([e[3] for e in sample], dtype=np.int64)

    got = source.tree.biome_ids[source.tree.get_many(
        source.sampler.sample_many(xs, ys, zs), -1)].astype(np.int64)
    rate = float((got != expected).sum()) / len(sample)
    check("sample -> biome at seed 0", rate <= CORPUS_TOLERANCE,
          f"{100 * rate:.3f}% wrong ({len(sample)} samples, tolerance "
          f"{100 * CORPUS_TOLERANCE:.3f}%)")
    if rate > 0:
        from collections import Counter

        pairs = Counter()
        for i in np.nonzero(got != expected)[0]:
            pairs[(table.names.get(int(expected[i])), table.names.get(int(got[i])))] += 1
        for (want, was), count in pairs.most_common(4):
            print(f"       {count:5d}  want {want} got {was}")


def check_shapes() -> None:
    print()
    print("4. shape and dispatch")
    table = B.biome_table()
    check("the biome table is the whole registry",
          len(table.ids) == 65 and len(table.names) == 65, f"{len(table.ids)} biomes")
    check("ids are a permutation of 0..64",
          sorted(table.ids.values()) == list(range(65)))
    for preset in ("overworld", "large_biomes", "amplified", "nether"):
        source = B.biome_source(preset, seed=0, biome_ids=table.ids)
        volume = source.fill_volume(0, 0, -64, 384 if preset != "nether" else 128)
        names = {table.names[int(v)] for v in np.unique(volume)}
        check(f"{preset}: generates biomes from its own tree", bool(names),
              ", ".join(sorted(names))[:80])
        if preset == "nether":
            check("nether chunks contain only nether biomes",
                  all(n.startswith("minecraft:") and n in {
                      "minecraft:nether_wastes", "minecraft:soul_sand_valley",
                      "minecraft:crimson_forest", "minecraft:warped_forest",
                      "minecraft:basalt_deltas"} for n in names),
                  ", ".join(sorted(names)))

    # The six climate slots carry no `Interpolated` wrapper, which is why the
    # proto phase this sampler builds in cannot change a biome. Assert it: if a
    # future dump gains one, the phase becomes load-bearing and this file needs
    # to compare against reference values again rather than against itself.
    import collections

    trees = json.load(open(ROOT / "assets" / "density_function.json", encoding="utf-8"))
    for preset in ("overworld", "nether"):
        wrapped = []
        for slot in ("temperature", "vegetation", "continents", "erosion", "depth", "ridges"):
            seen = collections.Counter()

            def walk(node):
                if isinstance(node, dict):
                    if "type" in node:
                        seen[node["type"]] += 1
                    for value in node.values():
                        walk(value)
                elif isinstance(node, list):
                    for value in node:
                        walk(value)

            walk(trees[preset][slot])
            if seen.get("Interpolated"):
                wrapped.append(slot)
        check(f"{preset}: no climate slot is Interpolated", not wrapped,
              ", ".join(wrapped) if wrapped else "so the density phase cannot matter")


def main() -> int:
    started = time.perf_counter()
    check_climate()
    check_chunk_dump()
    check_corpus()
    check_shapes()
    check_tiebreak()
    print()
    if failures:
        print(f"FAILED ({len(failures)}): " + "; ".join(failures[:5]))
        return 1
    print(f"biomes OK in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
