"""Is the biome tree search memory-bound on the params table?

The root has 9111 children, so the "branch and bound" is a linear scan over
every box: 9112 x 7 x 2 int64 = 1.02 MB read once per point, 1536 points per
chunk. If bandwidth is the limit, halving the table to int32 (distances still
accumulated in int64, so the answer is bit-identical) should show up as a
proportional speedup.

Run: py -3.14t tools/bench_biome_params.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from numba import njit
from loguru import logger

logger.remove()

from world.vanilla import biome as B


@njit(cache=True, nogil=True)
def _search_many_i32(params, is_leaf, child_off, child_count, children, biome_ids,
                     points, root, candidate, out):
    """`_search_many_kernel` with an int32 params table.

    Only the table's dtype changes: `point[axis] - high` is computed in int32
    (both operands are within +-2**15) and squared into an int64 accumulator, so
    every distance is the same integer the int64 kernel produces.
    """
    big = np.int64(1) << np.int64(62)
    stack = np.empty(params.shape[0], dtype=np.int64)
    axes = params.shape[1]
    last = candidate
    for row in range(points.shape[0]):
        point = points[row]
        best_dist = big
        best_node = root
        if last >= 0:
            best_dist = np.int64(0)
            for axis in range(axes):
                low = params[last, axis, 0]
                high = params[last, axis, 1]
                above = np.int32(point[axis]) - high
                below = low - np.int32(point[axis])
                if above > 0:
                    best_dist += np.int64(above) * np.int64(above)
                elif below > 0:
                    best_dist += np.int64(below) * np.int64(below)
            best_node = last
        top = 1
        stack[0] = root
        while top > 0:
            top -= 1
            node = stack[top]
            dist = np.int64(0)
            for axis in range(axes):
                low = params[node, axis, 0]
                high = params[node, axis, 1]
                above = np.int32(point[axis]) - high
                below = low - np.int32(point[axis])
                if above > 0:
                    dist += np.int64(above) * np.int64(above)
                elif below > 0:
                    dist += np.int64(below) * np.int64(below)
            if dist >= best_dist:
                continue
            if is_leaf[node] == 1:
                best_dist = dist
                best_node = node
                continue
            start = child_off[node]
            count = child_count[node]
            for i in range(count - 1, -1, -1):
                stack[top] = children[start + i]
                top += 1
        out[row] = best_node
        last = best_node
    return last


@njit(cache=True, nogil=True)
def _search_many_drop_degenerate(params, is_leaf, child_off, child_count, children,
                                biome_ids, points, root, candidate, out,
                                keep_axis):
    """As above, but only over the axes any node actually varies on.

    An axis where every node has low == high contributes exactly 0 to every
    distance, so dropping it cannot move a boundary.
    """
    big = np.int64(1) << np.int64(62)
    stack = np.empty(params.shape[0], dtype=np.int64)
    n_axes = keep_axis.shape[0]
    last = candidate
    for row in range(points.shape[0]):
        point = points[row]
        best_dist = big
        best_node = root
        if last >= 0:
            best_dist = np.int64(0)
            for k in range(n_axes):
                axis = keep_axis[k]
                low = params[last, axis, 0]
                high = params[last, axis, 1]
                above = np.int32(point[axis]) - high
                below = low - np.int32(point[axis])
                if above > 0:
                    best_dist += np.int64(above) * np.int64(above)
                elif below > 0:
                    best_dist += np.int64(below) * np.int64(below)
            best_node = last
        top = 1
        stack[0] = root
        while top > 0:
            top -= 1
            node = stack[top]
            dist = np.int64(0)
            for k in range(n_axes):
                axis = keep_axis[k]
                low = params[node, axis, 0]
                high = params[node, axis, 1]
                above = np.int32(point[axis]) - high
                below = low - np.int32(point[axis])
                if above > 0:
                    dist += np.int64(above) * np.int64(above)
                elif below > 0:
                    dist += np.int64(below) * np.int64(below)
            if dist >= best_dist:
                continue
            if is_leaf[node] == 1:
                best_dist = dist
                best_node = node
                continue
            start = child_off[node]
            count = child_count[node]
            for i in range(count - 1, -1, -1):
                stack[top] = children[start + i]
                top += 1
        out[row] = best_node
        last = best_node
    return last


def main() -> None:
    src = B.MultiNoiseBiomeSource.from_assets("overworld", seed=0)
    sampler, tree = src.sampler, src.tree

    ys = np.arange(96, dtype=np.int64)
    zs = np.arange(4, dtype=np.int64)
    xs = np.arange(4, dtype=np.int64)
    gy, gz, gx = np.meshgrid(ys, zs, xs, indexing="ij")
    pts = sampler.sample_many(gx.reshape(-1), gy.reshape(-1), gz.reshape(-1))
    padded = np.zeros((pts.shape[0], B.PARAMETER_AXES), dtype=np.int64)
    padded[:, :pts.shape[1]] = pts
    points = np.ascontiguousarray(padded)
    n = points.shape[0]

    lo, hi = tree.params[:, :, 0], tree.params[:, :, 1]
    fits32 = int(max(abs(lo).max(), abs(hi).max()))
    print(f"points {n}   params {tree.params.nbytes / 1e6:.2f} MB int64   "
          f"max |bound| {fits32} (int32 ok: {fits32 < 2**31})")

    keep = np.nonzero((lo != hi).any(axis=0))[0].astype(np.int32)
    print(f"axes any node varies on: {keep.tolist()} "
          f"(dropping {tree.params.shape[1] - keep.size})   "
          f"int32 = {tree.params.shape[0] * keep.size * 2 * 4 / 1e6:.2f} MB")

    p32 = tree.params.astype(np.int32)
    p32k = np.ascontiguousarray(tree.params[:, keep, :].astype(np.int32))

    ref = np.empty(n, dtype=np.int32)
    a = np.empty(n, dtype=np.int32)
    b = np.empty(n, dtype=np.int32)

    def bench(label, call, repeat=4):
        call()
        best = min((lambda: (lambda t0: (call(), time.perf_counter() - t0)[1])(
            time.perf_counter()))() for _ in range(repeat))
        print(f"  {label:34s} {best * 1e3:8.2f} ms  "
              f"{best / n * 1e6:7.2f} us/pt")
        return best

    base = bench("int64 params (current)", lambda: B._search_many_kernel(
        tree.params, tree.is_leaf, tree.child_off, tree.child_count,
        tree.children, tree.biome_ids, points, tree.root, -1, ref))
    i32 = bench("int32 params, 7 axes", lambda: _search_many_i32(
        p32, tree.is_leaf, tree.child_off, tree.child_count, tree.children,
        tree.biome_ids, points, tree.root, -1, a))
    drop = bench("int32 params, varying axes only", lambda: _search_many_drop_degenerate(
        p32k, tree.is_leaf, tree.child_off, tree.child_count, tree.children,
        tree.biome_ids, points, tree.root, -1, b, keep))

    print(f"\n  int32 identical:          {bool((a == ref).all())}")
    print(f"  dropped-axis identical:   {bool((b == ref).all())}")
    print(f"  int64->int32 speedup:     {base / i32:.2f}x")
    print(f"  + drop degenerate axes:   {base / drop:.2f}x")


if __name__ == "__main__":
    main()
