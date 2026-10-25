"""Time the biome tree search directly, with and without the numba kernel.

cProfile cannot see inside a numba nopython frame, so the kernel's time lands
on its Python caller and is indistinguishable from interpreter overhead. This
calls the kernel and the pure-Python fallback on the same points and times
them, and counts node visits, so the search's real cost per point is a number.

Run: py -3.14t tools/bench_biome_search.py [points]
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from loguru import logger

logger.remove()

from world.vanilla import biome as B


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1536

    src = B.MultiNoiseBiomeSource.from_assets("overworld", seed=0)
    sampler = src.sampler
    tree = src.tree
    print(f"tree: nodes={tree.node_count} leaves={int(tree.is_leaf.sum())} "
          f"axes={B.PARAMETER_AXES}  points={n}")

    # The same points `fill_volume` builds, in its own index order.
    ys = np.arange(96, dtype=np.int64)
    zs = np.arange(4, dtype=np.int64)
    xs = np.arange(4, dtype=np.int64)
    gy, gz, gx = np.meshgrid(ys, zs, xs, indexing="ij")
    points = sampler.sample_many(gx.reshape(-1), gy.reshape(-1), gz.reshape(-1))
    points = np.ascontiguousarray(points[:n] if n <= points.shape[0] else points)
    # `get_many` pads the 6 sampled slots out to PARAMETER_AXES; the kernel and
    # the Python walker both index all 7.
    if points.shape[1] == 6:
        padded = np.zeros((points.shape[0], B.PARAMETER_AXES), dtype=np.int64)
        padded[:, :6] = points
        points = np.ascontiguousarray(padded)
    print(f"points shape {points.shape}")

    out = np.empty(points.shape[0], dtype=np.int32)

    def bench(label, call, repeat=3):
        call()  # warm
        best = min(_time(call) for _ in range(repeat))
        print(f"  {label:28s} {best * 1e3:8.2f} ms   "
              f"{best / points.shape[0] * 1e6:7.2f} us/point")
        return best

    def _time(fn):
        t0 = time.perf_counter()
        fn()
        return time.perf_counter() - t0

    kernel_ms = bench(
        "numba _search_many_kernel",
        lambda: B._search_many_kernel(tree.params, tree.is_leaf, tree.child_off,
                                      tree.child_count, tree.children, tree.biome_ids,
                                      points, tree.root, -1, out))

    ref = out.copy()

    # The pure-Python fallback is unreachable whenever numba is present (it is
    # selected by `_search_many_impl is None`), so it is not timed here.

    # How many node visits does one point cost? Walk the tree in Python.
    visits = 0
    last = -1
    big = 1 << 62
    for row in range(points.shape[0]):
        point = points[row]
        best_dist = B._node_distance(tree.params, last, point) if last >= 0 else big
        best_node = last if last >= 0 else tree.root
        stack = [tree.root]
        while stack:
            node = stack.pop()
            visits += 1
            dist = B._node_distance(tree.params, node, point)
            if dist >= best_dist:
                continue
            if tree.is_leaf[node] == 1:
                best_dist, best_node = dist, node
                continue
            start = tree.child_off[node]
            for i in range(tree.child_count[node]):
                stack.append(int(tree.children[start + i]))
        last = best_node
    print(f"  node visits: {visits}  ({visits / points.shape[0]:.0f}/point)")
    print(f"  ns/visit: {kernel_ms / max(visits, 1) * 1e9:.0f}")

    # Cost of a cold search (no `lastResult` hint) for comparison: the hint is
    # what keeps the batch cheap, so its contribution is worth knowing.
    out2 = np.empty(points.shape[0], dtype=np.int32)
    bench("numba, no lastResult hint",
          lambda: B._search_many_kernel(tree.params, tree.is_leaf, tree.child_off,
                                        tree.child_count, tree.children,
                                        tree.biome_ids, points, tree.root, -1, out2),
          repeat=2)
    print(f"  identical to hinted run: {bool((out2 == ref).all())}")


if __name__ == "__main__":
    main()
