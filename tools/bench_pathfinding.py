"""Search cost, per world. `py -3.14t tools/bench_pathfinding.py`

One number matters for deciding whether the pathfinder is affordable: how
long a single `compute_path` takes, because that is what a mob pays on the
ticks it repaths. Two things keep it from being paid constantly, and both
are structural rather than tuned:

* `repath_cooldown` (15 ticks) means a mob that already has a path does not
  search again for a third of a second, whatever it is doing.
* The search is bounded by `follow_range` (16 for these animals), so a mob's
  plan covers at most sixteen blocks of walking however far away the goal is.
  The `wide` row below is deliberately *outside* that bound and is here to
  show the cost of the worst case rather than the expected one.

So: measure the rows a cow actually produces. `flat` and `comb` are what a
plains mob spends; `blocked` is what a mob in a bad spot spends, and it is
cheap because an unreachable target exhausts its frontier quickly.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path as FsPath

sys.path.insert(0, str(FsPath(__file__).resolve().parents[1]))

from tools.check_pathfinding import (  # noqa: E402
    FakeMob,
    Ground,
    search,
)

REPEATS = 30


def comb() -> Ground:
    """A picket fence of walls - the shape that forces the search to detour."""
    ground = Ground(radius=20)
    for z in range(-20, 21):
        if z % 2 == 0:
            ground.wall(4, z, 1, 3)
    return ground


def boxes() -> Ground:
    """A sealed room - the unreachable case."""
    ground = Ground(radius=20)
    for x in range(-8, 9, 4):
        for z in range(-8, 9):
            ground.wall(x, z, 1, 3)
    for z in range(-8, 9, 4):
        for x in range(-8, 9):
            ground.wall(x, z, 1, 3)
    return ground


CASES = (
    # label, world, destination, follow_range, what a cow would be doing
    ("flat", lambda: Ground(radius=20), (12.5, 1.0, 6.5), 16.0,
     "a wander leg, at the mob's real radius"),
    ("comb", comb, (12.5, 1.0, 6.5), 16.0,
     "a detour, at the mob's real radius"),
    ("blocked", boxes, (14.5, 1.0, 14.5), 16.0,
     "no route at all - the cheapest case"),
    ("wide", lambda: Ground(radius=48), (40.5, 1.0, 3.5), 64.0,
     "the worst case: 2.5x the radius a cow uses"),
)


def main() -> int:
    print(f"pathfinding: {REPEATS} searches per row\n")
    print(f"{'case':10s} {'ms/search':>10s} {'nodes':>6s} {'reached':>8s}   what")
    for label, build, dest, follow_range, note in CASES:
        ground = build()
        mob = FakeMob(ground, follow_range=follow_range)
        path = search(mob, dest)
        start = time.perf_counter()
        for _ in range(REPEATS):
            search(mob, dest)
        elapsed = (time.perf_counter() - start) / REPEATS
        print(f"{label:10s} {elapsed * 1000:10.2f} {len(path.nodes):6d} "
              f"{str(path.can_reach()):>8s}   {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())