"""
Picking somewhere plausible to walk to.

Vanilla's `WanderAroundGoal` does not roll a direction and walk. It rolls ten
of them, *scores* each candidate, and takes the best - and the scoring is what
stops a cow on a cliff edge choosing the empty air beyond it. The whole file is
that routine and its three helpers.

`generate_random_pos` is the shape:

    best = max over 10 attempts of (candidate, weight)

with a candidate that fails any check counting as absent rather than as zero,
so ten failures return None and the goal does not start. Returning None is
load-bearing: `WanderAroundGoal.can_start` reads it as "no", so a mob on a
pillar in the middle of the ocean simply does not wander rather than wandering
into the sea.

`generate_random_pos_toward_direction` is where a direction becomes a block:
the mob's own position plus the direction, floored. The direction itself is a
uniform sample in `[-horizontal, horizontal]` on x and z and
`[-vertical, vertical]` on y, so a walk is isotropic - a cow does not prefer
north.

**What is not here, and what it costs.** Vanilla passes each candidate through
`goal_utils.has_malus`, which asks the pathfinder what it thinks of standing
there. Without a pathfinder there is nothing to ask, and the check is absent
rather than stubbed: the survival of the mob is now decided by
`is_ground_stable` alone. On flat plains - where these three mobs spawn - that
is the whole rule. Beside a one-block drop, vanilla's walker evaluates the
block below and the drop, and this one evaluates the block below only. A cow
may therefore step off a ledge one block high that vanilla would refuse. It
will not step into a lake or off a cliff *into* nothing, because a candidate
with no solid floor scores nothing and ten of those return None.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

__all__ = [
    "RANDOM_POS_ATTEMPTS",
    "generate_random_direction",
    "generate_random_pos",
    "generate_random_pos_toward_direction",
    "random_pos_near",
]

#: Vanilla's `PathNavigation.RANDOM_POS_ATTEMPTS`. Ten is not tuned; it is the
#: number that makes "roll ten, take the best" feel like a choice rather than
#: like a coin flip.
RANDOM_POS_ATTEMPTS = 10


def generate_random_direction(rng, horizontal_dist: int,
                              vertical_dist: int) -> Tuple[int, int, int]:
    """One isotropic integer direction within the given radii.

    `rng.nextBetween(-h, h)` on each axis, not `rng.next_f32() * 2 * h`:
    vanilla's `RandomSource.nextInt(bound)` is the one the worldgen stream is
    defined in terms of, and a bounded integer draw is also the one that does
    not care how `next_f32` rounds.
    """
    return (
        rng.next_between(-horizontal_dist, horizontal_dist),
        rng.next_between(-vertical_dist, vertical_dist),
        rng.next_between(-horizontal_dist, horizontal_dist),
    )


def generate_random_pos_toward_direction(mob, xz_dist: float, rng,
                                         direction: Tuple[int, int, int]
                                         ) -> Tuple[int, int, int]:
    """The block a direction asks the mob to walk to.

    Vanilla's `generateRandomPosTowardDirection` also biases the offset *away*
    from a mob's `positionTarget` when it has one, so a restricted mob is
    pushed back toward the middle of its range. These mobs have no position
    target - that is `Mob.setPosTarget`, which nothing calls - so the bias is
    absent, and `xz_dist` is carried only so the signature still says what the
    caller's `horizontal_dist` means.
    """
    del xz_dist, rng
    pos = mob.position
    return (
        math.floor(pos.x + direction[0]),
        math.floor(pos.y + direction[1]),
        math.floor(pos.z + direction[2]),
    )


def generate_random_pos(supply: Callable[[], Optional[Tuple[int, int, int]]],
                        weight: Callable[[Tuple[int, int, int]], float]
                        ) -> Optional[Tuple[float, float, float]]:
    """Ten candidates, best weight wins, as `(x + 0.5, y, z + 0.5)`.

    The returned point is the *centre of the block* at feet height. Vanilla
    does the same (`generateRandomPos` adds the 0.5), and it is why a cow walks
    to the middle of a block rather than to its corner: a mob whose target is
    `(x, y, z)` exactly is standing on the seam and its bounding box straddles
    two columns, which the physics sweep resolves differently depending on
    which way it was facing.

    A candidate whose weight is 0 is kept, not discarded - vanilla compares
    `weight > bestWeight` starting from -Infinity, so the first candidate always
    wins if it is merely legal. That is why a mob will sometimes walk onto a
    block it likes no better than any other.
    """
    best_pos = None
    best_weight = float("-inf")
    for _ in range(RANDOM_POS_ATTEMPTS):
        candidate = supply()
        if candidate is None:
            continue
        candidate_weight = weight(candidate)
        if candidate_weight > best_weight:
            best_weight = candidate_weight
            best_pos = candidate
    if best_pos is None:
        return None
    return (best_pos[0] + 0.5, float(best_pos[1]), best_pos[2] + 0.5)


def random_pos_near(mob, context, horizontal_dist: int,
                    vertical_dist: int
                    ) -> Optional[Tuple[float, float, float]]:
    """The vanilla `default_random_pos::get_pos` for a mob.

    The four checks vanilla applies, and the three that have no answer here:

    * `is_outside_limits` - inside the build height. Kept: a candidate above
      the sky limit is a cow trying to path onto a floating island it cannot
      be seen from.
    * `is_restricted` - inside a `positionTarget` radius. Always true (not
      restricted): no mob has a position target yet.
    * `is_not_stable` - something solid below. Kept, and it does the real work.
    * `has_malus` - what the pathfinder thinks of the destination. Absent;
      see the module docstring.
    """
    rng = mob.random

    def supply():
        direction = generate_random_direction(rng, horizontal_dist,
                                              vertical_dist)
        candidate = generate_random_pos_toward_direction(
            mob, float(horizontal_dist), rng, direction)
        if not (context.min_y <= candidate[1] < context.max_y):
            return None
        if not context.is_ground_stable(*candidate):
            return None
        return candidate

    return generate_random_pos(
        supply,
        lambda candidate: 1.0 if context.is_ground_stable(*candidate) else 0.0,
    )
