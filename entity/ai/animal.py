"""
The three animals, as data.

Vanilla's `CowEntity`, `SheepEntity` and `ChickenEntity` carry eight or nine
goals each, and comparing the three lists shows that they are the *same* list:
`swim, escapeDanger(speed), breed, tempt(speed), followParent(speed),
wander, lookAt, lookAround`, plus `eatGrass` on the sheep. Three of those
goals do not exist on this server yet (`goals.py` says why), so what is left
per type is two floats and a name.

Putting them here rather than as three near-identical `add_goal` lists is the
point of the round: the *shape* of a passive animal's behaviour is one list,
written once, and the three mobs are three rows in it. When `breed` lands, it
is one line in `build_animal_ai` and a speed per row - not three edits that can
drift, which is how vanilla itself ended up with `Chicken`'s
`escapeDanger(1.4)` against `Cow`'s `2.0` and everyone being quietly unsure why
a chicken bolts faster than a cow.

`speed_modifier` values are vanilla's own constructor arguments, so a cow
panicking walks at 2.0 x `movement_speed` and a chicken at 1.4 x. They are
multipliers on the attribute, not velocities: `MoveControl` applies them, and
the ground-speed factor turns the result into blocks per tick.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from entity.ai import MobAI

__all__ = [
    "CHICKEN_PROFILE",
    "COW_PROFILE",
    "PIG_PROFILE",
    "RABBIT_PROFILE",
    "SHEEP_PROFILE",
    "AnimalProfile",
    "build_animal_ai",
    "build_rabbit_ai",
]


@dataclass(frozen=True)
class AnimalProfile:
    """Everything that distinguishes one passive animal's AI from another's.

    `wander_speed` is 1.0 for all three in vanilla and is here so the field is
    visible rather than absent - a future mob with a faster stroll (a pig, at
    1.0 too, or a rabbit at 1.0 with a much smaller `movement_speed`) has one
    place to set it.
    """

    name: str
    danger_speed: float
    """`EscapeDangerGoal` speed modifier - how fast the mob bolts when hurt."""

    wander_speed: float
    """`WanderAroundGoal` speed modifier."""

    tempt_items: Tuple[str, ...]
    """What tempts this mob. Carried for the plan's benefit, not used yet:
    `TemptGoal` is not implemented, and these are the vanilla item sets
    (`#minecraft:cow_food` is wheat; the chicken's are its six seed types).
    Having them written down is what makes adding the goal a lookup rather
    than a search."""


COW_PROFILE = AnimalProfile(
    name="cow",
    danger_speed=2.0,
    wander_speed=1.0,
    tempt_items=("minecraft:wheat",),
)

SHEEP_PROFILE = AnimalProfile(
    name="sheep",
    danger_speed=1.25,
    wander_speed=1.0,
    tempt_items=("minecraft:wheat",),
)

CHICKEN_PROFILE = AnimalProfile(
    name="chicken",
    danger_speed=1.4,
    wander_speed=1.0,
    tempt_items=(
        "minecraft:wheat_seeds",
        "minecraft:melon_seeds",
        "minecraft:pumpkin_seeds",
        "minecraft:beetroot_seeds",
        "minecraft:torchflower_seeds",
        "minecraft:pitcher_pod",
    ),
)

#: Vanilla's `Pig.createAttributes` row: movement_speed 0.25 (with chicken),
#: escapeDanger 1.5 (chicken is 1.4; the 0.1 gap is the whole reason a startled
#: pig can catch up with a startled chicken). Tempt items are the three
#: `minecraft:pig_food` members; the carrot-on-a-stick variant is a separate
#: held-item check that `TemptGoal` does not yet handle.
PIG_PROFILE = AnimalProfile(
    name="pig",
    danger_speed=1.5,
    wander_speed=1.0,
    tempt_items=(
        "minecraft:carrot",
        "minecraft:potato",
        "minecraft:beetroot",
    ),
)

#: Vanilla's `Rabbit.createAttributes` row: movement_speed 0.3, escapeDanger
#: 1.7. The 0.3 row is the rabbit's alone - no other passive animal shares
#: it, which is why a panicked rabbit is the hardest passive mob to corner
#: even though its body is the smallest. Tempt items are the rabbit-flavoured
#: vegetables (carrot, dandelion, golden carrot); `TemptGoal` is what reads
#: them and is not built yet.
RABBIT_PROFILE = AnimalProfile(
    name="rabbit",
    danger_speed=1.7,
    wander_speed=1.0,
    tempt_items=(
        "minecraft:carrot",
        "minecraft:dandelion",
        "minecraft:golden_carrot",
    ),
)


def build_animal_ai(mob, profile: AnimalProfile) -> MobAI:
    """The shared passive-animal goal list, in vanilla's priority order.

    Five goals, ranked:

    | pri | goal           | controls  | why it is here          |
    |-----|----------------|-----------|-------------------------|
    | 0   | swim           | JUMP      | a mob in water cannot    |
    |     |                |           | decide anything else    |
    | 1   | escapeDanger   | MOVE      | being hurt outranks      |
    |     |                |           | everything but drowning |
    | 5   | wander         | MOVE      | the default: if nothing |
    |     |                |           | else wants to walk, walk|
    | 6   | lookAt(player) | LOOK      | runs *alongside* wander |
    | 7   | lookAround     | MOVE+LOOK | holding MOVE stops the  |
    |     |                |           | wander, which is the    |
    |     |                |           | point of it             |

    The gaps in the numbering are the goals that are not built yet, and they
    are left as gaps rather than renumbered: priorities are compared as
    integers and the next goal to arrive has to slot in where vanilla put it,
    not where the current implementation would put it.
    """
    from entity.ai.goals import (
        EscapeDangerGoal,
        LookAtEntityGoal,
        RandomLookAroundGoal,
        SwimGoal,
        WanderAroundGoal,
    )

    return MobAI(mob, [
        (0, SwimGoal()),
        (1, EscapeDangerGoal(profile.danger_speed)),
        (5, WanderAroundGoal(profile.wander_speed)),
        (6, LookAtEntityGoal()),
        (7, RandomLookAroundGoal()),
    ])


def build_rabbit_ai(mob, profile: AnimalProfile) -> MobAI:
    """The rabbit's goal list - the first passive animal whose shape diverges.

    Same five goals as the shared list, plus `RabbitHopGoal` at priority 5.
    The hop goal asks for `JUMP` only - a different control from
    `WanderAroundGoal`'s `MOVE` - so the selector lets both run at the same
    priority: the rabbit walks toward its wander target *and* hops over the
    ground in the process. That is what a rabbit on a field looks like, and
    what is lost in a vanilla `MoveToBlockGoal`-based design (one hop per
    target, picked in advance) is precisely the pathfinder we do not have -
    so "hop while walking" is the closest faithful version on a flat plain,
    and the corner cases (rabbit cannot hop *over* a fence) are the same
    gap the cow's straight-line wander takes.
    """
    from entity.ai.goals import (
        EscapeDangerGoal,
        LookAtEntityGoal,
        RabbitHopGoal,
        RandomLookAroundGoal,
        SwimGoal,
        WanderAroundGoal,
    )

    return MobAI(mob, [
        (0, SwimGoal()),
        (1, EscapeDangerGoal(profile.danger_speed)),
        (5, WanderAroundGoal(profile.wander_speed)),
        (5, RabbitHopGoal()),
        (6, LookAtEntityGoal()),
        (7, RandomLookAroundGoal()),
    ])
