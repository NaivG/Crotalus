"""
Vanilla's eight mob categories, and the two numbers the spawner needs from each.

This is code-level data, not registry data: `assets/entities.json` gives every
entity type a `category` *name* ("CREATURE"), but the numbers behind that name
- the population cap, whether the category is hostile, how far away a mob of
it may be from a player before it despawns - are constants in vanilla's
`MobCategory` enum and appear in no report. The table below is transcribed from
Pumpkin's generated `MobCategory`
(`crates/pumpkin-data/src/generated/entity_type.rs`), which is itself generated
from vanilla's 774 report, with the two behaviours that are *not* vanilla
defaults called out at the bottom.

**A separate module because two modules need it.** `mob.py` reads
`despawn_distance` and `spawner.py` needs the whole table; putting it in
`spawner.py` would make `mob.py` import the module that imports it. The repo
already has this shape in `entity_type.py` - a table with no behaviour - so this
is the same answer one level down.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

#: How far away, in blocks, a mob may be from a player before vanilla removes
#: it rather than tracking it. Vanilla's `NO_DESPAWN_DISTANCE` is 32; every
#: category below except WATER_AMBIENT uses 128, which is why an unloaded cow
#: vanishes from your tab list but a loaded squid does not.
DEFAULT_DESPAWN_DISTANCE = 128

#: `MISC`'s cap. Vanilla's enum stores -1 to mean "no population limit", and
#: that is not a typo to be clamped away: a category with no cap must still
#: compare as uncapped, so the spawner skips the global test for it entirely
#: rather than multiplying -1 by anything.
UNLIMITED = -1


class MobCategory:
    """One spawn category: its cap, its temperament, its despawn radius."""

    __slots__ = ("name", "max_count", "is_friendly", "is_persistent",
                 "despawn_distance")

    def __init__(self, name: str, max_count: int, is_friendly: bool,
                 is_persistent: bool, despawn_distance: int) -> None:
        self.name = name
        self.max_count = max_count
        #: True for everything that is not MONSTER. Drives which entities
        #: count toward the local per-player cap and which players a spawn
        #: may happen near.
        self.is_friendly = is_friendly
        #: Persistent categories are not subject to the 32-block despawn when
        #: a player logs out and the mob is saved with the chunk. This server
        #: saves no entities at all yet, so nothing reads it - it is here so
        #: the table is the whole of vanilla's data rather than a subset that
        #: happens to suit round one.
        self.is_persistent = is_persistent
        self.despawn_distance = despawn_distance

    @property
    def biome_key(self) -> str:
        """The `assets/biome.json` key for this category.

        `entities.json` spells the category `UNDERGROUND_WATER_CREATURE` and
        `biome.json` keys the matching list `underground_water_creature`; the
        two files disagree only in case, so the conversion is mechanical for
        all eight. Asserted by `tools/check_mob_spawning.py` rather than
        assumed - a ninth category with an underscore-style name would break
        this quietly, returning no rows and spawning nothing.
        """
        return self.name.lower()

    def __repr__(self) -> str:
        return f"<MobCategory {self.name} max={self.max_count}>"


#: id -> category, in vanilla's declaration order. The id is the order the
#: vanilla enum declares them in, which is also the order a per-category count
#: list is indexed by.
CATEGORIES: Dict[int, MobCategory] = {}


def _register(category: MobCategory) -> MobCategory:
    CATEGORIES[len(CATEGORIES)] = category
    return category


MONSTER = _register(MobCategory("MONSTER", 70, False, False, 128))
CREATURE = _register(MobCategory("CREATURE", 10, True, True, 128))
AMBIENT = _register(MobCategory("AMBIENT", 15, True, False, 128))
AXOLOTLS = _register(MobCategory("AXOLOTLS", 5, True, False, 128))
UNDERGROUND_WATER_CREATURE = _register(
    MobCategory("UNDERGROUND_WATER_CREATURE", 5, True, False, 128)
)
WATER_CREATURE = _register(MobCategory("WATER_CREATURE", 5, True, True, 128))
WATER_AMBIENT = _register(MobCategory("WATER_AMBIENT", 20, True, False, 64))
MISC = _register(MobCategory("MISC", UNLIMITED, True, True, 128))

#: The categories whose ids are the position in `CATEGORIES`.
CATEGORY_LIST: Tuple[MobCategory, ...] = tuple(CATEGORIES[id] for id in sorted(CATEGORIES))

_BY_NAME: Dict[str, MobCategory] = {c.name: c for c in CATEGORY_LIST}


def mob_category(name: str) -> Optional[MobCategory]:
    """A category by the name `assets/entities.json` spells it in.

    Unknown names give None rather than a default: an entity type with a
    category this table has never heard of is a report mismatch, and quietly
    treating it as MISC would put it in the uncapped category where it can
    spawn without limit.
    """
    return _BY_NAME.get(name)


__all__ = [
    "CATEGORIES",
    "CATEGORY_LIST",
    "DEFAULT_DESPAWN_DISTANCE",
    "UNLIMITED",
    "MobCategory",
    "mob_category",
]
