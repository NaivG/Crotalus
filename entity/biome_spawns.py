"""
Which mobs a biome may spawn, reduced to the ones this server can actually build.

`assets/biome.json` lists, per biome and category, every type vanilla can spawn
there - a plains `creature` list is six rows: sheep, pig, chicken, cow, horse,
donkey. This server implements three of them. Drawing from the raw list would
either spawn a `Mob` base class with no behaviour or, worse, silently drop the
draw and under-fill the world. So the list is *filtered* before the draw, and
the filter is reported rather than hidden (see `stats`).

The filter is also what makes the weights mean anything. A row's `weight` is
its share of the category's total, so removing rows raises the others' shares
automatically: plains drops to sheep 12 / chicken 10 / cow 8 out of 30, and a
donkey - which had no chance anyway - does not take 1/6 of the passive spawns
just by being one of six rows.

**The draw is weighted; Pumpkin's is not.** `get_random_spawn_mob_at` ends in
`.choose(&mut rng())` over the row slice, which picks uniformly and never
reads `weight` - and cannot, because Pumpkin's generated `Spawner` struct
(`crates/pumpkin-data/src/generated/biome.rs:46`) has only `type`, `min_count`
and `max_count`, so the report's weight is dropped at parse time. Uniform over
plains' six rows would make a horse as likely as a sheep. The weights in the
report are hand-tuned (12/10/10/8/5/1), which is only meaningful if something
reads them, so this module does. See `MOB_SPAWNING_PLAN.md`.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from world.vanilla.biome import SPAWN_CATEGORIES, BiomeTable, SpawnerEntry


class BiomeSpawns:
    """The spawner rows of every biome, filtered to `allowed_types`.

    Read-only after construction and shared by every dimension actor, so it is
    built once and handed around - the filtering is pure data, and the tables
    behind it do not change while the server runs.
    """

    __slots__ = ("_rows", "_allowed", "_filtered_out", "_empty_categories",
                 "_rows_before")

    def __init__(self, allowed_types: Iterable[str],
                 table: Optional[BiomeTable] = None) -> None:
        self._allowed = frozenset(allowed_types)
        source = table if table is not None else _default_table()

        # biome -> category -> the surviving rows, weights untouched.
        self._rows: Dict[str, Dict[str, Tuple[SpawnerEntry, ...]]] = {}
        # How many rows each biome lost, summed over categories. Reported by
        # `stats` so "only three of the six plains creatures spawn" is a
        # number someone can read rather than a difference to notice.
        self._filtered_out: Dict[str, int] = {}
        # The raw row count, so `stats` can say "12 of 918 rows are live"
        # instead of only describing the survivors.
        self._rows_before = 0
        # Categories that lost every row everywhere: drawing from them is
        # pointless work, so the spawner skips them up front.
        self._empty_categories: Tuple[str, ...] = ()

        kept_anywhere: Dict[str, int] = {c: 0 for c in SPAWN_CATEGORIES}
        for biome in source.ids:
            per_biome: Dict[str, Tuple[SpawnerEntry, ...]] = {}
            dropped = 0
            for category in SPAWN_CATEGORIES:
                rows = source.spawners_for(biome, category)
                kept = tuple(row for row in rows if row.type_name in self._allowed)
                dropped += len(rows) - len(kept)
                kept_anywhere[category] += len(kept)
                per_biome[category] = kept
                self._rows_before += len(rows)
            self._rows[biome] = per_biome
            self._filtered_out[biome] = dropped

        self._empty_categories = tuple(
            c for c in SPAWN_CATEGORIES if kept_anywhere[c] == 0
        )

    def categories(self) -> Tuple[str, ...]:
        """The categories worth iterating, i.e. the ones with any row left.

        Iterating this instead of `SPAWN_CATEGORIES` is what keeps an
        all-monster world from paying for six empty draws per chunk per round.
        """
        return tuple(c for c in SPAWN_CATEGORIES if c not in self._empty_categories)

    def entries(self, biome: str, category: str) -> Tuple[SpawnerEntry, ...]:
        """The surviving rows of one biome's one category."""
        return self._rows.get(biome, {}).get(category, ())

    def draw(self, biome: str, category: str, unit: float) -> Optional[SpawnerEntry]:
        """One weighted pick from a biome's rows, or None if it has none.

        `unit` is the random float in [0, 1) rather than a draw from the
        spawner's own RNG, which is what makes the distribution testable: a
        caller can feed a fixed sequence and assert the exact rows it gets.
        The walk is cumulative over `weight`, so the surviving rows' shares
        are their file weights renormalised over the survivors' total.
        """
        rows = self.entries(biome, category)
        total = sum(row.weight for row in rows)
        if total <= 0:
            return None
        pick = unit * total
        for row in rows:
            pick -= row.weight
            if pick < 0.0:
                return row
        # `unit` in [0,1) and float subtraction can leave `pick` at exactly 0
        # on the last row; falling through returns the last row, which is the
        # same answer a closed-interval draw would give.
        return rows[-1]

    def has_type(self, biome: str, category: str, type_name: str) -> bool:
        """Whether this biome's category may spawn that type at all.

        The second half of the spawner's position check: a type absent here
        never gets as far as validating ground under it.
        """
        return any(row.type_name == type_name
                   for row in self.entries(biome, category))

    def filtered_types(self) -> Tuple[str, ...]:
        """The types this server draws on, sorted - what the check asserts."""
        return tuple(sorted(self._allowed))

    def stats(self) -> Dict[str, object]:
        """What the draw is working with, for `StatsMsg` and the check tools.

        `rows_before` is the raw list length summed over biomes and
        categories, so the two numbers together say exactly how much of the
        report is live: 65 biomes x 8 categories is a lot of rows, and only
        the passive-creature ones survive the filter.
        """
        rows_after = 0
        per_category: Dict[str, int] = {}
        for per_biome in self._rows.values():
            for category, rows in per_biome.items():
                rows_after += len(rows)
                per_category[category] = per_category.get(category, 0) + len(rows)
        return {
            "types": list(self._allowed),
            "categories": list(self.categories()),
            "empty_categories": list(self._empty_categories),
            "rows_before": self._rows_before,
            "rows_after_filter": rows_after,
            "per_category_rows": per_category,
            "filtered_out_total": sum(self._filtered_out.values()),
        }

    def __repr__(self) -> str:
        return f"<BiomeSpawns types={len(self._allowed)} categories={len(self.categories())}>"


def _default_table() -> BiomeTable:
    from world.vanilla.biome import biome_table

    return biome_table()


def biome_spawns(allowed_types: Sequence[str]) -> BiomeSpawns:
    """A `BiomeSpawns` over `allowed_types`, cached per frozenset.

    One instance per distinct set: the tables are read-only, and the actor
    threads share these without copying.
    """
    key = frozenset(allowed_types)
    cached = _CACHE.get(key)
    if cached is None:
        cached = BiomeSpawns(key)
        _CACHE[key] = cached
    return cached


_CACHE: Dict[frozenset, BiomeSpawns] = {}


__all__ = ["BiomeSpawns", "biome_spawns"]
