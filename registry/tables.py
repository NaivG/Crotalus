"""
Data tables - everything in `assets/` that is not a block, item, recipe, tag or
synced registry.

The registry proper is blocks + items + recipes + tags + synced registries.
Everything else Pumpkin ships is still registry-shaped data the server needs
sooner or later, and loading it here means "the complete registry" is actually
complete: fluids and entity types (whose ordering tags depend on), attributes,
mob effects, damage types, enchantments, potions, dimensions, biomes, sounds,
particles, translations, fuels, worldgen settings, ...

Two shapes are used, and which one applies is data, not taste:

* `IndexedRegistry` - the file records a numeric `id` (items, blocks, fluids,
  entities, attributes, effects, damage types, potions) or is a plain list whose
  position *is* the id (game events, particles, sounds). Those ids go on the
  wire, so they are kept as declared and checked for contiguity.
* `Registry` - a name -> payload map where no numeric id exists (enchantments,
  biomes, dimensions, configured features, carvers, noise parameters, ...).

The huge worldgen files are loaded eagerly too. Together they cost ~0.5s and
~90MB, and paying that once at boot is what lets world generation read a single
in-memory source of truth instead of re-parsing JSON on a worker thread.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from registry.core import (
    IndexedRegistry,
    Registry,
    ValidationError,
    canonical_id,
)

# (attribute, asset file, registry id, shape)
#   shape "records"  : a list of {id, name, ...} records        -> IndexedRegistry
#   shape "map-ids"  : {name: {id, ...}}                          -> IndexedRegistry
#   shape "map"      : {name: {...}}                              -> Registry
#   shape "names"    : [name, ...], position is the numeric id     -> IndexedRegistry
INDEXED_REGISTRIES: Tuple[Tuple[str, str, str, str], ...] = (
    ("fluids", "fluids.json", "minecraft:fluid", "records"),
    ("entity_types", "entities.json", "minecraft:entity_type", "map-ids"),
    ("game_events", "game_event.json", "minecraft:game_event", "names"),
    ("attributes", "attributes.json", "minecraft:attribute", "map-ids"),
    ("mob_effects", "effect.json", "minecraft:mob_effect", "map-ids"),
    ("damage_types", "damage_type.json", "minecraft:damage_type", "map-ids"),
    ("potions", "potion.json", "minecraft:potion", "map-ids"),
    ("particles", "particles.json", "minecraft:particle_type", "names"),
    ("sounds", "sounds.json", "minecraft:sound_event", "names"),
)

# (attribute, asset file) for name -> payload tables with no numeric id.
NAMED_REGISTRIES: Tuple[Tuple[str, str], ...] = (
    ("enchantments", "enchantments.json"),
    ("biomes", "biome.json"),
    ("dimensions", "dimension.json"),
    ("configured_features", "configured_features.json"),
    ("placed_features", "placed_feature.json"),
    ("carvers", "carver.json"),
    ("noise_parameters", "noise_parameters.json"),
    ("chunk_gen_settings", "chunk_gen_settings.json"),
    ("density_functions", "density_function.json"),
    ("density_function_tests", "density_function_tests.json"),
    ("multi_noise_biome_tree", "multi_noise_biome_tree.json"),
    ("jukebox_songs", "jukebox_song.json"),
    ("message_types", "message_type.json"),
    ("meta_data_types", "meta_data_type.json"),
    ("tracked_data", "tracked_data.json"),
    ("entity_tracking", "entity_tracking.json"),
    ("data_components", "data_component.json"),
)

# (attribute, asset file) for plain payloads that are not name -> value maps:
# lookups keyed by item id, indexed constants, or a list.
RAW_TABLES: Tuple[Tuple[str, str], ...] = (
    ("translations", "en_us.json"),
    ("fuels", "fuels.json"),
    ("spawn_eggs", "spawn_egg.json"),
    ("composter_chances", "composter_increase_chance.json"),
    ("recipe_remainders", "recipe_remainder.json"),
    ("flower_pot_transformations", "flower_pot_transformations.json"),
    ("game_rules", "game_rules.json"),
    ("packets", "packets.json"),
    ("screens", "screens.json"),
    ("scoreboard_display_slots", "scoreboard_display_slot.json"),
    ("sound_categories", "sound_category.json"),
    ("chunk_statuses", "chunk_status.json"),
    ("entity_poses", "entity_pose.json"),
    ("entity_statuses", "entity_statuses.json"),
    ("loot_condition_types", "loot_condition_types.json"),
    ("potion_brewing", "potion_brewing.json"),
    ("chunk_sample", "chunk.json"),
)

# Files that are deliberately not loaded, with the reason. `check_registry.py`
# fails if an asset file is in neither this set nor one of the plans above, so
# new data cannot be dropped on the floor unnoticed.
IGNORED_FILES: Dict[str, str] = {
    "blocks.json": "loaded by registry.blocks",
    "properties.json": "loaded by registry.blocks",
    "items.json": "loaded by registry.items",
    "recipes.json": "loaded by registry.recipes",
    "tags.json": "loaded by registry.tags",
    "synced_registries.json": "loaded by registry.synced",
    # Golden values for `tools/check_density_functions.py`, read by that tool
    # rather than by the server. It was on disk without being accounted for,
    # which made `tools/check_registry.py` fail on every run - a real gate
    # buried under a false alarm.
    "vein_golden_values.json": "reference data for tools/check_density_functions.py",
}


@dataclass(slots=True)
class DataTable:
    """A loaded asset file, with the accounting needed to trust it."""

    name: str
    source: str
    data: Any
    registry: Optional[Registry] = None

    @property
    def count(self) -> int:
        """Number of top-level entries, however the file is shaped."""
        if isinstance(self.data, (dict, list)):
            return len(self.data)
        return 1

    def __repr__(self) -> str:
        return f"<DataTable {self.name} ({self.source}) entries={self.count}>"

    def get(self, key: str, default: Any = None) -> Any:
        if isinstance(self.data, dict):
            return self.data.get(key, default)
        return default

    def keys(self) -> List[str]:
        return list(self.data) if isinstance(self.data, dict) else []


class DataTables:
    """Every auxiliary table, loaded from `assets/`."""

    def __init__(self) -> None:
        self.tables: Dict[str, DataTable] = {}

    # ----- identity -----

    def __len__(self) -> int:
        return len(self.tables)

    def __iter__(self):
        return iter(self.tables)

    def __getitem__(self, name: str) -> DataTable:
        try:
            return self.tables[name]
        except KeyError:
            raise KeyError(
                f"no data table {name!r}; loaded tables are {sorted(self.tables)}"
            ) from None

    def get(self, name: str) -> Optional[DataTable]:
        return self.tables.get(name)

    # ----- mapping access -----
    #
    # `__getattr__` is what makes `tables.biomes` read naturally, but it also
    # swallows a missing *method*: `tables.values()` used to raise "no data
    # table 'values'", which is a confusing way to learn that there is no
    # `values`. These three are real methods so the container behaves like the
    # mapping it looks like.

    def keys(self) -> List[str]:
        return list(self.tables)

    def values(self) -> List[DataTable]:
        return list(self.tables.values())

    def items(self) -> List[Tuple[str, DataTable]]:
        return list(self.tables.items())

    def __repr__(self) -> str:
        return f"<DataTables tables={len(self.tables)}>"

    def __getattr__(self, name: str) -> DataTable:
        """Attribute access to a table: `tables.fluids` -> `DataTable`.

        The table's payload and its numeric-id registry hang off it, so both
        `tables.fluids.data` and `tables.fluids.registry.numeric_id("water")`
        read naturally.
        """
        tables = self.__dict__.get("tables", {})
        table = tables.get(name)
        if table is None:
            raise AttributeError(f"no data table {name!r}")
        return table

    # ----- loading -----

    def load(self, assets_dir: Path, read_json) -> None:
        """Load every planned file from `assets_dir`.

        `read_json(filename) -> Any` is supplied by the loader so file reads and
        their timings stay in one place.
        """
        for attribute, filename, registry_id, shape in INDEXED_REGISTRIES:
            payload = read_json(filename)
            registry = self._build_indexed(registry_id, shape, payload, filename)
            self.tables[attribute] = DataTable(attribute, filename, payload, registry)

        for attribute, filename in NAMED_REGISTRIES:
            payload = read_json(filename)
            registry = Registry(
                canonical_id(filename.removesuffix(".json")),
                title=attribute,
                description=f"Name -> payload from {filename}",
            )
            for key, value in payload.items():
                registry.add(key, value)
            self.tables[attribute] = DataTable(attribute, filename, payload, registry)

        for attribute, filename in RAW_TABLES:
            payload = read_json(filename)
            self.tables[attribute] = DataTable(attribute, filename, payload)

        logger.debug(f"Data tables: {len(self.tables)} tables loaded")

    @staticmethod
    def _build_indexed(
        registry_id: str,
        shape: str,
        payload: Any,
        filename: str,
    ) -> IndexedRegistry:
        registry: IndexedRegistry = IndexedRegistry(
            registry_id,
            title=registry_id,
            description=f"Numeric ids from {filename}",
        )
        expected = 0

        if shape == "records":
            for record in payload:
                name = record["name"]
                numeric_id = int(record["id"])
                if numeric_id != expected:
                    raise ValidationError(
                        f"{filename}: {name} has id {numeric_id}, expected {expected}"
                    )
                registry.add(name, record, numeric_id=numeric_id)
                expected += 1
        elif shape == "map-ids":
            for name, record in payload.items():
                numeric_id = int(record["id"])
                if numeric_id != expected:
                    raise ValidationError(
                        f"{filename}: {name} has id {numeric_id}, expected {expected}"
                    )
                registry.add(name, record, numeric_id=numeric_id)
                expected += 1
        elif shape == "names":
            for index, name in enumerate(payload):
                registry.add(name, index, numeric_id=index)
        else:  # pragma: no cover - the plan above only uses the three shapes
            raise ValueError(f"unknown table shape {shape!r}")

        return registry

    # ----- reporting -----

    def summary(self) -> Dict[str, int]:
        """`{table: entry count}` for the startup log and check tool."""
        return {name: table.count for name, table in sorted(self.tables.items())}

    def consumed_files(self) -> set:
        """Asset filenames this module loads."""
        return {table.source for table in self.tables.values()}

    def total_entries(self) -> int:
        return sum(table.count for table in self.tables.values())


def known_files() -> set:
    """Every asset filename the registry accounts for."""
    consumed = {filename for _, filename, _, _ in INDEXED_REGISTRIES}
    consumed |= {filename for _, filename in NAMED_REGISTRIES}
    consumed |= {filename for _, filename in RAW_TABLES}
    consumed |= set(IGNORED_FILES)
    return consumed


__all__ = [
    "DataTable",
    "DataTables",
    "INDEXED_REGISTRIES",
    "NAMED_REGISTRIES",
    "RAW_TABLES",
    "IGNORED_FILES",
    "known_files",
]
