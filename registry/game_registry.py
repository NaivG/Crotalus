"""
Game registry - the one object the rest of the server talks to.

`GameRegistry` owns every registry this server has, answers the cross-registry
questions ("what numeric id does this tag entry have?", "what is the wire
payload for the synced registries?"), and carries the freeze latch that stops
ids from moving once the world exists.

Loading lives in `registry/loader.py`; this module is the data holder plus the
global accessor:

    from registry import init_registries, get_game_registry

    init_registries()            # explicit: server startup, before the world
    registry = get_game_registry()   # lazy fallback for anything that imports
                                     # a registry consumer without booting first

The lazy path exists so a tool or a test can `from world import World` without
knowing about boot order. It logs a warning, because a server that reaches world
generation through it has lost the guarantee the explicit call provides.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from registry.blocks import BlockRegistry
from registry.core import Registry, RegistryFrozenError, canonical_id
from registry.items import ItemRegistry
from registry.plugin import PluginManager
from registry.recipes import RecipeRegistry
from registry.synced import SyncedRegistries
from registry.tables import DataTables
from registry.tags import ResolutionReport, TagRegistry
from registry.worldgen import WorldgenData

#: Assets shipped with the server (Pumpkin's data at its 1.21.11 commit).
DEFAULT_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

#: The registries whose entry order the client learns from the wire and whose
#: numeric ids tags can therefore refer to without a local data file.
STATIC_ID_REGISTRIES = (
    "minecraft:block",
    "minecraft:item",
    "minecraft:fluid",
    "minecraft:entity_type",
    "minecraft:game_event",
)


class GameRegistry:
    """Every registry, plus the lookups that cross them."""

    def __init__(self, assets_dir: Optional[Path] = None) -> None:
        self.assets_dir = Path(assets_dir) if assets_dir else DEFAULT_ASSETS_DIR

        self.blocks = BlockRegistry()
        self.items = ItemRegistry()
        self.recipes = RecipeRegistry()
        self.tags = TagRegistry()
        self.synced = SyncedRegistries()
        self.tables = DataTables()
        self.plugins = PluginManager()
        #: Typed views over the twelve worldgen assets. Built by
        #: `load_registries` after plugins and before the freeze; `None` on a
        #: directly-constructed registry (tests) until it is.
        self.worldgen: Optional[WorldgenData] = None

        self.loaded_at: float = 0.0
        self.load_seconds: float = 0.0
        self.timings: Dict[str, float] = {}
        self.problems: List[str] = []
        self.validation_problems: List[str] = []
        self.tag_report: Optional[ResolutionReport] = None

        self._frozen = False
        self._frozen_reason = ""
        self._lock = threading.RLock()

    # ----- identity -----

    @property
    def frozen(self) -> bool:
        return self._frozen

    def __repr__(self) -> str:
        state = "frozen" if self._frozen else "open"
        return (f"<GameRegistry blocks={len(self.blocks.blocks)} "
                f"states={self.blocks.total_states} items={len(self.items)} "
                f"recipes={len(self.recipes)} synced={len(self.synced)} {state}>")

    # ----- registry lookup -----

    def find_registry(self, registry_id: str) -> Optional[Registry]:
        """The container for a registry id, whichever kind it is.

        Synced registries are checked before the server-side tables, because for
        an id that is both (damage types, enchantments, biomes, ...) the synced
        one is the ordering tags and the client agree on - the same container
        `numeric_id` resolves against.
        """
        canonical = canonical_id(registry_id)
        if canonical == "minecraft:block":
            return self.blocks.blocks
        if canonical == "minecraft:item":
            return self.items.items
        if canonical == "minecraft:recipe":
            return self.recipes.recipes

        synced = self.synced.get_registry(canonical)
        if synced is not None:
            return synced

        for table in (
            "fluids",
            "entity_types",
            "game_events",
            "attributes",
            "mob_effects",
            "damage_types",
            "potions",
            "particles",
            "sounds",
        ):
            data_table = self.tables.get(table)
            if data_table is not None and data_table.registry is not None:
                if data_table.registry.id == canonical:
                    return data_table.registry

        return self.tags.registries.get(canonical)

    def numeric_id(self, registry_id: str, entry_id: str) -> Optional[int]:
        """The id an entry has inside a registry, or None when unknown.

        This is what turns a tag entry into a number for the Update Tags packet.
        It is deliberately not the same as "does this entry exist": an entry can
        exist on the server and still have no id the client would agree with
        (a plugin block, for example), and a tag must never be built from those.
        """
        canonical = canonical_id(registry_id)
        entry = canonical_id(entry_id)

        if canonical == "minecraft:block":
            return self.blocks.blocks.numeric_id(entry)
        if canonical == "minecraft:item":
            return self.items.items.numeric_id(entry)
        if canonical == "minecraft:recipe":
            return self.recipes.recipes.numeric_id(entry)

        for table in ("fluids", "entity_types", "game_events"):
            data_table = self.tables.get(table)
            if data_table is not None and data_table.registry is not None:
                if data_table.registry.id == canonical:
                    return data_table.registry.numeric_id(entry)

        return self.synced.numeric_id(canonical, entry)

    def entry_exists(self, registry_id: str, entry_id: str) -> bool:
        """Whether an entry exists *and* has a numeric id the client shares."""
        return self.numeric_id(registry_id, entry_id) is not None

    def registry_ids(self) -> List[str]:
        """Every registry id this object knows about, in a stable order.

        An id can be reachable as more than one thing (a synced `damage_type`
        and its server-side table, a tag container for a registry already
        listed); each is reported once.
        """
        seen: Dict[str, None] = {}
        ordered: List[str] = [self.blocks.blocks.id, self.items.items.id,
                              self.recipes.recipes.id]
        ordered.extend(self.synced.registry_ids())
        for table in self.tables.values():
            if table.registry is not None:
                ordered.append(table.registry.id)
        ordered.extend(self.tags.registry_ids())

        result: List[str] = []
        for registry_id in ordered:
            canonical = canonical_id(registry_id)
            if canonical in seen:
                continue
            seen[canonical] = None
            result.append(canonical)
        return result

    # ----- wire building -----

    def build_registry_packets(self, include_nbt: bool = True) -> List[Tuple[str, bytes]]:
        """Registry Data payloads for every synced registry, in send order."""
        return self.synced.build_packets(include_nbt=include_nbt)

    def build_tag_payload(self) -> Tuple[List[Dict], ResolutionReport]:
        """Update Tags payload plus the report of what could not be resolved.

        Registries with unresolved entries are left out of the payload: the
        client replaces a registry's tags wholesale when it appears in the
        packet, so sending a partly-resolved registry would delete the correct
        tags the client already has from the known pack. Omitting it leaves
        them alone. (The old code sent all of them, which is why every
        `minecraft:block` and `minecraft:item` tag went out empty.)
        """
        payload, report = self.tags.build_payload(self.numeric_id)
        self.tag_report = report

        skipped = [
            registry_id
            for registry_id, info in report.registries.items()
            if info["resolved"] != info["total"]
        ]
        if skipped:
            for registry_id in skipped:
                info = report.registries[registry_id]
                sample = report.unresolved_entries.get(registry_id, [])[:3]
                logger.warning(
                    f"Tag registry {registry_id} left out of Update Tags: "
                    f"{info['resolved']}/{info['total']} entries resolved"
                    + (f" (e.g. {', '.join(sample)})" if sample else "")
                )
            payload = [entry for entry in payload if entry["registry_id"] not in skipped]
        return payload, report

    # ----- freeze -----

    def freeze(self, reason: str = "") -> None:
        """Seal every registry against structural change.

        Called once, before the world is generated or loaded and before any
        client is configured. Value changes that do not move an id stay legal
        (`modify_block`, `modify_item`, `replace_recipe`).
        """
        with self._lock:
            if self._frozen:
                return
            self._frozen = True
            self._frozen_reason = reason
            self.blocks.freeze(reason)
            self.items.freeze(reason)
            self.recipes.freeze(reason)
            self.tags.freeze(reason)
            self.synced.freeze(reason)
            logger.debug(f"Game registry frozen ({reason or 'no reason given'})")

    def unfreeze(self) -> None:
        """Reopen every registry. Tests and pre-world reloads only."""
        with self._lock:
            self._frozen = False
            self._frozen_reason = ""
            self.blocks.unfreeze()
            self.items.unfreeze()
            self.recipes.unfreeze()
            self.tags.unfreeze()
            self.synced.unfreeze()

    def require_open(self, action: str) -> None:
        """Raise unless the registry can still take structural changes."""
        if self._frozen:
            detail = f" ({self._frozen_reason})" if self._frozen_reason else ""
            raise RegistryFrozenError(
                f"cannot {action}: the game registry is frozen{detail}"
            )

    # ----- reporting -----

    def summary(self) -> Dict[str, Any]:
        """Counts for the startup log, the console command and the check tool."""
        return {
            "blocks": len(self.blocks.blocks),
            "block_states": self.blocks.total_states,
            "block_properties": len(self.blocks.properties),
            "items": len(self.items.items),
            "recipes": len(self.recipes),
            "recipe_types": len(self.recipes.types()),
            "tags": len(self.tags),
            "tag_registries": len(self.tags.registries),
            "synced_registries": len(self.synced),
            "synced_entries": self.synced.total_entries,
            "data_tables": len(self.tables),
            "plugins": len(self.plugins),
            "worldgen_presets": len(self.worldgen.settings_by_preset) if self.worldgen else 0,
            "biomes": len(self.worldgen.biomes) if self.worldgen else 0,
            "load_seconds": round(self.load_seconds, 3),
        }

    def describe(self) -> str:
        """One-line-per-registry description for the console `registry` command."""
        lines = [
            f"{self.blocks.blocks.id}: {len(self.blocks.blocks)} blocks, "
            f"{self.blocks.total_states} states",
            f"{self.items.items.id}: {len(self.items.items)} items",
            f"{self.recipes.recipes.id}: {len(self.recipes)} recipes",
            f"tags: {len(self.tags.registries)} registries, {len(self.tags)} tags",
            f"synced: {len(self.synced)} registries, {self.synced.total_entries} entries",
            f"data tables: {len(self.tables)}",
        ]
        for table in self.tables.values():
            lines.append(f"  {table.name} ({table.source}): {table.count}")
        if self.worldgen is not None:
            lines.append(self.worldgen.describe())
        return "\n".join(lines)


# ==================== Global access ====================

_registry: Optional[GameRegistry] = None
# Reentrant: the lazy path below can reach `init_registries`, which takes the
# same lock. A plain Lock deadlocks there, and it is an easy path to hit -
# `import world` resolves the block state table, which loads the registry.
_registry_lock = threading.RLock()


def get_game_registry() -> GameRegistry:
    """The process-wide registry, loading it on first use if necessary."""
    global _registry
    existing = _registry
    if existing is not None:
        return existing
    logger.warning(
        "Game registry was not initialized explicitly; loading it now. "
        "Server startup should call init_registries() before world load."
    )
    return init_registries()


def set_game_registry(registry: Optional[GameRegistry]) -> None:
    """Install a registry instance (tests, reloads) or clear it with None.

    `world/block_states.py` holds a lazily built id namespace that reads from the
    registry, so swapping the instance invalidates that cache. It is imported
    lazily and ignored when absent, because the registry must not depend on the
    world package.
    """
    global _registry
    with _registry_lock:
        _registry = registry

    module = sys.modules.get("world.block_states")
    refresh = getattr(module, "refresh", None)
    if callable(refresh):
        refresh()


def is_initialized() -> bool:
    """Whether an explicit (or lazy) load has happened."""
    return _registry is not None


def init_registries(
    assets_dir: Optional[Path] = None,
    *,
    plugins: Any = (),
    plugins_dir: Optional[Path] = None,
    datapacks_dir: Optional[Path] = None,
    strict: bool = False,
    log_summary: bool = True,
) -> GameRegistry:
    """Load the complete game registry and install it as the global one.

    Safe to call twice: the second call returns the existing instance rather
    than reloading, because reloading would hand out new ids to a running
    world. Use `load_registries()` directly when a fresh instance is wanted.
    """
    global _registry
    with _registry_lock:
        if _registry is not None:
            return _registry

    # The load runs outside the lock on purpose: it imports the registry
    # modules, and a caller may have reached here from inside an import of
    # another package (see the note on _registry_lock). Two threads racing here
    # both load, and the first to finish wins - a duplicate load wastes a second
    # of boot time but cannot corrupt anything, because the loser's instance is
    # simply dropped without ever being frozen or handed out.
    from registry.loader import load_registries

    registry = load_registries(
        assets_dir,
        plugins=plugins,
        plugins_dir=plugins_dir,
        datapacks_dir=datapacks_dir,
        strict=strict,
        log_summary=log_summary,
    )
    with _registry_lock:
        if _registry is None:
            _registry = registry
        return _registry


def reset_registries() -> None:
    """Drop the global instance. Tests only - a running server must not do this."""
    set_game_registry(None)


__all__ = [
    "DEFAULT_ASSETS_DIR",
    "STATIC_ID_REGISTRIES",
    "GameRegistry",
    "get_game_registry",
    "set_game_registry",
    "is_initialized",
    "init_registries",
    "reset_registries",
]
