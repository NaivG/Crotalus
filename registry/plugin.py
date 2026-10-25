"""
Plugin API - the reserved interface for adding and modifying registry content.

Everything a plugin is allowed to touch is reachable from the context it is
handed, and every mutation goes through the registry method that knows the
rules (append-only ids, freeze guards, index maintenance). A plugin never edits
a table directly.

Lifecycle
---------
    vanilla data loaded
      -> datapack overlay
      -> plugins, in (phase, registration order)
      -> cross-validation
      -> freeze
      -> world generation / loading starts

All phases run with the vanilla data already present - the phase number only
orders plugins relative to each other, it is not a barrier around loading:

* `EARLY`  patch vanilla entries before other plugins read them
* `DEFAULT` the normal case
* `LATE`   depends on entries other plugins registered

The freeze at the end is the important half of the contract: once the world
exists, block state ids may already be inside a saved chunk and synced registry
indices may already have been sent to a client, so ids cannot move any more.
`modify_*` calls that change a value without moving an id stay legal; anything
that appends or removes raises `RegistryFrozenError`.

Failure handling
----------------
A plugin that raises is logged with its traceback and skipped; the server keeps
booting, because a broken optional plugin should not take the world with it.
`strict=True` (used by the check tool) turns that into a `PluginError` so a
broken plugin fails loudly instead of silently doing nothing.
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional

from loguru import logger

from registry.core import RegistryError


class PluginPhase(IntEnum):
    """When a plugin runs, relative to the others."""

    EARLY = 0
    DEFAULT = 100
    LATE = 200


class PluginError(RegistryError):
    """Raised when a plugin fails and the manager is in strict mode."""


# ==================== Plugin interface ====================

class RegistryPlugin:
    """Base class for a registry plugin.

    Subclass it, set `name`, and override `on_registries`. A plugin object is
    also free to be a plain callable - see `register_plugin`.
    """

    #: Stable identifier used in logs and error messages.
    name: str = "plugin"

    #: Ordering relative to other plugins.
    phase: int = PluginPhase.DEFAULT

    def on_registries(self, context: "PluginContext") -> None:
        """Add or modify registry content. Runs before the freeze."""

    def on_frozen(self, context: "PluginContext") -> None:
        """Read-only hook called after the freeze, for sanity checks."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name} phase={int(self.phase)}>"


class PluginContext:
    """What a plugin gets to work with.

    Thin by design: the registries hang off it directly so a plugin reads like
    registry code, and only the helpers that encode a *rule* (append-only,
    canonical ids, freeze) live here.
    """

    def __init__(self, registry, strict: bool = False) -> None:
        self.registry = registry
        self.strict = strict
        self.notes: List[str] = []

    # ----- direct access -----

    @property
    def blocks(self):
        return self.registry.blocks

    @property
    def items(self):
        return self.registry.items

    @property
    def recipes(self):
        return self.registry.recipes

    @property
    def tags(self):
        return self.registry.tags

    @property
    def synced(self):
        return self.registry.synced

    @property
    def tables(self):
        return self.registry.tables

    def registry_by_id(self, registry_id: str):
        """A registry container by id, whichever kind it is."""
        return self.registry.find_registry(registry_id)

    # ----- helpers -----

    def note(self, message: str) -> None:
        """Record something for the startup summary."""
        self.notes.append(message)
        logger.info(f"[plugin] {message}")

    def add_item(self, name: str, components: Optional[Mapping[str, Any]] = None):
        """Register an item (append-only)."""
        return self.items.register_item(name, components)

    def add_block(self, name: str, **kwargs):
        """Register a block (append-only, pre-freeze only)."""
        return self.blocks.register_block(name, **kwargs)

    def add_recipe(self, record: Mapping[str, Any], key: Optional[str] = None):
        """Register a recipe from a `recipes.json`-shaped mapping."""
        return self.recipes.add_recipe(record, key)

    def add_tag_entry(self, registry_id: str, tag_name: str, entry: str) -> bool:
        """Add one entry to a tag, creating the tag when needed."""
        return self.tags.add_entry(registry_id, tag_name, entry)

    def add_tag_entries(self, registry_id: str, tag_name: str, entries: Iterable[str]) -> int:
        """Add several entries to a tag; returns how many were new."""
        return self.tags.add_entries(registry_id, tag_name, entries)

    def add_synced_entry(self, registry_id: str, entry_id: str, payload: Any):
        """Append an entry to a synced registry."""
        return self.synced.add_entry(registry_id, entry_id, payload)

    def add_translation(self, key: str, value: str) -> None:
        """Add a language entry to the loaded `en_us` table."""
        table = self.tables.get("translations")
        if table is None or not isinstance(table.data, dict):
            raise RegistryError("the translations table is not loaded")
        table.data[key] = value

    def add_block_property(self, name: str, kind: str, **kwargs):
        """Declare a block property so a registered block can use it."""
        from registry.entries import BlockProperty

        if kind == "boolean":
            prop = BlockProperty.of(name, kind)
        else:
            prop = BlockProperty.of(name, kind, **kwargs)
        self.blocks.declare_property(prop)
        return prop


# ==================== Manager ====================

@dataclass
class PluginRecord:
    """One plugin's registration and outcome, for the startup summary."""

    name: str
    phase: int
    order: int
    plugin: Any
    applied: bool = False
    error: Optional[str] = None

    def __repr__(self) -> str:
        state = "applied" if self.applied else ("failed" if self.error else "pending")
        return f"<PluginRecord {self.name} phase={self.phase} {state}>"


class PluginManager:
    """Registers plugins and applies them in a deterministic order."""

    def __init__(self, strict: bool = False) -> None:
        self.strict = strict
        self._records: List[PluginRecord] = []
        self._counter = 0

    # ----- registration -----

    def register(
        self,
        plugin: Any,
        name: Optional[str] = None,
        phase: Optional[int] = None,
    ) -> PluginRecord:
        """Register a plugin object or a callable.

        A callable is wrapped so `def setup(registry)` works without importing
        the base class; anything else needs `on_registries`.
        """
        if not callable(getattr(plugin, "on_registries", None)) and not callable(plugin):
            raise PluginError(
                f"{plugin!r} is neither a RegistryPlugin nor a callable"
            )

        resolved_name = name or getattr(plugin, "name", None) or getattr(
            plugin, "__name__", None
        ) or f"plugin-{self._counter}"
        resolved_phase = int(phase if phase is not None else getattr(plugin, "phase", PluginPhase.DEFAULT))

        self._counter += 1
        record = PluginRecord(
            name=str(resolved_name),
            phase=resolved_phase,
            order=self._counter,
            plugin=plugin,
        )
        self._records.append(record)
        logger.debug(f"Registered registry plugin {record.name} (phase {record.phase})")
        return record

    def load_directory(self, path: Path) -> int:
        """Import every `*.py` in a directory and register what it exposes.

        A module may expose any of:
          * `PLUGIN` - a `RegistryPlugin` instance
          * `plugins` - a list of plugin objects or callables
          * `register(registry)` / `setup(context)` - a module-level function

        Import failures are reported and skipped, never fatal: a plugins folder
        with a syntax error must not stop the server from starting.
        """
        if not path.exists():
            logger.debug(f"No plugin directory at {path}")
            return 0

        loaded = 0
        for file in sorted(path.glob("*.py")):
            if file.name.startswith("_"):
                continue
            module_name = f"crotalus_plugin_{file.stem}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, file)
                if spec is None or spec.loader is None:
                    logger.warning(f"Cannot load plugin file {file}")
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
            except Exception as error:
                logger.error(f"Plugin file {file.name} failed to import: {error}")
                if self.strict:
                    raise PluginError(f"plugin file {file.name} failed to import") from error
                continue

            loaded += self._register_module(module, file)

        logger.info(f"Loaded {loaded} registry plugin(s) from {path}")
        return loaded

    def _register_module(self, module, file: Path) -> int:
        count = 0
        if hasattr(module, "PLUGIN"):
            self.register(module.PLUGIN)
            count += 1
        for attribute in ("plugins", "PLUGINS"):
            found = getattr(module, attribute, None)
            if isinstance(found, (list, tuple)):
                for plugin in found:
                    self.register(plugin)
                    count += 1
        for function_name in ("register", "setup"):
            function = getattr(module, function_name, None)
            if callable(function) and not isinstance(function, type):
                self.register(function, name=f"{file.stem}:{function_name}")
                count += 1
        if count == 0:
            logger.warning(
                f"Plugin file {file.name} exposes no PLUGIN, plugins or register()/setup()"
            )
        return count

    # ----- application -----

    def ordered(self) -> List[PluginRecord]:
        """Records in application order: phase, then registration order."""
        return sorted(self._records, key=lambda record: (record.phase, record.order))

    def apply(self, registry, context: Optional[PluginContext] = None) -> PluginContext:
        """Run every registered plugin's `on_registries` hook."""
        shared = context or PluginContext(registry, strict=self.strict)

        for record in self.ordered():
            plugin = record.plugin
            try:
                hook = getattr(plugin, "on_registries", None)
                if callable(hook):
                    hook(shared)
                else:
                    plugin(shared.registry)
                record.applied = True
            except Exception as error:
                record.error = f"{type(error).__name__}: {error}"
                logger.error(
                    f"Registry plugin {record.name} failed: {record.error}\n"
                    f"{traceback.format_exc()}"
                )
                if self.strict:
                    raise PluginError(
                        f"registry plugin {record.name} failed: {error}"
                    ) from error

        return shared

    def notify_frozen(self, registry, context: PluginContext) -> None:
        """Run every plugin's `on_frozen` hook; failures are logged, not raised."""
        for record in self.ordered():
            plugin = record.plugin
            hook = getattr(plugin, "on_frozen", None)
            if not callable(hook):
                continue
            try:
                hook(context)
            except Exception as error:
                logger.error(f"Registry plugin {record.name} on_frozen failed: {error}")
                if self.strict:
                    raise PluginError(
                        f"registry plugin {record.name} on_frozen failed: {error}"
                    ) from error

    # ----- reporting -----

    @property
    def records(self) -> List[PluginRecord]:
        return list(self._records)

    @property
    def failures(self) -> List[PluginRecord]:
        return [record for record in self._records if record.error]

    def summary(self) -> List[str]:
        lines = []
        for record in self.ordered():
            if record.error:
                lines.append(f"  {record.name} (phase {record.phase}): FAILED - {record.error}")
            elif record.applied:
                lines.append(f"  {record.name} (phase {record.phase}): applied")
            else:
                lines.append(f"  {record.name} (phase {record.phase}): not run")
        return lines

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self.ordered())


__all__ = [
    "PluginPhase",
    "PluginError",
    "RegistryPlugin",
    "PluginContext",
    "PluginRecord",
    "PluginManager",
]
