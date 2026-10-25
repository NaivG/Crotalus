"""
Compatibility shim for the old registry-data module.

The JSON -> NBT conversion used for synced registry entries moved to
`registry.synced.RegistryJsonToNbtConverter`, and the manager moved to
`registry.game_registry.GameRegistry`. This module stays so anything importing
the old names keeps working; new code should import from `registry` directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from registry.game_registry import DEFAULT_ASSETS_DIR
from registry.registry import (
    RegistryDataPacketBuilder,
    RegistryManager,
    get_registry_manager,
)
from registry.synced import RegistryJsonToNbtConverter

#: Kept for old imports: the assets directory and the synced-registry file.
ASSETS_DIR = DEFAULT_ASSETS_DIR
DEFAULT_REGISTRY_FILE = ASSETS_DIR / "synced_registries.json"


class RegistryDataLoader:
    """Reads raw synced-registry JSON. Prefer `GameRegistry.synced`.

    Still useful to inspect the file without building a whole registry, e.g.
    from a tool that only wants the entry counts.
    """

    def __init__(self, registry_file: Path = DEFAULT_REGISTRY_FILE):
        self.registry_file = registry_file
        self._cache: Optional[Dict[str, Any]] = None

    def _load_json(self) -> Dict[str, Any]:
        if self._cache is not None:
            return self._cache
        if not self.registry_file.exists():
            raise FileNotFoundError(f"Registry file not found: {self.registry_file}")
        import json

        with open(self.registry_file, "r", encoding="utf-8") as handle:
            self._cache = json.load(handle)
        logger.debug(f"Loaded registry data from {self.registry_file}")
        return self._cache

    def get_registry(self, registry_id: str) -> Optional[Dict[str, Any]]:
        return self._load_json().get(registry_id)

    def get_all_registry_ids(self) -> List[str]:
        return list(self._load_json())

    def get_entry(self, registry_id: str, entry_id: str) -> Optional[Any]:
        registry = self.get_registry(registry_id)
        return registry.get(entry_id) if registry else None

    def get_entries_with_nbt(self, registry_id: str) -> List[tuple]:
        """`[(entry_id, nbt_bytes), ...]`, converted on the fly."""
        registry = self.get_registry(registry_id)
        if registry is None:
            return []
        return [
            (entry_id, RegistryJsonToNbtConverter.convert_registry_entry(entry_id, data))
            for entry_id, data in registry.items()
        ]


def get_registry(registry_id: str) -> Optional[Dict[str, Any]]:
    """Raw JSON payloads for a synced registry, or None."""
    return RegistryDataLoader().get_registry(registry_id)


def get_entry(registry_id: str, entry_id: str) -> Optional[Any]:
    """One raw JSON payload from a synced registry."""
    return RegistryDataLoader().get_entry(registry_id, entry_id)


def get_all_registry_ids() -> List[str]:
    """Every synced registry id, in file order."""
    return RegistryDataLoader().get_all_registry_ids()


__all__ = [
    "ASSETS_DIR",
    "DEFAULT_REGISTRY_FILE",
    "RegistryJsonToNbtConverter",
    "RegistryDataLoader",
    "RegistryManager",
    "RegistryDataPacketBuilder",
    "get_registry_manager",
    "get_registry",
    "get_entry",
    "get_all_registry_ids",
]
