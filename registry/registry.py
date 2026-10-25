"""
Network adapter for the synced registries - the login path's API.

`network/client_handler.py` builds Registry Data packets from a "registry
manager" and a "packet builder". That shape is kept here so the handler does not
have to know about the registry rework, but there is exactly one source of truth
behind it: `registry.game_registry.GameRegistry`. Everything in this module is a
view over `GameRegistry.synced`.

What changed underneath, and why it matters:

* Registry order is now deterministic. The old `RegistryManager` collected
  registry ids through a Python `set`, so the order registries were sent in
  depended on `PYTHONHASHSEED` - harmless to the client, but it made packet
  captures and logs differ between runs.
* Registry and entry ids are canonical (`minecraft:worldgen/biome`,
  `minecraft:plains`) instead of bare (`worldgen/biome`, `plains`). The client
  parses a ResourceLocation leniently and defaults the namespace to `minecraft`,
  so both spellings resolve to the same registry; canonical form is simply the
  one the plugin API can look things up with.
* Synced entries are cached NBT rather than re-serialised per packet, and the
  payloads are built from the same objects plugins append to, so a plugin's
  entry cannot be missing from one code path and present in another.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from loguru import logger

from registry.core import canonical_id
from registry.game_registry import GameRegistry, get_game_registry


@dataclass
class RegistryEntry:
    """One entry as the packet builder sees it: an id and its NBT bytes."""

    id: str
    data: Optional[bytes] = None


@dataclass
class RegistryView:
    """A registry's entries in numeric-id order.

    Kept as a materialised list because the old API exposed `.entries` and the
    login path indexes it; it is built per call, and there are 23 registries
    with ~400 entries in total.
    """

    registry_id: str
    entries: List[RegistryEntry] = field(default_factory=list)

    def get_entry(self, entry_id: str) -> Optional[RegistryEntry]:
        canonical = canonical_id(entry_id)
        for entry in self.entries:
            if entry.id == canonical:
                return entry
        return None

    def get_numeric_id(self, entry_id: str) -> Optional[int]:
        canonical = canonical_id(entry_id)
        for index, entry in enumerate(self.entries):
            if entry.id == canonical:
                return index
        return None

    def get_all_ids(self) -> Dict[str, int]:
        return {entry.id: index for index, entry in enumerate(self.entries)}

    def __len__(self) -> int:
        return len(self.entries)


class RegistryManager:
    """A view of the synced registries, in send order.

    Without an explicit instance it follows the process-wide game registry on
    every access, so swapping the global (a reload, a test) cannot leave this
    adapter pointing at a registry nobody else uses.
    """

    _instance: Optional["RegistryManager"] = None

    def __init__(self, game_registry: Optional[GameRegistry] = None) -> None:
        self._explicit = game_registry

    @classmethod
    def get_instance(cls) -> "RegistryManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def game_registry(self) -> GameRegistry:
        if self._explicit is not None:
            return self._explicit
        return get_game_registry()

    # ----- reading -----

    def get_registry(self, registry_id: str) -> Optional[RegistryView]:
        """A registry's entries, or None when it is not synced."""
        container = self.game_registry.synced.get_registry(registry_id)
        if container is None:
            return None
        return RegistryView(
            registry_id=container.id,
            entries=[RegistryEntry(entry.key, entry.to_nbt()) for entry in container.values()],
        )

    def get_all_registries(self) -> List[RegistryView]:
        """Every synced registry, in send order."""
        return [
            view
            for view in (self.get_registry(registry_id) for registry_id in self.registry_ids())
            if view is not None
        ]

    def registry_ids(self) -> List[str]:
        return self.game_registry.synced.registry_ids()

    def get_all_registry_ids(self) -> List[str]:
        return self.registry_ids()

    # ----- writing (back-compat plugins) -----

    def add_entry(self, registry_id: str, entry_id: str, data: Optional[bytes] = None) -> bool:
        """Add a synced registry entry whose payload is raw network NBT.

        This is the old `RegistryPluginExtension` interface. New plugins should
        register the JSON payload instead
        (`GameRegistry.synced.add_entry(registry_id, entry_id, payload)`), which
        goes through the same conversion as the shipped data.
        """
        if data is None:
            logger.warning(
                f"add_entry({registry_id}, {entry_id}) needs an NBT payload; "
                f"nothing was registered"
            )
            return False
        self.game_registry.synced.add_raw_entry(registry_id, entry_id, data)
        return True

    def modify_entry(self, registry_id: str, entry_id: str, data: bytes) -> bool:
        """Replace a synced entry's payload with raw network NBT."""
        if self.game_registry.synced.get(registry_id, entry_id) is None:
            return False
        self.game_registry.synced.replace_entry(registry_id, entry_id, raw_nbt=data)
        return True


class RegistryDataPacketBuilder:
    """Builds Registry Data payloads from the synced registries.

    `known_packs` is the intersection of packs both sides recognize, which is
    what allows NBT to be omitted: an entry that comes from `minecraft:core`
    can be written as a bare `TAG_End` and the client fills it in from the pack
    it already has. Measured against `assets/synced_registries.json` this takes
    the 23 packets from 91,504 bytes to 4,500.
    """

    def __init__(self, registry_manager: RegistryManager, known_packs: List[str] = None):
        self.registry_manager = registry_manager
        self.known_packs: List[str] = known_packs or []

    def set_known_packs(self, known_packs: List[str]) -> None:
        """Record the mutually known packs from the Known Packs exchange."""
        self.known_packs = known_packs
        logger.debug(f"Known packs set to: {known_packs}")

    def _should_include_nbt(self, registry_id: str) -> bool:
        """Whether an entry's NBT has to be sent.

        Always true for a vanilla server: every registry we have comes from
        `minecraft:core`, but only the client's own knowledge decides whether
        omitting it is safe, and that is the `include_nbt` argument.
        """
        return True

    def build_packet_data(self, registry_id: str, include_nbt: bool = True) -> bytes:
        """One registry's payload: id, entry count, then id/NBT pairs."""
        return self.registry_manager.game_registry.synced.build_packet_payload(
            registry_id, include_nbt=include_nbt
        )

    def build_all_packets(self, include_nbt: bool = True) -> List[Tuple[str, bytes]]:
        """Every registry's payload, in send order."""
        synced = self.registry_manager.game_registry.synced
        packets: List[Tuple[str, bytes]] = []
        for registry_id in synced.registry_ids():
            container = synced.get_registry(registry_id)
            if container is None or len(container) == 0:
                logger.warning(f"Skipping empty registry: {registry_id}")
                continue
            packets.append((registry_id, self.build_packet_data(registry_id, include_nbt)))
        return packets

    def build_packets_excluding_known(self) -> List[Tuple[str, bytes]]:
        """Payloads with known-pack entries stripped.

        Only meaningful after a Known Packs exchange; without one the client
        may not own the pack, so full data is sent.
        """
        if not self.known_packs:
            return self.build_all_packets(include_nbt=True)
        return self.build_all_packets(include_nbt=False)


# ==================== Global access ====================

def get_registry_manager() -> RegistryManager:
    """The registry manager bound to the process-wide game registry."""
    return RegistryManager.get_instance()


def create_registry_data_packet_builder() -> RegistryDataPacketBuilder:
    """A packet builder over the global registry."""
    return RegistryDataPacketBuilder(get_registry_manager())


__all__ = [
    "RegistryEntry",
    "RegistryView",
    "RegistryManager",
    "RegistryDataPacketBuilder",
    "get_registry_manager",
    "create_registry_data_packet_builder",
]
