"""
Synced (dynamic) registries - the 23 registries sent to the client at login.

Source: `assets/synced_registries.json`, `{registry_id: {entry_id: payload}}`.
These are the registries vanilla synchronizes in the configuration phase; blocks
and items are *not* among them (the client resolves those from the
`minecraft:core` known pack), which is why blocks and items live in their own
modules and why a plugin cannot invent one the client will understand.

Ordering is the contract
------------------------
Each registry's entry order is its numeric id order: tags reference these
entries by index, `chunk_codec.BIOME_PLAINS = 40` is the 41st entry of
`worldgen/biome`, and `player_chat` uses `chat_type` index 0. Entries are
therefore append-only until freeze, and the *load* order is the file's own key
order rather than a set iteration - the old implementation collected registry
ids through a Python `set`, which made the packet order (and any log diff)
depend on `PYTHONHASHSEED`.

Wire encoding
-------------
`Registry Data` carries, per registry:
`registry_id: string`, `count: VarInt`, then per entry `id: string` followed by
an optional NBT compound - `0x01` + payload, or a single `0x00` (TAG_End) to say
"take this entry from a known pack". The JSON -> NBT conversion is the one the
old `registry_data.py` used; it moved here rather than being rewritten so the
bytes the client already accepts cannot drift.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from loguru import logger

from datatypes.nbt import write_nbt
from registry.core import (
    IndexedRegistry,
    RegistryError,
    RegistryFrozenError,
    canonical_id,
)


# ==================== JSON to NBT ====================

class RegistryJsonToNbtConverter:
    """Converts a registry entry's JSON payload into network NBT bytes.

    The field lists below are load-bearing: NBT has no schema, so a bare JSON
    number is `TAG_Byte`, `TAG_Short`, `TAG_Int` or `TAG_Double` depending on
    magnitude, and the client rejects a compound whose field has the wrong tag
    type. These are the fields where that guess would be wrong.
    """

    #: Fields that must be a TAG_List even though they look like int arrays.
    LIST_FIELDS = {
        "slots", "effects", "parameters", "keyframes",
        "tracks", "tags", "requirements", "noise",
    }

    #: Fields that are genuinely int arrays.
    INT_ARRAY_FIELDS = {"divisor", "modifier_id_low", "modifier_id_high"}

    #: Fields that are genuinely long arrays.
    LONG_ARRAY_FIELDS: set = set()

    #: Fields whose ints must be TAG_Int regardless of magnitude.
    INT_FIELDS = {"color", "block_search_extent", "tick_delay"}

    @classmethod
    def convert(cls, data: Any, field_name: Optional[str] = None) -> Any:
        """Convert JSON data into NBT-writable Python values."""
        if data is None:
            return None
        if isinstance(data, bool):
            return 1 if data else 0
        if isinstance(data, (int, float, str)):
            return data
        if isinstance(data, list):
            return cls._convert_list(data, field_name)
        if isinstance(data, dict):
            return cls._convert_dict(data)
        logger.warning(f"registry NBT conversion: unknown type {type(data)}")
        return str(data)

    @classmethod
    def _convert_list(cls, data: Sequence[Any], field_name: Optional[str] = None) -> List[Any]:
        if not data:
            return []
        first = data[0]
        if isinstance(first, dict):
            return [cls._convert_dict(item) for item in data]
        if isinstance(first, bool):
            return [1 if value else 0 for value in data]
        return list(data)

    @classmethod
    def _convert_dict(cls, data: Mapping[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in data.items():
            if value is None:
                continue
            result[key] = cls.convert(value, key)
        return result

    @classmethod
    def convert_registry_entry(cls, entry_id: str, entry_data: Any) -> bytes:
        """Serialise one entry's payload as an unnamed network NBT compound."""
        del entry_id  # the id is written separately, not inside the NBT
        return write_nbt(cls._convert_dict(entry_data), "")


# ==================== Entry ====================

@dataclass(slots=True)
class SyncedEntry:
    """One entry of a synced registry.

    `data` is the JSON payload from `synced_registries.json` (or from a plugin
    or datapack). `raw_nbt` is the escape hatch: a plugin that already has
    network-NBT bytes can store them verbatim instead, which is what the old
    `RegistryPluginExtension.register_entry()` interface did.
    """

    key: str
    index: int
    data: Any = None
    raw_nbt: Optional[bytes] = None
    _nbt: Optional[bytes] = field(default=None, repr=False)

    def to_nbt(self) -> bytes:
        """The entry's payload as network NBT bytes, converted once."""
        if self._nbt is None:
            if self.raw_nbt is not None:
                self._nbt = self.raw_nbt
            else:
                self._nbt = RegistryJsonToNbtConverter.convert_registry_entry(self.key, self.data)
        return self._nbt

    @property
    def nbt_size(self) -> int:
        return len(self.to_nbt())

    def replace(self, data: Any = None, raw_nbt: Optional[bytes] = None) -> None:
        """Swap the payload and drop the cached bytes."""
        self.data = data
        self.raw_nbt = raw_nbt
        self._nbt = None

    def __repr__(self) -> str:
        return f"<SyncedEntry {self.key} index={self.index} nbt={self.nbt_size}B>"


# ==================== Registries ====================

class SyncedRegistries:
    """Every synchronized registry, in the order they are sent."""

    def __init__(self) -> None:
        self.registries: Dict[str, IndexedRegistry[SyncedEntry]] = {}
        self._lock = threading.RLock()
        self._frozen = False
        self._frozen_reason = ""

    # ----- identity -----

    @property
    def frozen(self) -> bool:
        return self._frozen

    def __len__(self) -> int:
        """Number of registries."""
        return len(self.registries)

    def __iter__(self):
        return iter(self.registries.values())

    def __contains__(self, registry_id: str) -> bool:
        return canonical_id(registry_id) in self.registries

    def __repr__(self) -> str:
        return (f"<SyncedRegistries registries={len(self.registries)} "
                f"entries={self.total_entries} frozen={self._frozen}>")

    def registry_ids(self) -> List[str]:
        """Registry ids in send order."""
        return list(self.registries)

    @property
    def total_entries(self) -> int:
        return sum(len(registry) for registry in self.registries.values())

    # ----- loading -----

    def load(self, data: Mapping[str, Mapping[str, Any]]) -> None:
        """Build every synced registry from `synced_registries.json`."""
        for registry_id, entries in data.items():
            target = self.registry(registry_id)
            for entry_id, payload in entries.items():
                target.add(
                    entry_id,
                    SyncedEntry(
                        key=canonical_id(entry_id),
                        index=len(target),
                        data=payload,
                    ),
                )
        logger.debug(
            f"Synced registries: {len(self.registries)} registries, "
            f"{self.total_entries} entries"
        )

    # ----- container access -----

    def registry(self, registry_id: str) -> IndexedRegistry[SyncedEntry]:
        """The container for a synced registry, creating it when missing."""
        canonical = canonical_id(registry_id)
        with self._lock:
            existing = self.registries.get(canonical)
            if existing is not None:
                return existing
            self._check_mutable(f"create synced registry {canonical}")
            created: IndexedRegistry[SyncedEntry] = IndexedRegistry(
                canonical,
                title=canonical,
                synced=True,
                description="Delivered to clients in a Registry Data packet",
            )
            self.registries[canonical] = created
            return created

    def get_registry(self, registry_id: str) -> Optional[IndexedRegistry[SyncedEntry]]:
        return self.registries.get(canonical_id(registry_id))

    # ----- entry access -----

    def get(self, registry_id: str, entry_id: str) -> Optional[SyncedEntry]:
        container = self.registries.get(canonical_id(registry_id))
        return container.get(entry_id) if container else None

    def require(self, registry_id: str, entry_id: str) -> SyncedEntry:
        container = self.registries.get(canonical_id(registry_id))
        if container is None:
            raise RegistryError(f"no synced registry {registry_id}")
        return container.require(entry_id)

    def entries(self, registry_id: str) -> List[SyncedEntry]:
        container = self.registries.get(canonical_id(registry_id))
        return container.values() if container else []

    def numeric_id(self, registry_id: str, entry_id: str) -> Optional[int]:
        """Index of an entry inside its registry - the id tags refer to."""
        container = self.registries.get(canonical_id(registry_id))
        if container is None:
            return None
        entry = container.get(entry_id)
        return entry.index if entry is not None else None

    def entry_keys(self, registry_id: str) -> Tuple[str, ...]:
        container = self.registries.get(canonical_id(registry_id))
        return container.ids() if container else ()

    # ----- plugin surface -----

    def add_entry(self, registry_id: str, entry_id: str, payload: Any) -> SyncedEntry:
        """Append an entry to a synced registry.

        Appending is the only option: entry order is the numeric id the client
        and our own tags use. `registry_id` must be a registry the client
        knows - an invented id is written into the packet, but the client will
        not recognize it.
        """
        return self._append(registry_id, entry_id, payload=payload)

    def add_raw_entry(self, registry_id: str, entry_id: str, nbt: bytes) -> SyncedEntry:
        """Append an entry whose payload is already network-NBT bytes."""
        return self._append(registry_id, entry_id, raw_nbt=nbt)

    def _append(
        self,
        registry_id: str,
        entry_id: str,
        payload: Any = None,
        raw_nbt: Optional[bytes] = None,
    ) -> SyncedEntry:
        container = self.registry(registry_id)
        canonical = canonical_id(entry_id)
        with self._lock:
            self._check_mutable(f"add {canonical} to {container.id}")
            if canonical in container:
                raise RegistryError(f"{canonical} is already in {container.id}")
            entry = SyncedEntry(
                key=canonical,
                index=len(container),
                data=payload,
                raw_nbt=raw_nbt,
            )
            container.add(canonical, entry)
        logger.debug(f"Added synced entry {container.id}/{canonical}")
        return entry

    def replace_entry(
        self,
        registry_id: str,
        entry_id: str,
        payload: Any = None,
        raw_nbt: Optional[bytes] = None,
    ) -> SyncedEntry:
        """Replace an entry's payload without moving its index."""
        entry = self.require(registry_id, entry_id)
        entry.replace(payload, raw_nbt)
        return entry

    # ----- wire format -----

    def build_packet_payload(self, registry_id: str, include_nbt: bool = True) -> bytes:
        """One registry's Registry Data payload (after the packet id byte)."""
        container = self.registries.get(canonical_id(registry_id))
        if container is None:
            raise RegistryError(f"unknown synced registry {registry_id}")

        from network.buffer import ByteBuffer

        buf = ByteBuffer()
        buf.write_string(container.id)
        buf.write_varint(len(container))
        for entry in container.values():
            buf.write_string(entry.key)
            if include_nbt:
                buf.write_bool(True)
                buf.write_bytes(entry.to_nbt())
            else:
                # TAG_End: "take this entry from the known pack you told us
                # about". Bypassing the NBT entirely is what turns a 91KB login
                # into a ~4KB one for a client that already has minecraft:core.
                buf.write_byte(0)
        return bytes(buf)

    def build_packets(self, include_nbt: bool = True) -> List[Tuple[str, bytes]]:
        """Every registry's payload, in send order."""
        return [
            (registry_id, self.build_packet_payload(registry_id, include_nbt))
            for registry_id in self.registries
        ]

    # ----- freezing -----

    def freeze(self, reason: str = "") -> None:
        with self._lock:
            self._frozen = True
            self._frozen_reason = reason
            for container in self.registries.values():
                container.freeze(reason)

    def unfreeze(self) -> None:
        with self._lock:
            self._frozen = False
            self._frozen_reason = ""
            for container in self.registries.values():
                container.unfreeze()

    def _check_mutable(self, action: str) -> None:
        if self._frozen:
            detail = f" ({self._frozen_reason})" if self._frozen_reason else ""
            raise RegistryFrozenError(
                f"cannot {action}: synced registries are frozen{detail}. "
                f"Clients have already been sent registry data."
            )

    # ----- integrity -----

    def validate(self, typed: Optional[Mapping[str, Any]] = None) -> List[str]:
        """Report structural problems, plus typed/ synced key mismatches.

        `typed` maps a synced registry id to a `Registry` holding the
        server-side detail for the same entries (biomes, dimensions, damage
        types, ...). When given, the two key sets must agree - a mismatch means
        the wire view and the server view have drifted apart.
        """
        problems: List[str] = []

        for registry_id, container in self.registries.items():
            for entry in container.values():
                if entry.index != container.numeric_id(entry.key):
                    problems.append(
                        f"{registry_id}/{entry.key}: index {entry.index} != registry "
                        f"position {container.numeric_id(entry.key)}"
                    )
                if entry.data is None:
                    problems.append(f"{registry_id}/{entry.key}: entry has no payload")

        if typed:
            for registry_id, table in typed.items():
                container = self.registries.get(canonical_id(registry_id))
                if container is None:
                    continue
                synced_keys = set(container.keys())
                typed_keys = {canonical_id(key) for key in table.keys()}
                missing = sorted(typed_keys - synced_keys)
                extra = sorted(synced_keys - typed_keys)
                if missing:
                    problems.append(
                        f"{registry_id}: {len(missing)} typed entries are not synced "
                        f"({missing[:3]}...)"
                    )
                if extra:
                    problems.append(
                        f"{registry_id}: {len(extra)} synced entries have no typed data "
                        f"({extra[:3]}...)"
                    )

        return problems


__all__ = [
    "RegistryJsonToNbtConverter",
    "SyncedEntry",
    "SyncedRegistries",
]
