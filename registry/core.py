"""
Registry core - the containers and identifiers every other registry module uses.

Two things live here:

`ResourceLocation` / `canonical_id()`
    Minecraft identifies everything as `namespace:path`, and almost every table
    Pumpkin ships omits the namespace (`oak_log`, `worldgen/biome`, `plains`).
    The wire format does not care - the client parses those leniently and
    defaults the namespace to `minecraft` - but our lookups do. Every registry
    in this package stores *canonical* keys (`minecraft:oak_log`,
    `minecraft:worldgen/biome`) and canonicalises on the way in, so a plugin
    cannot end up with an entry that is unreachable because it was spelled with
    the wrong prefix.

`Registry[T]`
    An ordered id -> value map with the semantics the game needs:

    * Insertion order *is* the numeric id. For every registry the client can
      see (synced registries, blocks, items, fluids, ...) the index of an entry
      is what goes on the wire, so order is part of the contract. New entries
      may only be appended; `insert_before` exists for the rare case where a
      registry is still being built and nothing has been sent yet.
    * Values are `freeze()`-able. Once the world exists, chunks on disk and on
      the wire encode state ids resolved from these tables - a later insert
      would renumber them, so mutation is refused rather than silently
      desynchronising every connected client.

The lock is a plain `threading.RLock`: chunk generation runs on worker threads
and reads these tables concurrently with (pre-world) plugin registration.
"""

from __future__ import annotations

import threading
from typing import (
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
)

T = TypeVar("T")

MINECRAFT_NAMESPACE = "minecraft"


# ==================== Identifiers ====================

def split_id(identifier: str) -> Tuple[str, str]:
    """Split `namespace:path` into its two halves, defaulting the namespace.

    Mirrors `ResourceLocation.parse`: a bare `path` is `minecraft:path`, and
    only the first colon separates (`minecraft:a:b` -> ("minecraft", "a:b")).
    """
    namespace, sep, path = identifier.partition(":")
    if not sep:
        return MINECRAFT_NAMESPACE, identifier
    return namespace, path


def canonical_id(identifier: str) -> str:
    """Return the canonical `namespace:path` form of an identifier."""
    namespace, path = split_id(identifier)
    return f"{namespace}:{path}"


def path_of(identifier: str) -> str:
    """Return just the path half of an identifier."""
    return split_id(identifier)[1]


def namespace_of(identifier: str) -> str:
    """Return just the namespace half of an identifier."""
    return split_id(identifier)[0]


def is_canonical(identifier: str) -> bool:
    """Whether an identifier already carries an explicit namespace."""
    return ":" in identifier


class ResourceLocation(str):
    """A canonical `namespace:path` identifier.

    A `str` subclass, so it can be used anywhere a plain id is expected (dict
    keys, f-strings, NBT/JSON payloads) while still offering `.namespace` and
    `.path`. Constructing one canonicalises the input.
    """

    __slots__ = ()

    def __new__(cls, identifier: str) -> "ResourceLocation":
        return super().__new__(cls, canonical_id(identifier))

    @property
    def namespace(self) -> str:
        return namespace_of(self)

    @property
    def path(self) -> str:
        return path_of(self)

    @classmethod
    def parse(cls, identifier: str) -> "ResourceLocation":
        return cls(identifier)


# ==================== Errors ====================

class RegistryError(Exception):
    """Base class for every registry problem a plugin can trigger."""


class RegistryFrozenError(RegistryError):
    """Raised when a mutation is attempted after `freeze()`.

    Frozen means the server has started generating or loading world data, so
    ids handed out earlier are already stored in chunks, sent to clients, or
    both. See `Registry.freeze` for why this is refused instead of allowed.
    """


class UnknownRegistryError(RegistryError, KeyError):
    """Raised when a registry id does not exist."""


class UnknownEntryError(RegistryError, KeyError):
    """Raised when an entry id does not exist in a registry."""


class DuplicateEntryError(RegistryError):
    """Raised when adding an entry that already exists and replace is off."""


class ValidationError(RegistryError):
    """Raised when an entry does not satisfy the registry's invariants."""


# ==================== Registry container ====================

class Registry(Generic[T]):
    """An ordered, named, freezable id -> value map.

    Args:
        registry_id: Registry identifier, e.g. `minecraft:item`.
        title: Human-readable name used in log lines.
        synced: True for registries delivered to the client in a Registry Data
            packet during configuration. Those are the ones whose entry order
            (and therefore every tag id) the client depends on.
        description: One-line summary shown by `registry` console command.
    """

    __slots__ = (
        "_id",
        "title",
        "synced",
        "description",
        "_entries",
        "_lock",
        "_frozen",
        "_frozen_reason",
    )

    def __init__(
        self,
        registry_id: str,
        *,
        title: Optional[str] = None,
        synced: bool = False,
        description: str = "",
    ) -> None:
        self._id = canonical_id(registry_id)
        self.title = title or self._id
        self.synced = synced
        self.description = description
        self._entries: Dict[str, T] = {}
        self._lock = threading.RLock()
        self._frozen = False
        self._frozen_reason = ""

    # ----- identity -----

    @property
    def id(self) -> str:
        """Canonical registry identifier."""
        return self._id

    @property
    def name(self) -> str:
        """Alias for `id`; some call sites read better as `name`."""
        return self._id

    def __repr__(self) -> str:
        state = "frozen" if self._frozen else "open"
        return f"<Registry {self._id} entries={len(self._entries)} {state}>"

    # ----- reading -----

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return canonical_id(key) in self._entries

    def __iter__(self) -> Iterator[str]:
        """Iterate entry ids in numeric-id order."""
        return iter(self._entries)

    def __getitem__(self, key: str) -> T:
        return self.require(key)

    def keys(self) -> List[str]:
        return list(self._entries)

    def values(self) -> List[T]:
        return list(self._entries.values())

    def items(self) -> List[Tuple[str, T]]:
        return list(self._entries.items())

    def ids(self) -> Tuple[str, ...]:
        """Entry ids in numeric-id order."""
        return tuple(self._entries)

    def get(self, key: str, default: Optional[T] = None) -> Optional[T]:
        """Look an entry up, or return `default` when it is not registered."""
        return self._entries.get(canonical_id(key), default)

    def require(self, key: str) -> T:
        """Look an entry up, raising `UnknownEntryError` when it is missing."""
        canonical = canonical_id(key)
        try:
            return self._entries[canonical]
        except KeyError:
            raise UnknownEntryError(
                f"{canonical} is not registered in {self._id} "
                f"({len(self._entries)} entries)"
            ) from None

    def numeric_id(self, key: str) -> Optional[int]:
        """Index of an entry, or None. This is the id used on the wire."""
        canonical = canonical_id(key)
        for index, existing in enumerate(self._entries):
            if existing == canonical:
                return index
        return None

    def by_numeric_id(self, numeric_id: int) -> Optional[T]:
        """Entry at an index, or None when out of range."""
        if 0 <= numeric_id < len(self._entries):
            return list(self._entries.values())[numeric_id]
        return None

    def id_of(self, value: T) -> Optional[str]:
        """Reverse lookup by value identity."""
        for key, existing in self._entries.items():
            if existing is value:
                return key
        return None

    def index_map(self) -> Dict[str, int]:
        """`{entry_id: numeric_id}` for the whole registry."""
        return {key: index for index, key in enumerate(self._entries)}

    def find(self, predicate) -> List[Tuple[str, T]]:
        """All entries matching a predicate, in numeric-id order."""
        return [(key, value) for key, value in self._entries.items()
                if predicate(value)]

    def first(self, predicate, default=None):
        """First entry matching a predicate, or `default`."""
        for value in self._entries.values():
            if predicate(value):
                return value
        return default

    # ----- writing -----

    def add(self, key: str, value: T, *, replace: bool = False) -> T:
        """Append (or replace) an entry.

        Appending is the normal path and preserves numeric ids of everything
        already registered. `replace` updates a value in place without moving
        it, which is the only kind of modification that is safe for an entry
        the client already knows.
        """
        canonical = canonical_id(key)
        with self._lock:
            existing = canonical in self._entries
            if existing and not replace:
                raise DuplicateEntryError(
                    f"{canonical} is already registered in {self._id}"
                )
            if not existing:
                self._check_mutable(f"add {canonical}")
            self._entries[canonical] = value
        return value

    def setdefault(self, key: str, factory) -> T:
        """Return an existing entry, or register one built by `factory()`."""
        canonical = canonical_id(key)
        with self._lock:
            if canonical in self._entries:
                return self._entries[canonical]
        return self.add(canonical, factory())

    def replace(self, key: str, value: T) -> T:
        """Replace an existing entry, keeping its numeric id."""
        return self.add(key, value, replace=True)

    def remove(self, key: str) -> T:
        """Remove an entry.

        Removing shifts every later entry's numeric id, so it is refused once
        the registry is frozen. Before that it is allowed, but callers should
        prefer replacing an entry with a disabled stub: ids are referenced by
        recipes, tags and chunk palettes.
        """
        canonical = canonical_id(key)
        with self._lock:
            if canonical not in self._entries:
                raise UnknownEntryError(f"{canonical} is not in {self._id}")
            self._check_mutable(f"remove {canonical}")
            return self._entries.pop(canonical)

    def clear(self) -> None:
        """Drop every entry. Only legal while the registry is open."""
        with self._lock:
            self._check_mutable("clear")
            self._entries.clear()

    def update(self, mapping: Dict[str, T], *, replace: bool = False) -> None:
        """Add many entries, left to right."""
        for key, value in mapping.items():
            self.add(key, value, replace=replace)

    # ----- freezing -----

    @property
    def frozen(self) -> bool:
        return self._frozen

    def freeze(self, reason: str = "") -> None:
        """Seal the registry against further structural changes.

        Called before the world is generated or loaded. From that point on a
        block state id may already sit in a saved chunk, and a synced registry
        index may already have been sent to a client; letting a plugin append
        an entry would renumber or invalidate data that is already out there.
        Value replacement on an existing key stays legal: it moves nothing.
        """
        with self._lock:
            self._frozen = True
            self._frozen_reason = reason

    def unfreeze(self) -> None:
        """Reopen the registry. Only for tests and pre-world reloads."""
        with self._lock:
            self._frozen = False
            self._frozen_reason = ""

    def _check_mutable(self, action: str) -> None:
        if self._frozen:
            detail = f" ({self._frozen_reason})" if self._frozen_reason else ""
            raise RegistryFrozenError(
                f"cannot {action}: registry {self._id} is frozen{detail}. "
                f"Register the change from a plugin before the world loads."
            )


class IndexedRegistry(Registry[T]):
    """A `Registry` that also keeps its numeric ids explicit.

    The synced registries, blocks, items and fluids all carry a numeric id in
    Pumpkin's data. It is *always* the insertion index for vanilla data, but
    keeping it as a field means `numeric_id()` stays honest even if a plugin
    inserts an entry somewhere unusual, and lets a check tool assert the two
    agree for the vanilla tables.
    """

    __slots__ = ("_numeric",)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._numeric: Dict[str, int] = {}

    def add(self, key: str, value: T, *, replace: bool = False,
            numeric_id: Optional[int] = None) -> T:
        canonical = canonical_id(key)
        super().add(canonical, value, replace=replace)
        if numeric_id is not None:
            self._numeric[canonical] = numeric_id
        else:
            self._numeric.setdefault(canonical, len(self) - 1)
        return value

    def remove(self, key: str) -> T:
        canonical = canonical_id(key)
        value = super().remove(canonical)
        self._numeric.pop(canonical, None)
        return value

    def clear(self) -> None:
        super().clear()
        self._numeric.clear()

    def numeric_id(self, key: str) -> Optional[int]:
        """Explicit numeric id when known, else the insertion index."""
        canonical = canonical_id(key)
        if canonical in self._numeric:
            return self._numeric[canonical]
        return super().numeric_id(canonical)

    def declared_numeric_ids(self) -> Dict[str, int]:
        """The explicit `{entry_id: numeric_id}` table, as loaded."""
        return dict(self._numeric)


# ==================== Shared helpers ====================

def freeze_all(registries, reason: str = "") -> None:
    """Freeze an iterable of registries, ignoring anything without `freeze`."""
    for registry in registries:
        freeze = getattr(registry, "freeze", None)
        if callable(freeze):
            freeze(reason)


def summarize(registries) -> str:
    """`name=count` line for log output."""
    return ", ".join(f"{r.id}={len(r)}" for r in registries if len(r))


__all__ = [
    "MINECRAFT_NAMESPACE",
    "ResourceLocation",
    "Registry",
    "IndexedRegistry",
    "RegistryError",
    "RegistryFrozenError",
    "UnknownRegistryError",
    "UnknownEntryError",
    "DuplicateEntryError",
    "ValidationError",
    "canonical_id",
    "split_id",
    "path_of",
    "namespace_of",
    "is_canonical",
    "freeze_all",
    "summarize",
]
