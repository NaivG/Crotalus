"""
Tag registry - `tags.json`, 14 tag registries and 1,131 tags.

Tag ids are not unique on their own: `minecraft:planks` exists as a *block* tag
and as an *item* tag, with different members. Every lookup here therefore takes
the registry id as well, and `Tag.registry` records it.

Why this module replaces the old tag path
-----------------------------------------
The previous implementation resolved tag entries against the *synced* registry
table only. `tags.json` mostly tags blocks, items and fluids, and those are not
synchronized registries - so every one of those entries failed to resolve and
was dropped with a warning. Of 10,404 entries across `tags.json`, only 1,687
survived, and `minecraft:block`/`minecraft:item` went out completely empty.
Numerically-indexed tags with no members are not an error the client reports; it
just silently believes no block is a plank.

Resolution now goes through `GameRegistry.numeric_id`, which knows about blocks,
items, fluids, entity types and game events (static registries, ordered by their
own data files) as well as synced registries (ordered by what we send). The
handful of registries we have no ordering for stay empty and are reported by
`resolution_report()` instead of being logged one warning per entry.

The payload keeps the exact shape `CUpdateTags` writes:
`[{"registry_id": str, "tags": [{"name": str, "entries": [int, ...]}]}]`.
"""

from __future__ import annotations

import threading
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from loguru import logger

from registry.core import (
    Registry,
    RegistryError,
    RegistryFrozenError,
    canonical_id,
)
from registry.entries import Tag

# `resolve(registry_id, entry_id) -> numeric_id | None`
NumericResolver = Callable[[str, str], Optional[int]]


class ResolutionReport:
    """Per-registry outcome of turning tag entries into numeric ids."""

    def __init__(self) -> None:
        self.registries: Dict[str, Dict[str, int]] = {}
        self.unresolved_entries: Dict[str, List[str]] = {}

    @property
    def unresolved_registries(self) -> List[str]:
        """Registries with entries that could not be resolved at all."""
        return sorted(self.unresolved_entries)

    @property
    def resolved_entries(self) -> int:
        return sum(info["resolved"] for info in self.registries.values())

    @property
    def unresolved_count(self) -> int:
        """Total entries that produced no numeric id, duplicates included."""
        return sum(info["total"] - info["resolved"] for info in self.registries.values())

    @property
    def total_entries(self) -> int:
        return sum(info["total"] for info in self.registries.values())

    def describe(self) -> str:
        lines = []
        for registry_id, info in sorted(self.registries.items()):
            state = "" if info["resolved"] == info["total"] else "  <-- incomplete"
            lines.append(
                f"  {registry_id}: {info['resolved']}/{info['total']} entries "
                f"({info['tags']} tags){state}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"<ResolutionReport {self.resolved_entries}/{self.total_entries} "
                f"entries, {len(self.unresolved_entries)} incomplete registries>")


class TagRegistry:
    """Every tag registry, keyed by the registry the tags belong to."""

    def __init__(self) -> None:
        self.registries: Dict[str, Registry[Tag]] = {}
        self._lock = threading.RLock()
        self._frozen = False
        self._frozen_reason = ""

    # ----- identity -----

    @property
    def frozen(self) -> bool:
        return self._frozen

    def __len__(self) -> int:
        return sum(len(registry) for registry in self.registries.values())

    def __iter__(self):
        return iter(self.registries.values())

    def __repr__(self) -> str:
        return (f"<TagRegistry registries={len(self.registries)} tags={len(self)} "
                f"frozen={self._frozen}>")

    # ----- loading -----

    def load(self, tags_data: Mapping[str, Mapping[str, Sequence[str]]]) -> None:
        """Build every tag registry from `tags.json`."""
        for registry_id, tags in tags_data.items():
            target = self.registry(registry_id)
            for tag_name, entries in tags.items():
                self.create_tag(registry_id, tag_name, entries)

        logger.debug(
            f"Tag registry: {len(self.registries)} registries, {len(self)} tags"
        )

    # ----- container access -----

    def registry(self, registry_id: str, create: bool = True) -> Optional[Registry[Tag]]:
        """The tag registry for a registry id, creating it when asked.

        This is the registry *the tags belong to*: `tags.registry("minecraft:block")`
        returns the container holding `minecraft:planks`, `minecraft:logs`, and so
        on.
        """
        canonical = canonical_id(registry_id)
        with self._lock:
            existing = self.registries.get(canonical)
            if existing is not None or not create:
                return existing
            if self._frozen:
                self._check_mutable(f"create tag registry {canonical}")
            created: Registry[Tag] = Registry(
                canonical,
                title=f"{canonical} tags",
                synced=True,
                description="Tag definitions sent in Update Tags",
            )
            self.registries[canonical] = created
            return created

    def registry_ids(self) -> List[str]:
        """Every registry that has tags, in load order."""
        return list(self.registries)

    # ----- tag access -----

    def get(self, registry_id: str, tag_name: str) -> Optional[Tag]:
        """Look a tag up, or None."""
        container = self.registries.get(canonical_id(registry_id))
        if container is None:
            return None
        return container.get(tag_name)

    def require(self, registry_id: str, tag_name: str) -> Tag:
        """Look a tag up, raising when the registry or tag is unknown."""
        container = self.registries.get(canonical_id(registry_id))
        if container is None:
            raise RegistryError(f"no tags for registry {registry_id}")
        return container.require(tag_name)

    def tags_of(self, registry_id: str) -> List[Tag]:
        """Every tag of a registry, in packet order."""
        container = self.registries.get(canonical_id(registry_id))
        return container.values() if container else []

    def has_tag(self, registry_id: str, tag_name: str) -> bool:
        return self.get(registry_id, tag_name) is not None

    def entries(self, registry_id: str, tag_name: str) -> "frozenset[str]":
        """An entry id -> members map is *the* question the game asks.

        Nested `#tag` references are expanded; an unknown tag resolves to the
        empty set rather than raising, matching vanilla's behaviour for a tag
        with no definition.
        """
        tag = self.get(registry_id, tag_name)
        if tag is None:
            return frozenset()
        return tag.resolve(lambda nested, seen: self._expand(registry_id, nested, seen))

    def _expand(self, registry_id: str, tag_name: str, seen: set) -> "frozenset[str]":
        nested = self.get(registry_id, tag_name)
        if nested is None:
            return frozenset()
        return nested.resolve(lambda name, inner: self._expand(registry_id, name, inner), seen)

    def contains(self, registry_id: str, tag_name: str, entry: str) -> bool:
        """Whether an entry is a member of a tag (tags expanded)."""
        return canonical_id(entry) in self.entries(registry_id, tag_name)

    def lookup(self, registry_id: str) -> Callable[[str], Optional["frozenset[str]"]]:
        """A tag-id -> members callable, for `Ingredient.matches`.

        Returns None for an unknown tag so callers can tell "no such tag" from
        "empty tag".
        """
        def resolve(tag_id: str) -> Optional["frozenset[str]"]:
            tag_name = canonical_id(tag_id)
            if self.get(registry_id, tag_name) is None:
                return None
            return self.entries(registry_id, tag_name)

        return resolve

    def item_lookup(self) -> Callable[[str], Optional["frozenset[str]"]]:
        """Tag lookup over `minecraft:item`, which is what recipes reference."""
        return self.lookup("minecraft:item")

    def block_lookup(self) -> Callable[[str], Optional["frozenset[str]"]]:
        """Tag lookup over `minecraft:block`."""
        return self.lookup("minecraft:block")

    # ----- plugin surface -----

    def create_tag(
        self,
        registry_id: str,
        tag_name: str,
        entries: Iterable[str] = (),
    ) -> Tag:
        """Create (or return) a tag and add entries to it.

        Idempotent: creating a tag that exists adds the missing entries rather
        than replacing what is there, so two plugins can both contribute.
        """
        container = self.registry(registry_id)
        canonical = canonical_id(tag_name)
        with self._lock:
            existing = container.get(canonical)
            if existing is None:
                self._check_mutable(f"create tag {canonical} in {container.id}")
                tag = Tag(registry=container.id, name=canonical, entries=list(entries))
                tag._guard = self._guard
                container.add(canonical, tag)
                return tag
            for entry in entries:
                existing.add(entry)
            return existing

    def add_entry(self, registry_id: str, tag_name: str, entry: str) -> bool:
        """Add one entry to a tag, creating the tag when needed."""
        tag = self.create_tag(registry_id, tag_name, ())
        return tag.add(entry)

    def add_entries(self, registry_id: str, tag_name: str, entries: Iterable[str]) -> int:
        """Add several entries; returns how many were new."""
        tag = self.create_tag(registry_id, tag_name, ())
        return sum(1 for entry in entries if tag.add(entry))

    def remove_entry(self, registry_id: str, tag_name: str, entry: str) -> bool:
        """Remove one entry from a tag."""
        tag = self.get(registry_id, tag_name)
        if tag is None:
            return False
        return tag.discard(entry)

    def replace_tag(
        self,
        registry_id: str,
        tag_name: str,
        entries: Iterable[str],
    ) -> Tag:
        """Replace a tag's entries wholesale, creating it when needed."""
        tag = self.create_tag(registry_id, tag_name, ())
        tag.replace(entries)
        return tag

    def remove_tag(self, registry_id: str, tag_name: str) -> Tag:
        """Remove a tag definition entirely."""
        container = self.registry(registry_id, create=False)
        if container is None:
            raise RegistryError(f"no tags for registry {registry_id}")
        canonical = canonical_id(tag_name)
        self._check_mutable(f"remove tag {canonical}")
        return container.remove(canonical)

    # ----- wire payload -----

    def build_payload(
        self,
        resolve: NumericResolver,
    ) -> Tuple[List[Dict], ResolutionReport]:
        """Build the Update Tags payload and a report of what did not resolve.

        Entry order is preserved exactly as loaded, because a tag's entries are
        not sorted on the wire and a stable payload keeps logins reproducible.
        Duplicate numeric ids are dropped: the same entry can appear in a tag
        twice only through a plugin mistake, and the client would just apply it
        twice.

        Nested `#tag` references are flattened into their members. The client
        receives numeric ids, not tag names, so a nested reference cannot be
        expressed on the wire - vanilla's own tag loader has already flattened
        its nested entries by the time the packet is built. The shipped
        `tags.json` has no nested references at all; this exists so a plugin
        adding one behaves the way it would in a datapack.
        """
        report = ResolutionReport()
        payload: List[Dict] = []

        for registry_id, container in self.registries.items():
            tags: List[Dict] = []
            resolved_count = 0
            total_count = 0
            misses: List[str] = []

            for tag in container.values():
                members = self._flatten(registry_id, tag)
                numeric_entries: List[int] = []
                seen: set = set()
                for entry in members:
                    total_count += 1
                    numeric = resolve(registry_id, entry)
                    if numeric is None:
                        if entry not in misses:
                            misses.append(entry)
                        continue
                    if numeric in seen:
                        continue
                    seen.add(numeric)
                    numeric_entries.append(numeric)
                    resolved_count += 1

                tags.append({"name": tag.name, "entries": numeric_entries})

            report.registries[registry_id] = {
                "tags": len(tags),
                "total": total_count,
                "resolved": resolved_count,
            }
            if misses:
                report.unresolved_entries[registry_id] = misses
            payload.append({"registry_id": registry_id, "tags": tags})

        return payload, report

    def _flatten(self, registry_id: str, tag: Tag) -> List[str]:
        """Direct members followed by the expanded members of nested tags."""
        members: List[str] = []
        for entry in tag.entries:
            if entry.startswith("#"):
                members.extend(sorted(self.entries(registry_id, entry[1:])))
            else:
                members.append(entry)
        return members

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

    def _guard(self, action: str) -> None:
        """Passed to every `Tag` so direct mutation respects the freeze."""
        with self._lock:
            self._check_mutable(action)

    def _check_mutable(self, action: str) -> None:
        if self._frozen:
            detail = f" ({self._frozen_reason})" if self._frozen_reason else ""
            raise RegistryFrozenError(
                f"cannot {action}: tag registry is frozen{detail}. "
                f"Clients have already been sent tag data; restart to change it."
            )

    # ----- integrity -----

    def validate(self, resolve: Optional[NumericResolver] = None) -> List[str]:
        """Report structural problems; `resolve` additionally checks entries."""
        problems: List[str] = []

        for registry_id, container in self.registries.items():
            for tag in container.values():
                if tag.registry != registry_id:
                    problems.append(
                        f"tag {tag.name} claims registry {tag.registry} but lives in "
                        f"{registry_id}"
                    )
                for entry in tag.entries:
                    if entry.startswith("#"):
                        nested = entry[1:]
                        if self.get(registry_id, nested) is None:
                            problems.append(
                                f"{registry_id}/{tag.name}: nested tag #{nested} is not "
                                f"defined in the same registry"
                            )
                        continue
                    if resolve is not None and resolve(registry_id, entry) is None:
                        problems.append(
                            f"{registry_id}/{tag.name}: entry {entry} does not resolve"
                        )
        return problems


__all__ = ["TagRegistry", "NumericResolver", "ResolutionReport"]
