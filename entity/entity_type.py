"""
Entity types - the typed view over `minecraft:entity_type`.

`assets/entities.json` (157 types, protocol 774, verified id-for-id against
Pumpkin's `42c903493` report) is already loaded by `registry/tables.py` as an
`IndexedRegistry` of raw dicts. What the raw table does not give you is
fields with names, and "an entity" is exactly a bag of named fields: its
collision box (`dimension` = [width, height]), where its camera sits
(`eye_height`), whether it is a mob, and - the one this module adds from a
second file - how far away a client may see it at all.

That second file is `assets/entity_tracking.json`. It exists because
`entities.json` at the 1.21.11 commit does NOT carry
`client_tracking_range` / `update_interval` / `track_deltas`; vanilla's
report only grew those columns in the 26.3 line. The values are taken by
entity NAME from the newer report (the names agree across both) and kept in
their own asset, and `tools/check_entity_tracking.py` fails loudly if the
two files ever drift apart. Do not merge them back into `entities.json`
without re-verifying the ids: the 26.3 report's *numeric ids are shifted*
(`player` is 155 at 774, 159 at 26.3) and merging naively would silently
spawn the wrong types.

The registry keeps the wire ids honest - `tables.py` asserts they are dense
from 0 at load - so the dense list here is safe to index with any id a
packet carries.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

from loguru import logger

from registry.core import canonical_id
from registry.game_registry import get_game_registry


@dataclass(slots=True)
class EntityType:
    """One entry of the `minecraft:entity_type` registry.

    Field names follow the JSON, with the two flat-out renames a Python
    reader benefits from: `dimension` becomes width/height (it is a
    two-element array in the file, and a field called "dimension" on a class
    that also has a *world* dimension is a trap), and `mob` becomes
    `is_mob` so `entity_type.mob` does not read like a noun.
    """

    name: str
    """Canonical id, e.g. `minecraft:zombie`."""

    numeric_id: int
    """The protocol entity-type id. `add_entity` writes exactly this."""

    width: float
    height: float
    eye_height: float

    category: str
    """Vanilla's MobCategory: MISC, MONSTER, CREATURE, ..."""

    is_mob: bool
    attackable: bool
    summonable: bool
    saveable: bool
    fire_immune: bool
    limit_per_chunk: int
    can_spawn_far_from_player: bool
    max_health: Optional[float]
    spawn_restriction: Dict[str, object]

    client_tracking_range: int
    """Blocks/16 the entity stays visible at, before the player's own
    view distance clamps it. 0 means never tracked (`marker`)."""

    update_interval: int
    """Ticks between automatic position syncs. `Integer.MAX_VALUE`
    (2147483647) is vanilla's "never on a timer" sentinel."""

    track_deltas: bool
    """Whether the tracker pushes velocity packets for this type. False for
    `player` on purpose: players drive their own movement packets."""

    @property
    def resource_name(self) -> str:
        """The bare name as the client's registries spell it, `zombie`."""
        _, _, path = self.name.partition(":")
        return path or self.name

    def __repr__(self) -> str:
        return f"<EntityType {self.name} id={self.numeric_id}>"


_lock = threading.Lock()
_by_id: Optional[List[EntityType]] = None
_by_name: Optional[Dict[str, EntityType]] = None


def load_entity_types() -> List[EntityType]:
    """Build the typed entity types from the loaded game registry.

    Idempotent: the first call builds and caches; later calls return the
    same list. Raises if called before `init_registries()` - silently
    returning an empty table here would push the failure to the first
    spawn, where it would look like a network bug instead of an ordering
    one.
    """
    global _by_id, _by_name
    with _lock:
        if _by_id is not None:
            return _by_id

        game = get_game_registry()
        raw_registry = game.tables.entity_types.registry
        tracking = game.tables.entity_tracking.data

        by_id: List[EntityType] = []
        by_name: Dict[str, EntityType] = {}

        for key in raw_registry.keys():
            record = raw_registry.get(key)
            name = canonical_id(key)
            numeric_id = int(record["id"])
            width, height = (float(v) for v in record.get("dimension", (0.6, 1.8)))
            # entity_tracking.json is keyed by bare names (`acacia_boat`),
            # the registry by canonical ones (`minecraft:acacia_boat`).
            _, _, bare_name = name.partition(":")
            track = tracking.get(bare_name or name, {})

            if not track:
                # No silent default: an entity with no tracking entry cannot
                # be given a visibility range, and `tools/check_entity_tracking.py`
                # exists to catch exactly this gap at asset level.
                raise ValueError(
                    f"entity type {name} has no entry in entity_tracking.json; "
                    f"run tools/check_entity_tracking.py"
                )

            entity_type = EntityType(
                name=name,
                numeric_id=numeric_id,
                width=width,
                height=height,
                eye_height=float(record.get("eye_height", height * 0.9)),
                category=str(record.get("category", "MISC")),
                is_mob=bool(record.get("mob", False)),
                attackable=bool(record.get("attackable", False)),
                summonable=bool(record.get("summonable", True)),
                saveable=bool(record.get("saveable", True)),
                fire_immune=bool(record.get("fire_immune", False)),
                limit_per_chunk=int(record.get("limit_per_chunk", 0)),
                can_spawn_far_from_player=bool(
                    record.get("can_spawn_far_from_player", False)
                ),
                max_health=(
                    float(record["max_health"])
                    if "max_health" in record else None
                ),
                spawn_restriction=dict(record.get("spawn_restriction", {})),
                client_tracking_range=int(track["client_tracking_range"]),
                update_interval=int(track["update_interval"]),
                track_deltas=bool(track.get("track_deltas", True)),
            )

            while len(by_id) <= numeric_id:
                by_id.append(None)  # type: ignore[list-item]
            if by_id[numeric_id] is not None:
                raise ValueError(
                    f"duplicate entity type id {numeric_id}: "
                    f"{by_id[numeric_id].name} vs {name}"
                )
            by_id[numeric_id] = entity_type
            by_name[name] = entity_type

        # entities.json ids are dense (tables.py asserts contiguity at load);
        # if holes ever appear here, a type is missing from the file and the
        # dense-index contract below is broken.
        holes = [i for i, t in enumerate(by_id) if t is None]
        if holes:
            raise ValueError(
                f"entity type ids are not dense; missing: {holes}"
            )

        _by_id = by_id
        _by_name = by_name
        logger.debug(f"Entity types: {len(by_id)} types loaded")
        return _by_id


def _ensure_loaded() -> None:
    if _by_id is None:
        load_entity_types()


def entity_type_by_id(numeric_id: int) -> Optional[EntityType]:
    """The type with this protocol id, or None if out of range.

    Indexing a list with an untrusted packet field must not raise - a bad
    id in a hostile packet is a "log and ignore", not a crash.
    """
    _ensure_loaded()
    if 0 <= numeric_id < len(_by_id):
        return _by_id[numeric_id]
    return None


def entity_type_by_name(name: str) -> Optional[EntityType]:
    """The type by canonical or bare name (`minecraft:zombie` or `zombie`)."""
    _ensure_loaded()
    return _by_name.get(canonical_id(name))


def entity_type_count() -> int:
    """How many types are loaded (157 at 1.21.11)."""
    _ensure_loaded()
    return len(_by_id)
