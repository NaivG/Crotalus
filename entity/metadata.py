"""
Entity metadata - the Set Entity Data blob.

`set_entity_data` (0x61) carries a sequence of typed key/value entries that
describe an entity's client-side state: is it on fire, what pose is it in,
what name floats over it. The wire shape is:

    entry := index:u8, type:VarInt, value:<type-specific>
    blob  := entry*, then a single 0xFF terminator

A blob with zero entries is just the terminator - that is legal and is what
an entity with nothing non-default to say sends.

Two tables from `assets/` name the numbers, and keeping them straight is
the whole game here:

* `meta_data_type.json` maps a TYPE NAME to the wire type id
  (`byte` -> 0, `boolean` -> 8, `profile` -> 37, ...). This is the `type`
  VarInt, and it decides how the value is parsed.
* `tracked_data.json` maps a DATA KEY NAME to the entry index
  (`FLAGS` -> 0, `POSE` -> 6, `PROFILE` -> 17, ...). This is the `index`
  byte, and it decides what the client does with the value.

They are orthogonal on purpose - index 17 carries a type-37 value for a
player's skin profile - and conflating "the profile is number 17" with "the
profile type is 37" is the natural first bug of this layer. The writer
below takes both numbers explicitly and refuses to guess.

Value encodings used this round (the serializer grows with the types the
server actually sets):

* byte    - one byte, unsigned
* int     - VarInt
* float   - little-endian f32
* boolean - one byte, 0/1
* profile - the ResolvableProfile (see `write_profile_value`)

**The ResolvableProfile caveat.** No oracle for this layout exists in this
repository: Pumpkin never serialises it (its Rust MetaDataType enum stops
at the type table and its profile metadata path is unimplemented), and the
vendored protocol snapshot references a format page it does not contain.
The encoding here mirrors the one profile serializer this codebase has
that is known to work against a real 1.21.11 client -
`PlayerInfoEntry.write`'s ADD_PLAYER action, which sends `name` +
`properties: [{name, value, optional signature}]` and gets tab-list skins
rendering. It adds nothing else: no trailing texture/background fields,
which some client versions expect. If a spawned player renders without a
skin or the client logs a metadata decode error, THIS is the suspect - the
rest of the blob is spec-pinned. The plan's phase-8 procedure tests it in
isolation for exactly that reason.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from network.buffer import ByteBuffer

# Type ids from assets/meta_data_type.json (verified identical to Pumpkin's
# protocol-774 report at e46459915; Pumpkin's hand-written Rust enum claims
# profile = 36 and is stale - trust the generated JSON).
META_TYPE_BYTE = 0
META_TYPE_INTEGER = 1
META_TYPE_FLOAT = 3
META_TYPE_BOOLEAN = 8
META_TYPE_PROFILE = 37

# Entry indexes from assets/tracked_data.json.
DATA_INDEX_FLAGS = 0
DATA_INDEX_CUSTOM_NAME = 2
DATA_INDEX_NAME_VISIBLE = 3
DATA_INDEX_POSE = 6
DATA_INDEX_HEALTH = 9
DATA_INDEX_PLAYER_MODE_CUSTOMIZATION = 16
DATA_INDEX_PROFILE = 17

# The blob terminator. A real entry cannot start with 0xFF because indexes
# are the tracked_data table's, whose maximum is well below 255.
TERMINATOR = 0xFF


def _type_id(type_name: str) -> int:
    """A metadata type id by name, from the loaded table.

    Going through the asset rather than the constants above keeps one
    source of truth; the constants exist so hot paths need not load JSON.
    """
    from registry.game_registry import get_game_registry

    table = get_game_registry().tables.meta_data_types.data
    value = table.get(type_name)
    if value is None:
        raise KeyError(f"unknown metadata type {type_name!r}; known: {sorted(table)}")
    return int(value)


def _data_index(key: str) -> int:
    """A metadata entry index by tracked-data key name, from the loaded table."""
    from registry.game_registry import get_game_registry

    table = get_game_registry().tables.tracked_data.data
    value = table.get(key)
    if value is None:
        raise KeyError(f"unknown tracked data key {key!r}; known: {sorted(table)}")
    return int(value)


def write_entry(buf: ByteBuffer, index: int, type_id: int, value: Any) -> None:
    """One metadata entry: index, type, then the value in that type's shape.

    The index/type are written by the caller's numbers; only `value` is
    dispatched on the type id, and an unencodable type raises rather than
    writing a value the client will misparse.
    """
    buf.write_byte(index)
    buf.write_varint(type_id)

    if type_id == META_TYPE_BYTE:
        buf.write_byte(int(value) & 0xFF)
    elif type_id == META_TYPE_INTEGER:
        buf.write_varint(int(value))
    elif type_id == META_TYPE_FLOAT:
        buf.write_float(float(value))
    elif type_id == META_TYPE_BOOLEAN:
        buf.write_bool(bool(value))
    elif type_id == META_TYPE_PROFILE:
        write_profile_value(buf, value)
    else:
        raise ValueError(
            f"no encoder for metadata type id {type_id}; "
            f"add it to entity/metadata.py before using it"
        )


def write_blob(entries: Sequence[Tuple[int, int, Any]]) -> bytes:
    """A whole metadata blob: the entries, then the 0xFF terminator.

    Returns raw bytes because `CSetEntityMetadata` writes the blob as an
    unprefixed byte run, not as a length-prefixed array - the client reads
    entries until it sees the terminator.
    """
    buf = ByteBuffer(256)
    for index, type_id, value in entries:
        write_entry(buf, index, type_id, value)
    buf.write_byte(TERMINATOR)
    return bytes(buf)


def write_profile_value(buf: ByteBuffer, profile: Dict[str, Any]) -> None:
    """The ResolvableProfile metadata value for a player's skin.

    Shape (see the module docstring for why this layout is trusted only as
    far as PlayerInfoEntry's tab-list encoding is verified):

        name: String
        properties: VarInt count, then per property
            name: String, value: String,
            has_signature: Boolean, [signature: String]

    `profile` is {"name": str, "properties": [(name, value, signature|None)]}.
    An empty properties list is the offline-server normal: no session server
    was involved, so there are no signed textures, and the client renders
    the default skin keyed by the profile name's hash.
    """
    name = profile.get("name", "")
    properties = profile.get("properties", ())

    buf.write_string(str(name))
    buf.write_varint(len(properties))
    for prop in properties:
        prop_name, prop_value, prop_sig = (list(prop) + [None, None, None])[:3]
        buf.write_string(str(prop_name))
        buf.write_string(str(prop_value))
        buf.write_bool(prop_sig is not None)
        if prop_sig is not None:
            buf.write_string(str(prop_sig))


def player_spawn_blob(name: str, skin_parts: int = 0xFF,
                      properties: Iterable[Tuple] = ()) -> bytes:
    """The spawn-time blob for a player entity.

    Two entries, in Pumpkin's add_pairing order: the skin-parts byte
    (index 16, which tells the client which body layers to render), then
    the profile (index 17, which names the model and carries any signed
    textures). A freshly spawned player has no other non-default state,
    so this is the correct minimal payload.

    Pumpkin's 774 tree has this exact sequence commented out, so the
    combination is unverified by the oracle; the phase-8 two-client test
    is where it gets proven. On a vanilla offline server `properties` is
    empty and the client falls back to the default skin.
    """
    return write_blob([
        (DATA_INDEX_PLAYER_MODE_CUSTOMIZATION, META_TYPE_BYTE,
         int(skin_parts) & 0xFF),
        (
            DATA_INDEX_PROFILE,
            META_TYPE_PROFILE,
            {"name": name, "properties": list(properties)},
        ),
    ])


def mob_spawn_blob(health: float) -> bytes:
    """The spawn-time blob for a living entity that is not a player.

    **Health is not optional here.** The client's `LivingEntity` reads index
    9 and, finding nothing, takes the tracked-data default of 0 - at which
    point `isAlive()` is false and the client removes the entity it was just
    told to add. The symptom is an add_entity with no visible body and no
    error anywhere, which is why this is called out rather than left to
    "non-default values only": health is non-default for every mob that is
    alive, because the default is 0 and no mob is at 0 health.

    Everything else a vanilla mob sends at spawn - shared flags, pose, custom
    name - really is at its default for a freshly spawned animal, and sending
    it would only cost bytes. The 1.21.11 client accepts a one-entry blob;
    `tools/check_entity_delta.py` decodes one back and asserts the index.
    """
    return write_blob([(DATA_INDEX_HEALTH, META_TYPE_FLOAT, float(health))])
