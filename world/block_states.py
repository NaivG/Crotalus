"""
Block state ids - one numbering, shared by chunk storage, the wire and the
registry.

Where the numbers come from
---------------------------
A chunk section stores **global block state ids**, the same numbers the client
sees in a chunk palette and the same numbers `assets/blocks.json` assigns
(`registry.entries.BlockState.id`, 0..29670 in 1.21.11). There is no second,
private id space any more.

Earlier versions kept a private 1.8-era numbering in `world/chunk.py` and
translated here. The translation cost more than it looked:

* The table was allocated at boot, so what a saved chunk *meant* depended on
  what had been loaded when it was written. A plugin loading in a different
  order, or not loading at all, turned one block into another - silently, in a
  file on disk.
* 33 ids existed. The worldgen data in `assets/` names 176 different blocks
  (and 61 non-default states of them), so the translation had nothing to return
  for most of what Minecraft's own data asks to place.
* Every read needed a second lookup to name a block, which is the thing a
  region file actually wants to store.

Storing the global id removes the translation: the generator places a state id,
the section holds it, the codec writes it, and the palette on disk names it.

What still has to be careful
----------------------------
* **Air is 0** - empty sections and the homogeneous palette encoding depend on
  it. `validate()` checks it against the registry.
* A state id is only meaningful for the `assets/blocks.json` it came from. That
  is why region files store `{Name, Properties}` and not this number: see
  `world/anvil.py`. In memory and on the wire the id is correct by definition,
  because both sides were built from the same data. A world saved by a build
  with different block data is *not* migrated - this server is a demo, and a
  region file that names a block this build does not have reads as air with a
  warning rather than being guessed at.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# Protocol this table was built for.
PROTOCOL_VERSION = 774
MC_VERSION = "1.21.11"

#: Air. Fixed at 0 by the registry (and by the protocol: an empty section is a
#: palette holding this one value).
AIR = 0

#: Names this server addresses blocks by in code. The values are read out of the
#: registry, so a rename in `assets/blocks.json` cannot leave a stale number
#: behind. `state_id()` takes any other block name - this table is only the
#: shortlist the generator and the commands use.
_MEMBERS: Dict[str, str] = {
    "AIR": "minecraft:air",
    "STONE": "minecraft:stone",
    "GRASS": "minecraft:grass_block",
    "DIRT": "minecraft:dirt",
    "COBBLESTONE": "minecraft:cobblestone",
    "PLANKS": "minecraft:oak_planks",
    "SAPLING": "minecraft:oak_sapling",
    "BEDROCK": "minecraft:bedrock",
    "WATER": "minecraft:water",
    "LAVA": "minecraft:lava",
    "SAND": "minecraft:sand",
    "GRAVEL": "minecraft:gravel",
    "GOLD_ORE": "minecraft:gold_ore",
    "IRON_ORE": "minecraft:iron_ore",
    "COAL_ORE": "minecraft:coal_ore",
    "LOG": "minecraft:oak_log",
    "LEAVES": "minecraft:oak_leaves",
    "SPONGE": "minecraft:sponge",
    "GLASS": "minecraft:glass",
    "LAPIS_ORE": "minecraft:lapis_ore",
    "LAPIS_BLOCK": "minecraft:lapis_block",
    "DISPENSER": "minecraft:dispenser",
    "SANDSTONE": "minecraft:sandstone",
    "NOTE_BLOCK": "minecraft:note_block",
    "DIAMOND_ORE": "minecraft:diamond_ore",
    "REDSTONE_ORE": "minecraft:redstone_ore",
    "DEEPSLATE_COAL_ORE": "minecraft:deepslate_coal_ore",
    "DEEPSLATE_GOLD_ORE": "minecraft:deepslate_gold_ore",
    "DEEPSLATE_IRON_ORE": "minecraft:deepslate_iron_ore",
    "DEEPSLATE_DIAMOND_ORE": "minecraft:deepslate_diamond_ore",
    "DEEPSLATE_LAPIS_ORE": "minecraft:deepslate_lapis_ore",
    "DEEPSLATE_REDSTONE_ORE": "minecraft:deepslate_redstone_ore",
    "EMERALD_ORE": "minecraft:emerald_ore",
    # Not in the old enum: the dimension presets name these, and before the
    # rewrite there was no id to place them with. `minecraft:netherrack` and
    # `minecraft:end_stone` are what `chunk_gen_settings.json` says the nether
    # and the end are made of.
    "NETHERRACK": "minecraft:netherrack",
    "END_STONE": "minecraft:end_stone",
    "DEEPSLATE": "minecraft:deepslate",
}


def _blocks():
    """The block registry, loading the game registry on first use."""
    from registry import get_game_registry

    return get_game_registry().blocks


class _BlockIds:
    """`BLOCKS.STONE` -> the global state id of `minecraft:stone`'s default state.

    Attribute access rather than a dict so a typo is an `AttributeError` at the
    line that has it, and so the numbers are resolved from the registry instead
    of being written down twice. Lookups are cached: the table is small and the
    generator asks for the same handful of ids millions of times.
    """

    __slots__ = ("_cache",)

    def __init__(self) -> None:
        self._cache: Dict[str, int] = {}

    def __getattr__(self, member: str) -> int:
        if member.startswith("_"):
            raise AttributeError(member)
        cached = self._cache.get(member)
        if cached is not None:
            return cached
        block_name = _MEMBERS.get(member)
        if block_name is None:
            raise AttributeError(
                f"{member!r} is not one of the block names this module knows "
                f"({', '.join(sorted(_MEMBERS))}). Use block_states.state_id("
                f"'minecraft:...') for any other block."
            )
        resolved = default_state_id(block_name)
        self._cache[member] = resolved
        return resolved

    def __dir__(self):
        return sorted(_MEMBERS)

    def __repr__(self) -> str:
        return f"<BlockIds members={len(_MEMBERS)} resolved={len(self._cache)}>"


#: The registry-backed block-id namespace. See `_BlockIds`.
BLOCKS = _BlockIds()


def refresh() -> None:
    """Drop cached ids, for when the global registry instance is replaced.

    Only `BLOCKS` caches; every other function here reads the registry per call.
    """
    BLOCKS._cache.clear()


# ==================== Lookup ====================

def default_state_id(block_name: str) -> int:
    """Global state id of a block's default state.

    Raises `KeyError` for a block the registry does not have: every caller of
    this is about to write the number into a chunk, and a missing block is a
    data problem, not a runtime condition.
    """
    block = _blocks().get_block(block_name)
    if block is None:
        raise KeyError(f"no block {block_name!r} in the block registry")
    return block.default_state_id


def state_id(block_name: str, properties: Optional[Dict[str, Any]] = None) -> int:
    """Global state id of `block_name` at `properties` (default state if omitted).

    `properties` are matched against the block's declared properties; a property
    the block does not have is a `KeyError` rather than a silent default, since
    a fluid-state property such as `falling` looking like a block property is a
    real trap in the worldgen data.
    """
    block = _blocks().get_block(block_name)
    if block is None:
        raise KeyError(f"no block {block_name!r} in the block registry")
    if not properties:
        return block.default_state_id
    return block.state_id_for(properties)


def try_state_id(block_name: str, properties: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """`state_id` but `None` instead of raising. For data that may be stale."""
    try:
        return state_id(block_name, properties)
    except (KeyError, ValueError):
        return None


def migrate_name(name: str, properties: Optional[Dict[str, Any]] = None) -> int:
    """Best-effort id for a name read out of a file; air when it is gone.

    This is the one place that has to tolerate a block that no longer exists:
    it runs while loading someone's world, and dropping a block is better than
    refusing to load the chunk.
    """
    resolved = try_state_id(name, properties)
    return AIR if resolved is None else resolved


def is_air(state_id_value: int) -> bool:
    """Whether a state id is air (or out of range, which chunks treat as air)."""
    if state_id_value == AIR:
        return True
    return _blocks().state(state_id_value) is None


def state(state_id_value: int):
    """The registry's `BlockState` for a global id, or None when unknown."""
    return _blocks().state(state_id_value)


def name_of(state_id_value: int) -> str:
    """Block name for a global state id; `minecraft:air` when unknown."""
    resolved = _blocks().state(state_id_value)
    return "minecraft:air" if resolved is None else resolved.block.name


def codec(state_id_value: int) -> Dict[str, Any]:
    """`{Name, Properties}` for a state id - what a section palette stores.

    Always carries `Name`. `Properties` is present only when the state has
    non-default-owned properties at all, matching how the client and the vanilla
    palette write it.
    """
    resolved = _blocks().state(state_id_value)
    if resolved is None:
        return {"Name": "minecraft:air"}
    return resolved.to_nbt()


# ==================== Validation ====================

def validate() -> None:
    """Check the ids this module hands out against the registry.

    Raises AssertionError on a problem. Cheap enough to run at boot, and it is
    how a broken `assets/blocks.json` becomes a loud failure instead of a client
    desync at runtime.
    """
    registry = _blocks()

    # Air must be state 0: it is the fallback for empty sections, and the
    # homogeneous encoding for a skipped section is exactly this VarInt.
    assert AIR == 0, "air must be state 0"
    air = registry.state(AIR)
    assert air is not None, "state 0 is not in the registry"
    assert air.block.name == "minecraft:air", f"state 0 is {air.block.name}, not air"
    assert air.is_default, "state 0 is not a default state"

    total = registry.total_states
    assert total <= 0x10000, f"{total} states do not fit the uint16 a section stores"

    for member, block_name in _MEMBERS.items():
        resolved = registry.get_block(block_name)
        assert resolved is not None, (
            f"BLOCKS.{member} names {block_name}, which is not in the block registry"
        )
        assert 0 <= resolved.default_state_id < total, (
            f"BLOCKS.{member} ({block_name}) resolves to state "
            f"{resolved.default_state_id}, outside the registry (0..{total - 1})"
        )


__all__ = [
    "PROTOCOL_VERSION",
    "MC_VERSION",
    "AIR",
    "BLOCKS",
    "default_state_id",
    "state_id",
    "try_state_id",
    "migrate_name",
    "is_air",
    "state",
    "name_of",
    "codec",
    "refresh",
    "validate",
]
