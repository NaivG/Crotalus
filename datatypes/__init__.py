"""
Datatypes Module

Common data types and serialization utilities for Minecraft protocol.
Used by network, world, entity, and other modules.
"""

from datatypes.nbt import (
    # Core NBT types
    NBTType,
    NBTCompound,
    NBTBuffer,
    NBTReader,
    NBTWriter,
    # NBT serialization functions
    read_nbt,
    write_nbt,
    read_nbt_list,
    write_nbt_list,
    nbt_to_snbt,
)

from datatypes.mini_message import (
    # MiniMessage parser
    MiniMessageParser,
    parse_mini_message,
    mini_message_to_json,
    # Text component
    TextComponent,
    # Color constants
    COLOR_NAMES,
    # Decoration mappings
    DECORATION_MAP,
)

__all__ = [
    # Core NBT types
    "NBTType",
    "NBTCompound",
    "NBTBuffer",
    "NBTReader",
    "NBTWriter",
    # NBT serialization functions
    "read_nbt",
    "write_nbt",
    "read_nbt_list",
    "write_nbt_list",
    "nbt_to_snbt",
    # MiniMessage parser
    "MiniMessageParser",
    "parse_mini_message",
    "mini_message_to_json",
    "TextComponent",
    "COLOR_NAMES",
    "DECORATION_MAP",
]
