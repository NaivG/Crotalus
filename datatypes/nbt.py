"""
NBT (Named Binary Tag) Parser and Encoder for Minecraft Protocol

Implements NBT serialization/deserialization following the Java edition format.
Supports all NBT tag types used in Minecraft 1.21+ registry data.

NBT Format Specification:
- TAG_End (0): No payload, marks end of compound
- TAG_Byte (1): 1 byte signed integer
- TAG_Short (2): 2 byte signed integer (big-endian)
- TAG_Int (3): 4 byte signed integer (big-endian)
- TAG_Long (4): 8 byte signed integer (big-endian)
- TAG_Float (5): 4 byte IEEE 754 float (big-endian)
- TAG_Double (6): 8 byte IEEE 754 double (big-endian)
- TAG_Byte_Array (7): TAG_Int (length) + length bytes
- TAG_String (8): TAG_Short (length) + UTF-8 bytes
- TAG_List (9): TAG_Byte (element type) + TAG_Int (length) + elements
- TAG_Compound (10): Series of TAG_Type + name + value, terminated by TAG_End
- TAG_Int_Array (11): TAG_Int (length) + length * 4 bytes
- TAG_Long_Array (12): TAG_Int (length) + length * 8 bytes
"""

from __future__ import annotations

import struct
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import IntEnum


class NBTType(IntEnum):
    """NBT tag type IDs matching Java edition."""
    END = 0
    BYTE = 1
    SHORT = 2
    INT = 3
    LONG = 4
    FLOAT = 5
    DOUBLE = 6
    BYTE_ARRAY = 7
    STRING = 8
    LIST = 9
    COMPOUND = 10
    INT_ARRAY = 11
    LONG_ARRAY = 12

    @classmethod
    def to_python(cls, type_id: int) -> type:
        """Get Python type for NBT type ID."""
        mapping = {
            cls.BYTE: int,
            cls.SHORT: int,
            cls.INT: int,
            cls.LONG: int,
            cls.FLOAT: float,
            cls.DOUBLE: float,
            cls.BYTE_ARRAY: bytes,
            cls.STRING: str,
            cls.LIST: list,
            cls.COMPOUND: dict,
            cls.INT_ARRAY: list,
            cls.LONG_ARRAY: list,
        }
        return mapping.get(type_id, Any)


@dataclass
class NBTCompound:
    """Represents an NBT compound tag (named list of tags)."""
    name: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.payload[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.payload

    def __len__(self) -> int:
        return len(self.payload)

    def __iter__(self):
        return iter(self.payload)

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)

    def keys(self):
        return self.payload.keys()

    def values(self):
        return self.payload.values()

    def items(self):
        return self.payload.items()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dictionary."""
        return self.payload

    @staticmethod
    def from_dict(name: str, data: Dict[str, Any]) -> 'NBTCompound':
        """Create from dictionary."""
        return NBTCompound(name=name, payload=data)


class NBTBuffer:
    """Buffer for reading/writing NBT data."""

    def __init__(self, data: bytes = b""):
        self._data = bytearray(data)
        self._offset = 0

    def readable_bytes(self) -> int:
        return len(self._data) - self._offset

    def get_writable(self) -> bytes:
        return bytes(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __bytes__(self) -> bytes:
        return bytes(self._data)

    # ==================== Read Methods ====================

    def read_byte(self) -> int:
        """Read a signed byte."""
        if self._offset >= len(self._data):
            raise BufferError("Unexpected end of buffer")
        value = self._data[self._offset]
        self._offset += 1
        if value >= 128:
            value -= 256
        return value

    def read_unsigned_byte(self) -> int:
        """Read an unsigned byte."""
        if self._offset >= len(self._data):
            raise BufferError("Unexpected end of buffer")
        value = self._data[self._offset]
        self._offset += 1
        return value

    def read_short(self) -> int:
        """Read a signed short (2 bytes, big-endian)."""
        if self._offset + 2 > len(self._data):
            raise BufferError("Unexpected end of buffer")
        value = struct.unpack(">h", bytes(self._data[self._offset:self._offset + 2]))[0]
        self._offset += 2
        return value

    def read_int(self) -> int:
        """Read a signed int (4 bytes, big-endian)."""
        if self._offset + 4 > len(self._data):
            raise BufferError("Unexpected end of buffer")
        value = struct.unpack(">i", bytes(self._data[self._offset:self._offset + 4]))[0]
        self._offset += 4
        return value

    def read_long(self) -> int:
        """Read a signed long (8 bytes, big-endian)."""
        if self._offset + 8 > len(self._data):
            raise BufferError("Unexpected end of buffer")
        value = struct.unpack(">q", bytes(self._data[self._offset:self._offset + 8]))[0]
        self._offset += 8
        return value

    def read_float(self) -> float:
        """Read a float (4 bytes, big-endian)."""
        if self._offset + 4 > len(self._data):
            raise BufferError("Unexpected end of buffer")
        value = struct.unpack(">f", bytes(self._data[self._offset:self._offset + 4]))[0]
        self._offset += 4
        return value

    def read_double(self) -> float:
        """Read a double (8 bytes, big-endian)."""
        if self._offset + 8 > len(self._data):
            raise BufferError("Unexpected end of buffer")
        value = struct.unpack(">d", bytes(self._data[self._offset:self._offset + 8]))[0]
        self._offset += 8
        return value

    def read_string(self) -> str:
        """Read an NBT string (UTF-8 with short length prefix)."""
        length = self.read_short()
        if length < 0:
            raise ValueError(f"Invalid string length: {length}")
        if self._offset + length > len(self._data):
            raise BufferError("Unexpected end of buffer")
        value = bytes(self._data[self._offset:self._offset + length]).decode("utf-8")
        self._offset += length
        return value

    def read_bytes(self, length: int) -> bytes:
        """Read raw bytes."""
        if self._offset + length > len(self._data):
            raise BufferError("Unexpected end of buffer")
        value = bytes(self._data[self._offset:self._offset + length])
        self._offset += length
        return value

    def peek_byte(self) -> int:
        """Peek at next byte without advancing offset."""
        if self._offset >= len(self._data):
            raise BufferError("Unexpected end of buffer")
        return self._data[self._offset]

    # ==================== Write Methods ====================

    def write_byte(self, value: int) -> None:
        """Write a signed byte."""
        self._data.append(value & 0xFF)

    def write_unsigned_byte(self, value: int) -> None:
        """Write an unsigned byte."""
        self._data.append(value & 0xFF)

    def write_short(self, value: int) -> None:
        """Write a signed short (2 bytes, big-endian)."""
        self._data.extend(struct.pack(">h", value))

    def write_int(self, value: int) -> None:
        """Write a signed int (4 bytes, big-endian)."""
        self._data.extend(struct.pack(">i", value))

    def write_long(self, value: int) -> None:
        """Write a signed long (8 bytes, big-endian)."""
        self._data.extend(struct.pack(">q", value))

    def write_float(self, value: float) -> None:
        """Write a float (4 bytes, big-endian)."""
        self._data.extend(struct.pack(">f", value))

    def write_double(self, value: float) -> None:
        """Write a double (8 bytes, big-endian)."""
        self._data.extend(struct.pack(">d", value))

    def write_string(self, value: str) -> None:
        """Write an NBT string (UTF-8 with short length prefix)."""
        encoded = value.encode("utf-8")
        self.write_short(len(encoded))
        self._data.extend(encoded)

    def write_bytes(self, data: bytes) -> None:
        """Write raw bytes."""
        self._data.extend(data)


class NBTReader:
    """Reads NBT tags from a buffer."""

    def __init__(self, buf: NBTBuffer):
        self.buf = buf

    def read_tag(self) -> Tuple[str, Any]:
        """Read a single NBT tag. Returns (name, value) tuple."""
        tag_type = self.buf.read_unsigned_byte()

        if tag_type == NBTType.END:
            return ("", NBTType.END)

        # Read tag name
        name = self.buf.read_string()

        # Read value based on type
        value = self.read_value(tag_type)
        return (name, value)

    def read_value(self, tag_type: int) -> Any:
        """Read a value without reading the type/name prefix."""
        if tag_type == NBTType.END:
            return None
        elif tag_type == NBTType.BYTE:
            return self.buf.read_byte()
        elif tag_type == NBTType.SHORT:
            return self.buf.read_short()
        elif tag_type == NBTType.INT:
            return self.buf.read_int()
        elif tag_type == NBTType.LONG:
            return self.buf.read_long()
        elif tag_type == NBTType.FLOAT:
            return self.buf.read_float()
        elif tag_type == NBTType.DOUBLE:
            return self.buf.read_double()
        elif tag_type == NBTType.BYTE_ARRAY:
            length = self.buf.read_int()
            return bytes(self.buf.read_bytes(length))
        elif tag_type == NBTType.STRING:
            return self.buf.read_string()
        elif tag_type == NBTType.LIST:
            return self.read_list()
        elif tag_type == NBTType.COMPOUND:
            return self.read_compound()
        elif tag_type == NBTType.INT_ARRAY:
            length = self.buf.read_int()
            return [self.buf.read_int() for _ in range(length)]
        elif tag_type == NBTType.LONG_ARRAY:
            length = self.buf.read_int()
            return [self.buf.read_long() for _ in range(length)]
        else:
            raise ValueError(f"Unknown NBT tag type: {tag_type}")

    def read_list(self) -> list:
        """Read a TAG_List."""
        element_type = self.buf.read_unsigned_byte()
        length = self.buf.read_int()

        if length == 0:
            return []

        result = []
        for _ in range(length):
            value = self.read_value(element_type)
            result.append(value)
        return result

    def read_compound(self) -> Dict[str, Any]:
        """Read a TAG_Compound as a dictionary."""
        result = {}
        while True:
            tag_type = self.buf.read_unsigned_byte()
            if tag_type == NBTType.END:
                break
            name = self.buf.read_string()
            value = self.read_value(tag_type)
            result[name] = value
        return result


class NBTWriter:
    """Writes NBT tags to a buffer."""

    def __init__(self, buf: NBTBuffer):
        self.buf = buf

    def write_tag(self, name: str, value: Any, tag_type: Optional[int] = None) -> None:
        """Write a named NBT tag."""
        if value is None:
            # Write TAG_End for None
            self.buf.write_byte(NBTType.END)
            return

        # Auto-detect type if not specified
        if tag_type is None:
            tag_type = self._get_type(value)

        self.buf.write_unsigned_byte(tag_type)
        self.buf.write_string(name)
        self.write_value(value, tag_type)

    def write_value(self, value: Any, tag_type: Optional[int] = None) -> None:
        """Write a value without type prefix."""
        if tag_type is None:
            tag_type = self._get_type(value)

        if tag_type == NBTType.END:
            pass  # No data for end tag
        elif tag_type == NBTType.BYTE:
            self.buf.write_byte(value)
        elif tag_type == NBTType.SHORT:
            self.buf.write_short(value)
        elif tag_type == NBTType.INT:
            self.buf.write_int(value)
        elif tag_type == NBTType.LONG:
            self.buf.write_long(value)
        elif tag_type == NBTType.FLOAT:
            self.buf.write_float(value)
        elif tag_type == NBTType.DOUBLE:
            self.buf.write_double(value)
        elif tag_type == NBTType.BYTE_ARRAY:
            self.buf.write_int(len(value))
            self.buf.write_bytes(value)
        elif tag_type == NBTType.STRING:
            self.buf.write_string(value)
        elif tag_type == NBTType.LIST:
            self.write_list(value)
        elif tag_type == NBTType.COMPOUND:
            self.write_compound(value)
        elif tag_type == NBTType.INT_ARRAY:
            self.buf.write_int(len(value))
            for v in value:
                self.buf.write_int(v)
        elif tag_type == NBTType.LONG_ARRAY:
            self.buf.write_int(len(value))
            for v in value:
                self.buf.write_long(v)
        else:
            raise ValueError(f"Unknown NBT tag type: {tag_type}")

    def write_list(self, value: list) -> None:
        """Write a TAG_List."""
        if len(value) == 0:
            # Empty list: type 0 (END), length 0
            self.buf.write_unsigned_byte(NBTType.END)
            self.buf.write_int(0)
            return

        # Determine element type from first element
        element_type = self._get_type(value[0])

        self.buf.write_unsigned_byte(element_type)
        self.buf.write_int(len(value))

        for item in value:
            # If element is a compound, write it directly
            if element_type == NBTType.COMPOUND:
                self.write_compound(item)
            else:
                self.write_value(item, element_type)

    def write_compound(self, value: Dict[str, Any]) -> None:
        """Write a TAG_Compound."""
        for name, val in value.items():
            if val is None:
                continue  # Skip None values
            tag_type = self._get_type(val, name)
            self.buf.write_unsigned_byte(tag_type)
            self.buf.write_string(name)
            self.write_value(val, tag_type)
        # End tag
        self.buf.write_byte(NBTType.END)

    # Fields that should use TAG_Float instead of TAG_Double
    # Based on Minecraft 1.21.11 protocol requirements
    FLOAT_FIELDS = {
        'ambient_light', 'temperature', 'downfall', 'use_duration',
        'offset', 'item_model_index', 'max_inclusive', 'min_inclusive',
    }

    # Fields that MUST use TAG_Short (even if value fits in byte)
    SHORT_FIELDS = {
        'min_y', 'height', 'logical_height',
    }

    # Fields that MUST use TAG_Int
    INT_FIELDS = {
        'color',  # RGB color values
        'block_search_extent', 'tick_delay',
    }

    @staticmethod
    def _get_type(value: Any, field_name: str = None) -> int:
        """Determine NBT type from Python value.
        
        Args:
            value: Python value to determine type for
            field_name: Optional field name for context-aware type selection
        """
        if isinstance(value, bool):
            return NBTType.BYTE
        elif isinstance(value, int) and not isinstance(value, bool):
            # Check if field requires specific type
            if field_name and field_name in NBTWriter.SHORT_FIELDS:
                # Force SHORT for dimension fields
                if -32768 <= value <= 32767:
                    return NBTType.SHORT
                else:
                    return NBTType.INT
            elif field_name and field_name in NBTWriter.INT_FIELDS:
                return NBTType.INT
            # Check if it fits in a byte
            if -128 <= value <= 127:
                return NBTType.BYTE
            elif -32768 <= value <= 32767:
                return NBTType.SHORT
            elif -2147483648 <= value <= 2147483647:
                return NBTType.INT
            else:
                return NBTType.LONG
        elif isinstance(value, float):
            # Use FLOAT for specific fields that require it in Minecraft
            if field_name and field_name in NBTWriter.FLOAT_FIELDS:
                return NBTType.FLOAT
            return NBTType.DOUBLE
        elif isinstance(value, bytes):
            return NBTType.BYTE_ARRAY
        elif isinstance(value, str):
            return NBTType.STRING
        elif isinstance(value, list):
            if len(value) == 0:
                return NBTType.LIST  # Empty list is TAG_List with END elements
            # Check first element to determine type
            first = value[0]
            if isinstance(first, bool):
                return NBTType.BYTE_ARRAY
            elif isinstance(first, int):
                # For enchantment ranges [I;min,max] - use INT_ARRAY
                return NBTType.INT_ARRAY
            elif isinstance(first, float):
                return NBTType.LIST  # Float list
            elif isinstance(first, dict):
                return NBTType.LIST  # Compound list (like dimension attributes)
            elif isinstance(first, str):
                return NBTType.LIST  # String list
            else:
                return NBTType.LIST
        elif isinstance(value, dict):
            return NBTType.COMPOUND
        else:
            raise ValueError(f"Cannot determine NBT type for {type(value)}")


# ==================== Convenience Functions ====================

def read_nbt(data: bytes, network_format: bool = True) -> Dict[str, Any]:
    """Read NBT data from bytes. Returns compound as dictionary.

    Args:
        data: NBT bytes to parse
        network_format: If True (default), expect 1.20.2+ network format (root compound has no name)
                        If False, expect file format (root compound has name)

    Network NBT (1.20.2+): Root TAG_Compound has NO name field
    File NBT: Root TAG_Compound has name field
    """
    buf = NBTBuffer(data)
    reader = NBTReader(buf)
    tag_type = buf.read_unsigned_byte()

    if tag_type == NBTType.END:
        return {}

    if tag_type != NBTType.COMPOUND:
        raise ValueError(f"Expected TAG_Compound at root, got type {tag_type}")

    # Network NBT (1.20.2+) excludes the name from root compound
    if not network_format:
        name = buf.read_string()

    return reader.read_compound()


def write_nbt(data: Dict[str, Any], name: str = "", network_format: bool = True) -> bytes:
    """Write dictionary as NBT compound to bytes.

    Args:
        data: Dictionary to serialize as NBT
        name: Name for the root compound (ignored if network_format=True)
        network_format: If True (default), use 1.20.2+ network format (no name field for root compound)
                       If False, include name field (for file format)

    Network NBT (1.20.2+): Root TAG_Compound has NO name field
    File NBT: Root TAG_Compound has name field (usually empty)
    """
    buf = NBTBuffer()
    writer = NBTWriter(buf)

    buf.write_unsigned_byte(NBTType.COMPOUND)

    # Network NBT (1.20.2+) excludes the name from root compound
    if not network_format:
        buf.write_string(name)

    writer.write_compound(data)

    return bytes(buf)


def read_nbt_list(data: bytes) -> list:
    """Read NBT list data from bytes."""
    buf = NBTBuffer(data)
    reader = NBTReader(buf)
    return reader.read_list()


def write_nbt_list(data: list) -> bytes:
    """Write list as NBT list to bytes."""
    buf = NBTBuffer()
    writer = NBTWriter(buf)
    writer.write_list(data)
    return bytes(buf)


# ==================== NBT to SNBT conversion ====================

def nbt_to_snbt(data: Dict[str, Any], indent: int = 0) -> str:
    """Convert NBT data to SNBT (String NBT) format.

    This is useful for debugging and for registries that use SNBT in JSON.
    """
    if isinstance(data, dict):
        if len(data) == 0:
            return "{}"
        parts = []
        for key, value in data.items():
            parts.append(f'"{key}":{nbt_to_snbt(value, 0)}')
        return "{" + ",".join(parts) + "}"
    elif isinstance(data, list):
        if len(data) == 0:
            return "[]"
        parts = [nbt_to_snbt(item, 0) for item in data]
        return "[" + ",".join(parts) + "]"
    elif isinstance(data, str):
        return f'"{data}"'
    elif isinstance(data, bool):
        return "1b" if data else "0b"
    elif isinstance(data, int):
        return f"{data}"
    elif isinstance(data, float):
        return f"{data}"
    else:
        return str(data)
