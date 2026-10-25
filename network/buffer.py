"""
ByteBuffer - Buffer for reading/writing Minecraft protocol data types

Based on Cuberite's cByteBuffer and Pumpkin's serialization approach.
"""

from __future__ import annotations

import json
import math
import struct
import zlib
from typing import Optional
from io import BytesIO

from datatypes.nbt import write_nbt


class ByteBuffer:
    """A buffer for reading and writing Minecraft protocol data types.
    
    Supports:
    - VarInt (variable-length integer)
    - VarLong (variable-length long)
    - String (UTF-8 with VarInt length prefix)
    - Position (64-bit block position)
    - UUID
    - Primitive types (int8, int16, int32, int64, float, double, bool)
    """

    # VarInt encoding masks
    VARINT_CONTINUE_MASK = 0x80  # 0b10000000
    VARINT_VALUE_MASK = 0x7F     # 0b01111111

    def __init__(self, initial_capacity: int = 8192):
        self._data = bytearray()
        self._read_offset = 0

    def write(self, data: bytes) -> None:
        """Write raw bytes to the buffer."""
        self._data.extend(data)

    def read(self, length: int) -> bytes:
        """Read raw bytes from the buffer."""
        if len(self._data) - self._read_offset < length:
            raise BufferError(f"Need {length} bytes, only {len(self._data) - self._read_offset} available")
        
        result = bytes(self._data[self._read_offset:self._read_offset + length])
        self._read_offset += length
        return result

    def peek(self, length: int) -> bytes:
        """Peek at bytes without advancing the read offset."""
        if len(self._data) - self._read_offset < length:
            raise BufferError(f"Need {length} bytes, only {len(self._data) - self._read_offset} available")
        return bytes(self._data[self._read_offset:self._read_offset + length])

    def readable_bytes(self) -> int:
        """Return the number of bytes available for reading."""
        return len(self._data) - self._read_offset

    def used_space(self) -> int:
        """Return the total bytes in the buffer."""
        return len(self._data)

    def reset_read(self) -> None:
        """Reset the read offset to the beginning."""
        self._read_offset = 0

    def commit_read(self) -> None:
        """Remove read bytes from the buffer."""
        if self._read_offset > 0:
            self._data = self._data[self._read_offset:]
            self._read_offset = 0

    def clear(self) -> None:
        """Clear all data from the buffer."""
        self._data = bytearray()
        self._read_offset = 0

    @staticmethod
    def get_varint_size(value: int) -> int:
        """Get the number of bytes needed to encode a VarInt."""
        if value < 0:
            # Handle negative values - use 10 bytes for signed ints
            return 10
        
        if value < 0x80:
            return 1
        elif value < 0x4000:
            return 2
        elif value < 0x200000:
            return 3
        elif value < 0x10000000:
            return 4
        return 5

    # ==================== Write Methods ====================

    def write_varint(self, value: int) -> int:
        """Write an unsigned VarInt to the buffer. Returns bytes written.
        
        Note: For signed integers, use write_signed_varint().
        """
        if value < 0:
            raise ValueError(f"Cannot write negative value to unsigned VarInt: {value}")
        
        written = 0
        while True:
            if (value & ~self.VARINT_VALUE_MASK) == 0:
                self._data.append(value & 0xFF)
                return written + 1
            self._data.append((value & 0x7F) | self.VARINT_CONTINUE_MASK)
            value >>= 7
            written += 1

    def write_signed_varint(self, value: int) -> int:
        """Write a zigzag-encoded signed VarInt."""
        # Zigzag encode: (n << 1) ^ (n >> 31)
        unsigned = (value << 1) ^ (value >> 31)
        return self.write_varint(unsigned)

    def write_varlong(self, value: int) -> int:
        """Write a VarLong to the buffer. Returns bytes written."""
        # Handle signed integers
        if value < 0:
            value = (value << 1) ^ (value >> 63)
        
        written = 0
        while True:
            if (value & ~self.VARINT_VALUE_MASK) == 0:
                self._data.append(value & 0xFF)
                return written + 1
            self._data.append((value & 0x7F) | self.VARINT_CONTINUE_MASK)
            value >>= 7
            written += 1

    def write_byte(self, value: int) -> None:
        """Write a single byte."""
        self._data.append(value & 0xFF)

    def write_short(self, value: int) -> None:
        """Write a signed 16-bit integer (big-endian)."""
        self._data.extend(struct.pack(">h", value))

    def write_int(self, value: int) -> None:
        """Write a signed 32-bit integer (big-endian)."""
        self._data.extend(struct.pack(">i", value))

    def write_long(self, value: int) -> None:
        """Write a signed 64-bit integer (big-endian)."""
        self._data.extend(struct.pack(">q", value))

    def write_float(self, value: float) -> None:
        """Write a 32-bit float (big-endian)."""
        self._data.extend(struct.pack(">f", value))

    def write_double(self, value: float) -> None:
        """Write a 64-bit double (big-endian)."""
        self._data.extend(struct.pack(">d", value))

    def write_bool(self, value: bool) -> None:
        """Write a boolean (1 byte)."""
        self._data.append(1 if value else 0)

    def write_string(self, value: str) -> int:
        """Write a string (UTF-8 with VarInt length prefix). Returns bytes written."""
        encoded = value.encode("utf-8")
        length_written = self.write_varint(len(encoded))
        self._data.extend(encoded)
        return length_written + len(encoded)

    def write_component(self, value: str) -> None:
        """Write a text component as network NBT.

        `value` is a JSON text component (e.g. '{"text": "Server closed"}').
        Since 1.20.3 components travel as NBT, not JSON strings: the JSON is
        converted to a compound tag, and write_nbt emits the leading 0x0A
        type byte, entries and TAG_End with no root name (1.20.2+ network
        format). Sending the JSON bytes verbatim makes the client read the
        VarInt length as a tag id ("Invalid tag id: 25" for a 25-byte JSON).
        """
        self.write(write_nbt(json.loads(value)))

    def write_position(self, x: int, y: int, z: int) -> None:
        """Write a 64-bit block position."""
        # Position is encoded as: ((x & 0x3FFFFFF) << 38) | ((z & 0x3FFFFFF) << 12) | (y & 0xFFF)
        position = ((x & 0x3FFFFFF) << 38) | ((z & 0x3FFFFFF) << 12) | (y & 0xFFF)
        self.write_long(position)

    def write_uuid(self, uuid: str) -> bytes:
        """Write a UUID from string format (with or without dashes)."""
        # Remove dashes
        uuid_clean = uuid.replace("-", "")
        
        # Convert hex to bytes (16 bytes)
        uuid_bytes = bytes.fromhex(uuid_clean)
        
        self._data.extend(uuid_bytes)
        return uuid_bytes

    def write_bytes(self, data: bytes) -> None:
        """Write raw bytes."""
        self._data.extend(data)

    # ==================== Angle helpers ====================
    #
    # Four packets encode rotation as a scaled byte (Add Entity, both
    # Move Entity Pos Rot/Rot, Rotate Head), and each of them writing
    # floor(d * 256 / 360) by hand is how a one-line change misses a copy.
    # The static pair is also the single implementation the entity tracker
    # compares against, so "did the angle change on the wire" and "what the
    # wire says" can never disagree.

    @staticmethod
    def pack_angle(degrees: float) -> int:
        """Degrees -> the wire's angle byte: floor(d * 256 / 360), wrapped.

        Yaw wraps through & 0xFF exactly like vanilla's byte cast (yaw -90
        becomes 192); pitch relies on the same wrap, which is why callers
        must NOT pre-normalize pitch into [-90, 90] *and* apply wrap twice.
        """
        return int(math.floor(degrees * 256.0 / 360.0)) & 0xFF

    @staticmethod
    def unpack_angle(packed: int) -> float:
        """The angle byte -> degrees, the way the client reads it.

        Java reads these bytes as SIGNED (a byte 224 is -32, i.e. -45
        degrees of pitch), so this mirrors that: interpret first, scale
        second. For yaw the signed and unsigned readings describe the same
        direction - yaw is modular on the client - but pitch is not, and a
        pitch that unpacked as +315 would have the player staring at the
        sky behind them.
        """
        signed = packed - 256 if packed >= 128 else packed
        return signed * 360.0 / 256.0

    def write_angle(self, degrees: float) -> None:
        """Write a rotation as its scaled angle byte."""
        self._data.append(self.pack_angle(degrees))

    def read_angle(self) -> float:
        """Read a rotation angle byte, back as degrees."""
        return self.unpack_angle(self.read_unsigned_byte())

    # ==================== Read Methods ====================

    def read_varint(self) -> int:
        """Read a VarInt from the buffer.
        
        Note: Minecraft uses unsigned VarInt for most fields.
        Use read_signed_varint() for zigzag-encoded signed integers.
        """
        result = 0
        shift = 0
        
        for _ in range(10):  # Max 10 bytes for VarInt
            if self._read_offset >= len(self._data):
                raise BufferError("Unexpected end of buffer while reading VarInt")
            
            byte = self._data[self._read_offset]
            self._read_offset += 1
            
            result |= (byte & 0x7F) << shift
            
            if (byte & 0x80) == 0:
                return result
            
            shift += 7
        
        raise ValueError("VarInt too large")

    def read_signed_varint(self) -> int:
        """Read a zigzag-encoded signed VarInt."""
        unsigned = self.read_varint()
        # Zigzag decode: (unsigned >> 1) ^ -(unsigned & 1)
        return (unsigned >> 1) ^ (-(unsigned & 1))

    def read_varlong(self) -> int:
        """Read a VarLong from the buffer."""
        result = 0
        shift = 0
        
        for _ in range(10):  # Max 10 bytes for VarLong
            if self._read_offset >= len(self._data):
                raise BufferError("Unexpected end of buffer while reading VarLong")
            
            byte = self._data[self._read_offset]
            self._read_offset += 1
            
            result |= (byte & 0x7F) << shift
            
            if (byte & 0x80) == 0:
                # Decode signed integer
                if result & 1:
                    result = -(result ^ 1)
                return result
            
            shift += 7
        
        raise ValueError("VarLong too large")

    def read_byte(self) -> int:
        """Read a single signed byte."""
        if self._read_offset >= len(self._data):
            raise BufferError("Unexpected end of buffer")
        value = self._data[self._read_offset]
        self._read_offset += 1
        # Convert to signed
        if value >= 128:
            value -= 256
        return value

    def read_unsigned_byte(self) -> int:
        """Read a single unsigned byte."""
        if self._read_offset >= len(self._data):
            raise BufferError("Unexpected end of buffer")
        value = self._data[self._read_offset]
        self._read_offset += 1
        return value

    def read_short(self) -> int:
        """Read a signed 16-bit integer."""
        if self._read_offset + 2 > len(self._data):
            raise BufferError("Unexpected end of buffer")
        value = struct.unpack(">h", bytes(self._data[self._read_offset:self._read_offset + 2]))[0]
        self._read_offset += 2
        return value

    def read_int(self) -> int:
        """Read a signed 32-bit integer."""
        if self._read_offset + 4 > len(self._data):
            raise BufferError("Unexpected end of buffer")
        value = struct.unpack(">i", bytes(self._data[self._read_offset:self._read_offset + 4]))[0]
        self._read_offset += 4
        return value

    def read_long(self) -> int:
        """Read a signed 64-bit integer."""
        if self._read_offset + 8 > len(self._data):
            raise BufferError("Unexpected end of buffer")
        value = struct.unpack(">q", bytes(self._data[self._read_offset:self._read_offset + 8]))[0]
        self._read_offset += 8
        return value

    def read_float(self) -> float:
        """Read a 32-bit float."""
        if self._read_offset + 4 > len(self._data):
            raise BufferError("Unexpected end of buffer")
        value = struct.unpack(">f", bytes(self._data[self._read_offset:self._read_offset + 4]))[0]
        self._read_offset += 4
        return value

    def read_double(self) -> float:
        """Read a 64-bit double."""
        if self._read_offset + 8 > len(self._data):
            raise BufferError("Unexpected end of buffer")
        value = struct.unpack(">d", bytes(self._data[self._read_offset:self._read_offset + 8]))[0]
        self._read_offset += 8
        return value

    def read_bool(self) -> bool:
        """Read a boolean."""
        return self.read_unsigned_byte() != 0

    def read_string(self) -> str:
        """Read a string (UTF-8 with VarInt length prefix)."""
        length = self.read_varint()
        if self._read_offset + length > len(self._data):
            raise BufferError(f"Unexpected end of buffer: need {length} bytes")
        
        value = bytes(self._data[self._read_offset:self._read_offset + length]).decode("utf-8")
        self._read_offset += length
        return value

    def read_component(self) -> str:
        """Read a text component (NBT) and return it as a JSON string.

        Inverse of write_component. The component field has three wire
        forms, distinguished by the first byte:

          0x00 -- empty component, no payload at all
          0x08 -- bare TAG_String: a literal-text component, i.e. the JSON
                  is just that string
          0x0A -- compound: the full component object (entries + TAG_End,
                  as produced by write_component)

        Vanilla and Pumpkin servers emit the bare-string form for plain
        text kicks, so tooling that decodes their captures needs all three.
        """
        # TODO(user): implement — see docstring for the three forms.
        # 0x0A case: datatypes.read_nbt (network_format=True) gives the
        # dict; json.dumps turns it back into the JSON string the packet
        # fields hold. The other two forms need direct NBTBuffer reads.
        raise NotImplementedError("read_component is not implemented yet")

    def read_position(self) -> tuple[int, int, int]:
        """Read a 64-bit block position. Returns (x, y, z)."""
        pos = self.read_long()
        x = (pos >> 38) & 0x3FFFFFF
        z = (pos >> 12) & 0x3FFFFFF
        y = pos & 0xFFF
        
        # Sign extension
        if x >= 0x2000000:
            x -= 0x4000000
        if z >= 0x2000000:
            z -= 0x4000000
        if y >= 0x800:
            y -= 0x1000
        
        return (x, y, z)

    def read_uuid(self) -> str:
        """Read a 128-bit UUID. Returns UUID string."""
        if self._read_offset + 16 > len(self._data):
            raise BufferError("Unexpected end of buffer")
        
        uuid_bytes = self._data[self._read_offset:self._read_offset + 16]
        self._read_offset += 16
        
        # Format as standard UUID string
        uuid_int = int.from_bytes(uuid_bytes, byteorder="big")
        return f"{uuid_int:032x}"

    def read_bytes(self, length: int) -> bytes:
        """Read raw bytes."""
        return self.read(length)

    # ==================== Utility Methods ====================

    def write_compressed(self, data: bytes, threshold: int) -> int:
        """Write data with optional compression. Returns total bytes written."""
        if len(data) >= threshold:
            # Compress the data
            compressed = zlib.compress(data)
            self.write_varint(len(compressed))
            self.write_bytes(compressed)
            return 1 + len(compressed)
        else:
            # Write as uncompressed
            self.write_varint(0)
            self.write_bytes(data)
            return 1 + len(data)

    def read_compressed(self) -> bytes:
        """Read compressed data. Returns decompressed bytes."""
        data_length = self.read_varint()
        
        if data_length == 0:
            # Uncompressed
            return self.read_bytes(self.readable_bytes())
        
        compressed = self.read_bytes(data_length)
        return zlib.decompress(compressed)

    def get_writable(self) -> bytes:
        """Get all data as bytes."""
        return bytes(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __bytes__(self) -> bytes:
        return bytes(self._data)
