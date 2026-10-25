"""
NBT (Named Binary Tag) Serialization Module

Implements NBT format for Minecraft level.dat and other game data.
Uses gzip compression for .dat files.
"""

import gzip
import struct
import zlib
from typing import Optional, Dict, Any, List, Union, Tuple
from dataclasses import dataclass, field
from enum import IntEnum
from io import BytesIO
import numpy as np


class NBTType(IntEnum):
    """NBT tag types."""
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


@dataclass
class NBTTag:
    """Base class for NBT tags."""
    name: str = ""
    
    def get_type(self) -> NBTType:
        raise NotImplementedError
    
    def write(self, buffer: BytesIO) -> None:
        raise NotImplementedError
    
    @staticmethod
    def read(buffer: BytesIO, nbt_type: NBTType) -> "NBTTag":
        """Read a tag from buffer."""
        if nbt_type == NBTType.END:
            return NBTEnd()
        elif nbt_type == NBTType.BYTE:
            return NBTByte.read(buffer)
        elif nbt_type == NBTType.SHORT:
            return NBTShort.read(buffer)
        elif nbt_type == NBTType.INT:
            return NBTInt.read(buffer)
        elif nbt_type == NBTType.LONG:
            return NBTLong.read(buffer)
        elif nbt_type == NBTType.FLOAT:
            return NBTFloat.read(buffer)
        elif nbt_type == NBTType.DOUBLE:
            return NBTDouble.read(buffer)
        elif nbt_type == NBTType.BYTE_ARRAY:
            return NBTByteArray.read(buffer)
        elif nbt_type == NBTType.STRING:
            return NBTString.read(buffer)
        elif nbt_type == NBTType.LIST:
            return NBTList.read(buffer)
        elif nbt_type == NBTType.COMPOUND:
            return NBTCompound.read(buffer)
        elif nbt_type == NBTType.INT_ARRAY:
            return NBTIntArray.read(buffer)
        elif nbt_type == NBTType.LONG_ARRAY:
            return NBTLongArray.read(buffer)
        raise ValueError(f"Unknown NBT type: {nbt_type}")


class NBTEnd(NBTTag):
    """NBT End tag (no data)."""
    
    def get_type(self) -> NBTType:
        return NBTType.END
    
    def write(self, buffer: BytesIO) -> None:
        pass
    
    @staticmethod
    def read(buffer: BytesIO) -> "NBTEnd":
        return NBTEnd()


class NBTByte(NBTTag):
    """NBT Byte tag."""
    
    def __init__(self, name: str = "", value: int = 0):
        super().__init__(name)
        self.value = value
    
    def get_type(self) -> NBTType:
        return NBTType.BYTE
    
    def write(self, buffer: BytesIO) -> None:
        # Signed, like the reader's `>b`: masking to 8 bits first would turn a
        # negative value (section Y below the world floor) into 128..255 and
        # overflow the signed pack.
        buffer.write(struct.pack(">b", self.value))
    
    @staticmethod
    def read(buffer: BytesIO) -> "NBTByte":
        value = struct.unpack(">b", buffer.read(1))[0]
        return NBTByte(value=value)


class NBTShort(NBTTag):
    """NBT Short tag."""
    
    def __init__(self, name: str = "", value: int = 0):
        super().__init__(name)
        self.value = value
    
    def get_type(self) -> NBTType:
        return NBTType.SHORT
    
    def write(self, buffer: BytesIO) -> None:
        buffer.write(struct.pack(">h", self.value))
    
    @staticmethod
    def read(buffer: BytesIO) -> "NBTShort":
        value = struct.unpack(">h", buffer.read(2))[0]
        return NBTShort(value=value)


class NBTInt(NBTTag):
    """NBT Int tag."""
    
    def __init__(self, name: str = "", value: int = 0):
        super().__init__(name)
        self.value = value
    
    def get_type(self) -> NBTType:
        return NBTType.INT
    
    def write(self, buffer: BytesIO) -> None:
        buffer.write(struct.pack(">i", self.value))
    
    @staticmethod
    def read(buffer: BytesIO) -> "NBTInt":
        value = struct.unpack(">i", buffer.read(4))[0]
        return NBTInt(value=value)


class NBTLong(NBTTag):
    """NBT Long tag (64-bit)."""
    
    def __init__(self, name: str = "", value: int = 0):
        super().__init__(name)
        self.value = value
    
    def get_type(self) -> NBTType:
        return NBTType.LONG
    
    def write(self, buffer: BytesIO) -> None:
        buffer.write(struct.pack(">q", self.value))
    
    @staticmethod
    def read(buffer: BytesIO) -> "NBTLong":
        value = struct.unpack(">q", buffer.read(8))[0]
        return NBTLong(value=value)


class NBTFloat(NBTTag):
    """NBT Float tag."""
    
    def __init__(self, name: str = "", value: float = 0.0):
        super().__init__(name)
        self.value = value
    
    def get_type(self) -> NBTType:
        return NBTType.FLOAT
    
    def write(self, buffer: BytesIO) -> None:
        buffer.write(struct.pack(">f", self.value))
    
    @staticmethod
    def read(buffer: BytesIO) -> "NBTFloat":
        value = struct.unpack(">f", buffer.read(4))[0]
        return NBTFloat(value=value)


class NBTDouble(NBTTag):
    """NBT Double tag (64-bit float)."""
    
    def __init__(self, name: str = "", value: float = 0.0):
        super().__init__(name)
        self.value = value
    
    def get_type(self) -> NBTType:
        return NBTType.DOUBLE
    
    def write(self, buffer: BytesIO) -> None:
        buffer.write(struct.pack(">d", self.value))
    
    @staticmethod
    def read(buffer: BytesIO) -> "NBTDouble":
        value = struct.unpack(">d", buffer.read(8))[0]
        return NBTDouble(value=value)


class NBTByteArray(NBTTag):
    """NBT Byte Array tag."""
    
    def __init__(self, name: str = "", value: bytes = b""):
        super().__init__(name)
        self.value = value
    
    def get_type(self) -> NBTType:
        return NBTType.BYTE_ARRAY
    
    def write(self, buffer: BytesIO) -> None:
        buffer.write(struct.pack(">i", len(self.value)))
        buffer.write(self.value)
    
    @staticmethod
    def read(buffer: BytesIO) -> "NBTByteArray":
        length = struct.unpack(">i", buffer.read(4))[0]
        value = buffer.read(length)
        return NBTByteArray(value=value)


class NBTString(NBTTag):
    """NBT String tag."""
    
    def __init__(self, name: str = "", value: str = ""):
        super().__init__(name)
        self.value = value
    
    def get_type(self) -> NBTType:
        return NBTType.STRING
    
    def write(self, buffer: BytesIO) -> None:
        encoded = self.value.encode("utf-8")
        buffer.write(struct.pack(">H", len(encoded)))
        buffer.write(encoded)
    
    @staticmethod
    def read(buffer: BytesIO) -> "NBTString":
        length = struct.unpack(">H", buffer.read(2))[0]
        value = buffer.read(length).decode("utf-8")
        return NBTString(value=value)


class NBTList(NBTTag):
    """NBT List tag."""
    
    def __init__(self, name: str = "", tags: List[NBTTag] = None, list_type: NBTType = NBTType.END):
        super().__init__(name)
        self.tags = tags or []
        self.list_type = list_type
    
    def get_type(self) -> NBTType:
        return NBTType.LIST
    
    def write(self, buffer: BytesIO) -> None:
        if not self.tags:
            buffer.write(struct.pack(">b", NBTType.END))
            buffer.write(struct.pack(">i", 0))
            return
        
        buffer.write(struct.pack(">b", self.tags[0].get_type()))
        buffer.write(struct.pack(">i", len(self.tags)))
        for tag in self.tags:
            tag.write(buffer)
    
    @staticmethod
    def read(buffer: BytesIO) -> "NBTList":
        list_type = NBTType(struct.unpack(">b", buffer.read(1))[0])
        length = struct.unpack(">i", buffer.read(4))[0]
        
        tags = []
        for _ in range(length):
            tag = NBTTag.read(buffer, list_type)
            tags.append(tag)
        
        return NBTList(tags=tags, list_type=list_type)


class NBTCompound(NBTTag):
    """NBT Compound tag (dictionary of tags)."""
    
    def __init__(self, name: str = ""):
        super().__init__(name)
        self.tags: Dict[str, NBTTag] = {}
    
    def get_type(self) -> NBTType:
        return NBTType.COMPOUND
    
    def __setitem__(self, key: str, value: NBTTag) -> None:
        self.tags[key] = value
    
    def __getitem__(self, key: str) -> NBTTag:
        return self.tags[key]
    
    def __contains__(self, key: str) -> bool:
        return key in self.tags
    
    def get(self, key: str, default=None) -> Optional[NBTTag]:
        return self.tags.get(key, default)
    
    def write(self, buffer: BytesIO) -> None:
        for name, tag in self.tags.items():
            buffer.write(struct.pack(">b", tag.get_type()))
            # Write name as string
            encoded = name.encode("utf-8")
            buffer.write(struct.pack(">H", len(encoded)))
            buffer.write(encoded)
            tag.write(buffer)
        # End tag
        buffer.write(struct.pack(">b", NBTType.END))
    
    @staticmethod
    def read(buffer: BytesIO) -> "NBTCompound":
        compound = NBTCompound()
        
        while True:
            nbt_type = NBTType(struct.unpack(">b", buffer.read(1))[0])
            if nbt_type == NBTType.END:
                break
            
            # Read name
            name_len = struct.unpack(">H", buffer.read(2))[0]
            name = buffer.read(name_len).decode("utf-8")
            
            # Read tag
            tag = NBTTag.read(buffer, nbt_type)
            tag.name = name
            compound.tags[name] = tag
        
        return compound
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert compound to Python dictionary."""
        result = {}
        for name, tag in self.tags.items():
            result[name] = self._tag_to_python(tag)
        return result
    
    def _tag_to_python(self, tag: NBTTag) -> Any:
        """Convert NBT tag to Python object."""
        if isinstance(tag, NBTByte):
            return tag.value
        elif isinstance(tag, NBTShort):
            return tag.value
        elif isinstance(tag, NBTInt):
            return tag.value
        elif isinstance(tag, NBTLong):
            return tag.value
        elif isinstance(tag, NBTFloat):
            return tag.value
        elif isinstance(tag, NBTDouble):
            return tag.value
        elif isinstance(tag, NBTByteArray):
            return tag.value
        elif isinstance(tag, NBTString):
            return tag.value
        elif isinstance(tag, NBTList):
            return [self._tag_to_python(t) for t in tag.tags]
        elif isinstance(tag, NBTCompound):
            return tag.to_dict()
        elif isinstance(tag, NBTIntArray):
            return list(tag.value)
        elif isinstance(tag, NBTLongArray):
            return list(tag.value)
        return None
    
    @staticmethod
    def from_dict(name: str, data: Dict[str, Any]) -> "NBTCompound":
        """Create compound from Python dictionary."""
        compound = NBTCompound(name)
        for key, value in data.items():
            compound.tags[key] = NBTCompound._python_to_tag(key, value)
        return compound
    
    @staticmethod
    def _python_to_tag(name: str, value: Any) -> NBTTag:
        """Convert Python object to NBT tag."""
        if isinstance(value, bool):
            return NBTByte(name, 1 if value else 0)
        elif isinstance(value, int):
            if -128 <= value <= 127:
                return NBTByte(name, value)
            elif -32768 <= value <= 32767:
                return NBTShort(name, value)
            else:
                return NBTInt(name, value)
        elif isinstance(value, float):
            return NBTDouble(name, value)
        elif isinstance(value, str):
            return NBTString(name, value)
        elif isinstance(value, bytes):
            return NBTByteArray(name, value)
        elif isinstance(value, list):
            if not value:
                return NBTList(name)
            if isinstance(value[0], int):
                if len(value) > 0 and abs(value[0]) > 2**31:
                    return NBTLongArray(name, value)
                return NBTIntArray(name, value)
            return NBTList(name, [NBTCompound._python_to_tag("", v) for v in value])
        elif isinstance(value, dict):
            return NBTCompound.from_dict(name, value)
        return NBTEnd()


class NBTIntArray(NBTTag):
    """NBT Int Array tag."""
    
    def __init__(self, name: str = "", value: List[int] = None):
        super().__init__(name)
        self.value = value or []
    
    def get_type(self) -> NBTType:
        return NBTType.INT_ARRAY
    
    def write(self, buffer: BytesIO) -> None:
        buffer.write(struct.pack(">i", len(self.value)))
        for v in self.value:
            buffer.write(struct.pack(">i", v))
    
    @staticmethod
    def read(buffer: BytesIO) -> "NBTIntArray":
        length = struct.unpack(">i", buffer.read(4))[0]
        value = list(struct.unpack(f">{length}i", buffer.read(length * 4)))
        return NBTIntArray(value=value)


class NBTLongArray(NBTTag):
    """NBT Long Array tag."""
    
    def __init__(self, name: str = "", value: List[int] = None):
        super().__init__(name)
        self.value = value or []
    
    def get_type(self) -> NBTType:
        return NBTType.LONG_ARRAY
    
    def write(self, buffer: BytesIO) -> None:
        buffer.write(struct.pack(">i", len(self.value)))
        for v in self.value:
            buffer.write(struct.pack(">q", v))
    
    @staticmethod
    def read(buffer: BytesIO) -> "NBTIntArray":
        length = struct.unpack(">i", buffer.read(4))[0]
        value = list(struct.unpack(f">{length}q", buffer.read(length * 8)))
        return NBTLongArray(value=value)


def write_nbt(compound: NBTCompound, compressed: bool = True) -> bytes:
    """
    Write NBT compound to bytes.
    
    Args:
        compound: NBT compound to write
        compressed: Whether to gzip compress the output
    
    Returns:
        NBT bytes
    """
    buffer = BytesIO()
    compound.write(buffer)
    data = buffer.getvalue()
    
    if compressed:
        return gzip.compress(data, compresslevel=6)
    return data


def read_nbt(data: bytes, compressed: bool = True) -> NBTCompound:
    """
    Read NBT compound from bytes.
    
    Args:
        data: NBT bytes
        compressed: Whether data is gzip compressed
    
    Returns:
        NBT compound
    """
    if compressed:
        data = gzip.decompress(data)
    
    buffer = BytesIO(data)
    return NBTCompound.read(buffer)


def save_level_dat(
    file_path: str,
    level_name: str,
    seed: int,
    spawn_x: Optional[int],
    spawn_y: Optional[int],
    spawn_z: Optional[int],
    world_time: int,
    day_time: int,
    game_rules: Dict[str, str] = None,
    difficulty: int = 2,
    hardcore: bool = False,
    game_type: int = 0,
    **kwargs
) -> None:
    """
    Save Minecraft level.dat file.

    Args:
        file_path: Path to save level.dat
        level_name: World name
        seed: World seed
        spawn_x, spawn_y, spawn_z: Spawn coordinates, or None to leave the
            world's spawn unpicked so the terrain chooses it
        world_time: Total world time
        day_time: Day time (0-24000)
        game_rules: Game rules dictionary
        difficulty: 0=peaceful, 1=easy, 2=normal, 3=hard
        hardcore: Whether hardcore mode is enabled
        game_type: 0=survival, 1=creative, 2=adventure, 3=spectator
    """
    root = NBTCompound()
    data = NBTCompound("Data")

    # Core data - following NBT.md specification
    data["LevelName"] = NBTString("LevelName", level_name)
    data["RandomSeed"] = NBTLong("RandomSeed", seed)
    # An absent spawn is a real state, not a zero: writing SpawnY=0 would
    # pin every new world to the bottom of the world on first load.
    if spawn_x is not None and spawn_y is not None and spawn_z is not None:
        data["SpawnX"] = NBTInt("SpawnX", spawn_x)
        data["SpawnY"] = NBTInt("SpawnY", spawn_y)
        data["SpawnZ"] = NBTInt("SpawnZ", spawn_z)
    data["Time"] = NBTLong("Time", world_time)
    data["DayTime"] = NBTLong("DayTime", day_time)
    data["Difficulty"] = NBTByte("Difficulty", difficulty)
    data["Hardcore"] = NBTByte("Hardcore", 1 if hardcore else 0)
    data["GameType"] = NBTInt("GameType", game_type)
    
    # Version - use standard level version (19132 = Anvil format)
    data["version"] = NBTInt("version", 19133)
    
    # Game rules
    if game_rules:
        game_rules_compound = NBTCompound("GameRules")
        for key, value in game_rules.items():
            game_rules_compound[key] = NBTString(key, value)
        data["GameRules"] = game_rules_compound
    
    # Border settings
    data["BorderCenterX"] = NBTDouble("BorderCenterX", kwargs.get("border_center_x", 0.0))
    data["BorderCenterZ"] = NBTDouble("BorderCenterZ", kwargs.get("border_center_z", 0.0))
    data["BorderSize"] = NBTDouble("BorderSize", kwargs.get("border_size", 59999968.0))
    data["BorderSizeLerpTarget"] = NBTDouble("BorderSizeLerpTarget", kwargs.get("border_size_lerp_target", 59999968.0))
    data["BorderSizeLerpTime"] = NBTLong("BorderSizeLerpTime", kwargs.get("border_size_lerp_time", 0))
    data["BorderSafeZone"] = NBTDouble("BorderSafeZone", kwargs.get("border_safe_zone", 5.0))
    data["BorderWarningBlocks"] = NBTDouble("BorderWarningBlocks", kwargs.get("border_warning_blocks", 5.0))
    data["BorderWarningTime"] = NBTLong("BorderWarningTime", kwargs.get("border_warning_time", 0))
    
    # Weather - use TAG_Byte for boolean fields per NBT.md
    data["RainTime"] = NBTInt("RainTime", kwargs.get("rain_time", 0))
    data["raining"] = NBTByte("raining", 1 if kwargs.get("is_raining", False) else 0)
    data["ThunderTime"] = NBTInt("ThunderTime", kwargs.get("thunder_time", 0))
    data["thundering"] = NBTByte("thundering", 1 if kwargs.get("is_thundering", False) else 0)
    
    # Last played
    import time
    data["LastPlayed"] = NBTLong("LastPlayed", int(time.time() * 1000))
    
    root["Data"] = data
    
    # Write to file with gzip compression
    with open(file_path, "wb") as f:
        f.write(write_nbt(root, compressed=True))


def load_level_dat(file_path: str) -> Dict[str, Any]:
    """
    Load Minecraft level.dat file.
    
    Args:
        file_path: Path to level.dat
    
    Returns:
        Dictionary with world data
    """
    with open(file_path, "rb") as f:
        data = f.read()
    
    root = read_nbt(data, compressed=True)
    data_dict = root["Data"].to_dict()
    
    # Normalize field names for backward compatibility
    # Map RandomSeed to seed if seed is not present
    if "RandomSeed" in data_dict and "seed" not in data_dict:
        data_dict["seed"] = data_dict["RandomSeed"]
    
    # Map raining/thundering variants
    if "raining" in data_dict and "Raining" not in data_dict:
        data_dict["Raining"] = data_dict["raining"]
    if "thundering" in data_dict and "Thundering" not in data_dict:
        data_dict["Thundering"] = data_dict["thundering"]
    
    return data_dict
