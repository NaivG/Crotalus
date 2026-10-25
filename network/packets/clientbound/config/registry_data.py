"""
Registry Data (Clientbound)

Delivers a single dynamic registry to the client. One packet carries one
registry; send several packets for several registries.

This file also holds the NBT builders for the four vanilla dimension types.
They are hand-encoded rather than serialised from a dict because NBT compound
field order is not guaranteed to round-trip, and the vanilla client rejects
registries whose fields arrive out of order.

Note on entry data: 1.20.5+ lets the server omit NBT for entries both sides
already know (see CKnownPacks). The builders here always emit full data, which
is always correct - just larger.
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.CONFIG, Direction.SERVER_TO_CLIENT, 0x07)
class CRegistryData(ClientBoundPacket):
    """Registry data - contains a single registry.

    Format:
    - registry_id: string (e.g., "minecraft:dimension_type")
    - entries: array of {entry_id: string, has_data: bool, data: optional NBT}

    Note: Each packet should contain only ONE registry. Send multiple packets for multiple registries.
    """

    name = "Registry Data"
    packet_id = 0x07

    def __init__(self, registry_id: str, entries: list):
        """Initialize with a single registry.

        Args:
            registry_id: Registry identifier (e.g., "minecraft:dimension_type")
            entries: List of entry dicts with 'id' and optional 'data' keys
        """
        super().__init__()
        self.registry_id = registry_id
        self.entries = entries

    @classmethod
    def create_dimension_type(cls, name: str, height: int, min_y: int, max_y: int) -> 'CRegistryData':
        """Create a dimension_type registry entry with a single dimension."""
        nbt_data = cls._create_dimension_nbt_static(name, height, min_y, max_y)
        return cls("minecraft:dimension_type", [
            {"id": name, "data": nbt_data}
        ])

    @classmethod
    def create_all_dimension_types(cls) -> 'CRegistryData':
        """Create a single dimension_type registry packet with ALL dimension types.

        Minecraft 1.21+ requires all dimension types to be sent in a SINGLE packet
        before entering PLAY state. This includes:
        - minecraft:overworld
        - minecraft:overworld_caves
        - minecraft:the_end
        - minecraft:the_nether

        Returns a single CRegistryData packet containing all dimension type entries.
        """
        entries = []

        # Overworld dimension type
        overworld_nbt = cls._create_dimension_nbt_static(
            "minecraft:overworld", height=384, min_y=-64, max_y=319
        )
        entries.append({"id": "minecraft:overworld", "data": overworld_nbt})

        # Overworld Caves dimension type
        overworld_caves_nbt = cls._create_dimension_nbt_overworld_caves()
        entries.append({"id": "minecraft:overworld_caves", "data": overworld_caves_nbt})

        # The End dimension type
        the_end_nbt = cls._create_dimension_nbt_the_end()
        entries.append({"id": "minecraft:the_end", "data": the_end_nbt})

        # The Nether dimension type
        the_nether_nbt = cls._create_dimension_nbt_the_nether()
        entries.append({"id": "minecraft:the_nether", "data": the_nether_nbt})

        return cls("minecraft:dimension_type", entries)

    @staticmethod
    def _create_dimension_nbt_overworld_caves() -> bytes:
        """Create dimension NBT for overworld caves (has_ceiling=1, no skylight)."""
        buf = ByteBuffer()

        # TAG_Compound (0x0A)
        buf.write_byte(10)
        buf.write_string("")

        # ambient_light: TAG_Float
        buf.write_byte(5)
        buf.write_string("ambient_light")
        buf.write_float(0.0)

        # attributes: TAG_List
        buf.write_byte(9)
        buf.write_string("attributes")
        buf.write_byte(10)  # Element type: TAG_Compound
        buf.write_int(1)    # List size: 1 element (mood)

        # mood attribute compound
        buf.write_byte(10)
        buf.write_string("")
        buf.write_byte(3)
        buf.write_string("block_search_extent")
        buf.write_int(8)
        buf.write_byte(6)
        buf.write_string("offset")
        buf.write_double(2.0)
        buf.write_byte(8)
        buf.write_string("sound")
        buf.write_string("minecraft:ambient.cave")
        buf.write_byte(3)
        buf.write_string("tick_delay")
        buf.write_int(6000)
        buf.write_byte(0)  # End mood compound

        # coordinate_scale: TAG_Double
        buf.write_byte(6)
        buf.write_string("coordinate_scale")
        buf.write_double(1.0)

        # has_ceiling: TAG_Byte (1 = true for caves)
        buf.write_byte(1)
        buf.write_string("has_ceiling")
        buf.write_byte(1)

        # has_skylight: TAG_Byte (0 = false for caves)
        buf.write_byte(1)
        buf.write_string("has_skylight")
        buf.write_byte(0)

        # height: TAG_Int
        buf.write_byte(3)
        buf.write_string("height")
        buf.write_int(384)

        # infiniburn: TAG_String
        buf.write_byte(8)
        buf.write_string("infiniburn")
        buf.write_string("#minecraft:infiniburn_overworld")

        # logical_height: TAG_Int
        buf.write_byte(3)
        buf.write_string("logical_height")
        buf.write_int(384)

        # min_y: TAG_Int
        buf.write_byte(3)
        buf.write_string("min_y")
        buf.write_int(-64)

        # monster_spawn_block_light_limit: TAG_Int
        buf.write_byte(3)
        buf.write_string("monster_spawn_block_light_limit")
        buf.write_int(0)

        # monster_spawn_light_level: TAG_Compound
        buf.write_byte(10)
        buf.write_string("monster_spawn_light_level")
        buf.write_byte(3)
        buf.write_string("max_inclusive")
        buf.write_int(7)
        buf.write_byte(3)
        buf.write_string("min_inclusive")
        buf.write_int(0)
        buf.write_byte(8)
        buf.write_string("type")
        buf.write_string("minecraft:uniform")
        buf.write_byte(0)  # End compound

        # natural: TAG_Byte
        buf.write_byte(1)
        buf.write_string("natural")
        buf.write_byte(1)

        # skybox: TAG_String
        buf.write_byte(8)
        buf.write_string("skybox")
        buf.write_string("minecraft:overworld")

        # timelines: TAG_String
        buf.write_byte(8)
        buf.write_string("timelines")
        buf.write_string("#minecraft:in_overworld")

        # TAG_End
        buf.write_byte(0)

        return bytes(buf)

    @staticmethod
    def _create_dimension_nbt_the_end() -> bytes:
        """Create dimension NBT for The End."""
        buf = ByteBuffer()

        # TAG_Compound (0x0A)
        buf.write_byte(10)
        buf.write_string("")

        # ambient_light: TAG_Float (brighter in the end)
        buf.write_byte(5)
        buf.write_string("ambient_light")
        buf.write_float(15.0)

        # attributes: empty list
        buf.write_byte(9)
        buf.write_string("attributes")
        buf.write_byte(10)
        buf.write_int(0)  # No attributes

        # coordinate_scale: TAG_Double
        buf.write_byte(6)
        buf.write_string("coordinate_scale")
        buf.write_double(1.0)

        # has_ceiling: TAG_Byte (0)
        buf.write_byte(1)
        buf.write_string("has_ceiling")
        buf.write_byte(0)

        # has_fixed_time: TAG_Byte (1 = true, time doesn't pass)
        buf.write_byte(1)
        buf.write_string("has_fixed_time")
        buf.write_byte(1)

        # has_skylight: TAG_Byte (0)
        buf.write_byte(1)
        buf.write_string("has_skylight")
        buf.write_byte(0)

        # height: TAG_Int
        buf.write_byte(3)
        buf.write_string("height")
        buf.write_int(256)

        # infiniburn: TAG_String
        buf.write_byte(8)
        buf.write_string("infiniburn")
        buf.write_string("#minecraft:infiniburn_end")

        # logical_height: TAG_Int
        buf.write_byte(3)
        buf.write_string("logical_height")
        buf.write_int(256)

        # min_y: TAG_Int
        buf.write_byte(3)
        buf.write_string("min_y")
        buf.write_int(0)

        # monster_spawn_block_light_limit: TAG_Int
        buf.write_byte(3)
        buf.write_string("monster_spawn_block_light_limit")
        buf.write_int(0)

        # monster_spawn_light_level: TAG_Int
        buf.write_byte(3)
        buf.write_string("monster_spawn_light_level")
        buf.write_int(15)

        # skybox: TAG_String
        buf.write_byte(8)
        buf.write_string("skybox")
        buf.write_string("end")

        # timelines: TAG_String
        buf.write_byte(8)
        buf.write_string("timelines")
        buf.write_string("#minecraft:in_end")

        # TAG_End
        buf.write_byte(0)

        return bytes(buf)

    @staticmethod
    def _create_dimension_nbt_the_nether() -> bytes:
        """Create dimension NBT for The Nether."""
        buf = ByteBuffer()

        # TAG_Compound (0x0A)
        buf.write_byte(10)
        buf.write_string("")

        # ambient_light: TAG_Float
        buf.write_byte(5)
        buf.write_string("ambient_light")
        buf.write_float(0.1)

        # attributes: list with piglin_zombify and fast_lava
        buf.write_byte(9)
        buf.write_string("attributes")
        buf.write_byte(10)
        buf.write_int(3)

        # fast_lava attribute
        buf.write_byte(10)
        buf.write_string("")
        buf.write_byte(1)
        buf.write_string("minecraft:gameplay/fast_lava")
        buf.write_byte(1)
        buf.write_byte(0)  # End compound

        # piglins_zombify attribute
        buf.write_byte(10)
        buf.write_string("")
        buf.write_byte(1)
        buf.write_string("minecraft:gameplay/piglins_zombify")
        buf.write_byte(0)
        buf.write_byte(0)  # End compound

        # sky_light_level attribute
        buf.write_byte(10)
        buf.write_string("")
        buf.write_byte(3)
        buf.write_string("minecraft:gameplay/sky_light_level")
        buf.write_int(15)
        buf.write_byte(0)  # End compound

        # water_evaporates attribute
        buf.write_byte(10)
        buf.write_string("")
        buf.write_byte(1)
        buf.write_string("minecraft:gameplay/water_evaporates")
        buf.write_byte(1)
        buf.write_byte(0)  # End compound

        # cardinal_light: TAG_String
        buf.write_byte(8)
        buf.write_string("cardinal_light")
        buf.write_string("nether")

        # coordinate_scale: TAG_Double (2.0 for nether)
        buf.write_byte(6)
        buf.write_string("coordinate_scale")
        buf.write_double(2.0)

        # has_ceiling: TAG_Byte (1)
        buf.write_byte(1)
        buf.write_string("has_ceiling")
        buf.write_byte(1)

        # has_fixed_time: TAG_Byte (0)
        buf.write_byte(1)
        buf.write_string("has_fixed_time")
        buf.write_byte(0)

        # has_skylight: TAG_Byte (0)
        buf.write_byte(1)
        buf.write_string("has_skylight")
        buf.write_byte(0)

        # height: TAG_Int
        buf.write_byte(3)
        buf.write_string("height")
        buf.write_int(256)

        # infiniburn: TAG_String
        buf.write_byte(8)
        buf.write_string("infiniburn")
        buf.write_string("#minecraft:infiniburn_nether")

        # logical_height: TAG_Int
        buf.write_byte(3)
        buf.write_string("logical_height")
        buf.write_int(128)

        # min_y: TAG_Int
        buf.write_byte(3)
        buf.write_string("min_y")
        buf.write_int(0)

        # monster_spawn_block_light_limit: TAG_Int
        buf.write_byte(3)
        buf.write_string("monster_spawn_block_light_limit")
        buf.write_int(15)

        # monster_spawn_light_level: TAG_Int
        buf.write_byte(3)
        buf.write_string("monster_spawn_light_level")
        buf.write_int(7)

        # skybox: TAG_String
        buf.write_byte(8)
        buf.write_string("skybox")
        buf.write_string("none")

        # timelines: TAG_String
        buf.write_byte(8)
        buf.write_string("timelines")
        buf.write_string("#minecraft:in_nether")

        # TAG_End
        buf.write_byte(0)

        return bytes(buf)

    @classmethod
    def create_worldgen_biome(cls, name: str) -> 'CRegistryData':
        """Create a worldgen/biome registry entry."""
        nbt_data = cls._create_biome_nbt_static()
        return cls("minecraft:worldgen/biome", [
            {"id": name, "data": nbt_data}
        ])

    @classmethod
    def create_chat_type(cls, name: str) -> 'CRegistryData':
        """Create a chat_type registry entry (no NBT data)."""
        return cls("minecraft:chat_type", [
            {"id": name, "data": None}
        ])

    @staticmethod
    def _create_dimension_nbt_static(name: str, height: int, min_y: int, max_y: int) -> bytes:
        """Create dimension NBT compound for Minecraft 1.21.11.

        NBT Format:
        [TAG_Compound (0x0A)]
        [Name: String (VarInt length + UTF-8)]
          [Tag1_ID: 1 byte][Tag1_Name: String][Tag1_Value]
          ...
        [TAG_End (0x00)]

        Required fields for 1.21.11:
        - ambient_light: float
        - attributes: list (with mood sounds, background music, etc.)
        - coordinate_scale: double
        - has_ceiling: byte
        - has_skylight: byte
        - height: int
        - infiniburn: string (tag)
        - logical_height: int
        - min_y: int
        - monster_spawn_block_light_limit: int
        - monster_spawn_light_level: compound
        - natural: byte
        - skybox: string
        - timelines: string (tag)
        """
        buf = ByteBuffer()

        # TAG_Compound (0x0A)
        buf.write_byte(10)
        # Root compound name - empty string
        buf.write_string("")

        # ambient_light: TAG_Float (0x05)
        buf.write_byte(5)
        buf.write_string("ambient_light")
        buf.write_float(0.0)

        # attributes: TAG_List (0x09) - complex list with nested compounds
        buf.write_byte(9)  # TAG_List type
        buf.write_string("attributes")
        buf.write_byte(10)  # List element type: TAG_Compound
        buf.write_int(3)    # List size: 3 elements (mood, background_music, creative)

        # === First attribute: mood ===
        buf.write_byte(10)  # TAG_Compound
        buf.write_string("")  # Compound name

        # block_search_extent: TAG_Int
        buf.write_byte(3)
        buf.write_string("block_search_extent")
        buf.write_int(8)

        # offset: TAG_Double
        buf.write_byte(6)
        buf.write_string("offset")
        buf.write_double(2.0)

        # sound: TAG_String
        buf.write_byte(8)
        buf.write_string("sound")
        buf.write_string("minecraft:ambient.cave")

        # tick_delay: TAG_Int
        buf.write_byte(3)
        buf.write_string("tick_delay")
        buf.write_int(6000)

        buf.write_byte(0)  # End of mood compound

        # === Second attribute: background_music (creative) ===
        buf.write_byte(10)  # TAG_Compound
        buf.write_string("")

        # max_delay: TAG_Int
        buf.write_byte(3)
        buf.write_string("max_delay")
        buf.write_int(24000)

        # min_delay: TAG_Int
        buf.write_byte(3)
        buf.write_string("min_delay")
        buf.write_int(12000)

        # replace_current_music: TAG_Byte
        buf.write_byte(1)
        buf.write_string("replace_current_music")
        buf.write_byte(0)

        # sound: TAG_String
        buf.write_byte(8)
        buf.write_string("sound")
        buf.write_string("minecraft:music.creative")

        buf.write_byte(0)  # End of creative compound

        # === Third attribute: background_music (default) ===
        buf.write_byte(10)  # TAG_Compound
        buf.write_string("")

        # max_delay: TAG_Int
        buf.write_byte(3)
        buf.write_string("max_delay")
        buf.write_int(24000)

        # min_delay: TAG_Int
        buf.write_byte(3)
        buf.write_string("min_delay")
        buf.write_int(12000)

        # sound: TAG_String
        buf.write_byte(8)
        buf.write_string("sound")
        buf.write_string("minecraft:music.game")

        buf.write_byte(0)  # End of default compound

        # coordinate_scale: TAG_Double (0x06)
        buf.write_byte(6)
        buf.write_string("coordinate_scale")
        buf.write_double(1.0)

        # has_ceiling: TAG_Byte (0x01)
        buf.write_byte(1)
        buf.write_string("has_ceiling")
        buf.write_byte(0)

        # has_skylight: TAG_Byte (0x01)
        buf.write_byte(1)
        buf.write_string("has_skylight")
        buf.write_byte(1)

        # height: TAG_Int (0x03)
        buf.write_byte(3)
        buf.write_string("height")
        buf.write_int(height)

        # infiniburn: TAG_String (0x08)
        buf.write_byte(8)
        buf.write_string("infiniburn")
        buf.write_string("#minecraft:infiniburn_overworld")

        # logical_height: TAG_Int (0x03)
        buf.write_byte(3)
        buf.write_string("logical_height")
        buf.write_int(height)

        # min_y: TAG_Int (0x03)
        buf.write_byte(3)
        buf.write_string("min_y")
        buf.write_int(min_y)

        # monster_spawn_block_light_limit: TAG_Int (0x03)
        buf.write_byte(3)
        buf.write_string("monster_spawn_block_light_limit")
        buf.write_int(0)

        # monster_spawn_light_level: TAG_Compound (0x0A)
        buf.write_byte(10)
        buf.write_string("monster_spawn_light_level")

        # max_inclusive: TAG_Int
        buf.write_byte(3)
        buf.write_string("max_inclusive")
        buf.write_int(7)

        # min_inclusive: TAG_Int
        buf.write_byte(3)
        buf.write_string("min_inclusive")
        buf.write_int(0)

        # type: TAG_String
        buf.write_byte(8)
        buf.write_string("type")
        buf.write_string("minecraft:uniform")

        buf.write_byte(0)  # End of monster_spawn_light_level compound

        # natural: TAG_Byte (0x01)
        buf.write_byte(1)
        buf.write_string("natural")
        buf.write_byte(1)

        # skybox: TAG_String (0x08)
        buf.write_byte(8)
        buf.write_string("skybox")
        buf.write_string("minecraft:overworld")

        # timelines: TAG_String (0x08)
        buf.write_byte(8)
        buf.write_string("timelines")
        buf.write_string("#minecraft:in_overworld")

        # TAG_End (0x00)
        buf.write_byte(0)

        return bytes(buf)

    @staticmethod
    def _create_biome_nbt_static() -> bytes:
        """Create minimal biome NBT - STATIC VERSION.

        NBT Format:
        [TAG_Compound (0x0A)]
        [Name: String (VarInt length + UTF-8)]
          [Tag1_ID: 1 byte][Tag1_Name: String][Tag1_Value]
          ...
        [TAG_End (0x00)]
        """
        buf = ByteBuffer()

        # TAG_Compound (0x0A)
        buf.write_byte(10)
        # Root compound name - MUST be a string!
        buf.write_string("")  # Fixed: empty string with VarInt length

        # temperature: TAG_Float (0x05)
        buf.write_byte(5)
        buf.write_string("temperature")
        buf.write_float(0.5)

        # downfall: TAG_Float (0x05)
        buf.write_byte(5)
        buf.write_string("downfall")
        buf.write_float(0.5)

        # temperature_modifier: TAG_String (0x08)
        buf.write_byte(8)
        buf.write_string("temperature_modifier")
        buf.write_string("none")

        # TAG_End (0x00)
        buf.write_byte(0)

        return bytes(buf)

    def write(self, buf: ByteBuffer) -> None:
        """Write a single registry to the buffer."""
        # Write registry ID
        buf.write_string(self.registry_id)

        # Write entries array
        buf.write_varint(len(self.entries))

        for entry in self.entries:
            buf.write_string(entry["id"])
            if entry.get("data"):
                buf.write_bool(True)
                buf.write_bytes(entry["data"])
            else:
                buf.write_bool(False)

    def read(self, buf: ByteBuffer) -> None:
        # Not implemented for server-side
        pass
