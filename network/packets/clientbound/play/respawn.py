"""
Respawn (Clientbound, Play) - packet id 0x50

The server's "you're now in this dimension" packet, sent after Join Game,
after a death-and-respawn cycle, or when the player crosses into another
dimension. The client uses it to tear down its world state and reissue the
chunk-streaming pipeline.

The dimensional fields are written in *this* order in 1.21.11:

    dimension_type        VarInt     minecraft:dimension_type registry index
    dimension_name        String     the namespace, e.g. minecraft:overworld
    hashed_seed           i64
    game_mode             Byte
    previous_game_mode     Byte     -1 = "no previous"
    debug                 Bool
    is_flat               Bool
    death_dimension       Optional String
    death_position        Optional Position
    portal_cooldown       VarInt
    sealevel              VarInt     1.21.2+
    data_kept             Byte      1.21.4+

The block of "death_*" fields is optional; when absent, the client shows
its existing respawn screen. The portal_cooldown gate on cross-dimension
travel is read at the client and stops the player from immediately
teleporting back.

Packet ID 0x50 per Pumpkin @ 42c903493 assets/packets.json (protocol 774).
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x50)
class CRespawn(ClientBoundPacket):
    """Respawn - reposition the player, optionally in a new dimension.

    Packet ID: 0x50
    """

    name = "Respawn"
    packet_id = 0x50

    # Flag bits the client reads for what NOT to clear across respawn.
    DATA_KEEP_ATTRIBUTES = 0x01
    DATA_KEEP_METADATA = 0x02
    DATA_KEEP_INVENTORY = 0x04
    DATA_KEEP_ENCHANTMENTS = 0x08

    def __init__(
        self,
        dimension_type: int = 0,
        dimension_name: str = "minecraft:overworld",
        hashed_seed: int = 0,
        game_mode: int = 0,
        previous_game_mode: int = -1,
        debug: bool = False,
        is_flat: bool = False,
        has_death_position: bool = False,
        death_dimension: str | None = None,
        death_position: tuple | None = None,
        portal_cooldown: int = 0,
        sealevel: int = 63,
        data_kept: int = 0,
    ):
        super().__init__()
        self.dimension_type = dimension_type
        self.dimension_name = dimension_name
        self.hashed_seed = hashed_seed
        self.game_mode = game_mode
        self.previous_game_mode = previous_game_mode
        self.debug = debug
        self.is_flat = is_flat
        self.has_death_position = has_death_position
        self.death_dimension = death_dimension
        self.death_position = death_position
        self.portal_cooldown = portal_cooldown
        self.sealevel = sealevel
        self.data_kept = data_kept

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(self.dimension_type)
        buf.write_string(self.dimension_name)
        buf.write_long(self.hashed_seed)
        buf.write_byte(self.game_mode)
        buf.write_byte(self.previous_game_mode)
        buf.write_bool(self.debug)
        buf.write_bool(self.is_flat)

        if self.has_death_position:
            buf.write_bool(True)
            buf.write_string(self.death_dimension or "")
            position = self.death_position or (0, 0, 0)
            buf.write_position(*position)
        else:
            buf.write_bool(False)

        buf.write_varint(self.portal_cooldown)
        buf.write_varint(self.sealevel)
        buf.write_byte(self.data_kept)

    def read(self, buf: ByteBuffer) -> None:
        self.dimension_type = buf.read_varint()
        self.dimension_name = buf.read_string()
        self.hashed_seed = buf.read_long()
        self.game_mode = buf.read_byte()
        self.previous_game_mode = buf.read_byte()
        self.debug = buf.read_bool()
        self.is_flat = buf.read_bool()
        self.has_death_position = buf.read_bool()
        if self.has_death_position:
            self.death_dimension = buf.read_string()
            self.death_position = buf.read_position()
        else:
            self.death_dimension = None
            self.death_position = None
        self.portal_cooldown = buf.read_varint()
        self.sealevel = buf.read_varint()
        self.data_kept = buf.read_unsigned_byte()