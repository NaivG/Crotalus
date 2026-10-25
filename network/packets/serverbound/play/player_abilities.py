"""
Player Abilities (Serverbound, Play) - packet id 0x27

The client echoes back its current abilities flags whenever they change
and every tick they differ from what the server last set. The server
compares against its own view and re-syncs via CPlayerAbilities (0x3E)
on mismatch.

The flags match the clientbound packet:

    0x01  invulnerable
    0x02  flying
    0x04  allow_flying
    0x08  instant_break (creative-only)

`flying_speed` and `field_of_view_modifier` are floats the client sends
back so the server can confirm what speed the client thinks it is moving
at.

Packet ID 0x27 per Pumpkin @ 42c903493 assets/packets.json (protocol 774).
"""

from network.buffer import ByteBuffer
from network.packet import ServerBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.CLIENT_TO_SERVER, 0x27)
class SPlayerAbilities(ServerBoundPacket):
    """Player Abilities - client echoes current ability flags.

    Packet ID: 0x27
    """

    name = "Player Abilities"
    packet_id = 0x27

    FLAG_INVULNERABLE = 0x01
    FLAG_FLYING = 0x02
    FLAG_ALLOW_FLYING = 0x04
    FLAG_INSTANT_BREAK = 0x08

    def __init__(
        self,
        flags: int = 0,
        flying_speed: float = 0.05,
        field_of_view_modifier: float = 0.1,
    ):
        super().__init__()
        self.flags = flags
        self.flying_speed = flying_speed
        self.field_of_view_modifier = field_of_view_modifier

    def write(self, buf: ByteBuffer) -> None:
        buf.write_byte(self.flags)
        buf.write_float(self.flying_speed)
        buf.write_float(self.field_of_view_modifier)

    def read(self, buf: ByteBuffer) -> None:
        self.flags = buf.read_unsigned_byte()
        self.flying_speed = buf.read_float()
        self.field_of_view_modifier = buf.read_float()