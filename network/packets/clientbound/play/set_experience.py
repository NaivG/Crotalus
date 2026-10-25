"""
Set Experience (Clientbound, Play) - packet id 0x65

Updates the XP bar: total progress (0.0..1.0 of the current level), current
level, total XP. The bar fill is `progress`, so a player half-way through
level 5 sends progress=0.5, level=5. `total_xp` is informational -- the
client uses it for the number next to the bar.

Packet ID 0x65 per Pumpkin @ 42c903493 assets/packets.json (protocol 774).
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x65)
class CSetExperience(ClientBoundPacket):
    """Set Experience - update XP bar.

    Packet ID: 0x65
    """

    name = "Set Experience"
    packet_id = 0x65

    def __init__(
        self,
        progress: float = 0.0,
        level: int = 0,
        total_xp: int = 0,
    ):
        super().__init__()
        self.progress = progress
        self.level = level
        self.total_xp = total_xp

    def write(self, buf: ByteBuffer) -> None:
        buf.write_float(self.progress)
        buf.write_varint(self.level)
        buf.write_varint(self.total_xp)

    def read(self, buf: ByteBuffer) -> None:
        self.progress = buf.read_float()
        self.level = buf.read_varint()
        self.total_xp = buf.read_varint()