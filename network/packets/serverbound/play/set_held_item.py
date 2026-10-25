"""
Set Held Item (Serverbound, Play) - packet id 0x34

The player changed hotbar slot (mouse wheel, hotbar key 1..9, swap
buttons). The slot is zero-based: 0..8 inclusive.

Packet ID 0x34 per Pumpkin @ 42c903493 assets/packets.json (protocol 774).
"""

from network.buffer import ByteBuffer
from network.packet import ServerBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.CLIENT_TO_SERVER, 0x34)
class SSetHeldItem(ServerBoundPacket):
    """Set Held Item - hotbar slot selection (serverbound).

    Packet ID: 0x34
    """

    name = "Set Held Item"
    packet_id = 0x34

    def __init__(self, slot: int = 0):
        super().__init__()
        # 0..8 inclusive. Values outside that range are clamped by the
        # client; a server-side validator should reject them.
        self.slot = slot

    def write(self, buf: ByteBuffer) -> None:
        buf.write_short(self.slot)

    def read(self, buf: ByteBuffer) -> None:
        self.slot = buf.read_short()