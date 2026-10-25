"""
Set Held Item (Clientbound, Play) - packet id 0x67

Tells the client which hotbar slot is currently selected (0..8). The client
runs the held-item render off this -- the visual highlight on the hotbar --
and triggers offhand swaps in creative mode when the slot id changes.

Packet ID 0x67 per Pumpkin @ 42c903493 assets/packets.json (protocol 774).
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x67)
class CSetHeldItem(ClientBoundPacket):
    """Set Held Item (clientbound) - hotbar slot selection.

    Packet ID: 0x67
    """

    name = "Set Held Item (Clientbound)"
    packet_id = 0x67

    def __init__(self, slot: int = 0):
        super().__init__()
        # 0..8 inclusive. Values outside that range are valid on the wire
        # (the client logs a warning and clamps) but should not be sent.
        self.slot = slot

    def write(self, buf: ByteBuffer) -> None:
        # VarInt in 1.21.4+; byte in older versions. VarInt is the
        # forward-compatible one and is what the client in this protocol
        # reads, so use that.
        buf.write_varint(self.slot)

    def read(self, buf: ByteBuffer) -> None:
        self.slot = buf.read_varint()