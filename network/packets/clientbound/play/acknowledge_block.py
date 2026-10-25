"""
Acknowledge Block Change (Clientbound, Play) - packet id 0x04

Confirms a block the client predicted on its own. Client-side the server's
block updates and the player's own right-click-during-tick work race; the
client tags every predicted block with a sequence number and sends the
sequence back here, freeing the server from having to send a Block Update
for changes the client already applied.

Packet ID 0x04 per Pumpkin @ 42c903493 assets/packets.json (protocol 774).
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x04)
class CAcknowledgeBlock(ClientBoundPacket):
    """Acknowledge Block Change - tells the client which predicted sequence landed.

    Packet ID: 0x04
    """

    name = "Acknowledge Block Change"
    packet_id = 0x04

    def __init__(self, sequence: int = 0):
        super().__init__()
        self.sequence = sequence

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(self.sequence)

    def read(self, buf: ByteBuffer) -> None:
        self.sequence = buf.read_varint()