"""
Close Container (Serverbound, Play)

The player closed an inventory or other open screen. The server uses it to stop
treating the player as container-bound and to commit any pending edits.
"""

from network.buffer import ByteBuffer
from network.packet import ServerBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.CLIENT_TO_SERVER, 0x12)
class SCloseContainer(ServerBoundPacket):
    """Close container (inventory).

    Packet ID: 0x12
    """

    name = "Close Container"
    packet_id = 0x12

    def __init__(self, window_id: int = 0):
        super().__init__()
        self.window_id = window_id

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(self.window_id)

    def read(self, buf: ByteBuffer) -> None:
        self.window_id = buf.read_varint()
