"""
Finish Configuration (Serverbound)

The client's signal that config is done. Receiving it moves the connection to
the PLAY state, after which the server sends the Join Game packet.
"""

from network.buffer import ByteBuffer
from network.packet import ServerBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.CONFIG, Direction.CLIENT_TO_SERVER, 0x03)
class SFinishConfiguration(ServerBoundPacket):
    """Client signals finished configuration."""

    name = "Finish Configuration"
    packet_id = 0x03

    def __init__(self):
        super().__init__()

    def write(self, buf: ByteBuffer) -> None:
        pass

    def read(self, buf: ByteBuffer) -> None:
        pass
