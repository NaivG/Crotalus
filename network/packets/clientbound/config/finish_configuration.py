"""
Finish Configuration (Clientbound)

The server's mirror of SFinishConfiguration. The client replies with its own
SFinishConfiguration, and only then does the server send the Join Game packet
to enter the play state.
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.CONFIG, Direction.SERVER_TO_CLIENT, 0x03)
class CFinishConfiguration(ClientBoundPacket):
    """Client signals finished configuration."""

    name = "Finish Configuration"
    packet_id = 0x03

    def __init__(self):
        super().__init__()

    def write(self, buf: ByteBuffer) -> None:
        pass

    def read(self, buf: ByteBuffer) -> None:
        pass
