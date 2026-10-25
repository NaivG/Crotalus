"""
Status Request (Serverbound)

Empty packet - the client asking "what is this server?". Its presence is the
entire payload; the connection knows it arrived from the packet ID alone.
"""

from network.buffer import ByteBuffer
from network.packet import ServerBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.STATUS, Direction.CLIENT_TO_SERVER, 0x00)
class SStatusRequest(ServerBoundPacket):
    """Client requests server status.

    No payload - just the packet ID.
    """

    name = "Status Request"
    packet_id = 0x00

    def __init__(self):
        super().__init__()

    def write(self, buf: ByteBuffer) -> None:
        pass  # No data

    def read(self, buf: ByteBuffer) -> None:
        pass  # No data
