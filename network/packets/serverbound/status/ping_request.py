"""
Ping Request (Serverbound)

Carries a client-chosen cookie that the server must echo back verbatim in
CStatusResponse's sibling, CPingResponse. The round-trip is what the client's
ping overlay measures.
"""

from network.buffer import ByteBuffer
from network.packet import ServerBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.STATUS, Direction.CLIENT_TO_SERVER, 0x01)
class SPingRequest(ServerBoundPacket):
    """Client sends a ping request.

    Fields:
    - payload: Long (time/cookie)
    """

    name = "Ping Request"
    packet_id = 0x01

    def __init__(self, payload: int = 0):
        super().__init__()
        self.payload = payload

    def write(self, buf: ByteBuffer) -> None:
        buf.write_long(self.payload)

    def read(self, buf: ByteBuffer) -> None:
        self.payload = buf.read_long()
