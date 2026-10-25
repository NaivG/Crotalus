"""
Ping Response (Clientbound)

Echoes SPingRequest's cookie unchanged. Any mismatch means the connection is
tampering with or buffering frames, so the value is the integrity check.
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.STATUS, Direction.SERVER_TO_CLIENT, 0x01)
class CPingResponse(ClientBoundPacket):
    """Server responds to ping.

    Fields:
    - payload: Long (time/cookie, should match request)
    """

    name = "Ping Response"
    packet_id = 0x01

    def __init__(self, payload: int = 0):
        super().__init__()
        self.payload = payload

    def write(self, buf: ByteBuffer) -> None:
        buf.write_long(self.payload)

    def read(self, buf: ByteBuffer) -> None:
        self.payload = buf.read_long()

    def __repr__(self) -> str:
        return f"<CPingResponse payload={self.payload}>"
