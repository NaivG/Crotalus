"""
Cookie Response (Serverbound)

The client's answer to CCookieRequest. An empty payload is meaningful, not a
malformed packet: it means the client holds no cookie under that key.

The read path clamps oversized payloads to MAX_PAYLOAD_SIZE rather than
trusting the declared length, so a hostile client cannot force a huge
allocation.
"""

from network.buffer import ByteBuffer
from network.packet import ServerBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.CONFIG, Direction.CLIENT_TO_SERVER, 0x01)
class SCookieResponse(ServerBoundPacket):
    """Client responds to a cookie request.

    Packet ID: 0x01

    Format:
    - key: String (cookie identifier, must match request)
    - payload: Byte array (max 5120 bytes / 5kiB)
      - If empty, the client has no cookie for this key
    """

    name = "Cookie Response (Config)"
    packet_id = 0x01
    MAX_PAYLOAD_SIZE = 5120  # 5kiB

    def __init__(self, key: str = "", payload: bytes = b""):
        super().__init__()
        self.key = key
        self.payload = payload

    def write(self, buf: ByteBuffer) -> None:
        buf.write_string(self.key)
        buf.write_varint(len(self.payload))
        buf.write_bytes(self.payload)

    def read(self, buf: ByteBuffer) -> None:
        self.key = buf.read_string()
        payload_len = buf.read_varint()
        if payload_len > self.MAX_PAYLOAD_SIZE:
            # Clamp to max size to prevent issues
            self.payload = buf.read_bytes(self.MAX_PAYLOAD_SIZE)
            # Skip remaining bytes
            remaining = payload_len - self.MAX_PAYLOAD_SIZE
            if remaining > 0:
                buf.read_bytes(remaining)
        else:
            self.payload = buf.read_bytes(payload_len)

    @property
    def has_cookie(self) -> bool:
        """Check if response contains cookie data."""
        return len(self.payload) > 0
