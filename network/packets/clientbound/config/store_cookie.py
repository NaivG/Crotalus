"""
Store Cookie (Clientbound)

Instructs the client to persist a cookie for future sessions. The client will
return it via SCookieResponse on a later connection.
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.CONFIG, Direction.SERVER_TO_CLIENT, 0x10)
class CStoreCookie(ClientBoundPacket):
    """Server stores a cookie on the client for future sessions.

    Packet ID: 0x10

    The client will store this cookie and send it back on future sessions
    via the Cookie Response packet.

    Format:
    - key: String (cookie identifier)
    - payload: Byte array (max 5120 bytes / 5kiB)
    """

    name = "Store Cookie (Config)"
    packet_id = 0x10
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
            self.payload = buf.read_bytes(self.MAX_PAYLOAD_SIZE)
            remaining = payload_len - self.MAX_PAYLOAD_SIZE
            if remaining > 0:
                buf.read_bytes(remaining)
        else:
            self.payload = buf.read_bytes(payload_len)
