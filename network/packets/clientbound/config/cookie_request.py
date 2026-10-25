"""
Cookie Request (Clientbound)

Asks the client for a stored cookie by key, optionally with a signature to
prove it wasn't tampered with. The reply is SCookieResponse.
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.CONFIG, Direction.SERVER_TO_CLIENT, 0x00)
class CCookieRequest(ClientBoundPacket):
    """Server requests a cookie from client (max 5kiB payload).

    Packet ID: 0x00

    The client should respond with SCookieResponse containing the cookie data.

    Format:
    - key: String (cookie identifier)
    - signature: Boolean (has signature)
    - signature_data: Optional bytes (if signature is true)
    """

    name = "Cookie Request (Config)"
    packet_id = 0x00
    MAX_PAYLOAD_SIZE = 5120  # 5kiB

    def __init__(self, key: str = "", has_signature: bool = False, signature: bytes = b""):
        super().__init__()
        self.key = key
        self.has_signature = has_signature
        self.signature = signature

    def write(self, buf: ByteBuffer) -> None:
        buf.write_string(self.key)
        buf.write_bool(self.has_signature)
        if self.has_signature:
            buf.write_varint(len(self.signature))
            buf.write_bytes(self.signature)

    def read(self, buf: ByteBuffer) -> None:
        self.key = buf.read_string()
        self.has_signature = buf.read_bool()
        if self.has_signature:
            sig_len = buf.read_varint()
            self.signature = buf.read_bytes(sig_len)
