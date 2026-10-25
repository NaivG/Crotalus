"""
Encryption Answer (Serverbound)

Only sent in online mode: the client decrypts the server's RSA challenge with
the shared secret it generated and returns it alongside the verify token that
CEncryptionRequest handed out. The server uses the echoed token to prove it was
really in the TLS session and not a man in the middle.
"""

from network.buffer import ByteBuffer
from network.packet import ServerBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.LOGIN, Direction.CLIENT_TO_SERVER, 0x01)
class SEncryptionAnswer(ServerBoundPacket):
    """Client sends encryption response (online mode).

    Fields:
    - shared_secret: Byte array
    - verify_token: Byte array
    """

    name = "Encryption Answer"
    packet_id = 0x01

    def __init__(self, shared_secret: bytes = b"", verify_token: bytes = b""):
        super().__init__()
        self.shared_secret = shared_secret
        self.verify_token = verify_token

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(len(self.shared_secret))
        buf.write_bytes(self.shared_secret)
        buf.write_varint(len(self.verify_token))
        buf.write_bytes(self.verify_token)

    def read(self, buf: ByteBuffer) -> None:
        secret_len = buf.read_varint()
        self.shared_secret = buf.read_bytes(secret_len)
        token_len = buf.read_varint()
        self.verify_token = buf.read_bytes(token_len)
