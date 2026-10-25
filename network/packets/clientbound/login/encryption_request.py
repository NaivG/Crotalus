"""
Encryption Request (Clientbound)

Starts the online-mode handshake. The server sends its RSA public key and a
random verify token; the client encrypts the shared secret with that key and
returns it in SEncryptionAnswer, which the server then decrypts with its own
private key.
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.LOGIN, Direction.SERVER_TO_CLIENT, 0x01)
class CEncryptionRequest(ClientBoundPacket):
    """Server requests encryption (online mode).

    Fields:
    - server_id: String (empty for mojang auth)
    - public_key: Byte array
    - verify_token: Byte array
    """

    name = "Encryption Request"
    packet_id = 0x01

    def __init__(
        self,
        server_id: str = "",
        public_key: bytes = b"",
        verify_token: bytes = b""
    ):
        super().__init__()
        self.server_id = server_id
        self.public_key = public_key
        self.verify_token = verify_token

    def write(self, buf: ByteBuffer) -> None:
        buf.write_string(self.server_id)
        buf.write_varint(len(self.public_key))
        buf.write_bytes(self.public_key)
        buf.write_varint(len(self.verify_token))
        buf.write_bytes(self.verify_token)

    def read(self, buf: ByteBuffer) -> None:
        self.server_id = buf.read_string()
        pubkey_len = buf.read_varint()
        self.public_key = buf.read_bytes(pubkey_len)
        token_len = buf.read_varint()
        self.verify_token = buf.read_bytes(token_len)
