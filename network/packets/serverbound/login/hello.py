"""
Hello (Serverbound)

The client's opening move in the login state: who it claims to be. Since 1.19.3
the UUID is mandatory (previously it was only sent in online mode), so offline
servers cannot rely on the client omitting it.
"""

from network.buffer import ByteBuffer
from network.packet import ServerBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.LOGIN, Direction.CLIENT_TO_SERVER, 0x00)
class SHello(ServerBoundPacket):
    """Client sends login start packet.

    Fields (1.19.3+):
    - name: String (player username, max 16 chars)
    - uuid: UUID (player UUID)
    """

    name = "Hello"
    packet_id = 0x00

    def __init__(self, name: str = "", profile_id: str = ""):
        super().__init__()
        self.name = name
        self.profile_id = profile_id  # In 1.19.3+, this is actually a required UUID

    def write(self, buf: ByteBuffer) -> None:
        buf.write_string(self.name)
        buf.write_uuid(self.profile_id)

    def read(self, buf: ByteBuffer) -> None:
        self.name = buf.read_string()
        # In 1.19.3+, UUID is always present
        self.profile_id = buf.read_uuid()
