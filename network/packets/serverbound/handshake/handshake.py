"""
Handshake (Serverbound)

Sent by the client as the very first packet of every connection. The `next_state`
field it carries is what routes the connection into the STATUS or LOGIN state.
"""

from network.buffer import ByteBuffer
from network.packet import ServerBoundPacket, register_packet
from network.protocol import ProtocolState, Direction, PROTOCOL_VERSION


@register_packet(ProtocolState.HANDSHAKE, Direction.CLIENT_TO_SERVER, 0x00)
class SHandshake(ServerBoundPacket):
    """Handshake packet sent when client connects.

    Fields:
    - protocol_version: VarInt (774 for 1.21.11)
    - server_address: String (server hostname)
    - server_port: Unsigned short
    - next_state: VarInt (1 = Status, 2 = Login)
    """

    name = "Handshake"
    packet_id = 0x00

    def __init__(
        self,
        protocol_version: int = PROTOCOL_VERSION,
        server_address: str = "",
        server_port: int = 25565,
        next_state: int = 2
    ):
        super().__init__()
        self.protocol_version = protocol_version
        self.server_address = server_address
        self.server_port = server_port
        self.next_state = next_state

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(self.protocol_version)
        buf.write_string(self.server_address)
        buf.write_short(self.server_port)
        buf.write_varint(self.next_state)

    def read(self, buf: ByteBuffer) -> None:
        self.protocol_version = buf.read_varint()
        self.server_address = buf.read_string()
        self.server_port = buf.read_short() & 0xFFFF
        self.next_state = buf.read_varint()

    def __repr__(self) -> str:
        return f"<SHandshake version={self.protocol_version} addr={self.server_address}:{self.server_port} next={self.next_state}>"
