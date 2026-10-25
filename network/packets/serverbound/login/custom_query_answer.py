"""
Custom Query Answer (Serverbound)

The client's reply to a server-issued Custom Query (mod/plugin channel probe).
Carrying the raw bytes through unmodified lets the server run whatever
handshake that plugin expects without the packet layer needing to know it.
"""

from network.buffer import ByteBuffer
from network.packet import ServerBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.LOGIN, Direction.CLIENT_TO_SERVER, 0x02)
class SCustomQueryAnswer(ServerBoundPacket):
    """Client answers a custom query.

    Fields:
    - channel: Identifier
    - data: Byte array
    """

    name = "Custom Query Answer"
    packet_id = 0x02

    def __init__(self, channel: str = "", data: bytes = b""):
        super().__init__()
        self.channel = channel
        self.data = data

    def write(self, buf: ByteBuffer) -> None:
        buf.write_string(self.channel)
        buf.write_varint(len(self.data))
        buf.write_bytes(self.data)

    def read(self, buf: ByteBuffer) -> None:
        self.channel = buf.read_string()
        data_len = buf.read_varint()
        self.data = buf.read_bytes(data_len)
