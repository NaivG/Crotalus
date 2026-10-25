"""
Chat Message (Serverbound, Play)

A player typed in the chat box.

The read path is deliberately forgiving: signed chat adds a trailing signature
block that older parsers choke on, so a BufferError falls back to decoding
whatever remains as UTF-8 with replacement characters. A slightly garbled
message is better than dropping the connection.
"""

from network.buffer import ByteBuffer
from network.packet import ServerBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.CLIENT_TO_SERVER, 0x08)
class SChatMessage(ServerBoundPacket):
    """Player chat message.

    Packet ID: 0x08

    Simplified version: just reads message string.
    """

    name = "Chat Message"
    packet_id = 0x08

    def __init__(self, message: str = ""):
        super().__init__()
        self.message = message

    def write(self, buf: ByteBuffer) -> None:
        buf.write_string(self.message)

    def read(self, buf: ByteBuffer) -> None:
        try:
            self.message = buf.read_string()
        except BufferError:
            self.message = buf.read_bytes(buf.readable_bytes()).decode("utf-8", errors="replace")
