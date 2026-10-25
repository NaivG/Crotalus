"""
Chat Ack (Serverbound, Play)

Acknowledges a server-sent chat message, carrying the offset the server should
apply to its next message. Part of the signed-chat bookkeeping that prevents
reordering.
"""

from network.buffer import ByteBuffer
from network.packet import ServerBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.CLIENT_TO_SERVER, 0x05)
class SChatAck(ServerBoundPacket):
    """Chat message acknowledgement / offset.

    Packet ID: 0x05

    The parameter is "offset" - a VarInt indicating the chat offset.
    """

    name = "Chat Ack"
    packet_id = 0x05

    def __init__(self, offset: int = 0):
        super().__init__()
        self.offset = offset

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(self.offset)

    def read(self, buf: ByteBuffer) -> None:
        self.offset = buf.read_varint()
