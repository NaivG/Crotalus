"""
Pong (Serverbound)

The client's answer to CPingConfig, echoing the cookie. Unlike the play-state
ping, the config-state ping exists to provoke traffic while the connection is
still being set up.
"""

from network.buffer import ByteBuffer
from network.packet import ServerBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.CONFIG, Direction.CLIENT_TO_SERVER, 0x05)
class SPong(ServerBoundPacket):
    """Response to server ping."""

    name = "Pong"
    packet_id = 0x05

    def __init__(self, payload: int = 0):
        super().__init__()
        self.payload = payload

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(self.payload)

    def read(self, buf: ByteBuffer) -> None:
        self.payload = buf.read_varint()
