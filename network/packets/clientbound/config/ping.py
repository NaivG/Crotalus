"""
Ping (Clientbound, Config)

Asks the client to answer with SPong. Exists so the connection is provably
bidirectional before the server commits to the heavier play-state handshake.
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.CONFIG, Direction.SERVER_TO_CLIENT, 0x05)
class CPingConfig(ClientBoundPacket):
    """Ping request for Config state."""

    name = "Ping (Config)"
    packet_id = 0x05

    def __init__(self, payload: int = 0):
        super().__init__()
        self.payload = payload

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(self.payload)

    def read(self, buf: ByteBuffer) -> None:
        self.payload = buf.read_varint()
