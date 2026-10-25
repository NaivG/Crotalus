"""
Keep Alive (Clientbound, Config)

The server's heartbeat. The client echoes the cookie in SKeepAliveConfig; a
missing echo within the timeout means the peer is gone.
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.CONFIG, Direction.SERVER_TO_CLIENT, 0x04)
class CKeepAliveConfig(ClientBoundPacket):
    """Keep Alive for Config state."""

    name = "Keep Alive (Config)"
    packet_id = 0x04

    def __init__(self, payload: int = 0):
        super().__init__()
        self.payload = payload

    def write(self, buf: ByteBuffer) -> None:
        buf.write_long(self.payload)

    def read(self, buf: ByteBuffer) -> None:
        self.payload = buf.read_long()
