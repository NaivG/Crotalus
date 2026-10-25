"""
Keep Alive (Serverbound, Config)

The client's half of the config-state heartbeat - echoes back the cookie from
CKeepAliveConfig. No reply follows; a missed echo is what reveals a dead peer.
"""

from network.buffer import ByteBuffer
from network.packet import ServerBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.CONFIG, Direction.CLIENT_TO_SERVER, 0x04)
class SKeepAliveConfig(ServerBoundPacket):
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
