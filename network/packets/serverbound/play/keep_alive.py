"""
Keep Alive (Serverbound, Play)

The client's half of the play-state heartbeat -- echoes the cookie from
CKeepAlive. No reply follows; a missed echo is what reveals a dead peer.

    ID 0x1B (27), from Pumpkin's 1.21.11 commit 42c903493 (protocol 774):
    git show 42c903493:assets/packets.json
"""

from network.buffer import ByteBuffer
from network.packet import ServerBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.CLIENT_TO_SERVER, 0x1B)
class SKeepAlive(ServerBoundPacket):
    """Keep Alive for Play state."""

    name = "Keep Alive"
    packet_id = 0x1B

    def __init__(self, payload: int = 0):
        super().__init__()
        self.payload = payload

    def write(self, buf: ByteBuffer) -> None:
        buf.write_long(self.payload)

    def read(self, buf: ByteBuffer) -> None:
        self.payload = buf.read_long()
