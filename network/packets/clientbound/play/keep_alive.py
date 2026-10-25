"""
Keep Alive (Clientbound, Play)

The server's heartbeat once the client is in the world. A single i64 cookie
the client echoes back; a missing echo within the timeout means the peer is
gone.

Config state has its own keep-alive (CKeepAliveConfig, 0x04 in CONFIG), and
it does not carry over: the client tracks the two independently and the play
one is the only thing that keeps a *joined* connection alive. Without it a
server that is simply idle -- a stalled chunk stream, a quiet dimension --
sends nothing at all, and the client eventually decides the server has hung
and disconnects.

    ID 0x2B (43), from Pumpkin's 1.21.11 commit 42c903493 (protocol 774):
    git show 42c903493:assets/packets.json
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x2B)
class CKeepAlive(ClientBoundPacket):
    """Keep Alive for Play state."""

    name = "Keep Alive"
    packet_id = 0x2B

    def __init__(self, payload: int = 0):
        super().__init__()
        self.payload = payload

    def write(self, buf: ByteBuffer) -> None:
        buf.write_long(self.payload)

    def read(self, buf: ByteBuffer) -> None:
        self.payload = buf.read_long()
