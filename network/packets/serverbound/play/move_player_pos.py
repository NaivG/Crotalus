"""
Set Player Position (Serverbound, Play)

Sent when the client moves without changing where it is looking. This is the
most frequent movement packet for a player who walks in a straight line, and
the one a server cannot afford to drop: position is what the chunk streamer
uses to decide which chunks are in view.
"""

from network.buffer import ByteBuffer
from network.packet import register_packet
from network.protocol import ProtocolState, Direction
from network.packets.serverbound.play.movement import SMovementPacket


@register_packet(ProtocolState.PLAY, Direction.CLIENT_TO_SERVER, 0x1D)
class SMovePlayerPos(SMovementPacket):
    """Player position.

    Packet ID: 0x1D
    """

    name = "Set Player Position"
    packet_id = 0x1D

    CARRIES_POSITION = True

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        flags: int = 0,
    ):
        super().__init__(flags=flags)
        self.x = x
        self.y = y
        self.z = z

    def write(self, buf: ByteBuffer) -> None:
        buf.write_double(self.x)
        buf.write_double(self.y)
        buf.write_double(self.z)
        self._write_flags(buf)

    def read(self, buf: ByteBuffer) -> None:
        self.x = buf.read_double()
        self.y = buf.read_double()
        self.z = buf.read_double()
        self._read_flags(buf)
