"""
Set Player Rotation (Serverbound, Play)

Sent when the client turns its view without moving, e.g. while looking around
while standing still. Carries no position, so a server that only tracks where
the player is can ignore it — but the server does track facing, for the same
reasons it tracks position: entity broadcast and player list rendering.
"""

from network.buffer import ByteBuffer
from network.packet import register_packet
from network.protocol import ProtocolState, Direction
from network.packets.serverbound.play.movement import SMovementPacket


@register_packet(ProtocolState.PLAY, Direction.CLIENT_TO_SERVER, 0x1F)
class SMovePlayerRot(SMovementPacket):
    """Player rotation.

    Packet ID: 0x1F
    """

    name = "Set Player Rotation"
    packet_id = 0x1F

    CARRIES_ROTATION = True

    def __init__(self, yaw: float = 0.0, pitch: float = 0.0, flags: int = 0):
        super().__init__(flags=flags)
        self.yaw = yaw
        self.pitch = pitch

    def write(self, buf: ByteBuffer) -> None:
        buf.write_float(self.yaw)
        buf.write_float(self.pitch)
        self._write_flags(buf)

    def read(self, buf: ByteBuffer) -> None:
        self.yaw = buf.read_float()
        self.pitch = buf.read_float()
        self._read_flags(buf)
