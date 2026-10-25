"""
Move Player Position and Rotation (Serverbound, Play)

The client's main movement packet. The client sends it whenever the position
*or* the rotation changes, so a moving player produces several per second.

`flags` is a bitfield, not a boolean:
    0x01 - on ground
    0x02 - horizontal collision
    0x04 - vertical collision
Only bit 0 matters to a server that does not run its own physics.
"""

from network.buffer import ByteBuffer
from network.packet import register_packet
from network.protocol import ProtocolState, Direction
from network.packets.serverbound.play.movement import SMovementPacket


@register_packet(ProtocolState.PLAY, Direction.CLIENT_TO_SERVER, 0x1E)
class SMovePlayerPosRot(SMovementPacket):
    """Player position and rotation.

    Packet ID: 0x1E
    """

    name = "Move Player Position and Rotation"
    packet_id = 0x1E

    CARRIES_POSITION = True
    CARRIES_ROTATION = True

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        yaw: float = 0.0,
        pitch: float = 0.0,
        flags: int = 0,
    ):
        super().__init__(flags=flags)
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.pitch = pitch

    def write(self, buf: ByteBuffer) -> None:
        buf.write_double(self.x)
        buf.write_double(self.y)
        buf.write_double(self.z)
        buf.write_float(self.yaw)
        buf.write_float(self.pitch)
        self._write_flags(buf)

    def read(self, buf: ByteBuffer) -> None:
        self.x = buf.read_double()
        self.y = buf.read_double()
        self.z = buf.read_double()
        self.yaw = buf.read_float()
        self.pitch = buf.read_float()
        self._read_flags(buf)
