"""
Set Initial Position (Clientbound)

Deprecated since 1.20.2 - spawning is now handled by CSynchronizePlayerPosition
in the play state. Still registered so an older client that asks for it gets a
well-formed packet instead of a silent desync.
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.LOGIN, Direction.SERVER_TO_CLIENT, 0x04)
class CSetInitialPosition(ClientBoundPacket):
    """Server sets initial spawn position (deprecated in 1.20.2+).

    Fields:
    - position: Position (x, y, z)
    - angle: Float (yaw)
    """

    name = "Set Initial Position"
    packet_id = 0x04

    def __init__(self, x: int = 0, y: int = 64, z: int = 0, angle: float = 0.0):
        super().__init__()
        self.x = x
        self.y = y
        self.z = z
        self.angle = angle

    def write(self, buf: ByteBuffer) -> None:
        buf.write_position(self.x, self.y, self.z)
        buf.write_float(self.angle)

    def read(self, buf: ByteBuffer) -> None:
        self.x, self.y, self.z = buf.read_position()
        self.angle = buf.read_float()
