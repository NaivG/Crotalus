"""
Set Default Spawn Position (Clientbound, Play) - packet id 0x5F

Tells the client the world spawn point (the bed anchor in vanilla, the
world spawn otherwise). The client renders the green beam of light at
this position and respawns here on death when no bed is bound.

The angle is the player yaw that the client respawns at, packed the same
way as every angle byte.

Packet ID 0x5F per Pumpkin @ 42c903493 assets/packets.json (protocol 774).
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x5F)
class CSetDefaultSpawnPosition(ClientBoundPacket):
    """Set Default Spawn Position - the world's respawn anchor.

    Packet ID: 0x5F
    """

    name = "Set Default Spawn Position"
    packet_id = 0x5F

    def __init__(
        self,
        x: int = 0,
        y: int = 64,
        z: int = 0,
        angle: float = 0.0,
    ):
        super().__init__()
        self.x = x
        self.y = y
        self.z = z
        self.angle = angle

    def write(self, buf: ByteBuffer) -> None:
        buf.write_position(self.x, self.y, self.z)
        buf.write_angle(self.angle)

    def read(self, buf: ByteBuffer) -> None:
        self.x, self.y, self.z = buf.read_position()
        self.angle = buf.read_angle()