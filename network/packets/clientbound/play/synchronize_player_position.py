"""
Synchronize Player Position (Clientbound, Play)

Positions and orients the player, and doubles as the anti-cheat gate: the
teleport_id must be echoed in STeleportConfirm before the client accepts
movement input again.

The flags field decides whether each coordinate is absolute or a delta from
the current value. 1.21.2 widened it from a single byte to a signed 32-bit
int, which is why it is written with write_int rather than write_byte.
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x46)
class CSynchronizePlayerPosition(ClientBoundPacket):
    """Synchronize Player Position - updates player position and rotation.

    Packet ID: 0x46

    This is sent after Join Game to position the player in the world.
    """

    name = "Synchronize Player Position"
    packet_id = 0x46

    # Bitfield of relative fields (1.21.2+ widened the old 1-byte flag to a full int).
    # Bit set => the matching field below is a delta from the current value, not absolute.
    FLAG_RELATIVE_X = 0x01
    FLAG_RELATIVE_Y = 0x02
    FLAG_RELATIVE_Z = 0x04
    FLAG_RELATIVE_YAW = 0x08
    FLAG_RELATIVE_PITCH = 0x10
    FLAG_RELATIVE_DELTA_X = 0x20
    FLAG_RELATIVE_DELTA_Y = 0x40
    FLAG_RELATIVE_DELTA_Z = 0x80
    FLAG_ROTATE_DELTA = 0x100

    def __init__(
        self,
        teleport_id: int = 0,
        x: float = 0.0,
        y: float = 64.0,
        z: float = 0.0,
        velocity_x: float = 0.0,
        velocity_y: float = 0.0,
        velocity_z: float = 0.0,
        yaw: float = 0.0,
        pitch: float = 0.0,
        flags: int = 0,
    ):
        super().__init__()
        self.teleport_id = teleport_id
        self.x = x
        self.y = y
        self.z = z
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.velocity_z = velocity_z
        self.yaw = yaw
        self.pitch = pitch
        self.flags = flags

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(self.teleport_id)
        buf.write_double(self.x)
        buf.write_double(self.y)
        buf.write_double(self.z)
        buf.write_double(self.velocity_x)
        buf.write_double(self.velocity_y)
        buf.write_double(self.velocity_z)
        buf.write_float(self.yaw)
        buf.write_float(self.pitch)
        # Since 1.21.2 this is a signed 32-bit bitfield, not a single byte.
        buf.write_int(self.flags)

    def read(self, buf: ByteBuffer) -> None:
        self.teleport_id = buf.read_varint()
        self.x = buf.read_double()
        self.y = buf.read_double()
        self.z = buf.read_double()
        self.velocity_x = buf.read_double()
        self.velocity_y = buf.read_double()
        self.velocity_z = buf.read_double()
        self.yaw = buf.read_float()
        self.pitch = buf.read_float()
        self.flags = buf.read_int()
