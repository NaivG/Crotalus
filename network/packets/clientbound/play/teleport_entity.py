"""
Teleport Entity (Clientbound, Play) - packet id 0x7B

Absolute teleport for a non-player entity, with an optional set of passengers
riding along. Distinct from the player-only Synchronize Player Position
(0x46), which doubles as the teleport-confirmation gate.

Position deltas use 1/4096-block fixed-point shorts: the server pre-computes
the difference from the entity's last sent position. `on_ground` rides
along because teleporting into a fall resets ground detection.

Packet ID 0x7B per Pumpkin @ 42c903493 assets/packets.json (protocol 774).
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x7B)
class CTeleportEntity(ClientBoundPacket):
    """Teleport Entity - absolute teleport with delta short-coded coordinates.

    Packet ID: 0x7B
    """

    name = "Teleport Entity"
    packet_id = 0x7B

    def __init__(
        self,
        entity_id: int = 0,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        dx: int = 0,
        dy: int = 0,
        dz: int = 0,
        yaw: float = 0.0,
        pitch: float = 0.0,
        on_ground: bool = True,
        relative_flags: int = 0,
    ):
        super().__init__()
        self.entity_id = entity_id
        self.x = x
        self.y = y
        self.z = z
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.yaw = yaw
        self.pitch = pitch
        self.on_ground = on_ground
        # Bit 0: dx/dy/dz are deltas against the absolute coords above.
        # Bit 1: yaw and pitch are also deltas against current rotation.
        # Bit 2: dx/dy/dz are velocities, not positions (knockback-style).
        self.relative_flags = relative_flags

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(self.entity_id)
        buf.write_double(self.x)
        buf.write_double(self.y)
        buf.write_double(self.z)
        buf.write_short(self.dx)
        buf.write_short(self.dy)
        buf.write_short(self.dz)
        buf.write_float(self.yaw)
        buf.write_float(self.pitch)
        buf.write_bool(self.on_ground)
        buf.write_byte(self.relative_flags)

    def read(self, buf: ByteBuffer) -> None:
        self.entity_id = buf.read_varint()
        self.x = buf.read_double()
        self.y = buf.read_double()
        self.z = buf.read_double()
        self.dx = buf.read_short()
        self.dy = buf.read_short()
        self.dz = buf.read_short()
        self.yaw = buf.read_float()
        self.pitch = buf.read_float()
        self.on_ground = buf.read_bool()
        self.relative_flags = buf.read_unsigned_byte()