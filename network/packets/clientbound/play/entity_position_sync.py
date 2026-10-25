"""
Entity Position Sync (Clientbound, Play) - packet id 0x23

The absolute correction. The tracker sends this when the delta path cannot
honestly describe the movement - an 8-block teleport exceeds the i16
fixed-point shorts, `on_ground` flipped, the 60-tick forced resync came due,
or the entity has a pending server-side teleport. It is also what despawns
rely on being wrong about nothing: everything here is exact, no
reconstruction on the client side.

Named `teleport_entity` before 1.21.2; the rename is the easiest way to
misread older protocol notes.

Must never be sent to a client about the client's OWN player - that
entity is driven by Synchronize Player Position (0x46) and the client's
own movement packets, and a sync for it desyncs the client's movement
prediction.
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x23)
class CEntityPositionSync(ClientBoundPacket):
    """Entity Position Sync - absolute position, rotation and velocity.

    Packet ID: 0x23
    """

    name = "Entity Position Sync"
    packet_id = 0x23

    def __init__(self, entity_id: int = 0, x: float = 0.0, y: float = 0.0,
                 z: float = 0.0, velocity_x: float = 0.0, velocity_y: float = 0.0,
                 velocity_z: float = 0.0, yaw: float = 0.0, pitch: float = 0.0,
                 on_ground: bool = True):
        super().__init__()
        self.entity_id = entity_id
        self.x = x
        self.y = y
        self.z = z
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.velocity_z = velocity_z
        self.yaw = yaw
        self.pitch = pitch
        self.on_ground = on_ground

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(self.entity_id)
        buf.write_double(self.x)
        buf.write_double(self.y)
        buf.write_double(self.z)
        buf.write_double(self.velocity_x)
        buf.write_double(self.velocity_y)
        buf.write_double(self.velocity_z)
        buf.write_float(self.yaw)
        buf.write_float(self.pitch)
        buf.write_bool(self.on_ground)

    def read(self, buf: ByteBuffer) -> None:
        self.entity_id = buf.read_varint()
        self.x = buf.read_double()
        self.y = buf.read_double()
        self.z = buf.read_double()
        self.velocity_x = buf.read_double()
        self.velocity_y = buf.read_double()
        self.velocity_z = buf.read_double()
        self.yaw = buf.read_float()
        self.pitch = buf.read_float()
        self.on_ground = buf.read_bool()
