"""
Move Entity Rot (Clientbound, Play) - packet id 0x36

Rotation-only: the entity turned in place (a player looking around while
standing still is the common case). No position, so the client keeps
interpolating whatever position it last believed.
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x36)
class CMoveEntityRot(ClientBoundPacket):
    """Move Entity Rot - rotation-only update for a tracked entity.

    Packet ID: 0x36
    """

    name = "Move Entity Rot"
    packet_id = 0x36

    def __init__(self, entity_id: int = 0, yaw: float = 0.0, pitch: float = 0.0,
                 on_ground: bool = True):
        super().__init__()
        self.entity_id = entity_id
        self.yaw = yaw
        self.pitch = pitch
        self.on_ground = on_ground

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(self.entity_id)
        buf.write_angle(self.yaw)
        buf.write_angle(self.pitch)
        buf.write_bool(self.on_ground)

    def read(self, buf: ByteBuffer) -> None:
        self.entity_id = buf.read_varint()
        self.yaw = buf.read_angle()
        self.pitch = buf.read_angle()
        self.on_ground = buf.read_bool()
