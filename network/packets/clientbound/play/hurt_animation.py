"""
Hurt Animation (Clientbound, Play) - packet id 0x29

The red flash + sideways tilt that shows an entity took damage. Direction is
the yaw the tilt faces in, packed as an angle byte.

Vanilla sends this on every damage event so the client can render the visual
even when no actual HP change occurs; the HP change itself rides on Set
Health (0x66) or Set Experience (0x65). Two packets in the same tick is
expected, not a duplicate.

Packet ID 0x29 per Pumpkin @ 42c903493 assets/packets.json (protocol 774).
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x29)
class CHurtAnimation(ClientBoundPacket):
    """Hurt Animation - flash + tilt on a damaged entity.

    Packet ID: 0x29
    """

    name = "Hurt Animation"
    packet_id = 0x29

    def __init__(self, entity_id: int = 0, yaw: float = 0.0):
        super().__init__()
        self.entity_id = entity_id
        self.yaw = yaw

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(self.entity_id)
        buf.write_angle(self.yaw)

    def read(self, buf: ByteBuffer) -> None:
        self.entity_id = buf.read_varint()
        self.yaw = buf.read_angle()