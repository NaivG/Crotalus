"""
Entity Animation (Clientbound, Play) - packet id 0x02

Triggers a single visual animation on an entity: arm swing, hurt shake, eat,
etc. The client picks the animation from `animation_id`; the server only needs
the entity and the id. Damage from a player attack is *not* here -- it goes
through Hurt Animation (0x29) which adds the red flash and damage tilt.

    Packet ID: 0x02 per Pumpkin @ 42c903493 assets/packets.json (protocol 774).
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x02)
class CEntityAnimation(ClientBoundPacket):
    """Entity Animation - one visual animation on a tracked entity.

    Packet ID: 0x02
    """

    name = "Entity Animation"
    packet_id = 0x02

    ANIMATION_SWING_MAIN_ARM = 0
    ANIMATION_LEAVE_BED = 1
    ANIMATION_SWING_OFFHAND = 3
    ANIMATION_CRITICAL_EFFECT = 4
    ANIMATION_MAGICAL_CRITICAL_EFFECT = 5

    def __init__(self, entity_id: int = 0, animation: int = 0):
        super().__init__()
        self.entity_id = entity_id
        self.animation = animation

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(self.entity_id)
        buf.write_byte(self.animation)

    def read(self, buf: ByteBuffer) -> None:
        self.entity_id = buf.read_varint()
        self.animation = buf.read_unsigned_byte()