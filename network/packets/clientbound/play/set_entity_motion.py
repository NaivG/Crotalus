"""
Set Entity Motion (Clientbound, Play) - packet id 0x63

Sets an entity's velocity directly, ignoring client prediction. Used for
knockback from melee damage, explosions, fishing rods, /tp -- anywhere the
server wants the entity to *start* moving at a specific velocity rather
than keep accelerating from its current one.

Each component is a fixed-point short with 8000 = 1 block/tick (so 1.0
in normal units -> 8000 on the wire). The client clamps the value but does
not validate against any physics rule; sending impossibly large motion
just makes the entity shoot away. The protocol rejects non-normalised speed
vectors (negative zero, NaN) at the layer below.

Packet ID 0x63 per Pumpkin @ 42c903493 assets/packets.json (protocol 774).
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x63)
class CSetEntityMotion(ClientBoundPacket):
    """Set Entity Motion - set an entity's velocity directly.

    Packet ID: 0x63
    """

    name = "Set Entity Motion"
    packet_id = 0x63

    # Conversion factor: 8000 = 1 block/tick = 20 blocks/second.
    _VELOCITY_SCALE = 8000.0

    def __init__(
        self,
        entity_id: int = 0,
        velocity_x: float = 0.0,
        velocity_y: float = 0.0,
        velocity_z: float = 0.0,
    ):
        super().__init__()
        self.entity_id = entity_id
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.velocity_z = velocity_z

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(self.entity_id)
        buf.write_short(int(self.velocity_x * self._VELOCITY_SCALE))
        buf.write_short(int(self.velocity_y * self._VELOCITY_SCALE))
        buf.write_short(int(self.velocity_z * self._VELOCITY_SCALE))

    def read(self, buf: ByteBuffer) -> None:
        self.entity_id = buf.read_varint()
        self.velocity_x = buf.read_short() / self._VELOCITY_SCALE
        self.velocity_y = buf.read_short() / self._VELOCITY_SCALE
        self.velocity_z = buf.read_short() / self._VELOCITY_SCALE