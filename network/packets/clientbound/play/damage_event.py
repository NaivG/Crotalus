"""
Damage Event (Clientbound, Play) - packet id 0x19

Plays the directional damage sound for an entity. Source direction is what
makes the "incoming damage" audio pan and where the red flash tries to come from.
Triggers the client-side bones for damage tilt as a side effect.

Source type maps to:
    0  damage from a melee attack (knight's voice cue / hurt thump)
    1  damage from a projectile (arrow thud / hurt hiss)
    2  damage from an explosion (loud thud / knockback)
    3  damage from a magic source (totem / generic)
    4  damage from indirect magic (potion / splash)

Packet ID 0x19 per Pumpkin @ 42c903493 assets/packets.json (protocol 774).
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x19)
class CDamageEvent(ClientBoundPacket):
    """Damage Event - play directional damage feedback for an entity.

    Packet ID: 0x19
    """

    name = "Damage Event"
    packet_id = 0x19

    SOURCE_MELEE = 0
    SOURCE_PROJECTILE = 1
    SOURCE_EXPLOSION = 2
    SOURCE_MAGIC = 3
    SOURCE_INDIRECT_MAGIC = 4

    def __init__(
        self,
        entity_id: int = 0,
        source_type: int = 0,
        source_cause_id: int = 0,
        source_direct_id: int = 0,
        source_position: tuple | None = None,
    ):
        super().__init__()
        self.entity_id = entity_id
        self.source_type = source_type
        # source_cause_id is a VarInt: the entity that caused the damage
        # (or 0 if it was environmental). Optional.
        self.source_cause_id = source_cause_id
        # source_direct_id is a VarInt: the entity the player sees as the
        # immediate source (e.g. the shooter via a trident across a
        # player). Optional.
        self.source_direct_id = source_direct_id
        # source_position is an optional Vec3. Vanilla sends it only for
        # damage from explosions and similar positional causes; a server
        # that always sends it will play extra damage bones.
        self.source_position = source_position

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(self.entity_id)
        buf.write_varint(self.source_type)
        buf.write_varint(self.source_cause_id)
        buf.write_varint(self.source_direct_id)
        if self.source_position is not None:
            buf.write_bool(True)
            x, y, z = self.source_position
            buf.write_double(x)
            buf.write_double(y)
            buf.write_double(z)
        else:
            buf.write_bool(False)

    def read(self, buf: ByteBuffer) -> None:
        self.entity_id = buf.read_varint()
        self.source_type = buf.read_varint()
        self.source_cause_id = buf.read_varint()
        self.source_direct_id = buf.read_varint()
        has_pos = buf.read_bool()
        if has_pos:
            self.source_position = (buf.read_double(), buf.read_double(), buf.read_double())
        else:
            self.source_position = None