"""
Entity Event (Clientbound, Play) - packet id 0x22

Status effects on a single entity: play villager trade accept, living entity
hurt (the *visual* part -- Damage Event 0x19 is the audio/sound), sheep
graze, firework blast, totem pop, etc. The wire is one VarInt event id.

Many of the ids are 1.21.x specific; the value list below is what vanilla
1.21.11 sends. New ids are added per-version; an id the client does not
recognise silently does nothing.

Packet ID 0x22 per Pumpkin @ 42c903493 assets/packets.json (protocol 774).
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x22)
class CEntityEvent(ClientBoundPacket):
    """Entity Event - play a single status effect on an entity.

    Packet ID: 0x22
    """

    name = "Entity Event"
    packet_id = 0x22

    # Living entity visual events.
    ENTITY_HURT = 2
    ENTITY_DEATH = 3
    LIVING_HURT_SOUND = 2  # alias kept for callers used to the literal value
    LIVING_DEATH_SOUND = 3
    LIVING_SHAKE_OFF_WATER = 6
    LIVING_EATING_ACCEPTED = 9
    LIVING_USE_MAGIC = 17

    # Sheep / villager / animal-specific effects.
    SHEEP_GRAZE = 10
    SHEEP_EAT_GRASS = 11
    VILLAGER_HAPPY = 13
    VILLAGER_UNHAPPY = 14
    VILLAGER_HEART_PARTICLES = 15
    VILLAGER_ANGRY = 16
    VILLAGER_SPLASH = 18

    # Special-case visual effects.
    FIREWORK_EXPLOSION = 17  # alias used for fireworks
    PLAYER_FINISH_USING_ITEM = 9  # alias kept for callers
    TOTEM_OF_UNDYING_POP = 35
    SHEARED_SHEEP_EAT_GRASS = 11  # alias

    def __init__(self, entity_id: int = 0, event_id: int = 0):
        super().__init__()
        self.entity_id = entity_id
        self.event_id = event_id

    def write(self, buf: ByteBuffer) -> None:
        buf.write_int(self.entity_id)
        buf.write_byte(self.event_id)

    def read(self, buf: ByteBuffer) -> None:
        self.entity_id = buf.read_int()
        self.event_id = buf.read_unsigned_byte()