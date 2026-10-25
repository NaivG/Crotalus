"""
Interact (Serverbound, Play) - packet id 0x19

How the player interacted with an entity: attack (left-click on a mob),
interact (right-click on a mob or item frame), interact-at (right-click
at a precise position, used by armor stands and entities with multiple
hitboxes).

The wire layout in 1.21.11:

    entity_id          VarInt
    action             VarInt   0 = interact, 1 = attack, 2 = interact_at
    target_position    Vector3<f32>  present iff action == 2
    hand               VarInt   present iff action != 1 (0 = main hand, 1 = off)
    sneaking           bool     the player was sneaking when they clicked

`target_position` is the exact spot on the entity the client hit -- for
armor stands, this is what determines which slot the player clicked on
(for example).

Packet ID 0x19 per Pumpkin @ 42c903493 assets/packets.json (protocol 774).
"""

from network.buffer import ByteBuffer
from network.packet import ServerBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.CLIENT_TO_SERVER, 0x19)
class SInteract(ServerBoundPacket):
    """Interact - attack, use or use-at an entity.

    Packet ID: 0x19
    """

    name = "Interact"
    packet_id = 0x19

    ACTION_INTERACT = 0
    ACTION_ATTACK = 1
    ACTION_INTERACT_AT = 2

    def __init__(
        self,
        entity_id: int = 0,
        action: int = 0,
        target_x: float = 0.0,
        target_y: float = 0.0,
        target_z: float = 0.0,
        hand: int = 0,
        sneaking: bool = False,
    ):
        super().__init__()
        self.entity_id = entity_id
        self.action = action
        self.target_x = target_x
        self.target_y = target_y
        self.target_z = target_z
        self.hand = hand
        self.sneaking = sneaking

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(self.entity_id)
        buf.write_varint(self.action)
        if self.action == self.ACTION_INTERACT_AT:
            buf.write_float(self.target_x)
            buf.write_float(self.target_y)
            buf.write_float(self.target_z)
        if self.action != self.ACTION_ATTACK:
            buf.write_varint(self.hand)
        buf.write_bool(self.sneaking)

    def read(self, buf: ByteBuffer) -> None:
        self.entity_id = buf.read_varint()
        self.action = buf.read_varint()
        if self.action == self.ACTION_INTERACT_AT:
            self.target_x = buf.read_float()
            self.target_y = buf.read_float()
            self.target_z = buf.read_float()
        if self.action != self.ACTION_ATTACK:
            self.hand = buf.read_varint()
        self.sneaking = buf.read_bool()