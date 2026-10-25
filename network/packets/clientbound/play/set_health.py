"""
Set Health (Clientbound, Play) - packet id 0x66

Updates the player's health, food and saturation. The client clamps the
food bar to the saturation in `write`, so sending a separate Set Health for
each damage tick is the normal cadence -- the client renders the regen
heart and the food shake against this number.

`food` was widened to a VarInt in 1.21.4: the older i32 still parses, just
no prefix. The client compares the *new* food tick against `saturation`
internally and animates the difference, so the two values must be consistent
within one packet -- sending 6 food + 0 saturation makes the food bar start
shrinking.

Packet ID 0x66 per Pumpkin @ 42c903493 assets/packets.json (protocol 774).
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x66)
class CSetHealth(ClientBoundPacket):
    """Set Health - update HP, food and saturation.

    Packet ID: 0x66
    """

    name = "Set Health"
    packet_id = 0x66

    def __init__(
        self,
        health: float = 20.0,
        food: int = 20,
        saturation: float = 5.0,
    ):
        super().__init__()
        self.health = health
        self.food = food
        self.saturation = saturation

    def write(self, buf: ByteBuffer) -> None:
        buf.write_float(self.health)
        buf.write_varint(self.food)
        buf.write_float(self.saturation)

    def read(self, buf: ByteBuffer) -> None:
        self.health = buf.read_float()
        self.food = buf.read_varint()
        self.saturation = buf.read_float()