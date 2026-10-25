"""
Use Item (Serverbound, Play) - packet id 0x40

The player's right-click *not* on a block: throw an ender pearl, eat
food, draw a bow, splash a potion. Use Item On (0x3F) carries the
block-targeted case; this is the in-air case.

Since 1.21 the client also sends yaw and pitch at the moment of use, so
the server can compute projectile trajectories without having to guess.

Packet ID 0x40 per Pumpkin @ 42c903493 assets/packets.json (protocol 774).
"""

from network.buffer import ByteBuffer
from network.packet import ServerBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.CLIENT_TO_SERVER, 0x40)
class SUseItem(ServerBoundPacket):
    """Use Item - right-click in the air.

    Packet ID: 0x40
    """

    name = "Use Item"
    packet_id = 0x40

    HAND_MAIN = 0
    HAND_OFF = 1

    def __init__(
        self,
        hand: int = 0,
        sequence: int = 0,
        yaw: float = 0.0,
        pitch: float = 0.0,
    ):
        super().__init__()
        self.hand = hand
        self.sequence = sequence
        self.yaw = yaw
        self.pitch = pitch

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(self.hand)
        buf.write_varint(self.sequence)
        buf.write_float(self.yaw)
        buf.write_float(self.pitch)

    def read(self, buf: ByteBuffer) -> None:
        self.hand = buf.read_varint()
        self.sequence = buf.read_varint()
        self.yaw = buf.read_float()
        self.pitch = buf.read_float()