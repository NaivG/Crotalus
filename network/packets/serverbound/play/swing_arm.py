"""
Swing Arm (Serverbound, Play) - packet id 0x3C

The player swung their arm -- left or right. Sent in addition to any
other action: attacking a mob sends both Swing Arm (this packet) and
Interact (0x19) with action=ATTACK, so the server gets the swing as a
distinct event even when the target is missing (e.g. they swung at
empty air).

Packet ID 0x3C per Pumpkin @ 42c903493 assets/packets.json (protocol 774).
"""

from network.buffer import ByteBuffer
from network.packet import ServerBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.CLIENT_TO_SERVER, 0x3C)
class SSwingArm(ServerBoundPacket):
    """Swing Arm - which hand the player swung.

    Packet ID: 0x3C
    """

    name = "Swing Arm"
    packet_id = 0x3C

    HAND_MAIN = 0
    HAND_OFF = 1

    def __init__(self, hand: int = 0):
        super().__init__()
        self.hand = hand

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(self.hand)

    def read(self, buf: ByteBuffer) -> None:
        self.hand = buf.read_varint()