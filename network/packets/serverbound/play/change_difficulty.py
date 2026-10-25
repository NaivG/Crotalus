"""
Change Difficulty (Serverbound, Play)

Difficulty lock or unlock, sent when a player (or a command) changes it.
0 = peaceful, 1 = easy, 2 = normal, 3 = hard.
"""

from network.buffer import ByteBuffer
from network.packet import ServerBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.CLIENT_TO_SERVER, 0x03)
class SChangeDifficulty(ServerBoundPacket):
    """Change difficulty.

    Packet ID: 0x03

    Fields:
    - difficulty: VarInt (0: peaceful, 1: easy, 2: normal, 3: hard)
    """

    name = "Change Difficulty"
    packet_id = 0x03

    def __init__(self, difficulty: int = 0):
        super().__init__()
        self.difficulty = difficulty

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(self.difficulty)

    def read(self, buf: ByteBuffer) -> None:
        self.difficulty = buf.read_varint()
