"""
Use Item On (Serverbound, Play) - packet id 0x3F

The player's right-click on a block: place a block, interact with a chest,
hit a note block, etc. Distinct from Use Item (0x40), which is the
right-click in the air.

The wire carries a precise hit position inside the block face so the
server can resolve "which side of the slab did they click" and "where on
the chest did they hit". `sequence` is the prediction counter the server
echoes back in Acknowledge Block Change (0x04) so the client knows its
prediction landed.

Packet ID 0x3F per Pumpkin @ 42c903493 assets/packets.json (protocol 774).
"""

from network.buffer import ByteBuffer
from network.packet import ServerBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.CLIENT_TO_SERVER, 0x3F)
class SUseItemOn(ServerBoundPacket):
    """Use Item On - right-click on a block.

    Packet ID: 0x3F
    """

    name = "Use Item On"
    packet_id = 0x3F

    HAND_MAIN = 0
    HAND_OFF = 1

    def __init__(
        self,
        hand: int = 0,
        x: int = 0,
        y: int = 0,
        z: int = 0,
        face: int = 0,
        cursor_x: float = 0.0,
        cursor_y: float = 0.0,
        cursor_z: float = 0.0,
        inside_block: bool = False,
        is_against_world_border: bool = False,
        sequence: int = 0,
    ):
        super().__init__()
        self.hand = hand
        self.x = x
        self.y = y
        self.z = z
        self.face = face
        self.cursor_x = cursor_x
        self.cursor_y = cursor_y
        self.cursor_z = cursor_z
        self.inside_block = inside_block
        self.is_against_world_border = is_against_world_border
        self.sequence = sequence

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(self.hand)
        buf.write_position(self.x, self.y, self.z)
        buf.write_varint(self.face)
        buf.write_float(self.cursor_x)
        buf.write_float(self.cursor_y)
        buf.write_float(self.cursor_z)
        buf.write_bool(self.inside_block)
        buf.write_bool(self.is_against_world_border)
        buf.write_varint(self.sequence)

    def read(self, buf: ByteBuffer) -> None:
        self.hand = buf.read_varint()
        self.x, self.y, self.z = buf.read_position()
        self.face = buf.read_varint()
        self.cursor_x = buf.read_float()
        self.cursor_y = buf.read_float()
        self.cursor_z = buf.read_float()
        self.inside_block = buf.read_bool()
        self.is_against_world_border = buf.read_bool()
        self.sequence = buf.read_varint()