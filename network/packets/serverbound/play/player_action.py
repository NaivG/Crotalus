"""
Player Action (Serverbound, Play) - packet id 0x28

What the player did with a block: started digging, cancelled digging,
finished digging, swapped or dropped the held item. The `sequence` is the
client's per-block-edit counter and the server uses it for prediction
acks (see CAcknowledgeBlock, 0x04).

Pre-1.21.11: started digging / cancelled / finished / drop stack / drop
one / release item in use / swap item. Newer versions (26.3) added a
"change destroy direction" action that shifted every later action by
one; this implementation targets 1.21.11 and skips that one.

Packet ID 0x28 per Pumpkin @ 42c903493 assets/packets.json (protocol 774).
"""

from network.buffer import ByteBuffer
from network.packet import ServerBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.CLIENT_TO_SERVER, 0x28)
class SPlayerAction(ServerBoundPacket):
    """Player Action - what the player did with a block.

    Packet ID: 0x28
    """

    name = "Player Action"
    packet_id = 0x28

    # Action ids, as the client sends them in 1.21.11.
    START_DIGGING = 0
    CANCELLED_DIGGING = 1
    FINISHED_DIGGING = 2
    DROP_ITEM_STACK = 3
    DROP_ITEM = 4
    RELEASE_ITEM_IN_USE = 5
    SWAP_ITEM_WITH_OFFHAND = 6

    # Block face ids, in the order vanilla enumerates them. Negative Y is the
    # top of a block (e.g. a foot placed on the ground).
    FACE_DOWN = 0
    FACE_UP = 1
    FACE_NORTH = 2
    FACE_SOUTH = 3
    FACE_WEST = 4
    FACE_EAST = 5

    def __init__(
        self,
        status: int = 0,
        x: int = 0,
        y: int = 0,
        z: int = 0,
        face: int = 0,
        sequence: int = 0,
    ):
        super().__init__()
        self.status = status
        self.x = x
        self.y = y
        self.z = z
        self.face = face
        self.sequence = sequence

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(self.status)
        buf.write_position(self.x, self.y, self.z)
        buf.write_varint(self.face)
        buf.write_varint(self.sequence)

    def read(self, buf: ByteBuffer) -> None:
        self.status = buf.read_varint()
        self.x, self.y, self.z = buf.read_position()
        self.face = buf.read_varint()
        self.sequence = buf.read_varint()