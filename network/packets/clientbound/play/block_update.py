"""
Block Update (Clientbound, Play) - packet id 0x08

Sets one block at a position. The block state id is the *global* palette id
from the registry, not the per-chunk palette -- every chunk ships the same
mapping on every chunk send, so any client can read any id without first
learning a chunk-local table.

Packet ID 0x08 per Pumpkin @ 42c903493 assets/packets.json (protocol 774).
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x08)
class CBlockUpdate(ClientBoundPacket):
    """Block Update - change one block.

    Packet ID: 0x08
    """

    name = "Block Update"
    packet_id = 0x08

    def __init__(self, x: int = 0, y: int = 0, z: int = 0, block_state_id: int = 0):
        super().__init__()
        self.x = x
        self.y = y
        self.z = z
        self.block_state_id = block_state_id

    def write(self, buf: ByteBuffer) -> None:
        buf.write_position(self.x, self.y, self.z)
        buf.write_varint(self.block_state_id)

    def read(self, buf: ByteBuffer) -> None:
        self.x, self.y, self.z = buf.read_position()
        self.block_state_id = buf.read_varint()