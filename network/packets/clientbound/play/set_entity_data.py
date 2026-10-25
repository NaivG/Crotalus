"""
Set Entity Data (Clientbound, Play) - packet id 0x61

Pushes entity metadata entries: index, type, value, repeated, terminated by
0xFF. The blob is written as an unprefixed byte run - the client reads
entries until the terminator - so the packet takes pre-built bytes from
`entity.metadata` rather than parsing entries back out of a list.

Absent entries mean "unchanged", which is why the tracker can send only
what moved. The blob-building rules (which index, which type, and the
profile layout's unverified tail) live in entity/metadata.py; this module
is just the envelope.
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x61)
class CSetEntityData(ClientBoundPacket):
    """Set Entity Data - update tracked metadata for one entity.

    Packet ID: 0x61
    """

    name = "Set Entity Data"
    packet_id = 0x61

    def __init__(self, entity_id: int = 0, metadata: bytes = b"\xff"):
        super().__init__()
        self.entity_id = entity_id
        self.metadata = bytes(metadata)

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(self.entity_id)
        buf.write_bytes(self.metadata)

    def read(self, buf: ByteBuffer) -> None:
        self.entity_id = buf.read_varint()
        self.metadata = buf.read_bytes(buf.readable_bytes())
