"""
Remove Entities (Clientbound, Play) - packet id 0x4B

Despawn one or more entities in a single packet. The list is what makes
mass despawns cheap - a chunk unloading with forty mobs is one packet, not
forty - so the tracker batches its removals per viewer per tick.

An entity removed here must not be referenced again by the client until a
fresh Add Entity re-introduces it; the tracker's `seen_by` bookkeeping is
what keeps those two in agreement.
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x4B)
class CRemoveEntities(ClientBoundPacket):
    """Remove Entities - despawn a batch of entities.

    Packet ID: 0x4B
    """

    name = "Remove Entities"
    packet_id = 0x4B

    def __init__(self, entity_ids: list | None = None):
        super().__init__()
        self.entity_ids = list(entity_ids or [])

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(len(self.entity_ids))
        for entity_id in self.entity_ids:
            buf.write_varint(entity_id)

    def read(self, buf: ByteBuffer) -> None:
        count = buf.read_varint()
        self.entity_ids = [buf.read_varint() for _ in range(count)]
