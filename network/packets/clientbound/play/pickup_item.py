"""
Pickup Item (Clientbound, Play) - packet id 0x7A

Animation for an item entity flying into a collector: the item entity at
`collected_entity_id` lerps toward `collector_entity_id` over a fixed time and
then despawns. The actual inventory change still rides on Set Container Slot
(0x14); this packet is purely visual.

A `count` of 0 was used by vanilla in older versions to indicate a denied
pickup; in 1.21.11 it is the actual stack size and the only way to send
"collected" without changing inventory is to send count = 1 with the right
id.

Packet ID 0x7A per Pumpkin @ 42c903493 assets/packets.json (protocol 774).
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x7A)
class CPickupItem(ClientBoundPacket):
    """Pickup Item - animate an item entity being collected.

    Packet ID: 0x7A
    """

    name = "Pickup Item"
    packet_id = 0x7A

    def __init__(
        self,
        collected_entity_id: int = 0,
        collector_entity_id: int = 0,
        count: int = 1,
    ):
        super().__init__()
        self.collected_entity_id = collected_entity_id
        self.collector_entity_id = collector_entity_id
        self.count = count

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(self.collected_entity_id)
        buf.write_varint(self.collector_entity_id)
        buf.write_varint(self.count)

    def read(self, buf: ByteBuffer) -> None:
        self.collected_entity_id = buf.read_varint()
        self.collector_entity_id = buf.read_varint()
        self.count = buf.read_varint()