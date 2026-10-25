"""
Bundle Item Selected (Serverbound, Play)

Which slot the player picked inside an open bundle. The server maps it to the
item stack when the bundle is closed.
"""

from network.buffer import ByteBuffer
from network.packet import ServerBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.CLIENT_TO_SERVER, 0x02)
class SBundleItemSelected(ServerBoundPacket):
    """Bundle item selected.

    Packet ID: 0x02

    Fields:
    - slot_id: VarInt
    - selected_item_index: VarInt
    """

    name = "Bundle Item Selected"
    packet_id = 0x02

    def __init__(self, slot_id: int = 0, selected_item_index: int = 0):
        super().__init__()
        self.slot_id = slot_id
        self.selected_item_index = selected_item_index

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(self.slot_id)
        buf.write_varint(self.selected_item_index)

    def read(self, buf: ByteBuffer) -> None:
        self.slot_id = buf.read_varint()
        self.selected_item_index = buf.read_varint()
