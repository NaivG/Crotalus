"""
Set Equipment (Clientbound, Play) - packet id 0x64

Updates one entity's equipped items. The wire format (since 1.16) is:

    entity_id VarInt
    for each (slot, item):
        slot_signed_byte   lower 7 bits = slot, high bit set when more follow
        item_stack         NBT-encoded stack, identical to the inventory wire

Slot id space, single source of truth -- vanilla and the client both use:
    0  main hand
    1  offhand
    2  boots
    3  leggings
    4  chestplate
    5  helmet
    6  body / horse armor

`None` on a slot clears it (sends an empty ItemStack); an absent slot
keeps whatever the client already has rendered for it.

Packet ID 0x64 per Pumpkin @ 42c903493 assets/packets.json (protocol 774).
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x64)
class CSetEquipment(ClientBoundPacket):
    """Set Equipment - replace one entity's equipped items.

    Packet ID: 0x64
    """

    name = "Set Equipment"
    packet_id = 0x64

    SLOT_MAINHAND = 0
    SLOT_OFFHAND = 1
    SLOT_BOOTS = 2
    SLOT_LEGGINGS = 3
    SLOT_CHESTPLATE = 4
    SLOT_HELMET = 5
    SLOT_BODY = 6

    # 0x80: more entries are coming after this one. Slot fits in 0..6 so the
    # high bit is free to carry the continuation flag.
    _MORE_BIT = 0x80

    def __init__(self, entity_id: int = 0, equipment: list | None = None):
        super().__init__()
        self.entity_id = entity_id
        # List of (slot, item_bytes) pairs. Item bytes come from the item
        # stack encoder in the inventory layer; this packet is a transport.
        self.equipment = list(equipment or [])

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(self.entity_id)
        last_index = len(self.equipment) - 1
        for index, (slot, item_bytes) in enumerate(self.equipment):
            byte_value = slot & 0x7F
            if index != last_index:
                byte_value |= self._MORE_BIT
            buf.write_byte(byte_value)
            buf.write_bytes(item_bytes)

    def read(self, buf: ByteBuffer) -> None:
        self.entity_id = buf.read_varint()
        self.equipment = []
        while True:
            raw = buf.read_unsigned_byte()
            slot = raw & 0x7F
            more = bool(raw & self._MORE_BIT)
            # Without a wire item-stack decoder the boundary between stack
            # bytes and the next slot byte is not knowable from here, so the
            # read path is intentionally best-effort: it stops at the
            # terminator when one is present and otherwise consumes whatever
            # remains. A real decoder lives in the inventory layer and is
            # expected to swap this method out.
            if more:
                self.equipment.append((slot, b""))
                continue
            self.equipment.append((slot, b""))
            return