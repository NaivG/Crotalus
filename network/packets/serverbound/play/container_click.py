"""
Container Click (Serverbound, Play) - packet id 0x11

The player clicked a slot in an open container or their own inventory.
Used for moving stacks, dropping items, swapping with the hotbar,
shift-click, hotbar swap, item painting.

The wire is rich: a list of slot-index + item-hash pairs the client
claims changed, plus the carried item the cursor is now holding. The
server uses the hashes to detect a desync (the client and the server
disagree on the contents of slot N) and the carried item to apply
post-click state. Item hashes and stack contents are produced by the
inventory layer; here it is just a transport.

Packet ID 0x11 per Pumpkin @ 42c903493 assets/packets.json (protocol 774).
"""

from network.buffer import ByteBuffer
from network.packet import ServerBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.CLIENT_TO_SERVER, 0x11)
class SContainerClick(ServerBoundPacket):
    """Container Click - one click in an open container.

    Packet ID: 0x11
    """

    name = "Container Click"
    packet_id = 0x11

    # Mouse buttons as the wire carries them.
    BUTTON_LEFT = 0
    BUTTON_RIGHT = 1
    BUTTON_MIDDLE = 2

    # The offhand-swap "button" lives in the button field but is really a
    # mode; vanilla spots 40 and treats it as swap with the offhand.
    BUTTON_OFFHAND_SWAP = 40

    # Click modes. Shift-click is "quick move" -- the server moves the
    # clicked stack as completely as it can to the other half of the
    # container.
    MODE_CLICK = 0
    MODE_SHIFT_CLICK = 1
    MODE_SWAP = 2       # button = hotbar slot
    MODE_CLONE = 3
    MODE_THROW = 4
    MODE_QUICK_MOVE = 5  # deprecated alias of shift-click in older versions
    MODE_PICKUP_ALL = 6

    def __init__(
        self,
        window_id: int = 0,
        state_id: int = 0,
        slot: int = 0,
        button: int = 0,
        mode: int = 0,
        changed_slots: list | None = None,
        carried_item_bytes: bytes = b"",
    ):
        super().__init__()
        # The container id: 0 for inventory, otherwise the id from Open
        # Screen. The client increments it per click and the server uses
        # it to discard out-of-order clicks during a closed-then-reopened
        # transition.
        self.window_id = window_id
        # Revision counter (a.k.a. state id) the client increments per
        # click and the server checks against its own.
        self.state_id = state_id
        # The slot that was clicked. -999 means "click outside the GUI"
        # (i.e. drop carried item).
        self.slot = slot
        self.button = button
        self.mode = mode
        # Each changed slot is a (slot_id, item_hash_bytes) pair. The
        # hash format is owned by the inventory layer.
        self.changed_slots = list(changed_slots or [])
        # The item the cursor is now carrying, raw item-stack bytes.
        self.carried_item_bytes = bytes(carried_item_bytes)

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(self.window_id)
        buf.write_varint(self.state_id)
        buf.write_short(self.slot)
        buf.write_byte(self.button)
        buf.write_varint(self.mode)
        buf.write_varint(len(self.changed_slots))
        for slot_id, item_hash_bytes in self.changed_slots:
            buf.write_short(slot_id)
            buf.write_bytes(item_hash_bytes)
        buf.write_bytes(self.carried_item_bytes)

    def read(self, buf: ByteBuffer) -> None:
        self.window_id = buf.read_varint()
        self.state_id = buf.read_varint()
        self.slot = buf.read_short()
        self.button = buf.read_byte()
        self.mode = buf.read_varint()
        count = buf.read_varint()
        self.changed_slots = []
        for _ in range(count):
            slot_id = buf.read_short()
            # The hash layout is owned by the inventory layer; consume
            # bytes up to the next slot entry without trying to parse
            # them here. The inventory code replaces this with the real
            # hash decoder.
            remaining = buf.readable_bytes()
            item_hash_bytes = buf.read_bytes(max(0, remaining))
            self.changed_slots.append((slot_id, item_hash_bytes))
        self.carried_item_bytes = buf.read_bytes(buf.readable_bytes())