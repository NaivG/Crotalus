"""
Shared base for the serverbound movement packets.

The client sends exactly one movement packet per tick, picking whichever of
the four matches what actually changed that tick: position, rotation, both, or
neither. All four end with the same `flags` byte, so the bit meanings live
here rather than being restated in each packet.

`flags` is a bitfield, not a boolean:
    0x01 - on ground
    0x02 - horizontal collision
    0x04 - vertical collision
Only bit 0 matters to a server that does not run its own physics; the other
two are hints for movement prediction and are recorded but otherwise ignored.
"""

from network.buffer import ByteBuffer
from network.packet import ServerBoundPacket


class SMovementPacket(ServerBoundPacket):
    """Base class for the client-to-server movement packets."""

    FLAG_ON_GROUND = 0x01
    FLAG_HORIZONTAL_COLLISION = 0x02
    FLAG_VERTICAL_COLLISION = 0x04

    # Which of the tracked player values this packet reports. The client
    # sends only what changed that tick, so each subclass declares its own and
    # a handler can tell "yaw is 0 because the player looked at 0" from
    # "yaw is 0 because this packet never mentioned it".
    CARRIES_POSITION = False
    CARRIES_ROTATION = False

    def __init__(self, flags: int = 0):
        super().__init__()
        self.flags = flags

    @property
    def on_ground(self) -> bool:
        return bool(self.flags & self.FLAG_ON_GROUND)

    @property
    def horizontal_collision(self) -> bool:
        return bool(self.flags & self.FLAG_HORIZONTAL_COLLISION)

    @property
    def vertical_collision(self) -> bool:
        return bool(self.flags & self.FLAG_VERTICAL_COLLISION)

    def _write_flags(self, buf: ByteBuffer) -> None:
        buf.write_byte(self.flags)

    def _read_flags(self, buf: ByteBuffer) -> None:
        self.flags = buf.read_unsigned_byte()
