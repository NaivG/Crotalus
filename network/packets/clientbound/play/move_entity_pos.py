"""
Move Entity Pos (Clientbound, Play) - packet id 0x33

The position-only delta: "I moved by this much, facing unchanged". The
deltas are 1/4096-block fixed-point shorts computed by the entity tracker
against the entity's last-sent position; when they would overflow a short
(an 8-block jump) the tracker sends Entity Position Sync (0x23) instead,
so a short here is always a real, small move.

`on_ground` rides along because walking off a ledge changes fall logic
without changing position.
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x33)
class CMoveEntityPos(ClientBoundPacket):
    """Move Entity Pos - position-only delta for a tracked entity.

    Packet ID: 0x33
    """

    name = "Move Entity Pos"
    packet_id = 0x33

    def __init__(self, entity_id: int = 0, dx: int = 0, dy: int = 0, dz: int = 0,
                 on_ground: bool = True):
        super().__init__()
        self.entity_id = entity_id
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.on_ground = on_ground

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(self.entity_id)
        buf.write_short(self.dx)
        buf.write_short(self.dy)
        buf.write_short(self.dz)
        buf.write_bool(self.on_ground)

    def read(self, buf: ByteBuffer) -> None:
        self.entity_id = buf.read_varint()
        self.dx = buf.read_short()
        self.dy = buf.read_short()
        self.dz = buf.read_short()
        self.on_ground = buf.read_bool()
