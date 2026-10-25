"""
Move Entity Pos Rot (Clientbound, Play) - packet id 0x34

The combined position-and-rotation delta - the packet most entities travel
by, since walking almost always turns you a little. Same 1/4096 fixed-point
deltas as Move Entity Pos (0x33), with the packed yaw and pitch appended.

Angle encoding matches Add Entity's (floor(d * 256 / 360), yaw wrapped,
pitch not) but via `ByteBuffer.write_angle`, which is where that rule lives
once four packets share it.
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x34)
class CMoveEntityPosRot(ClientBoundPacket):
    """Move Entity Pos Rot - combined position and rotation delta.

    Packet ID: 0x34
    """

    name = "Move Entity Pos Rot"
    packet_id = 0x34

    def __init__(self, entity_id: int = 0, dx: int = 0, dy: int = 0, dz: int = 0,
                 yaw: float = 0.0, pitch: float = 0.0, on_ground: bool = True):
        super().__init__()
        self.entity_id = entity_id
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.yaw = yaw
        self.pitch = pitch
        self.on_ground = on_ground

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(self.entity_id)
        buf.write_short(self.dx)
        buf.write_short(self.dy)
        buf.write_short(self.dz)
        buf.write_angle(self.yaw)
        buf.write_angle(self.pitch)
        buf.write_bool(self.on_ground)

    def read(self, buf: ByteBuffer) -> None:
        self.entity_id = buf.read_varint()
        self.dx = buf.read_short()
        self.dy = buf.read_short()
        self.dz = buf.read_short()
        self.yaw = buf.read_angle()
        self.pitch = buf.read_angle()
        self.on_ground = buf.read_bool()
