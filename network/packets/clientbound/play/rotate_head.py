"""
Rotate Head (Clientbound, Play) - packet id 0x51

Head yaw only, and deliberately NOT throttled with the position packets:
vanilla sends it every tick it changes. The body yaw a client renders comes
from the move packets' yaw; the head is a separate bone and lags behind the
body when turning, which is what makes a player's head "lead" their torso.
Collapsing head yaw into the move packets is the classic
"everyone else's head is frozen" bug.

The value is packed the same way as every angle byte, via
`ByteBuffer.write_angle`.
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x51)
class CRotateHead(ClientBoundPacket):
    """Rotate Head - set a tracked entity's head yaw.

    Packet ID: 0x51
    """

    name = "Rotate Head"
    packet_id = 0x51

    def __init__(self, entity_id: int = 0, head_yaw: float = 0.0):
        super().__init__()
        self.entity_id = entity_id
        self.head_yaw = head_yaw

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(self.entity_id)
        buf.write_angle(self.head_yaw)

    def read(self, buf: ByteBuffer) -> None:
        self.entity_id = buf.read_varint()
        self.head_yaw = buf.read_angle()
