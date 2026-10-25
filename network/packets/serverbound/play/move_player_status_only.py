"""
Set Player Movement Flags (Serverbound, Play)

Sent when the movement *state* changes but the position and rotation do not —
stepping off a ledge, or landing. Its only job is to tell the server whether
the player is currently on the ground; the `flags` byte is reused whole for
the collision hints, but they carry no extra information here.
"""

from network.buffer import ByteBuffer
from network.packet import register_packet
from network.protocol import ProtocolState, Direction
from network.packets.serverbound.play.movement import SMovementPacket


@register_packet(ProtocolState.PLAY, Direction.CLIENT_TO_SERVER, 0x20)
class SMovePlayerStatusOnly(SMovementPacket):
    """Player movement flags only.

    Packet ID: 0x20
    """

    name = "Set Player Movement Flags"
    packet_id = 0x20

    def write(self, buf: ByteBuffer) -> None:
        self._write_flags(buf)

    def read(self, buf: ByteBuffer) -> None:
        self._read_flags(buf)
