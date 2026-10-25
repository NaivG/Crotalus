"""
Teleport Confirm (Serverbound, Play)

The client's acknowledgement of a CSynchronizePlayerPosition. Until this
arrives the client ignores its own movement input, so the server must pair
every teleport with a confirmation or the player simply stops responding.
"""

from network.buffer import ByteBuffer
from network.packet import ServerBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.CLIENT_TO_SERVER, 0x00)
class STeleportConfirm(ServerBoundPacket):
    """Confirm a teleport request.

    Packet ID: 0x00
    """

    name = "Teleport Confirm"
    packet_id = 0x00

    def __init__(self, teleport_id: int = 0):
        super().__init__()
        self.teleport_id = teleport_id

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(self.teleport_id)

    def read(self, buf: ByteBuffer) -> None:
        self.teleport_id = buf.read_varint()
