"""
Login Acknowledged (Serverbound)

Empty packet sent once the client accepts CLoginSuccess. Its arrival is the
server's signal to flip the connection into the CONFIG state - no data needed,
the ID alone is the event.
"""

from network.buffer import ByteBuffer
from network.packet import ServerBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.LOGIN, Direction.CLIENT_TO_SERVER, 0x03)
class SLoginAcknowledged(ServerBoundPacket):
    """Client acknowledges login (1.19.3+).

    No payload.
    """

    name = "Login Acknowledged"
    packet_id = 0x03

    def __init__(self):
        super().__init__()

    def write(self, buf: ByteBuffer) -> None:
        pass

    def read(self, buf: ByteBuffer) -> None:
        pass
