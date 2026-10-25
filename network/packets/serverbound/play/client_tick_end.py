"""
Client Tick End (Serverbound, Play)

An empty packet the client sends at the end of every client tick - roughly
20 times a second while it is connected. It carries no data at all.

It is the client's "I finished my tick" marker, introduced so a server can
detect a client that is not ticking. Because it arrives continuously, an
unregistered ClientTickEnd shows up as a wall of warnings that looks like a
flood but is just normal traffic.
"""

from network.buffer import ByteBuffer
from network.packet import ServerBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.CLIENT_TO_SERVER, 0x0C)
class SClientTickEnd(ServerBoundPacket):
    """End of a client tick. No payload.

    Packet ID: 0x0C
    """

    name = "Client Tick End"
    packet_id = 0x0C

    def write(self, buf: ByteBuffer) -> None:
        pass

    def read(self, buf: ByteBuffer) -> None:
        pass
