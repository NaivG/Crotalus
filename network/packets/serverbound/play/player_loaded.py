"""
Player Loaded (Serverbound, Play)

A zero-payload packet the client sends once it considers itself loaded into
the world, after the initial chunk set has arrived. Useful as a "the player is
actually playable now" signal, distinct from the teleport that put them there.
"""

from network.buffer import ByteBuffer
from network.packet import ServerBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.CLIENT_TO_SERVER, 0x2B)
class SPlayerLoaded(ServerBoundPacket):
    """Player loaded.

    Packet ID: 0x2B
    """

    name = "Player Loaded"
    packet_id = 0x2B

    def write(self, buf: ByteBuffer) -> None:
        pass

    def read(self, buf: ByteBuffer) -> None:
        pass
