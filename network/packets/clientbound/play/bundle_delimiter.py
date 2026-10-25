"""
Bundle Delimiter (Clientbound, Play) - packet id 0x00

The bracketing packet for "process this batch on the same tick". The vanilla
server uses it to make spawn + initial-entity-data packets land on the same
frame: the client stores everything after a delimiter, plays it on the next
one, and is asked to keep each batch under 4096 packets.

The body is empty by design. The packet ID alone is the signal that the next
packet is part of a bundle, and another 0x00 marks the end of that bundle.
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x00)
class CBundleDelimiterPacket(ClientBoundPacket):
    """Bundle Delimiter - bracket a group of packets for same-tick delivery.

    Packet ID: 0x00
    """

    name = "Bundle Delimiter"
    packet_id = 0x00

    def write(self, buf: ByteBuffer) -> None:
        pass

    def read(self, buf: ByteBuffer) -> None:
        pass