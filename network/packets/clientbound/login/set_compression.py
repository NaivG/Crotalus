"""
Set Compression (Clientbound)

Turns on zlib for everything that follows. From the next packet onward each
frame is prefixed with a VarInt uncompressed length, and bodies at or above
the threshold are deflated.

Kept in the login state (not config) because it must be negotiated before any
bulk data - registries, chunk data - starts flowing.
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.LOGIN, Direction.SERVER_TO_CLIENT, 0x03)
class CSetCompression(ClientBoundPacket):
    """Server tells client to enable compression.

    After this packet, all packets are compressed with zlib.

    Fields:
    - threshold: VarInt - packets larger than this will be compressed (0 = compress everything)
    """

    name = "Set Compression"
    packet_id = 0x03

    def __init__(self, threshold: int = 256):
        super().__init__()
        self.threshold = threshold

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(self.threshold)

    def read(self, buf: ByteBuffer) -> None:
        self.threshold = buf.read_varint()
