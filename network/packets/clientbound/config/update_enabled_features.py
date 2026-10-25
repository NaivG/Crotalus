"""
Update Enabled Features (Clientbound)

Tells the client which optional feature sets to enable - vanilla, bundle, and
friends. The client uses this to decide which packets and components it should
expect.
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.CONFIG, Direction.SERVER_TO_CLIENT, 0x0C)
class CUpdateEnabledFeatures(ClientBoundPacket):
    """Update enabled features."""

    name = "Update Enabled Features"
    packet_id = 0x0C

    def __init__(self, features: list = None):
        super().__init__()
        self.features = features or [
            "minecraft:vanilla",
            "minecraft:bundle",
        ]

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(len(self.features))
        for feature in self.features:
            buf.write_string(feature)

    def read(self, buf: ByteBuffer) -> None:
        num = buf.read_varint()
        self.features = []
        for _ in range(num):
            self.features.append(buf.read_string())
