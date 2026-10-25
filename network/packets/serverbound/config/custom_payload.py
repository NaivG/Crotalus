"""
Custom Payload (Serverbound, Config)

Plugin channel traffic during the config state - most importantly the client's
`minecraft:brand`, which identifies it as vanilla or modded and decides
whether the server needs to send server-side brand data back.
"""

from network.buffer import ByteBuffer
from network.packet import ServerBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.CONFIG, Direction.CLIENT_TO_SERVER, 0x02)
class SCustomPayload(ServerBoundPacket):
    """Client sends plugin message (e.g., minecraft:brand).

    Format:
    - channel: string (resource location, e.g., "minecraft:brand")
    - data: remaining bytes
    """

    name = "Custom Payload (Config)"
    packet_id = 0x02

    def __init__(self, channel: str = "", data: bytes = b""):
        super().__init__()
        self.channel = channel
        self.data = data

    def write(self, buf: ByteBuffer) -> None:
        buf.write_string(self.channel)
        buf.write_bytes(self.data)

    def read(self, buf: ByteBuffer) -> None:
        self.channel = buf.read_string()
        self.data = buf.read_bytes(buf.readable_bytes())
