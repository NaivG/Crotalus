"""
Custom Payload (Clientbound, Config)

The server's half of the plugin channel. 1.20.2+ requires an explicit VarInt
length prefix on the data - the client uses it to know where the frame ends
instead of assuming "rest of the packet".
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.CONFIG, Direction.SERVER_TO_CLIENT, 0x01)
class CPluginMessageConfig(ClientBoundPacket):
    """Plugin Message - channel identifier + raw bytes.

    Format:
    - channel: string (resource location)
    - data: remaining bytes
    """

    name = "Plugin Message (Config)"
    packet_id = 0x01

    def __init__(self, channel: str = "", data: bytes = b""):
        super().__init__()
        self.channel = channel
        self.data = data

    def write(self, buf: ByteBuffer) -> None:
        buf.write_string(self.channel)
        # Minecraft 1.20.2+: Data needs VarInt length prefix
        buf.write_varint(len(self.data))
        buf.write_bytes(self.data)

    def read(self, buf: ByteBuffer) -> None:
        self.channel = buf.read_string()
        data_len = buf.read_varint()
        self.data = buf.read_bytes(data_len)
