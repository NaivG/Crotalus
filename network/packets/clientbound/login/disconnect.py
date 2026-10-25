"""
Disconnect (Login, Clientbound)

Kicks the client while still in the login state, before the world exists.
The reason is a JSON text component, NBT-encoded on the wire.
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.LOGIN, Direction.SERVER_TO_CLIENT, 0x00)
class CDisconnect(ClientBoundPacket):
    """Server kicks the client during login.

    Fields:
    - json_reason: JSON text component, NBT-encoded on the wire
    """

    name = "Disconnect (Login)"
    packet_id = 0x00

    def __init__(self, json_reason: str = '{"text": "Disconnected"}'):
        super().__init__()
        self.json_reason = json_reason

    def write(self, buf: ByteBuffer) -> None:
        buf.write_component(self.json_reason)

    def read(self, buf: ByteBuffer) -> None:
        self.json_reason = buf.read_component()
