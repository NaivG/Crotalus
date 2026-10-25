"""
Disconnect (Config, Clientbound)

Kicks the client while it is still in the config state. Distinct from the
login-state disconnect: the failure happened during setup (usually a bad
registry or resource pack), not during authentication.
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.CONFIG, Direction.SERVER_TO_CLIENT, 0x02)
class CDisconnectConfig(ClientBoundPacket):
    """Server kicks client during config."""

    name = "Disconnect (Config)"
    packet_id = 0x02

    def __init__(self, json_reason: str = '{"text": "Disconnected"}'):
        super().__init__()
        self.json_reason = json_reason

    def write(self, buf: ByteBuffer) -> None:
        buf.write_component(self.json_reason)

    def read(self, buf: ByteBuffer) -> None:
        self.json_reason = buf.read_component()
