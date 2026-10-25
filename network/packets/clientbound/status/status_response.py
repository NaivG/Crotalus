"""
Status Response (Clientbound)

The JSON blob the server-list screen renders: MOTD, player counts, version.
Sent in reply to SStatusRequest.
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.STATUS, Direction.SERVER_TO_CLIENT, 0x00)
class CStatusResponse(ClientBoundPacket):
    """Server responds with status information.

    Fields:
    - json_response: String (JSON containing server info)
    """

    name = "Status Response"
    packet_id = 0x00

    def __init__(self, json_response: str = "{}"):
        super().__init__()
        self.json_response = json_response

    def write(self, buf: ByteBuffer) -> None:
        buf.write_string(self.json_response)

    def read(self, buf: ByteBuffer) -> None:
        self.json_response = buf.read_string()

    def __repr__(self) -> str:
        return f"<CStatusResponse length={len(self.json_response)}>"
