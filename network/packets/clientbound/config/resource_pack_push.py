"""
Resource Pack Push (Clientbound)

Offers a resource pack. The optional prompt message lets the server show custom
wording in the confirmation dialog instead of the client's default.
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.CONFIG, Direction.SERVER_TO_CLIENT, 0x09)
class CResourcePackPush(ClientBoundPacket):
    """Resource pack push."""

    name = "Resource Pack Push"
    packet_id = 0x09

    def __init__(
        self,
        url: str = "",
        hash: str = "",
        forced: bool = False,
        prompt_message: str = None,
    ):
        super().__init__()
        self.url = url
        self.hash = hash
        self.forced = forced
        self.prompt_message = prompt_message

    def write(self, buf: ByteBuffer) -> None:
        buf.write_string(self.url)
        buf.write_string(self.hash)
        buf.write_bool(self.forced)
        if self.prompt_message:
            buf.write_bool(True)
            buf.write_string(self.prompt_message)
        else:
            buf.write_bool(False)

    def read(self, buf: ByteBuffer) -> None:
        self.url = buf.read_string()
        self.hash = buf.read_string()
        self.forced = buf.read_bool()
        if buf.read_bool():
            self.prompt_message = buf.read_string()
