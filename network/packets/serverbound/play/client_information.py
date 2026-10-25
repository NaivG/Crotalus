"""
Client Information (Serverbound, Play)

The same payload the client sends during the configuration phase, repeated
once it is in play. The client uses it to re-report settings that can change
after joining, most importantly the render distance once it knows the server's
own view distance.

The wire format is byte-for-byte identical to the config-state packet; only the
packet ID differs.
"""

from network.buffer import ByteBuffer
from network.packet import ServerBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.CLIENT_TO_SERVER, 0x0D)
class SClientInformationPlay(ServerBoundPacket):
    """Client settings, re-sent on entering play.

    Packet ID: 0x0D
    """

    name = "Client Information"
    packet_id = 0x0D

    def __init__(
        self,
        locale: str = "en_US",
        view_distance: int = 10,
        chat_mode: int = 0,
        chat_colors: bool = True,
        displayed_skin_parts: int = 0xFF,
        main_hand: int = 1,
        enable_text_filtering: bool = False,
        allow_listed_server: bool = False,
    ):
        super().__init__()
        self.locale = locale
        self.view_distance = view_distance
        self.chat_mode = chat_mode
        self.chat_colors = chat_colors
        self.displayed_skin_parts = displayed_skin_parts
        self.main_hand = main_hand
        self.enable_text_filtering = enable_text_filtering
        self.allow_listed_server = allow_listed_server

    def write(self, buf: ByteBuffer) -> None:
        buf.write_string(self.locale)
        buf.write_byte(self.view_distance)
        buf.write_varint(self.chat_mode)
        buf.write_bool(self.chat_colors)
        buf.write_byte(self.displayed_skin_parts)
        buf.write_varint(self.main_hand)
        buf.write_bool(self.enable_text_filtering)
        buf.write_bool(self.allow_listed_server)

    def read(self, buf: ByteBuffer) -> None:
        self.locale = buf.read_string()
        self.view_distance = buf.read_byte()
        self.chat_mode = buf.read_varint()
        self.chat_colors = buf.read_bool()
        self.displayed_skin_parts = buf.read_unsigned_byte()
        self.main_hand = buf.read_varint()
        self.enable_text_filtering = buf.read_bool()
        self.allow_listed_server = buf.read_bool()
