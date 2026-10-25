"""
Set Titles Animation (Clientbound, Play) - packet id 0x71

Sets the timings for any title text and subtitle text that follow on the wire.
The three ints are ticks:

    fade_in       how long to fade from invisible to full opacity
    stay          how long the title stays at full opacity
    fade_out      how long to fade from full to invisible

A title text or subtitle text without a paired animation packet will not
animate at all -- it just appears.

Packet ID 0x71 per Pumpkin @ 42c903493 assets/packets.json (protocol 774).
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x71)
class CSetTitlesAnimation(ClientBoundPacket):
    """Set Titles Animation - fade-in / stay / fade-out timings.

    Packet ID: 0x71
    """

    name = "Set Titles Animation"
    packet_id = 0x71

    def __init__(
        self,
        fade_in: int = 10,
        stay: int = 70,
        fade_out: int = 20,
    ):
        super().__init__()
        self.fade_in = fade_in
        self.stay = stay
        self.fade_out = fade_out

    def write(self, buf: ByteBuffer) -> None:
        buf.write_int(self.fade_in)
        buf.write_int(self.stay)
        buf.write_int(self.fade_out)

    def read(self, buf: ByteBuffer) -> None:
        self.fade_in = buf.read_int()
        self.stay = buf.read_int()
        self.fade_out = buf.read_int()