"""
Clear Titles (Clientbound, Play) - packet id 0x0E

Dismisses any title currently on the screen. The boolean tells the client
whether to also reset the title/subtitle text buffers; setting it false
just hides the visible title, while setting it true wipes the stored text
so a fresh animation stack needs both Set Title Text + Set Titles Animation
to play again.

Packet ID 0x0E per Pumpkin @ 42c903493 assets/packets.json (protocol 774).
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x0E)
class CClearTitles(ClientBoundPacket):
    """Clear Titles - dismiss any current title.

    Packet ID: 0x0E
    """

    name = "Clear Titles"
    packet_id = 0x0E

    def __init__(self, reset: bool = True):
        super().__init__()
        # True wipes both the visible title and the buffered text; false
        # just hides the current title until the next Set Title Text.
        self.reset = reset

    def write(self, buf: ByteBuffer) -> None:
        buf.write_bool(self.reset)

    def read(self, buf: ByteBuffer) -> None:
        self.reset = buf.read_bool()