"""
Set Title Text (Clientbound, Play) - packet id 0x70

The big title in the centre of the screen. Three packets stack for a full
title sequence: Set Title Text (this one) for the main line, Set Subtitle
Text (0x6E) for the smaller line beneath, and Set Titles Animation (0x71)
for fade-in / stay / fade-out timings.

Until the animation packet lands the client keeps the previous title (or
nothing); the title text alone does not animate in. The text component
carries the same NBT-on-wire encoding as the chat packets.

Packet ID 0x70 per Pumpkin @ 42c903493 assets/packets.json (protocol 774).
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction
from datatypes import write_nbt


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x70)
class CSetTitleText(ClientBoundPacket):
    """Set Title Text - main line of the centred title.

    Packet ID: 0x70
    """

    name = "Set Title Text"
    packet_id = 0x70

    def __init__(self, text: str | dict = ""):
        super().__init__()
        self.text = text

    def write(self, buf: ByteBuffer) -> None:
        component = self.text if isinstance(self.text, dict) else {"text": self.text}
        buf.write(write_nbt(component))

    def read(self, buf: ByteBuffer) -> None:
        # Not implementing read for server-side
        pass