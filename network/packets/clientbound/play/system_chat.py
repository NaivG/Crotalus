"""
System Chat Message (Clientbound, Play)

A chat line the server vouches for directly — motd-style notices and, in
Crotalus, the player-chat relay. Using system chat instead of Player Chat
(0x3F) sidesteps the whole 1.19+ message signing scheme: system messages are
unsigned by definition, so an offline server can relay text the client will
still display.
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction
from datatypes import write_nbt


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x77)
class CSystemChatMessage(ClientBoundPacket):
    """System Chat Message.

    Fields:
    - message: text component, NBT-encoded. A bare string is wrapped as
      {"text": ...}; a dict is written as-is, which is how "translate"
      components (client-localised command feedback) go out.
    - overlay: True renders above the hotbar (actionbar) instead of in chat
    """

    name = "System Chat Message"
    packet_id = 0x77

    def __init__(self, message: str | dict = "", overlay: bool = False):
        super().__init__()
        self.message = message
        self.overlay = overlay

    def write(self, buf: ByteBuffer) -> None:
        component = self.message if isinstance(self.message, dict) else {"text": self.message}
        buf.write(write_nbt(component))
        buf.write_bool(self.overlay)

    def read(self, buf: ByteBuffer) -> None:
        # Not implementing read for server-side
        pass
