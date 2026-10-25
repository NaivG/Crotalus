"""
Game Event (Clientbound, Play)

One packet, many unrelated meanings. Event 13 (start waiting for level chunks)
is the load-screen trigger and the reason the server must send it right after
Join Game; the rest drive weather, respawn, gamemode and demo messages.
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x26)
class CGameEvent(ClientBoundPacket):
    """Game Event - various game state events.

    Used for weather, respawn, game mode, demo messages, and more.
    Event 13 = Start waiting for level chunks (critical after Join Game)

    Packet ID: 0x26 (protocol:) / 0x23 (resource:)
    """

    name = "Game Event"
    packet_id = 0x26

    # Event types
    EVENT_NO_RESPAWN_BLOCK = 0
    EVENT_BEGIN_RAIN = 1
    EVENT_END_RAIN = 2
    EVENT_CHANGE_GAME_MODE = 3
    EVENT_WIN_GAME = 4
    EVENT_DEMO_EVENT = 5
    EVENT_ARROW_HIT_PLAYER = 6
    EVENT_RAIN_LEVEL_CHANGE = 7
    EVENT_THUNDER_LEVEL_CHANGE = 8
    EVENT_PUFFERFISH_STING = 9
    EVENT_ELDER_GUARDIAN_APPEAR = 10
    EVENT_ENABLE_RESPAWN_SCREEN = 11
    EVENT_LIMITED_CRAFTING = 12
    EVENT_START_WAITING_FOR_LEVEL_CHUNKS = 13  # Critical!

    def __init__(self, event: int = 13, value: float = 0.0):
        super().__init__()
        self.event = event
        self.value = value

    def write(self, buf: ByteBuffer) -> None:
        buf.write_byte(self.event)
        buf.write_float(self.value)

    def read(self, buf: ByteBuffer) -> None:
        self.event = buf.read_unsigned_byte()
        self.value = buf.read_float()
