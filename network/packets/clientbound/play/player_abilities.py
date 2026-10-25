"""
Player Abilities (Clientbound, Play)

Tells the client what this player is currently allowed to do: take damage,
fly, stop flying, break blocks instantly. One flag byte plus two tuning
floats.

This is how a gamemode switch actually takes effect client-side. The Game
Event packet changes the client's notion of "which gamemode am I in" (tab
icon, hotbar behaviour), but it is the abilities packet that grants flight:
a creative switch without it leaves the player unable to double-jump.
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction

# Flag bits, as documented for the client's own Abilities enum.
FLAG_INVULNERABLE = 0x01
FLAG_FLYING = 0x02
FLAG_ALLOW_FLYING = 0x04
FLAG_INSTANT_BREAK = 0x08

# Gamemode numbers, matching both the protocol and CLogin's game_mode field.
GAMEMODE_SURVIVAL = 0
GAMEMODE_CREATIVE = 1
GAMEMODE_ADVENTURE = 2
GAMEMODE_SPECTATOR = 3


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x3E)
class CPlayerAbilities(ClientBoundPacket):
    """Player Abilities.

    Packet ID: 0x3E

    Fields:
    - flags: bit field (invulnerable / flying / allow flying / instant break)
    - flying_speed: movement speed while flying, 0.05 by default
    - field_of_view_modifier: FOV scaling, 0.1 by default
    """

    name = "Player Abilities"
    packet_id = 0x3E

    FLYING_SPEED = 0.05
    FIELD_OF_VIEW_MODIFIER = 0.1

    def __init__(self, flags: int = 0):
        super().__init__()
        self.flags = flags

    @classmethod
    def for_gamemode(cls, gamemode: int) -> "CPlayerAbilities":
        """The ability set a vanilla server grants for each gamemode.

        Creative gets allow-flying but starts grounded -- the player earns
        flight with a double-jump. Spectator gets the flying bit itself,
        because a spectator is airborne from the moment the mode lands.
        Neither survival nor adventure grants anything.
        """
        if gamemode == GAMEMODE_CREATIVE:
            flags = FLAG_INVULNERABLE | FLAG_ALLOW_FLYING | FLAG_INSTANT_BREAK
        elif gamemode == GAMEMODE_SPECTATOR:
            flags = FLAG_INVULNERABLE | FLAG_FLYING | FLAG_ALLOW_FLYING
        else:
            flags = 0
        return cls(flags)

    def write(self, buf: ByteBuffer) -> None:
        buf.write_byte(self.flags)
        buf.write_float(self.FLYING_SPEED)
        buf.write_float(self.FIELD_OF_VIEW_MODIFIER)

    def read(self, buf: ByteBuffer) -> None:
        # Not implementing read for server-side
        pass
