"""
Login / Join Game (Clientbound, Play)

The packet that makes the world exist. It carries the entity ID, the dimension
type, game mode and a pile of flags the client uses to initialise its renderer
and HUD. Nothing about the world can be sent before this arrives, which is why
the server sends it the instant the connection leaves the config state.
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x30)
class CLogin(ClientBoundPacket):
    """Join Game packet - transitions client from Config to Play.

    This is one of the most important packets.
    """

    name = "Login (Join Game)"
    packet_id = 0x30

    def __init__(
        self,
        entity_id: int = 1,
        is_hardcore: bool = False,
        dimension_names: list = None,
        max_players: int = 20,
        view_distance: int = 10,
        simulated_distance: int = 10,
        reduced_debug_info: bool = False,
        enabled_respawn_screen: bool = True,
        limited_crafting: bool = False,
        dimension_type: int = 0,
        dimension_name: str = "minecraft:overworld",
        hashed_seed: int = 0,
        game_mode: int = 0,
        previous_gamemode: int = -1,
        debug: bool = False,
        is_flat: bool = False,
        has_death_position: bool = False,
        death_dimension: str | None = None,
        death_position: tuple | None = None,
        portal_cooldown: int = 0,
        sealevel: int = 63,
        enforce_secure_chat: bool = True,
    ):
        super().__init__()
        self.entity_id = entity_id
        self.is_hardcore = is_hardcore
        self.dimension_names = dimension_names or ["minecraft:overworld", "minecraft:nether", "minecraft:the_end"]
        self.max_players = max_players
        self.view_distance = view_distance
        self.simulated_distance = simulated_distance
        self.reduced_debug_info = reduced_debug_info
        self.enabled_respawn_screen = enabled_respawn_screen
        self.limited_crafting = limited_crafting
        self.dimension_type = dimension_type
        self.dimension_name = dimension_name
        self.hashed_seed = hashed_seed
        self.game_mode = game_mode
        self.previous_gamemode = previous_gamemode
        self.debug = debug
        self.is_flat = is_flat
        self.has_death_position = has_death_position
        self.death_dimension = death_dimension
        self.death_position = death_position
        self.portal_cooldown = portal_cooldown
        self.sealevel = sealevel
        self.enforce_secure_chat = enforce_secure_chat

    def write(self, buf: ByteBuffer) -> None:
        # Entity ID
        buf.write_int(self.entity_id)

        # Is hardcore
        buf.write_bool(self.is_hardcore)

        # Dimension names (list)
        buf.write_varint(len(self.dimension_names))
        for dim in self.dimension_names:
            buf.write_string(dim)

        # Max players
        buf.write_varint(self.max_players)

        # View distance
        buf.write_varint(self.view_distance)

        # Simulation distance
        buf.write_varint(self.simulated_distance)

        # Reduced debug info
        buf.write_bool(self.reduced_debug_info)

        # Enabled respawn screen
        buf.write_bool(self.enabled_respawn_screen)

        # Limited crafting
        buf.write_bool(self.limited_crafting)

        # Dimension type (1.21.2+)
        buf.write_varint(self.dimension_type)

        # Dimension name
        buf.write_string(self.dimension_name)

        # Hashed seed
        buf.write_long(self.hashed_seed)

        # Game mode
        buf.write_byte(self.game_mode)

        # Previous game mode
        buf.write_byte(self.previous_gamemode)

        # Debug
        buf.write_bool(self.debug)

        # Is flat
        buf.write_bool(self.is_flat)

        # Death dimension + position (optional)
        if self.has_death_position:
            buf.write_bool(True)
            buf.write_string(self.death_dimension)
            if self.death_position:
                buf.write_position(*self.death_position)
            else:
                buf.write_position(0, 0, 0)
        else:
            buf.write_bool(False)

        # Portal cooldown
        buf.write_varint(self.portal_cooldown)

        # Sea level (1.21.2+)
        buf.write_varint(self.sealevel)

        # Enforce secure chat
        buf.write_bool(self.enforce_secure_chat)

    def read(self, buf: ByteBuffer) -> None:
        # Not implementing read for server-side
        pass
