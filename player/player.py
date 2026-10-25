"""
Player Class

The domain object for somebody at the server: who they are, where they are,
what they may do. It extends `entity.LivingEntity` (which itself extends
`Entity`) because a player is not *like* an entity - the client tracks them
with the same spawn packet and the same delta movement packets as any other
entity, and the server's bookkeeping mirrors that instead of keeping a
parallel position scheme.

The class hierarchy mirrors Pumpkin, and the boundary between `Player` and
`LivingEntity` does the same work:

    Entity               position, runtime id, world, removed flag
      |
    LivingEntity         health, hurt cooldown, status effects, knockback
      |
    Player               identity, food, gamemode, persisted profile

Everything that belongs to the *pipe* rather than the person -- crypto
handshakes, outstanding keep-alive ids, the config-phase step we are waiting
on -- lives on ClientHandler, because a Player is something you can save to
disk and load back tomorrow and an asyncio task is not that.

The split matters more than it looks. Once the client's position lives here
rather than scattered across three private floats on the connection, the
chunk streamer, the entity tracker, the console and a future physics tick
can all be handed the same object and agree about where the player is,
instead of each re-deriving it.

The fields this class declares are the ones ONLY a player has. Identity
(`entity_id`, `uuid`), geometry (`position`) and the removed flag come from
`Entity`; health, hurt cooldown, status effects and the death path come from
`LivingEntity`. Redeclaring any of them would be how two copies of the truth
appear with one name.

**Plain class, not `@dataclass`.** Both `Entity` (parent up the chain) and
`LivingEntity` (the immediate parent) have explicit `__init__`s, and
`@dataclass` does not chain `super().__init__()` through non-dataclass
parents. Keeping Player explicit is the only way to make the parent chain
run in the right order. (Player overrides `__str__` and `__repr__` so the
generated defaults are not needed; `__eq__` is identity, which is what the
tracker already relies on.)
"""

import uuid as uuid_module
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from entity import (
    EntityPosition,
    LivingEntity,
    allocate_entity_id,
    player_entity_type,
)


class GameMode(Enum):
    """Minecraft game modes."""

    SURVIVAL = 0
    CREATIVE = 1
    ADVENTURE = 2
    SPECTATOR = 3

    @classmethod
    def from_wire(cls, value: int) -> "GameMode":
        """Map a wire value to a mode, falling back to survival.

        Modes cross the wire as bare integers -- they go in the Join Game
        packet, the tab list and Game Event -- so every read of one is a
        chance to be handed something out of range. Guessing wrong here
        costs a player their gamemode; raising would cost them the session.
        """
        try:
            return cls(value)
        except ValueError:
            return cls.SURVIVAL


class PlayerState(Enum):
    """Player connection state.

    A mirror of the protocol's own state machine, kept separately so that
    player/player.py stays free of network imports. The handler copies across
    on every transition; see ClientHandler.set_state.
    """

    DISCONNECTED = 0
    HANDSHAKING = 1
    LOGIN = 2
    CONFIG = 3
    PLAY = 4


# PlayerPosition lived here from the beginning and grew into EntityPosition
# (entity/position.py) once entities needed the same six fields plus
# velocity and derived coordinates. The serialised keys did not change, so
# existing player JSON files load unchanged.


class Player(LivingEntity):
    """Represents a connected player in the server.

    Built the moment a socket opens, which is before the client has told us
    anything, so uuid and name default to empty and are filled in during
    login. That is deliberate: a connection that never logs in still has one
    of these, so nothing downstream has to guard for the player not existing.

    The `LivingEntity` half (`health`, `max_health`, `hurt_cooldown`, the
    effect map, the death path) is initialised by `super().__init__`; the
    `Entity` half (`entity_id`, `uuid`, `position`, `dimension`,
    `removed`) runs before that. The entity id is claimed from the global
    allocator the moment the object exists (ids are never reused, so
    burning one on a socket that never logs in costs nothing but a counter
    tick), and the type is pinned to `minecraft:player` because this class
    *is* that type.
    """

    def __init__(self,
                 name: str = "",
                 state: PlayerState = PlayerState.DISCONNECTED,
                 game_mode: GameMode = GameMode.SURVIVAL,
                 loaded: bool = False,
                 locale: str = "en_US",
                 view_distance: int = 10,
                 chat_mode: int = 0,
                 chat_colors: bool = True,
                 skin_parts: int = 0xFF,
                 main_hand: int = 1,  # 0: LEFT, 1: RIGHT
                 text_filtering: bool = False,
                 allow_listed_server: bool = False,
                 brand: str = "",
                 food: int = 20,
                 saturation: float = 5.0,
                 experience: float = 0.0,
                 level: int = 0,
                 client_ticks: int = 0,
                 chat_sent: int = 0,
                 chat_index: int = 0,
                 last_login: Optional[datetime] = None,
                 first_play: Optional[datetime] = None,
                 ip_address: str = "",
                 properties: Optional[dict] = None,
                 # Entity-shaped pass-throughs, defaulted to the values the
                 # rest of the server fills in on its own.
                 uuid: str = "",
                 entity_id: int = 0,
                 entity_type=None,
                 position: Optional[EntityPosition] = None,
                 dimension: str = "minecraft:overworld",
                 removed: bool = False,
                 # LivingEntity pass-throughs.
                 max_health: Optional[float] = None,
                 health: Optional[float] = 20.0) -> None:
        # Build the wire-stable identity, then the living half, then store
        # the player-only fields. `super().__init__` runs `Entity.__init__`
        # before `LivingEntity.__init__` because the explicit chain is the
        # only way to make Python call both given a non-dataclass parent.
        super().__init__(
            entity_id=entity_id,
            uuid=uuid,
            entity_type=entity_type if entity_type is not None else player_entity_type(),
            position=position if position is not None else EntityPosition(),
            dimension=dimension,
            removed=removed,
            max_health=max_health,
            health=health if health is not None else 20.0,
        )

        # Player-only identity: the profile name. The uuid and the runtime
        # entity id are Entity's, surfaced through this class.
        self.name: str = name

        # Connection state, mirroring the protocol's own state machine.
        self.state: PlayerState = state

        # Game state
        self.game_mode: GameMode = game_mode

        # Set once the client has sent PlayerLoaded, i.e. it is playable rather
        # than merely present. Until then it is still applying its initial chunk
        # set and drops anything the server says -- so this, not the protocol
        # state, is what decides whether a player receives broadcasts.
        self.loaded: bool = loaded

        # Client settings, from the Client Information packet. The client sends
        # the same payload twice, once in config and once on entering play; the
        # play copy is the one that matters for view_distance, because that is
        # where it learns the server's own limit.
        self.locale: str = locale
        self.view_distance: int = view_distance
        self.chat_mode: int = chat_mode
        self.chat_colors: bool = chat_colors
        self.skin_parts: int = skin_parts
        self.main_hand: int = main_hand
        self.text_filtering: bool = text_filtering
        self.allow_listed_server: bool = allow_listed_server
        self.brand: str = brand

        # Survival-state fields. `health` itself lives on `LivingEntity` -
        # this constructor's `health=` kwarg is forwarded to the base
        # initializer. The two food half-fields are player-only.
        self.food: int = food
        self.saturation: float = saturation
        self.experience: float = experience
        self.level: int = level

        # Client tick counter, driven by the ~20/s ClientTickEnd packet
        self.client_ticks: int = client_ticks

        # Player Chat bookkeeping. These are two different counters and conflating
        # them is a real bug: the Index counts lines this player has *sent*, the
        # Global Index counts Player Chat packets this player has *received*. One
        # is about the sender, the other about the receiver, and no two players
        # agree on either.
        self.chat_sent: int = chat_sent
        self.chat_index: int = chat_index

        # Timing
        self.last_login: Optional[datetime] = last_login
        self.first_play: Optional[datetime] = first_play

        # Meta
        self.ip_address: str = ip_address
        self.properties: dict = dict(properties) if properties is not None else {}

        # A freshly-built Player still needs the runtime id and timestamps
        # to be filled in - mirrors what the old `__post_init__` did, but
        # folded into the explicit constructor so non-dataclass inheritance
        # still works.
        if self.entity_id == 0:
            self.entity_id = allocate_entity_id()
        if self.last_login is None:
            self.last_login = datetime.now()
        if self.first_play is None:
            self.first_play = datetime.now()

    @property
    def display_name(self) -> str:
        """Get player's display name."""
        return self.properties.get("display_name", self.name)

    def is_spawned(self) -> bool:
        """True once the player is in the world and actually playable.

        Both halves are needed. PLAY alone includes the window between
        entering the state and finishing the loading screen, and a client
        still in that window discards chat as a validation error.
        """
        return self.state == PlayerState.PLAY and self.loaded

    def is_op(self) -> bool:
        """Check if player has operator privileges."""
        return self.properties.get("op", False)

    def set_op(self, value: bool = True) -> None:
        """Set operator status."""
        self.properties["op"] = value

    def apply_client_settings(self, packet) -> None:
        """Fold a Client Information packet into these settings.

        Takes the packet by duck-typing rather than importing it: the config
        and play variants are separate classes with the same fields, and
        this module deliberately knows nothing about the network.
        """
        self.locale = packet.locale
        self.view_distance = packet.view_distance
        self.chat_mode = packet.chat_mode
        self.chat_colors = packet.chat_colors
        self.skin_parts = packet.displayed_skin_parts
        self.main_hand = packet.main_hand
        self.text_filtering = packet.enable_text_filtering
        self.allow_listed_server = packet.allow_listed_server

    def claim_sent_index(self) -> int:
        """Claim the index to stamp on the next Player Chat line we send.

        Per *sender*, so it belongs to the player typing. Claimed before the
        send is awaited: two lines typed in the same tick must not collide.
        """
        index = self.chat_sent
        self.chat_sent += 1
        return index

    def claim_chat_index(self) -> int:
        """Claim the Player Chat index for the next message sent *to* us.

        Per *receiver*: the client counts the Player Chat packets it has
        been sent, so two recipients of one line each hand out their own.
        """
        index = self.chat_index
        self.chat_index += 1
        return index

    def _tick_regeneration(self) -> None:
        """Player regeneration, food-driven.

        The base class's regen is a no-op so mobs with no hunger model do
        nothing. Players regenerate 1 HP every 4 ticks when `food` is at 18
        or more and `saturation` is positive - the simplification of
        vanilla's rule that ignores the saturation-having-priority detail.
        """
        if self.dead or self.health >= self.max_health:
            return
        if self.food < 18 or self.food <= 0:
            return
        # 4 ticks at 20 tps = one regen tick on a 5-second clock; we keep
        # the cadence on `_age` so the regen line up with the rest of the
        # tick surface.
        if (self._age & 0x3) != 0:
            return
        self.heal(1.0)
        # Food drops one point per successful regen; saturation also drops
        # first and only depletes food when it hits zero. Vanilla's full
        # table is more nuanced, but the visible difference is "regen eats
        # food", which this preserves.
        if self.saturation > 0.0:
            self.saturation = max(0.0, self.saturation - 1.0)
        else:
            self.food = max(0, self.food - 1)

    def spawn_blob(self) -> bytes:
        """The metadata blob used at spawn, on top of add_entity.

        Players get skin + profile (the two player-specific metadata
        entries), in the same order Pumpkin's spawn path sends them. The
        health-only blob `LivingEntity.spawn_blob` would build is wrong
        here - a player at index 9 with a profile would put the client in
        an inconsistent state.
        """
        from entity.metadata import player_spawn_blob
        return player_spawn_blob(self.name, self.skin_parts, self.properties.get("textures", ()))

    def to_dict(self) -> dict:
        """Serialize the persistent half of the player to storage.

        Deliberately excludes the connection-scoped fields -- state, loaded,
        the tick count and both chat counters. Saving those would mean
        restoring a player who is still "in the world" and still expecting
        chat packets they have already been sent.
        """
        return {
            "uuid": self.uuid,
            "name": self.name,
            "game_mode": self.game_mode.value,
            "position": self.position.to_dict(),
            "health": self.health,
            "max_health": self.max_health,
            "food": self.food,
            "saturation": self.saturation,
            "experience": self.experience,
            "level": self.level,
            "locale": self.locale,
            "view_distance": self.view_distance,
            "chat_mode": self.chat_mode,
            "chat_colors": self.chat_colors,
            "skin_parts": self.skin_parts,
            "main_hand": self.main_hand,
            "text_filtering": self.text_filtering,
            "allow_listed_server": self.allow_listed_server,
            "brand": self.brand,
            "ip_address": self.ip_address,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "first_play": self.first_play.isoformat() if self.first_play else None,
            "properties": self.properties,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Player":
        """Deserialize player data from storage.

        Every field goes through .get, and the game mode through from_wire:
        a player file is a thing a human can hand-edit, and a typo in it
        should cost that one player their file, not the server's startup.

        The entity id is deliberately not read: it is runtime state from
        the global allocator, and a stored one would either collide with a
        live entity or (harmlessly but confusingly) burn the allocator's
        `observe()` for no benefit - nothing persisted refers to a player
        by entity id.
        """
        position = EntityPosition.from_dict(data.get("position", {}))

        last_login = None
        if data.get("last_login"):
            last_login = datetime.fromisoformat(data["last_login"])

        first_play = None
        if data.get("first_play"):
            first_play = datetime.fromisoformat(data["first_play"])

        stored_uuid = data.get("uuid") or str(uuid_module.uuid4())

        return cls(
            uuid=stored_uuid,
            name=data.get("name", "Unknown"),
            game_mode=GameMode.from_wire(data.get("game_mode", 0)),
            position=position,
            health=data.get("health"),
            max_health=data.get("max_health"),
            food=data.get("food", 20),
            saturation=data.get("saturation", 5.0),
            experience=data.get("experience", 0.0),
            level=data.get("level", 0),
            locale=data.get("locale", "en_US"),
            view_distance=data.get("view_distance", 10),
            chat_mode=data.get("chat_mode", 0),
            chat_colors=data.get("chat_colors", True),
            skin_parts=data.get("skin_parts", 0xFF),
            main_hand=data.get("main_hand", 1),
            text_filtering=data.get("text_filtering", False),
            allow_listed_server=data.get("allow_listed_server", False),
            brand=data.get("brand", ""),
            ip_address=data.get("ip_address", ""),
            last_login=last_login,
            first_play=first_play,
            properties=data.get("properties", {}),
        )

    def __str__(self) -> str:
        if not self.uuid:
            return f"Player(unidentified, {self.ip_address or 'no address'})"
        return f"Player({self.name}, {self.uuid[:8]}...)"

    def __repr__(self) -> str:
        return self.__str__()
