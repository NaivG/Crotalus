"""
EntityPosition - where an entity is, which way it faces, and how it moves.

This replaces `player/player.py`'s `PlayerPosition`. The old class carried
six fields and served exactly one caller family (the movement packets); an
entity needs the same six plus velocity, and needs the *derived* coordinates
- which block am I in, which chunk section - that the tracker, the chunk
streamer and future physics all re-derive at their call sites today.

Two design decisions, both deviating from Pumpkin on purpose:

**Derived coordinates are computed, not cached.** Pumpkin's `Entity::set_pos`
maintains `block_pos` / `chunk_pos` / `bounding_box` as sibling caches because
Rust makes a single sanctioned writer cheap and every other writer a bug. This
codebase cannot have that writer: `_apply_movement` (client_handler.py)
deliberately writes the fields one by one, and `Player.update_position`'s
docstring already explains why the object must be mutated in place rather
than swapped. A cache fed by five different writers is a stale-read generator,
and `math.floor(x / 16)` is not where this server's time goes. The caches
become properties; if profiling ever says otherwise, the writer paths get
centralised FIRST (see the plan's known-gaps note) and only then is caching
safe to add.

**Velocity lives here, flat.** The packets need it (`add_entity` and
`entity_position_sync` both carry one), physics will maintain it, and
splitting it from position means two objects to keep in sync about the same
question. Nothing maintains it yet this round - it stays at zero, which is
exactly what a freshly-spawned entity's velocity is.

The serialised keys (`x`, `y`, `z`, `yaw`, `pitch`, `on_ground`) are the ones
`PlayerPosition` used, so player JSON files written by older versions load
unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class EntityPosition:
    """Position, rotation and velocity of one entity, mutated in place.

    `x/y/z` are the feet position: Minecraft's entity Y is at the soles, not
    the eyes - the camera is `eye_height` above this and the entity's box
    extends `height` above it. Yaw and pitch are degrees, the client's own
    convention (yaw 0 = +Z / south, increasing clockwise; pitch positive =
    looking down).
    """

    x: float = 0.0
    y: float = 64.0
    z: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    on_ground: bool = True

    # Per-tick movement. Written by nothing this round - there is no physics
    # tick yet - and read by the spawn/position-sync packets, for which a
    # standing-still entity's zero velocity is the correct value.
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    velocity_z: float = 0.0

    @property
    def chunk(self) -> tuple[int, int]:
        """The chunk this position sits in, as (x, z).

        Chunk streaming only ever cares *which* chunk the player is in, not
        where exactly, and re-deriving that at each call site is how a
        server ends up with two different answers.

        floor, not trunc: a player at x = -0.5 is in chunk -1, and int()
        would hand back 0 and stream them the wrong half of the world.
        """
        return (math.floor(self.x / 16), math.floor(self.z / 16))

    @property
    def block_pos(self) -> tuple[int, int, int]:
        """The block the feet occupy, as (x, y, z).

        Vanilla's BlockPos is the floor of the position on all three axes -
        including Y, unlike the chunk property above, which is a column and
        has no Y to floor.
        """
        return (math.floor(self.x), math.floor(self.y), math.floor(self.z))

    @property
    def section(self) -> tuple[int, int, int]:
        """The 16-block chunk section, as (x, y, z) section coords.

        This is the coordinate the entity tracker ranges over (Pumpkin's
        `last_section_pos`, a Vector3 - it does track Y). Arithmetic shift,
        not floor-div-then-int: `>>` on negative ints floors the same way
        math.floor does, which is what makes -0.5 land in section -1.
        """
        return (
            math.floor(self.x) >> 4,
            math.floor(self.y) >> 4,
            math.floor(self.z) >> 4,
        )

    @property
    def velocity(self) -> tuple[float, float, float]:
        """Velocity as the (x, y, z) tuple the packets and physics want."""
        return (self.velocity_x, self.velocity_y, self.velocity_z)

    def set_velocity(self, x: float, y: float, z: float) -> None:
        """Set all three velocity components at once."""
        self.velocity_x = x
        self.velocity_y = y
        self.velocity_z = z

    def distance_squared_to(self, other: "EntityPosition") -> float:
        """Squared distance between two positions (feet to feet).

        The tracker compares distances in squared form - sqrt is where
        floating error and cycles go to waste.
        """
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return dx * dx + dy * dy + dz * dz

    def to_dict(self) -> dict:
        """The persisted shape. Same keys as PlayerPosition's, so existing
        player JSON files load without migration; velocity is ephemeral and
        deliberately not saved (a rejoining player is standing still)."""
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "yaw": self.yaw,
            "pitch": self.pitch,
            "on_ground": self.on_ground,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EntityPosition":
        return cls(
            x=data.get("x", 0.0),
            y=data.get("y", 64.0),
            z=data.get("z", 0.0),
            yaw=data.get("yaw", 0.0),
            pitch=data.get("pitch", 0.0),
            on_ground=data.get("on_ground", True),
        )
