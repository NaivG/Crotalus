"""
The Entity base class.

Everything that occupies world space and talks the entity packets is an
`Entity`: it has a runtime id, a type, a position, and a lifecycle. `Player`
extends it - a player is not *like* an entity, it IS one; the client tracks
it with the same `add_entity` spawn and the same delta movement packets as a
zombie, and the server's bookkeeping should not pretend otherwise.

The class is deliberately small. Pumpkin's `Entity` struct carries ~110
fields because it owns collision, fluids, portals, fire and riding for every
entity type in the game; those systems do not exist here yet, and copying
their fields would be reserving parking spaces for buildings not yet
designed. What exists today is what the wire needs: identity, geometry,
velocity, and the removed flag.

Why a dataclass (not the plain-class-with-properties shape Pumpkin's
interior mutability would suggest): this object lives on ONE thread per its
owner - the event loop for players, the dimension actor for non-player
entities - so plain attribute access is the correct, race-free shape.
Crossing threads happens only through `PlayerViewMsg` snapshots, never by
sharing a mutable `Entity` (see the tracker's module docstring).

`entity_id` defaults to 0, which vanilla treats as "no entity"; the join
path assigns a real one from the global allocator the moment the object is
created, and tests pass explicit ids to stay deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

from entity.entity_type import EntityType, entity_type_by_name
from entity.position import EntityPosition


@dataclass
class Entity:
    """Base state shared by every entity, players included."""

    # The runtime id, allocated from the global counter and written verbatim
    # into every packet that names this entity. 0 means "not assigned yet" -
    # vanilla itself treats 0 as no-entity, so nothing downstream has to
    # guess whether a 0 is real.
    entity_id: int = 0

    # The persistent id. For players this is the profile UUID the client
    # logged in with; for future non-player entities it will be the UUID
    # saved into NBT. Not the wire entity id - that is `entity_id`.
    uuid: str = ""

    # What this is. Concrete entity classes set this; it decides the spawn
    # packet's type id, the collision box and the eye height.
    entity_type: Optional[EntityType] = None

    position: EntityPosition = field(default_factory=EntityPosition)

    # Which world this entity is in, as the dimension's resource name.
    # A string rather than a Dimension reference: dimensions are owned by
    # their actor threads, and holding one from the event loop side would
    # be a mutable object shared across threads by the back door.
    dimension: str = "minecraft:overworld"

    # Set once the entity has left the world. The tracker checks this
    # before doing work; the flag lives here because removal is an entity
    # fact, not an observation fact.
    removed: bool = False

    def update_position(self, x: float, y: float, z: float,
                        yaw: float = 0.0, pitch: float = 0.0,
                        on_ground: bool = True) -> None:
        """Move the entity to a known-good pose, in place.

        Assigns through the existing EntityPosition instead of replacing it.
        The chunk streamer, the view tracker and anything that later grows
        out of them hold a reference to that object; swapping it out from
        under them is how a position change ends up half-applied.
        """
        pos = self.position
        pos.x = x
        pos.y = y
        pos.z = z
        pos.yaw = yaw
        pos.pitch = pitch
        pos.on_ground = on_ground

    # ----- geometry -----

    @property
    def eye_height(self) -> float:
        """How far above the feet this entity's camera sits.

        A property of the *type*, not of the instance: a cow at 1.3 and a
        chicken at 0.644 are the difference between a mob you can see over and
        one you cannot. Defaults to vanilla's player value for an untyped
        entity, which is what `eye_pos` has always used.
        """
        return self.entity_type.eye_height if self.entity_type else 1.62

    @property
    def chunk(self) -> Tuple[int, int]:
        """The chunk column this entity is in, as (x, z)."""
        return self.position.chunk

    @property
    def block_pos(self) -> Tuple[int, int, int]:
        """The block the entity's feet occupy."""
        return self.position.block_pos

    @property
    def section(self) -> Tuple[int, int, int]:
        """The 16-block section the entity's feet occupy."""
        return self.position.section

    @property
    def eye_pos(self) -> Tuple[float, float, float]:
        """Where this entity's camera / gaze sits.

        Lives on Entity rather than EntityPosition because it needs
        `eye_height`, which is a property of the *type*: a player at the
        same feet position as a boat looks out from a different height.
        """
        pos = self.position
        return (pos.x, pos.y + self.eye_height, pos.z)

    @property
    def velocity(self) -> Tuple[float, float, float]:
        return self.position.velocity

    def set_velocity(self, x: float, y: float, z: float) -> None:
        self.position.set_velocity(x, y, z)

    def tick(self) -> None:
        """One game tick of entity behavior. A no-op by design this round.

        `Dimension.tick` calls this for every non-player entity on its
        actor thread. When physics arrives (gravity, collision, fire), it
        lands HERE for the base behavior and in overrides for the special
        types - Pumpkin's `EntityBase.tick` defaults do the same layering.
        Nothing reads a return value; state changes go through the fields.
        """

    def __str__(self) -> str:
        type_name = self.entity_type.name if self.entity_type else "untyped"
        return f"Entity({type_name}, id={self.entity_id})"

    def __repr__(self) -> str:
        return self.__str__()


def player_entity_type() -> EntityType:
    """The `minecraft:player` type, looked up once the registry is loaded.

    A tiny helper rather than a module-level constant: the type table only
    exists after `init_registries()`, and a module constant would either
    capture it too early or need import-order cooperation from every test.
    """
    entity_type = entity_type_by_name("minecraft:player")
    if entity_type is None:
        raise RuntimeError(
            "minecraft:player is not in the entity type table; "
            "was init_registries() called before the first Player was built?"
        )
    return entity_type
