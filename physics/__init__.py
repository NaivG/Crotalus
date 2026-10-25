"""The physics engine: gravity, AABB collision, and server-side move validation.

    from physics import block_shapes, PhysicsSolver

`physics/` sits next to `entity/`, `world/` and `registry/` because it answers a
different question from all three: not "what is this entity", not "what is in
this chunk", but "where can this box go". The dependency direction is
`physics -> world` (it reads block state ids) and `entity -> physics` (it writes
positions back); nothing in `world/` or `entity/` imports this package at module
scope, so a process that never moves anything never needs it.

What this package owns, and who owns it
---------------------------------------

Every array and table here is owned by the **dimension actor thread** -- the one
OS thread that owns a dimension's mutable state (`world/dimension_actor.py`).
There are no locks in this package on purpose: single ownership is the
concurrency control, exactly as it is for chunks and entities. The one exception
is `physics.player`, which validates a player's path on the actor thread from a
frozen `PlayerViewMsg` and hands corrections back through `ChunkSink.send()`;
it never touches the `Player` object, which belongs to the event loop.

What this round does not do
---------------------------

Known gaps, deliberately not faked (they are also listed in `PHYSICS_PLAN.md`):
no jumping or `set_entity_motion`, no ground friction or block materials even
though the material columns exist, no liquids or swimming, no fall damage, no
mob AI or wandering, no entity-to-entity collision, no entity persistence, and
no replayed player input. An unloaded chunk **freezes** the bodies that reach
into it -- it does not read as air and it does not read as solid.

A body exists only for an entity something gave one to: `PhysicsSolver.add_entity`
is the only door, and today the callers are `NaturalSpawner._place` (a mob is
placed) and the player path. An entity admitted to a dimension by any other
route -- a command, a tool, a future spawn egg -- is in the world and simply not
physical, which is a gap rather than a rule: nothing moves it, and nothing says
so.

Lazy imports
------------

`numba` is imported by the kernel module, not by this one. `import physics`
must not pull a JIT into the boot path any more than `import world` pulls in
the legacy Perlin field (`world/__init__.py` uses the same trick for the same
reason).
"""

from __future__ import annotations

__all__ = [
    "BlockShapes",
    "block_shapes",
    "BodyStore",
    "WorldView",
    "PhysicsSolver",
    "PlayerMoveValidator",
]

#: Attribute -> the module that defines it. Resolved on first access.
_LAZY = {
    "BlockShapes": "physics.block_shapes",
    "block_shapes": "physics.block_shapes",
    "BodyStore": "physics.bodies",
    "WorldView": "physics.world_view",
    "PhysicsSolver": "physics.solver",
    "PlayerMoveValidator": "physics.player",
}


def __getattr__(name: str):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
