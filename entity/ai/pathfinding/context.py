"""
The one place a pathfinder is allowed to look at a block.

Every other file in this package - the evaluator, the finder, the navigation -
asks this object what kind of block something is, and gets a `PathType` back.
Nothing else touches `AIContext` or the dimension. That is a deliberate wall:
the alternative is a pathfinding algorithm that can also read the world, and
every question about "why did the cow walk into the lava" becomes a question
about which of eight call sites made the read.

**The rule this file exists to enforce: an unloaded block is `BLOCKED`, not
air.** `AIContext.state` returns `None` for a coordinate whose chunk is not in
memory, and `Chunk.get_block` answers air for anything outside a chunk - so a
pathfinder that treated a missing read as "no block there" would route a cow
into unloaded space, off the edge of the world, and the failure would show up
as a cow falling forever rather than as a path that stops. Every predicate
below funnels through `_state_id`, which maps `None` to a sentinel that is
solid, non-air and not pathfindable, and so reads as solid rock.

**The classification is vanilla's `computePathTypeFromState`
(`pathfinding_context.rs:84`) in the same order, and the order matters.** Every
branch returns, so a block that matches two rules is classified by whichever
comes first: air beats everything, then the special-cased damage blocks
(cactus hurts, honey sticks, powder snow is fatal), then lava and fire, then
the tags (doors, rails, leaves, fences), and only last the generic test "is
this a full block". Putting that generic test earlier is the tempting
simplification and it is wrong in a way that does not look wrong - see
`_is_pathfindable`.

**`get_land_node_type` is the second half, and it is where "walkable" comes
from.** A node's `PathType` is not only about the block it names. An air block
is `OPEN`, but whether it is *standable* depends on what is under it: air over
air over air is a fall, air over lava is `DAMAGE_FIRE`, air over a cactus is
`DAMAGE_OTHER`, and air over anything else is `WALKABLE` - the one type that
means "this node is floor". Vanilla resolves this by looking at the block
below, and if that is inconclusive, at all twenty-six neighbours
(`get_node_type_from_neighbors`) for the danger types that propagate - a node
touching lava is `DANGER_FIRE` even if it is plain air. The propagation is why
a path will walk two blocks wide around a lava pool rather than one block
beside it.
"""

from __future__ import annotations

from math import floor
from typing import Dict, Optional, Tuple

from entity.ai.pathfinding.node import PathType

__all__ = ["PathfindingContext"]


class _Loaded:
    """The `PathType` a coordinate gets before its block is known.

    Solid and full-cube on purpose: those are the two answers that make
    `_is_pathfindable` say no, so an unloaded coordinate is `BLOCKED` whichever
    of the two checks is reached first. See the module docstring - this is the
    whole point of the file.
    """

    __slots__ = ()


#: A sentinel state id that reads as a full solid cube. Only ever compared,
#: never stored in a chunk, never sent anywhere.
UNLOADED = -1


def _is_unloaded(state_id: Optional[int]) -> bool:
    return state_id is None


class PathfindingContext:
    """Block lookups for one search, with the caches that search earns.

    A context lives for exactly one `compute_path` call. That is not a
    lifetime chosen for convenience - it is what makes the caches in here
    honest. `path_types` and `_collision` are both keyed by coordinates and
    both assume the world did not change underneath them. A cow's own chunk is
    re-sent while it walks, a door another mob opens changes a node's type
    mid-search, and a search that outlived its world would be reasoning about
    terrain that no longer exists. One call, one lifetime, and the next call
    pays to rebuild the cache rather than inheriting a stale answer.

    `mob_position` is kept because the danger-propagation and the collision
    tests both need to know where the mob is, and passing it through every
    signature would be the same thing as keeping it here with extra steps.

    **The caches are the performance story and they are not optional.** A
    560-node search calls `path_type` roughly three thousand times: once per
    node for its own classification, once per node for the block above it
    (which decides whether the node may jump), once per node per neighbour
    (eight of them), and again inside `land_node_type` for the block below.
    Without the dict that is a few thousand `Chunk.get_block` walks per mob
    per tick, and a plains chunk is not big enough for that to be free.
    """

    __slots__ = ("ai_context", "mob_position", "min_y", "sea_level",
                 "_path_types", "_collision")

    def __init__(self, ai_context, mob_position: Tuple[int, int, int],
                 min_y: int = -64, sea_level: int = 63) -> None:
        #: The context block reads go through. Never `None` in practice - the
        #: caller checks `AIContext.dimension is None` first and refuses to
        #: search at all, because a search with no world is a search whose
        #: every answer would be `BLOCKED`, which is a search that returns
        #: "nowhere", which is not the same as "do not bother".
        self.ai_context = ai_context
        self.mob_position = mob_position
        self.min_y = min_y
        self.sea_level = sea_level
        self._path_types: Dict[Tuple[int, int, int], PathType] = {}
        self._collision: Dict[Tuple[float, ...], bool] = {}

    # ----- the primitive -----

    def _state_id(self, x: int, y: int, z: int) -> Optional[int]:
        return self.ai_context.state(x, y, z)

    @property
    def facts(self):
        return self.ai_context.facts

    def path_type(self, pos: Tuple[int, int, int]) -> PathType:
        """The cached `PathType` of a position, classified on first ask."""
        cached = self._path_types.get(pos)
        if cached is None:
            cached = self.compute_path_type_from_state(pos)
            self._path_types[pos] = cached
        return cached

    # ----- classification -----

    def compute_path_type_from_state(self, pos: Tuple[int, int, int]) -> PathType:
        """Vanilla's `computePathTypeFromState` (`pathfinding_context.rs:84`).

        The branch order is vanilla's and is load bearing; see the module
        docstring. Every branch returns, so the first match wins.
        """
        x, y, z = pos
        state_id = self._state_id(x, y, z)
        if _is_unloaded(state_id):
            # The whole reason this file exists. Not air, not Open.
            return PathType.BLOCKED

        facts = self.facts
        if facts.is_air(state_id):
            return PathType.OPEN

        name = facts.names[state_id]

        # Damage and terrain specials, checked before any tag because a
        # cactus is in `minecraft:fences` (it is a fence you can be hurt by)
        # and a honey block is solid - the tag tests would answer "wall" for
        # both and lose the part that matters.
        if (name in _TRAPDOOR_NAMES
                or facts.in_tag(state_id, "minecraft:trapdoors")):
            return PathType.TRAPDOOR
        if name == "minecraft:powder_snow":
            return PathType.POWDER_SNOW
        if name in _DAMAGE_OTHER_NAMES:
            return PathType.DAMAGE_OTHER
        if name == "minecraft:honey_block":
            return PathType.STICKY_HONEY
        if name == "minecraft:cocoa":
            return PathType.COCOA
        if name in _DAMAGE_CAUTIOUS_NAMES:
            return PathType.DAMAGE_CAUTIOUS

        fluid = facts.fluid_of(state_id)
        is_water = fluid == "minecraft:water"
        if fluid == "minecraft:lava":
            return PathType.LAVA

        if name in _FIRE_NAMES:
            return PathType.DAMAGE_FIRE

        if facts.in_tag(state_id, "minecraft:doors"):
            # Collision shapes, not `is_open`: a door is "open" to the pathfinder
            # when it has no box to collide with, and that is the same question
            # the physics solver asks, so the answer cannot drift from the one
            # that decides whether a mob physically fits through.
            if not facts.collision_shape_ids(state_id):
                return PathType.DOOR_OPEN
            return (PathType.DOOR_IRON_CLOSED if name == "minecraft:iron_door"
                    else PathType.DOOR_WOOD_CLOSED)

        if facts.in_tag(state_id, "minecraft:rails"):
            return PathType.RAIL
        if facts.in_tag(state_id, "minecraft:leaves"):
            return PathType.LEAVES
        if (facts.in_tag(state_id, "minecraft:fences")
                or facts.in_tag(state_id, "minecraft:walls")):
            return PathType.FENCE
        if facts.in_tag(state_id, "minecraft:fence_gates"):
            if not facts.collision_shape_ids(state_id):
                return PathType.WATER if is_water else PathType.OPEN
            return PathType.FENCE

        if not self._is_pathfindable(state_id):
            return PathType.BLOCKED

        if is_water:
            return PathType.WATER

        return PathType.OPEN

    def _is_pathfindable(self, state_id: int) -> bool:
        """Vanilla's `isPathfindable` for a land computation.

        Pumpkin's registry has a per-block override for a hundred-odd blocks
        and falls back to `!state.is_full_cube()` for everything else
        (`block/registry.rs:1508`); Crotalus has no override table, so the
        fallback *is* the rule, and matching it means matching Pumpkin for
        every block whose override Crotalus does not have.

        The reading that matters is the one this rule rejects: "no collision,
        or the top is at or below one block". That would make dirt walkable,
        which is exactly backwards - dirt is the thing you stand *on*. The
        test is "is this a full cell", and everything less than a full cell
        (a slab, a stair, a snow layer) passes.
        """
        return not self.facts.is_full_cube(state_id)

    def land_node_type(self, pos: Tuple[int, int, int]) -> PathType:
        """The type a node at `pos` has *as floor* (`get_land_node_type`).

        The raw classification of the block itself, plus a question about the
        block below: air is only standable if something is under it, and what
        that something is decides whether standing there is free, fatal, or
        merely dangerous. `WALKABLE` - "this is the ground" - is invented here
        and appears nowhere in `compute_path_type_from_state`, because it is
        not a fact about a block, it is a fact about a column.
        """
        x, y, z = pos
        raw_type = self.path_type(pos)
        if raw_type == PathType.OPEN and y > self.min_y:
            below = self.path_type((x, y - 1, z))
            if below in (PathType.OPEN, PathType.WATER, PathType.LAVA,
                         PathType.WALKABLE):
                return PathType.OPEN
            if below in _BELOW_PASSTHROUGH:
                # Damage you stand *in* rather than beside: lava under you is
                # the same -1 as lava beside you.
                return below
            if below == PathType.POWDER_SNOW:
                return PathType.DANGER_POWDER_SNOW
            if below == PathType.TRAPDOOR:
                return PathType.DANGER_TRAPDOOR
            if below == PathType.DAMAGE_CAUTIOUS:
                return PathType.DAMAGE_CAUTIOUS
            # Nothing below that decides it, so ask the neighbours whether
            # this column is next to something that hurts.
            return self.node_type_from_neighbors(pos, PathType.WALKABLE)
        return raw_type

    def node_type_from_neighbors(self, pos: Tuple[int, int, int],
                                 fallback: PathType) -> PathType:
        """Danger propagation across the 26 neighbours (`get_node_type_from_neighbors`).

        A block of air next to lava is not air, it is `DANGER_FIRE`, and that
        is what keeps a path from threading the single block between a mob and
        a lava pool. The scan is three deep because the type that matters may
        be diagonal or one above: a cow standing beside a one-block step of
        lava is standing next to fire in the literal 26-neighbourhood sense.

        Falls back to `fallback` - in practice always `WALKABLE` - when the
        neighbourhood is clean.
        """
        x, y, z = pos
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dz == 0:
                        continue
                    neighbor = self.path_type((x + dx, y + dy, z + dz))
                    if neighbor == PathType.DAMAGE_OTHER:
                        return PathType.DANGER_OTHER
                    if neighbor in (PathType.DAMAGE_FIRE, PathType.LAVA):
                        return PathType.DANGER_FIRE
                    if neighbor == PathType.WATER:
                        return PathType.WATER_BORDER
                    if neighbor == PathType.DAMAGE_CAUTIOUS:
                        return PathType.DAMAGE_CAUTIOUS
        return fallback

    # ----- geometry -----

    def floor_level(self, pos: Tuple[int, int, int]) -> float:
        """The Y an entity resting in this column would have its feet at.

        Vanilla's `getFloorLevel` (`pathfinding_context.rs:245`): look at the
        block *below* the node and take the top of the highest collision box.
        That is why a slab is a half-height step rather than a wall - the
        evaluator asks "how much higher than my current floor is this
        node's floor" and this is the number that answers it.

        An unloaded block, or air, has no boxes and so contributes nothing -
        which means the answer is one below the node, i.e. "there is no floor
        here, keep falling". That is the same answer air gives, which is
        correct: from the pathfinder's point of view an unloaded column and an
        empty column are equally not-standable, and the difference between them
        is handled by the node being `BLOCKED` before this is ever asked.
        """
        below = (pos[0], pos[1] - 1, pos[2])
        state_id = self._state_id(*below)
        max_y = 0.0
        if not _is_unloaded(state_id):
            for shape in self.facts.collision_shapes(state_id):
                if shape[4] > max_y:
                    max_y = shape[4]
        return below[1] + max_y

    def has_collision(self, box: Tuple[float, float, float,
                                       float, float, float]) -> bool:
        """Whether an axis-aligned box overlaps any solid block's collision box.

        Vanilla's `hasCollision` (`pathfinding_context.rs:255`), including its
        shape-exact test rather than a per-block yes/no. The two callers are
        both questions about a mob *fitting*, not about a block *existing*:
        can a 0.6-wide cow squeeze through that gap in the fence line, and can
        its head clear the ceiling when it jumps onto that step. A per-block
        answer cannot answer either, which is why this takes a box.

        Non-solid blocks are skipped before their shapes are consulted -
        vanilla's `is_air() || !is_solid()`. Grass, flowers and cobwebs have
        no collision and are not obstacles, and short-circuiting them here is
        what keeps this from walking every box of every flower in the field.

        Cached on the box coordinates. The boxes come from two callers that
        each produce a small number of distinct values per search (the mob's
        own box translated to each candidate corner, and the grow-box of each
        step-up), so the cache converts a repeated neighbour question into a
        dict hit.
        """
        key = box
        cached = self._collision.get(key)
        if cached is not None:
            return cached

        min_x = int(floor(box[0]))
        min_y = int(floor(box[1]))
        min_z = int(floor(box[2]))
        max_x = int(floor(box[3]))
        max_y = int(floor(box[4]))
        max_z = int(floor(box[5]))

        facts = self.facts
        collided = False
        for bx in range(min_x, max_x + 1):
            for by in range(min_y, max_y + 1):
                for bz in range(min_z, max_z + 1):
                    state_id = self._state_id(bx, by, bz)
                    if _is_unloaded(state_id):
                        # Unknown terrain is treated as solid. A box that hits
                        # an unloaded block is not proven clear, and a cow
                        # path that assumed the void was empty space is the
                        # walk-into-the-void bug again, one level down.
                        collided = True
                        break
                    if facts.is_air(state_id) or not facts.is_solid(state_id):
                        continue
                    for shape in facts.collision_shapes(state_id):
                        if (shape[0] + bx < box[3] and shape[3] + bx > box[0]
                                and shape[1] + by < box[4]
                                and shape[4] + by > box[1]
                                and shape[2] + bz < box[5]
                                and shape[5] + bz > box[2]):
                            collided = True
                            break
                    if collided:
                        break
                if collided:
                    break
            if collided:
                break

        self._collision[key] = collided
        return collided

    def is_water(self, pos: Tuple[int, int, int]) -> bool:
        state_id = self._state_id(*pos)
        return (not _is_unloaded(state_id)
                and self.facts.fluid_of(state_id) == "minecraft:water")

    def is_pathfindable(self, pos: Tuple[int, int, int]) -> bool:
        """Whether a mob may *occupy* this block, as opposed to walk beside it.

        False for a full cube (dirt, stone, a chest) and for anything not
        loaded. The evaluator's airborne start search uses it to find the
        first thing below a falling mob worth standing on top of.
        """
        state_id = self._state_id(*pos)
        return (not _is_unloaded(state_id)) and self._is_pathfindable(state_id)

    def is_air(self, pos: Tuple[int, int, int]) -> bool:
        state_id = self._state_id(*pos)
        return _is_unloaded(state_id) or self.facts.is_air(state_id)

    def clear_caches(self) -> None:
        self._path_types.clear()
        self._collision.clear()


#: Blocks classified as `TRAPDOOR` (`pathfinding_context.rs:94`). The lily
#: pad and the big dripleaf are in this list upstream because both lie flat
#: and a mob that reaches them has effectively stepped onto a trapdoor - they
#: are one block of walkable surface with a name that does not say so.
_TRAPDOOR_NAMES = frozenset((
    "minecraft:lily_pad",
    "minecraft:big_dripleaf",
))

#: Hurt to touch (`pathfinding_context.rs:105`).
_DAMAGE_OTHER_NAMES = frozenset((
    "minecraft:cactus",
    "minecraft:sweet_berry_bush",
))

#: Hurt only if you are unlucky enough to stand in it (`pathfinding_context.rs:117`).
_DAMAGE_CAUTIOUS_NAMES = frozenset((
    "minecraft:wither_rose",
    "minecraft:pointed_dripstone",
))

#: Fire damage, or a block standing in for it (`pathfinding_context.rs:126`).
_FIRE_NAMES = frozenset((
    "minecraft:fire",
    "minecraft:soul_fire",
    "minecraft:magma_block",
    "minecraft:campfire",
    "minecraft:soul_campfire",
    "minecraft:lava_cauldron",
))

#: Types that pass straight through `land_node_type` unchanged when they are
#: the block below an open node - i.e. standing above them is the same bad
#: idea as being them.
_BELOW_PASSTHROUGH = frozenset((
    PathType.DAMAGE_FIRE,
    PathType.DAMAGE_OTHER,
    PathType.STICKY_HONEY,
))