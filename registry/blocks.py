"""
Block registry - `minecraft:block`, all 1,166 blocks and all 29,671 states.

Built from `assets/blocks.json` (block list, per-state detail) and
`assets/properties.json` (the 123 block properties, referenced by hash). The
file is Pumpkin's report generated at its 1.21.11 commit, so the state ids in
it are the ones protocol 774 clients expect.

What the loader guarantees, and what `tools/check_registry.py` re-checks:

* `states` are globally contiguous, `0..29670`, and each block owns the
  ascending run `first_state_id .. first_state_id + state_count - 1`. That is
  what makes `BlockType.state_ids` a range and lets a state id be mapped back
  to its block with one list index.
* within a block, state order is vanilla's mixed-radix order over its
  properties - the first property is the most significant digit - so a state id
  decodes back into property values. Checked against known vanilla states in
  `tools/check_registry.py`; note that a block's *default* state is data, not a
  rule (`oak_leaves` defaults to `distance=7`, the last value of its range), so
  it is read from `default_state_id` and never recomputed.

Plugin surface
--------------
`register_block` / `modify_block` are the two ways a plugin touches blocks:

* `modify_block` changes server-side numbers of a vanilla block and is always
  safe - it moves no ids.
* `register_block` appends a new block with fresh state ids. That is only legal
  before the registry freezes, and the entry is *server-side*: `minecraft:block`
  is not a synchronized registry in 1.21.11, so a vanilla client cannot learn a
  block it does not already have. A new block therefore declares
  `client_block=`, the vanilla block whose state is sent in its place, or stays
  invisible to clients (useful for markers/logic blocks).

Light tables
------------
`light_opacity_table()` and `light_emitters()` are the two flat state-id tables
`world/light_engine.py` indexes. Both are lazy and cached per registry instance,
and both have to be built *after* `GameRegistry.freeze()` -- their docstrings say
why, and why the `opacity` column is the only trustworthy source of occlusion.
"""

from __future__ import annotations

import threading
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from loguru import logger

from registry.core import (
    IndexedRegistry,
    Registry,
    RegistryError,
    RegistryFrozenError,
    ValidationError,
    canonical_id,
)
from registry.entries import BlockProperty, BlockState, BlockType

if TYPE_CHECKING:  # pragma: no cover - the tables import numpy where they use it
    import numpy as np

# Air is state 0 in every version, and empty sections encode it as a bare 0.
AIR_STATE_ID = 0


def _intern(table: Dict[Any, Any], value: Any) -> Any:
    """Share equal immutable values between states.

    A block's states differ in one property while repeating the same shape
    tuples, piston behaviour and instrument strings; without this, 29,671
    states would hold ~90,000 redundant small objects.
    """
    return table.setdefault(value, value)


class BlockRegistry:
    """The block registry plus the state-id tables built from it."""

    def __init__(self) -> None:
        self.blocks: IndexedRegistry[BlockType] = IndexedRegistry(
            "minecraft:block",
            title="Block",
            description="Blocks and their states (global state ids for the wire)",
        )
        self.properties: Registry[BlockProperty] = Registry(
            "minecraft:block_property",
            title="Block property",
            description="Block state properties referenced by blocks.json",
        )

        self._by_hash: Dict[int, BlockProperty] = {}
        self._by_serialized: Dict[str, List[BlockProperty]] = {}
        self._state_owner: List[BlockType] = []
        self._state_index: List[int] = []
        # Block id -> the vanilla state a client should draw for it. Only set
        # for plugin blocks; see `client_state`.
        self._client_stand_in: Dict[str, BlockState] = {}
        self._vanilla_block_count = 0
        self._lock = threading.RLock()
        self._frozen = False
        self._frozen_reason = ""
        self._next_state_id = 0

    # ----- identity -----

    @property
    def total_states(self) -> int:
        """Number of global block states in the registry."""
        return len(self._state_owner)

    @property
    def frozen(self) -> bool:
        return self._frozen

    def __len__(self) -> int:
        return len(self.blocks)

    def __contains__(self, name: str) -> bool:
        return name in self.blocks

    def __iter__(self):
        return iter(self.blocks)

    def __repr__(self) -> str:
        return (f"<BlockRegistry blocks={len(self.blocks)} "
                f"states={self.total_states} frozen={self._frozen}>")

    # ----- loading -----

    def load(
        self,
        blocks_data: Mapping[str, Any],
        properties_data: Sequence[Mapping[str, Any]],
    ) -> None:
        """Build the whole registry from the two asset files."""
        self._load_properties(properties_data)
        records = blocks_data["blocks"] if isinstance(blocks_data, dict) else blocks_data
        self._load_blocks(records)

        logger.debug(
            f"Block registry: {len(self.blocks)} blocks, {self.total_states} states, "
            f"{len(self.properties)} properties"
        )

    def _load_properties(self, properties_data: Sequence[Mapping[str, Any]]) -> None:
        for record in properties_data:
            prop = BlockProperty.from_json(record)
            # Keyed by enum name: `properties.json` reuses serialised names
            # ("axis" is both horizontal_axis and axis, "age" has eight
            # variants), so the serialised name alone cannot be a key.
            self.properties.add(prop.enum_name, prop)
            self._by_hash[prop.hash_key] = prop
            self._by_serialized.setdefault(prop.name, []).append(prop)

    def _load_blocks(self, records: Sequence[Mapping[str, Any]]) -> None:
        shape_pool: Dict[Any, Any] = {}
        string_pool: Dict[str, str] = {}
        expected_state_id = 0

        for record in records:
            name = canonical_id(record["name"])
            raw_states = sorted(record["states"], key=lambda state: state["id"])

            state_ids = [state["id"] for state in raw_states]
            if state_ids[0] != expected_state_id:
                raise ValidationError(
                    f"block {name}: first state id {state_ids[0]} does not follow "
                    f"the previous block's last id (expected {expected_state_id}); "
                    f"the block state table is not contiguous"
                )
            if state_ids != list(range(state_ids[0], state_ids[0] + len(state_ids))):
                raise ValidationError(
                    f"block {name}: state ids are not a contiguous ascending run"
                )

            properties = self._resolve_properties(name, record.get("properties", ()))
            expected_states = 1
            for prop in properties:
                expected_states *= len(prop)
            if expected_states != len(raw_states):
                raise ValidationError(
                    f"block {name}: {len(raw_states)} states but its properties "
                    f"{[p.name for p in properties]} describe {expected_states}"
                )

            block = BlockType(
                name=name,
                numeric_id=int(record["id"]),
                first_state_id=state_ids[0],
                state_count=len(raw_states),
                default_state_id=int(record["default_state_id"]),
                properties=tuple(properties),
                item_id=record.get("item_id"),
                translation_key=record.get("translation_key", ""),
                hardness=float(record.get("hardness", 0.0)),
                blast_resistance=float(record.get("blast_resistance", 0.0)),
                slipperiness=float(record.get("slipperiness", 0.6)),
                velocity_multiplier=float(record.get("velocity_multiplier", 1.0)),
                jump_velocity_multiplier=float(record.get("jump_velocity_multiplier", 1.0)),
                experience=record.get("experience"),
                flammable=record.get("flammable"),
                loot_table=record.get("loot_table"),
                luminance=tuple(int(s["luminance"]) for s in raw_states),
                opacity=tuple(int(s["opacity"]) for s in raw_states),
                state_flags=tuple(int(s["state_flags"]) for s in raw_states),
                side_flags=tuple(int(s["side_flags"]) for s in raw_states),
                piston_behavior=tuple(
                    _intern(string_pool, s["piston_behavior"]) for s in raw_states
                ),
                instrument=tuple(_intern(string_pool, s["instrument"]) for s in raw_states),
                state_hardness=tuple(float(s["hardness"]) for s in raw_states),
                collision_shapes=tuple(
                    _intern(shape_pool, tuple(s["collision_shapes"])) for s in raw_states
                ),
                outline_shapes=tuple(
                    _intern(shape_pool, tuple(s["outline_shapes"])) for s in raw_states
                ),
                block_entity_types=tuple(
                    _intern(string_pool, s["block_entity_type"]) if s.get("block_entity_type")
                    else None
                    for s in raw_states
                ),
            )
            block._strides = self._strides(block.properties)

            if not (block.first_state_id <= block.default_state_id <= block.last_state_id):
                raise ValidationError(
                    f"block {name}: default state {block.default_state_id} is outside "
                    f"{block.first_state_id}..{block.last_state_id}"
                )

            self._register(block)
            expected_state_id = block.last_state_id + 1

        self._next_state_id = expected_state_id
        # Everything loaded so far came from the vanilla report; anything added
        # later is a plugin block with no client-side counterpart.
        self._vanilla_block_count = len(self.blocks)

    def _resolve_properties(
        self, block_name: str, hashes: Iterable[int]
    ) -> List[BlockProperty]:
        resolved: List[BlockProperty] = []
        for hash_key in hashes:
            prop = self._by_hash.get(hash_key)
            if prop is None:
                raise ValidationError(
                    f"block {block_name}: property hash {hash_key} is not in "
                    f"properties.json"
                )
            resolved.append(prop)
        return resolved

    @staticmethod
    def _strides(properties: Sequence[BlockProperty]) -> Tuple[int, ...]:
        """Mixed-radix strides: the last property varies fastest."""
        strides = [1] * len(properties)
        stride = 1
        for position in range(len(properties) - 1, -1, -1):
            strides[position] = stride
            stride *= len(properties[position])
        return tuple(strides)

    def _register(self, block: BlockType) -> BlockType:
        """Add a block and extend the state tables, checking ids line up."""
        with self._lock:
            if block.numeric_id != len(self.blocks):
                raise ValidationError(
                    f"block {block.name}: numeric id {block.numeric_id} does not "
                    f"match its position ({len(self.blocks)}) in the registry"
                )
            self.blocks.add(block.name, block, numeric_id=block.numeric_id)
            self._state_owner.extend([block] * block.state_count)
            self._state_index.extend(range(block.state_count))
        return block

    # ----- state lookup -----

    def block(self, name: str) -> BlockType:
        """Look a block up by name, raising when it is unknown."""
        return self.blocks.require(name)

    def property(self, name: str) -> BlockProperty:
        """Look a block property up by enum name, or by a unique serialised name.

        The serialised name is what NBT and `/setblock` syntax use, but several
        distinct properties share one (`axis` covers both the horizontal two-way
        and the three-way property), so an ambiguous serialised name raises
        instead of picking one.
        """
        found = self.properties.get(name)
        if found is not None:
            return found
        matches = self._by_serialized.get(name, [])
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise RegistryError(f"no block property named {name!r}")
        raise RegistryError(
            f"{name!r} is ambiguous: {[prop.enum_name for prop in matches]}; "
            f"use the enum name"
        )

    def property_by_hash(self, hash_key: int) -> Optional[BlockProperty]:
        """Look a property up by the hash blocks reference it with."""
        return self._by_hash.get(hash_key)

    def properties_named(self, serialized_name: str) -> List[BlockProperty]:
        """Every property sharing a serialised name."""
        return list(self._by_serialized.get(serialized_name, []))

    def get_block(self, name: str) -> Optional[BlockType]:
        return self.blocks.get(name)

    def state(self, state_id: int) -> Optional[BlockState]:
        """The block state with a global id, or None when out of range."""
        if 0 <= state_id < len(self._state_owner):
            return BlockState(state_id, self._state_owner[state_id])
        return None

    def require_state(self, state_id: int) -> BlockState:
        state = self.state(state_id)
        if state is None:
            raise ValidationError(
                f"block state id {state_id} is outside the registry "
                f"(0..{self.total_states - 1})"
            )
        return state

    def state_of(self, block, properties: Optional[Mapping[str, Any]] = None) -> BlockState:
        """The state of a block at the given property values."""
        block = self.block(block) if isinstance(block, str) else block
        if not properties:
            return block.default_state
        return block.state(block.state_id_for(properties))

    def state_id_of(self, block, properties: Optional[Mapping[str, Any]] = None) -> int:
        """Global state id of a block at the given property values."""
        return self.state_of(block, properties).id

    def states_of(self, block: str) -> Tuple[BlockState, ...]:
        """Every state of a block, ascending."""
        resolved = self.block(block)
        return tuple(BlockState(state_id, resolved) for state_id in resolved.state_ids)

    def find_states(
        self,
        block: Optional[str] = None,
        **properties: Any,
    ) -> List[BlockState]:
        """States matching a block name and/or property values.

        `blocks.find_states("minecraft:oak_log", axis="y")` gives one state;
        `blocks.find_states(axis="y")` gives every block's y-axis state. A
        property a block does not have simply excludes that block.
        """
        results: List[BlockState] = []
        candidates = [self.block(block)] if block else list(self.blocks.values())
        for candidate in candidates:
            if not properties:
                results.extend(BlockState(sid, candidate) for sid in candidate.state_ids)
                continue
            if set(properties) - set(candidate.property_names):
                continue
            try:
                state_id = candidate.state_id_for(properties)
            except (KeyError, ValueError):
                continue
            results.append(candidate.state(state_id))
        return results

    def emitting_states(self) -> List[BlockState]:
        """Every state that emits light, as `BlockState` objects.

        The readable view of the same states `light_emitters()` returns as two
        numpy arrays: use this one to *name* an emitter, and the arrays to index
        a kernel with -- this method builds a `BlockState` per emitter on every
        call, while the arrays are built once and come ordered by state id.

        Ordered by descending luminance, then state id, so the brightest
        sources come first and the result is stable between runs.
        """
        emitting = [
            BlockState(state_id, block)
            for state_id, block in enumerate(self._state_owner)
            if block.luminance[state_id - block.first_state_id] > 0
        ]
        emitting.sort(key=lambda state: (-state.luminance, state.id))
        return emitting

    def state_from_nbt(self, name: str, properties: Optional[Mapping[str, Any]] = None):
        """Resolve a `{Name, Properties}` pair (as NBT carries it)."""
        block = self.get_block(name)
        if block is None:
            return None
        if not properties:
            return block.default_state
        return block.state(block.state_id_for(properties))

    def parse_state_string(self, text: str) -> Optional[BlockState]:
        """Parse `minecraft:oak_log[axis=y]` back into a state."""
        body = text.strip()
        properties: Dict[str, str] = {}
        if "[" in body:
            body, _, rest = body.partition("[")
            rest = rest.rstrip("]")
            for pair in rest.split(","):
                if not pair.strip():
                    continue
                key, _, value = pair.partition("=")
                properties[key.strip()] = value.strip()
        block = self.get_block(body)
        if block is None:
            return None
        return block.state(block.state_id_for(properties)) if properties else block.default_state

    # ----- item links -----

    def link_items(self, items) -> None:
        """Attach block items in both directions.

        Called by the loader once both registries exist and after plugins have
        run. Vanilla blocks record their item as a numeric id; a plugin block
        records a name, and either way the fallback is the shared-name
        convention, so a plugin that registers `mypack:charger` as both a block
        and an item does not have to wire them together itself.
        """
        for block in self.blocks.values():
            item = None
            if block.item_id is not None:
                item = items.by_numeric_id(block.item_id)
            elif block.item:
                item = items.get(block.item)
            if item is None:
                item = items.get(block.name)
            if item is None:
                continue
            block.item = item.name
            item.block = block.name

    # ----- plugin surface -----

    def modify_block(self, name: str, **fields: Any) -> BlockType:
        """Change the server-side numbers of a block, in place.

        Accepts any of `hardness`, `blast_resistance`, `slipperiness`,
        `velocity_multiplier`, `jump_velocity_multiplier`, `translation_key`,
        `experience`, `flammable`, `loot_table` and `item`. State ids do not move,
        and `luminance`/`opacity` are refused here because they are per state -
        use `set_luminance` / `set_opacity`.
        """
        block = self.blocks.require(name)
        with self._lock:
            for field_name, value in fields.items():
                if field_name in ("luminance", "opacity", "state_flags", "side_flags"):
                    raise RegistryError(
                        f"{field_name} is per state; use set_{field_name}() instead"
                    )
                if not hasattr(block, field_name):
                    raise RegistryError(
                        f"block {block.name} has no field {field_name!r}"
                    )
                setattr(block, field_name, value)
        return block

    def set_luminance(self, name: str, value: int, state_ids: Optional[Iterable[int]] = None) -> None:
        """Set the luminance of all of a block's states, or of listed ones."""
        self._patch_states(name, "luminance", int(value), state_ids)

    def set_opacity(self, name: str, value: int, state_ids: Optional[Iterable[int]] = None) -> None:
        """Set the opacity of all of a block's states, or of listed ones."""
        self._patch_states(name, "opacity", int(value), state_ids)

    def _patch_states(
        self,
        name: str,
        column: str,
        value: int,
        state_ids: Optional[Iterable[int]],
    ) -> None:
        block = self.blocks.require(name)
        column_values = list(getattr(block, column))
        with self._lock:
            if state_ids is None:
                column_values = [value] * block.state_count
            else:
                for state_id in state_ids:
                    index = state_id - block.first_state_id
                    if not 0 <= index < block.state_count:
                        raise RegistryError(
                            f"state {state_id} does not belong to {block.name}"
                        )
                    column_values[index] = value
            setattr(block, column, tuple(column_values))

    def register_block(
        self,
        name: str,
        *,
        properties: Sequence[BlockProperty] = (),
        default_values: Optional[Mapping[str, Any]] = None,
        item: Optional[str] = None,
        translation_key: Optional[str] = None,
        hardness: float = 0.0,
        blast_resistance: float = 0.0,
        slipperiness: float = 0.6,
        luminance: int = 0,
        opacity: int = 15,
        client_block: Optional[str] = None,
        client_properties: Optional[Mapping[str, Any]] = None,
    ) -> BlockType:
        """Append a new block with its own global state ids.

        Args:
            name: Canonical block id, e.g. `mypack:charger`.
            properties: Properties to give the block; each must already exist in
                `blocks.properties` or be built with `BlockProperty.from_json`.
            default_values: Property values for the default state; omitted
                properties take their first enumerated value. Unlike vanilla
                data, a plugin block's default is exactly this state.
            item: Item id that places this block.
            client_block: Vanilla block whose state is sent to clients instead.
                `minecraft:block` is not a synchronized registry, so a vanilla
                client has no id for a new block; without this the block is
                server-side only and renders as air.
            client_properties: Property values used on that stand-in block.

        Returns:
            The registered `BlockType`. Its first state id is the id chunks store
            and the client is sent; there is no second, private id space.

        Raises:
            RegistryError: If the registry is frozen, the name is taken, a
                property does not exist, or the block would need more states
                than the mixed-radix decode can address.
        """
        with self._lock:
            self._check_mutable(f"register block {name}")
            if len(self.blocks) > 0xFFFF:
                raise RegistryError("refusing to register more than 65535 blocks")

            canonical = canonical_id(name)
            if canonical in self.blocks:
                raise RegistryError(f"block {canonical} is already registered")

            resolved = tuple(self._resolve_property(prop) for prop in properties)
            strides = self._strides(resolved)
            count = 1
            for prop in resolved:
                count *= len(prop)

            first = self._next_state_id
            default_values = dict(default_values or {})
            unknown = set(default_values) - {prop.name for prop in resolved}
            if unknown:
                raise RegistryError(
                    f"block {canonical} has no properties {sorted(unknown)}"
                )
            default_index = 0
            for prop, stride in zip(resolved, strides):
                if prop.name in default_values:
                    default_index += prop.index_of(default_values[prop.name]) * stride

            if client_block is not None and canonical_id(client_block) not in self.blocks:
                raise RegistryError(
                    f"block {canonical}: client block {client_block} is not registered"
                )

            block = BlockType(
                name=canonical,
                numeric_id=len(self.blocks),
                first_state_id=first,
                state_count=count,
                default_state_id=first + default_index,
                properties=resolved,
                item=canonical_id(item) if item else None,
                translation_key=translation_key or f"block.{canonical_id(name)}",
                hardness=hardness,
                blast_resistance=blast_resistance,
                slipperiness=slipperiness,
                luminance=(luminance,) * count,
                opacity=(opacity,) * count,
                state_flags=(0,) * count,
                side_flags=(0,) * count,
                piston_behavior=("NORMAL",) * count,
                instrument=("HARP",) * count,
                state_hardness=(hardness,) * count,
                collision_shapes=((0,),) * count,
                outline_shapes=((0,),) * count,
                block_entity_types=(None,) * count,
            )
            block._strides = strides

            # A new block has to say what the client should draw. The registry
            # keeps that mapping rather than the world, because it is a property
            # of the wire format, not of chunk storage.
            if client_block is not None:
                stand_in = self.state_of(client_block, client_properties)
                self._client_stand_in[canonical] = stand_in

            self._register(block)
            self._next_state_id = block.last_state_id + 1
            logger.debug(
                f"Registered block {canonical} with {count} state(s) "
                f"(ids {first}..{block.last_state_id})"
            )
            return block

    def declare_property(self, prop: BlockProperty) -> BlockProperty:
        """Register a block property so blocks (including new ones) can use it.

        Returns the registry's instance of an equal property, so two plugins
        declaring the same property share one object.
        """
        existing = self.properties.get(prop.enum_name)
        if existing is None or existing != prop:
            if prop.hash_key in self._by_hash and self._by_hash[prop.hash_key] != prop:
                raise RegistryError(
                    f"property hash {prop.hash_key} is already used by "
                    f"{self._by_hash[prop.hash_key].enum_name}"
                )
            self.properties.add(prop.enum_name, prop, replace=True)
            self._by_hash[prop.hash_key] = prop
            named = self._by_serialized.setdefault(prop.name, [])
            if prop not in named:
                named.append(prop)
            return prop
        return existing

    def _resolve_property(self, prop: BlockProperty) -> BlockProperty:
        """Internal alias for `declare_property`, used while loading blocks."""
        return self.declare_property(prop)

    def client_state(self, block) -> BlockState:
        """The state to put on the wire for a block.

        Vanilla blocks are their own client state. A plugin block that declared
        `client_block` resolves to that stand-in; one that did not resolves to
        air, which is the honest answer for a block the client cannot know.
        """
        resolved = self.block(block) if isinstance(block, str) else block
        stand_in = self._client_stand_in.get(resolved.name)
        if stand_in is not None:
            return stand_in
        if resolved.numeric_id >= self._vanilla_block_count:
            return self.require_state(AIR_STATE_ID)
        return resolved.default_state

    # ----- freezing -----

    def freeze(self, reason: str = "") -> None:
        with self._lock:
            self._frozen = True
            self._frozen_reason = reason
            self.blocks.freeze(reason)
            self.properties.freeze(reason)

    def unfreeze(self) -> None:
        with self._lock:
            self._frozen = False
            self._frozen_reason = ""
            self.blocks.unfreeze()
            self.properties.unfreeze()

    def _check_mutable(self, action: str) -> None:
        if self._frozen:
            detail = f" ({self._frozen_reason})" if self._frozen_reason else ""
            raise RegistryFrozenError(
                f"cannot {action}: block registry is frozen{detail}. "
                f"Register blocks from a plugin before the world loads."
            )

    # ----- integrity -----

    def validate(self) -> List[str]:
        """Return a list of problems; empty means the table is consistent."""
        problems: List[str] = []

        if self.total_states == 0:
            return ["block registry is empty"]

        next_state_id = 0
        for block in self.blocks.values():
            if block.first_state_id != next_state_id:
                problems.append(
                    f"{block.name}: first state {block.first_state_id} != {next_state_id}"
                )
            next_state_id = block.last_state_id + 1

            product = 1
            for prop in block.properties:
                product *= len(prop)
            if product != block.state_count:
                problems.append(
                    f"{block.name}: {block.state_count} states but its properties "
                    f"describe {product}"
                )

            if not (block.first_state_id <= block.default_state_id <= block.last_state_id):
                problems.append(
                    f"{block.name}: default state {block.default_state_id} is outside "
                    f"{block.first_state_id}..{block.last_state_id}"
                )

            for column in ("luminance", "opacity", "state_flags", "side_flags"):
                if len(getattr(block, column)) != block.state_count:
                    problems.append(f"{block.name}: {column} column is the wrong length")

        if next_state_id != self.total_states:
            problems.append(
                f"state table holds {self.total_states} entries but the blocks "
                f"describe {next_state_id}"
            )

        return problems


# ----- light tables -----
#
# The two tables below are the state-id-indexed form of the `opacity` and
# `luminance` columns the blocks already carry. They live here rather than in
# `world/` because the state id space is the registry's, and because the light
# engine is not the only possible consumer (a map renderer, a `/light` command or
# a spawn check all want the same two arrays).

#: The state id space a `ChunkSection` can store -- one uint16 per cell, so ids
#: run 0..65535 (`world/block_states.validate` checks the registry against the
#: same limit). The tables are indexed by state id and sized to it, which is why
#: an id the registry does not own reads as 0 -- no block, no opacity, no light.
_STATE_ID_SPACE = 1 << 16

#: `(registry, table)` pairs. Keyed by the registry instance so a table built for
#: one registry is never handed out for another: `set_game_registry` swaps the
#: global instance, and a stale table would then light chunks by the wrong data.
_light_opacity_cache: Optional[Tuple["BlockRegistry", "np.ndarray"]] = None
_light_emitters_cache: Optional[
    Tuple["BlockRegistry", Tuple["np.ndarray", "np.ndarray"]]
] = None
_light_table_lock = threading.Lock()


def _light_registry(registry: Optional[BlockRegistry]) -> BlockRegistry:
    """The registry a light table is built from: the argument, or the global one.

    The import is function-local because `registry.game_registry` imports this
    module -- the same shape as `world.block_states._blocks`.
    """
    if registry is not None:
        return registry
    from registry import get_game_registry

    return get_game_registry().blocks


def _light_column(block: BlockType, column: str) -> Tuple[int, Tuple[int, ...]]:
    """`(first_state_id, values)` of a per-state light column, validated.

    `luminance` and `opacity` are indexed by `state_id - first_state_id`, so a
    column covers exactly the states of its block. A column of the wrong length
    or a block whose ids leave the uint16 space would shift every later state in
    the table instead of failing, so both are refused here.
    """
    first = block.first_state_id
    values = getattr(block, column)
    if len(values) != block.state_count:
        raise RegistryError(
            f"block {block.name}: {column} has {len(values)} entries, "
            f"expected {block.state_count}"
        )
    if first < 0 or first + len(values) > _STATE_ID_SPACE:
        raise RegistryError(
            f"block {block.name}: state ids {first}..{first + len(values) - 1} "
            f"fall outside the {_STATE_ID_SPACE}-entry light table"
        )
    if values and (min(values) < 0 or max(values) > 255):
        raise RegistryError(
            f"block {block.name}: {column} values {sorted(set(values))} do not "
            f"fit a uint8 table"
        )
    return first, values


def _build_light_opacity_table(registry: BlockRegistry) -> "np.ndarray":
    """`uint8[65536]`: global state id -> that state's `opacity`."""
    import numpy as np

    table = np.zeros(_STATE_ID_SPACE, dtype=np.uint8)
    for block in registry.blocks.values():
        first, values = _light_column(block, "opacity")
        table[first:first + len(values)] = values
    return table


def _build_light_emitters(registry: BlockRegistry) -> Tuple["np.ndarray", "np.ndarray"]:
    """`(state ids, luminance)` for every state whose luminance is above 0."""
    import numpy as np

    entries: List[Tuple[int, int]] = []
    for block in registry.blocks.values():
        first, values = _light_column(block, "luminance")
        entries.extend(
            (first + index, level)
            for index, level in enumerate(values)
            if level > 0
        )
    # State ids ascending. That is the order `np.searchsorted(ids, blocks)` needs
    # to be a valid per-cell screen, and the one `np.isin` does not care about;
    # `emitting_states()` keeps the brightest-first order for reading.
    entries.sort()
    ids = np.array([entry[0] for entry in entries], dtype=np.int32)
    levels = np.array([entry[1] for entry in entries], dtype=np.uint8)
    return ids, levels


def light_opacity_table(registry: Optional[BlockRegistry] = None) -> "np.ndarray":
    """`uint8[65536]`: global state id -> light opacity, built once per registry.

    This is what lets a kernel ask "how much does this cell damp light?" with one
    index instead of a `BlockState` per cell, for the same reason
    `BlockFacts.heightmap_mask` exists (`world/vanilla/features.py`): a predicate
    that is a function of the state id alone belongs in a table. Index it with the
    state ids a section stores -- `table[section.blocks]` is one section's
    opacities in a single numpy operation. Ids past the end of the registry read
    0, which is also what air reads as.

    **Build it after `GameRegistry.freeze()`.** `set_luminance` / `set_opacity`
    (and `modify_block`'s refusal to take those two) exist because they are
    per-state, and a plugin that runs before the freeze can still move them; the
    cache is only dropped when a *different registry instance* is passed, never
    when a value changes under it. The registry's contract is that nothing
    value-like moves after the freeze, so the first call after it is the safe one
    -- and on the server that call is the light engine's startup, not a per-tick
    path. Passing `registry` explicitly is for tools and tests that hold their own
    instance.

    **Trust `opacity`, not `state_flags`.** Measured over the 29,671 states of
    `assets/blocks.json`, this column holds exactly vanilla's `getLightBlock()`
    and only three values: 0 (16,240 states), 1 (10,872 -- water, leaves, ice,
    snow layers, cobweb, carpets) and 15 (2,559 -- the full cubes). The flags
    column cannot answer the question at all: no state in this file sets bit 10 or
    bit 11 (Pumpkin's `IS_SOLID_RENDER` / `CAN_OCCLUDE`) -- every one of the 2,559
    states with `opacity == 15` included -- and the highest bit used anywhere is
    bit 9. A "correction" that keyed occlusion off those flags would make all
    2,559 light-blocking states transparent, which is why
    `tools/check_light_layout.py` pins both the distribution and the unset bits.
    """
    global _light_opacity_cache

    source = _light_registry(registry)
    cached = _light_opacity_cache
    if cached is not None and cached[0] is source:
        return cached[1]
    table = _build_light_opacity_table(source)
    # Two threads racing here build the same table twice; both are correct and the
    # last store wins. The build stays outside the lock on purpose: resolving the
    # registry can run the lazy registry load, and re-entering that under this
    # lock would deadlock a plugin that asked for a table while being loaded.
    with _light_table_lock:
        _light_opacity_cache = (source, table)
    return table


def light_emitters(
    registry: Optional[BlockRegistry] = None,
) -> Tuple["np.ndarray", "np.ndarray"]:
    """`(state ids int32[E], luminance uint8[E])` for the E states that emit light.

    Only states with `luminance > 0` are in the arrays -- 1,626 of the 29,671
    states, 611 of them at level 15. The point is the pre-filter: a section's
    4,096 cells can be screened against 1,626 candidates instead of all 29,671,
    and the hits arrive with their levels already attached (`np.isin`, or
    `levels[np.searchsorted(ids, blocks)]`; inside a kernel a `uint8[65536]` level
    table built with `level_table[ids] = levels` is the cheaper screen, and it
    needs exactly this pairing to build).

    **The ids are ascending**, each paired with its level, so a `searchsorted`
    lookup is valid -- do not assume the arrays are a level map indexed by state
    id. `emitting_states()` is the same 1,626 states in its own order (brightest
    first) and is meant for naming an emitter, not for indexing a kernel.

    Same rules as `light_opacity_table`: lazy, cached per registry instance, and
    built after `GameRegistry.freeze()` -- `set_luminance` moves these numbers in
    place, so a table built before the plugins ran can be stale, and there is no
    per-value invalidation hook.
    """
    global _light_emitters_cache

    source = _light_registry(registry)
    cached = _light_emitters_cache
    if cached is not None and cached[0] is source:
        return cached[1]
    emitters = _build_light_emitters(source)
    with _light_table_lock:
        _light_emitters_cache = (source, emitters)
    return emitters


def _default_index(block: BlockType) -> int:
    """Retained for callers that want the enumeration-start state index.

    Not the block's default state: vanilla's default is chosen per block and
    recorded in `default_state_id`.
    """
    return block.index_of([None] * len(block.properties))


__all__ = [
    "BlockRegistry",
    "AIR_STATE_ID",
    "light_opacity_table",
    "light_emitters",
]
