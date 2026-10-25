"""
Registry entry types - what the loaders build and plugins hand back.

The layouts here are dictated by the shape of Pumpkin's generated assets at the
1.21.11 commit (`42c903493`), which is what `assets/` holds. Three of them are
worth spelling out, because getting them wrong is silent:

`BlockProperty` value order
    Pumpkin records a property's possible values in the order vanilla assigns
    them, and does *not* record a state's property values - only its global id.
    The mapping back from a state id to its values is therefore a mixed-radix
    decode, and it only works if the value order matches vanilla's:

      * `boolean` -> `("true", "false")` (vanilla's `BooleanProperty` lists
        true first, which is why `grass_block`'s default - snowy=false - is the
        *second* of its two states)
      * `int` -> ascending `min..max`
      * `enum` -> the order in `properties.json`

    The first property is the most significant digit and the last one varies
    fastest. Verified against the recorded `default_state_id` of every block:
    e.g. `oak_leaves` (distance 1-7, persistent, waterlogged) defaults to
    distance=7, persistent=false, waterlogged=false, which is index
    (6*2 + 1)*2 + 1 = 27 -> state 252 + 27 = 279, exactly its default.

`BlockType` per-state columns
    A block has one to 1150 states and 29,671 states exist in total, so the
    per-state detail is stored as parallel tuples indexed by
    `state_id - first_state_id` rather than as 29,671 objects. `BlockState` is
    the (cheap, on-demand) view that pairs an id with its block.

`Recipe` identity
    `recipes.json` is a bare list: Pumpkin's recipes have no names. Stable keys
    are synthesised (`minecraft:oak_planks`, `minecraft:baked_potato_from_smelting`)
    so recipes can be referenced, replaced and logged; see
    `registry.recipes.synthesise_key`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from registry.core import DuplicateEntryError, canonical_id, path_of


# ==================== Block properties ====================

BOOLEAN_VALUES: Tuple[str, ...] = ("true", "false")


def fnv1a_i32(text: str) -> int:
    """32-bit FNV-1a of a string, as a signed int.

    Only used to give a plugin-declared property a stable unique key; nothing
    in the shipped data depends on it.
    """
    value = 0x811C9DC5
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 0x01000193) & 0xFFFFFFFF
    return value - 0x100000000 if value >= 0x80000000 else value


@dataclass(frozen=True, slots=True)
class BlockProperty:
    """One block state property (`axis`, `facing`, `waterlogged`, ...)."""

    name: str
    """Serialised name, as it appears in `Name[prop=value]` and in NBT."""
    enum_name: str
    """Pumpkin's internal enum name; differs for e.g. `distance`/`distance_1_7`.

    Unlike the serialised name this is unique across `properties.json` - the
    same serialised name is reused by several distinct properties (`axis` is
    both `horizontal_axis` and `axis`), so registries key properties by this.
    """
    kind: str
    """One of `boolean`, `int`, `enum`."""
    values: Tuple[str, ...]
    """Legal values, in the order vanilla enumerates them."""
    hash_key: int
    """Pumpkin's property hash; blocks reference properties by this."""

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "BlockProperty":
        """Build from one `properties.json` record."""
        kind = data["type"]
        if kind == "boolean":
            values = BOOLEAN_VALUES
        elif kind == "int":
            values = tuple(str(v) for v in range(data["min"], data["max"] + 1))
        elif kind == "enum":
            values = tuple(data["values"])
        else:  # pragma: no cover - the asset only has the three kinds above
            raise ValueError(f"unknown property type {kind!r} for {data['name']}")

        return cls(
            name=data["serialized_name"],
            enum_name=data["enum_name"],
            kind=kind,
            values=values,
            hash_key=data["hash_key"],
        )

    @classmethod
    def of(
        cls,
        name: str,
        kind: str,
        values: Optional[Sequence[Any]] = None,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
        enum_name: Optional[str] = None,
        hash_key: Optional[int] = None,
    ) -> "BlockProperty":
        """Build a property from scratch, for plugin-registered blocks.

        `hash_key` only has to be unique inside the registry, so a plugin
        property gets a stable hash of its enum name unless one is given;
        `properties.json`'s real hashes are Pumpkin's and unrelated.
        """
        if kind == "boolean":
            resolved = BOOLEAN_VALUES
        elif kind == "int":
            if min_value is None or max_value is None:
                raise ValueError(f"int property {name!r} needs min_value and max_value")
            resolved = tuple(str(v) for v in range(min_value, max_value + 1))
        elif kind == "enum":
            if not values:
                raise ValueError(f"enum property {name!r} needs values")
            resolved = tuple(str(v) for v in values)
        else:
            raise ValueError(f"unknown property type {kind!r}")

        resolved_enum_name = enum_name or name
        return cls(
            name=name,
            enum_name=resolved_enum_name,
            kind=kind,
            values=resolved,
            hash_key=hash_key if hash_key is not None else fnv1a_i32(resolved_enum_name),
        )

    def __len__(self) -> int:
        return len(self.values)

    @property
    def first_value(self) -> str:
        """First enumerated value - the enumeration start, *not* the default.

        Vanilla's `BooleanProperty` lists `true` first while most blocks default
        to `false`, and `distance` (1..7) defaults to its *last* value. A block's
        default state is data (`BlockType.default_state_id`), never a rule, so
        nothing here pretends otherwise.
        """
        return self.values[0]

    def index_of(self, value: Any) -> int:
        """Index of a value, accepting bools/ints as well as strings."""
        if isinstance(value, bool):
            value = "true" if value else "false"
        else:
            value = str(value)
        try:
            return self.values.index(value)
        except ValueError:
            raise ValueError(
                f"{value!r} is not a legal value for property {self.name} "
                f"(expected one of {', '.join(self.values)})"
            ) from None

    def normalise(self, value: Any) -> str:
        """Canonical spelling of a value."""
        return self.values[self.index_of(value)]

    def __str__(self) -> str:
        return f"{self.name}[{'|'.join(self.values)}]"


# ==================== Block states ====================

@dataclass(frozen=True, slots=True)
class BlockState:
    """A block plus one specific combination of property values.

    `id` is the *global* block state id - the number the client expects in a
    chunk section palette, the number a saved section names back to, and the
    number `properties.json`-derived tables are indexed by.
    """

    id: int
    block: "BlockType"

    @property
    def index(self) -> int:
        """Position of this state within its block's state list."""
        return self.id - self.block.first_state_id

    @property
    def properties(self) -> Dict[str, str]:
        """Property name -> value for this state."""
        return self.block.properties_of(self.index)

    @property
    def is_default(self) -> bool:
        return self.id == self.block.default_state_id

    @property
    def luminance(self) -> int:
        return self.block.luminance[self.index]

    @property
    def opacity(self) -> int:
        return self.block.opacity[self.index]

    @property
    def is_air(self) -> bool:
        return self.id == 0

    def to_nbt(self) -> Dict[str, Any]:
        """The `{Name, Properties}` compound the client uses for a block state."""
        return self.block.state_nbt(self.id)

    def __str__(self) -> str:
        properties = self.properties
        if not properties:
            return self.block.name
        body = ",".join(f"{key}={value}" for key, value in properties.items())
        return f"{self.block.name}[{body}]"


@dataclass(slots=True)
class BlockType:
    """One entry of the `minecraft:block` registry, with all of its states."""

    name: str
    numeric_id: int
    first_state_id: int
    state_count: int
    default_state_id: int
    properties: Tuple[BlockProperty, ...] = ()
    item_id: Optional[int] = None
    item: Optional[str] = None
    translation_key: str = ""
    hardness: float = 0.0
    blast_resistance: float = 0.0
    slipperiness: float = 0.6
    velocity_multiplier: float = 1.0
    jump_velocity_multiplier: float = 1.0
    experience: Optional[float] = None
    flammable: Optional[Dict[str, Any]] = None
    loot_table: Optional[Dict[str, Any]] = None

    # Per-state columns, indexed by (state_id - first_state_id).
    luminance: Tuple[int, ...] = ()
    opacity: Tuple[int, ...] = ()
    state_flags: Tuple[int, ...] = ()
    side_flags: Tuple[int, ...] = ()
    piston_behavior: Tuple[str, ...] = ()
    instrument: Tuple[str, ...] = ()
    state_hardness: Tuple[float, ...] = ()
    collision_shapes: Tuple[Tuple[int, ...], ...] = ()
    outline_shapes: Tuple[Tuple[int, ...], ...] = ()
    block_entity_types: Tuple[Optional[str], ...] = ()

    # Mixed-radix strides for decoding a state index into property values.
    _strides: Tuple[int, ...] = field(default=(), repr=False)

    def __len__(self) -> int:
        return self.state_count

    def __repr__(self) -> str:
        return (f"<BlockType {self.name} states={self.state_count} "
                f"default={self.default_state_id}>")

    # ----- states -----

    @property
    def state_ids(self) -> range:
        """Every global state id belonging to this block, ascending."""
        return range(self.first_state_id, self.first_state_id + self.state_count)

    @property
    def last_state_id(self) -> int:
        return self.first_state_id + self.state_count - 1

    @property
    def has_properties(self) -> bool:
        return bool(self.properties)

    @property
    def default_state(self) -> BlockState:
        return BlockState(self.default_state_id, self)

    @property
    def property_names(self) -> Tuple[str, ...]:
        return tuple(prop.name for prop in self.properties)

    def property(self, name: str) -> BlockProperty:
        """Look a property up by its serialised name."""
        for prop in self.properties:
            if prop.name == name:
                return prop
        raise KeyError(f"{self.name} has no property {name!r}")

    def values_at(self, index: int) -> Tuple[str, ...]:
        """Decode a state index into one value per property, in order."""
        if not self.properties:
            return ()
        values: List[str] = []
        remaining = index
        for prop, stride in zip(self.properties, self._strides):
            values.append(prop.values[(remaining // stride) % len(prop)])
        return tuple(values)

    def properties_of(self, index: int) -> Dict[str, str]:
        """Property name -> value for a state index."""
        return dict(zip(self.property_names, self.values_at(index)))

    def index_of(self, values: Sequence[Any]) -> int:
        """Mixed-radix encode property values into a state index.

        Values may be given in property order. Anything omitted - or None -
        takes that property's first enumerated value, which is what vanilla's
        `StateDefinition.any()` starts from before a block overrides it.
        """
        index = 0
        for position, (prop, stride) in enumerate(zip(self.properties, self._strides)):
            if position < len(values) and values[position] is not None:
                value_index = prop.index_of(values[position])
            else:
                value_index = 0
            index += value_index * stride
        return index

    def state_id_for(self, properties: Mapping[str, Any]) -> int:
        """Global state id for a name -> value mapping (unlisted keys start at 0)."""
        unknown = set(properties) - set(self.property_names)
        if unknown:
            raise KeyError(
                f"{self.name} has no properties {sorted(unknown)}; "
                f"expected a subset of {list(self.property_names)}"
            )
        index = 0
        for prop, stride in zip(self.properties, self._strides):
            if prop.name in properties and properties[prop.name] is not None:
                index += prop.index_of(properties[prop.name]) * stride
        return self.first_state_id + index

    def state_id(self, **properties: Any) -> int:
        """Global state id for keyword property values."""
        return self.state_id_for(properties)

    def state(self, state_id: int) -> BlockState:
        """Wrap one of this block's state ids."""
        if not (self.first_state_id <= state_id <= self.last_state_id):
            raise ValueError(
                f"state {state_id} does not belong to {self.name} "
                f"({self.first_state_id}..{self.last_state_id})"
            )
        return BlockState(state_id, self)

    def state_nbt(self, state_id: int) -> Dict[str, Any]:
        """`{Name, Properties}` for a state id; omits `Properties` when empty."""
        index = state_id - self.first_state_id
        result: Dict[str, Any] = {"Name": self.name}
        if self.properties:
            result["Properties"] = self.properties_of(index)
        return result

    def matches_nbt(self, state_id: int, name: str, properties: Mapping[str, Any]) -> bool:
        """Whether a (name, properties) pair from NBT denotes this state."""
        if canonical_id(name) != self.name:
            return False
        if not properties:
            return state_id == self.first_state_id or state_id == self.default_state_id
        return self.state_id_for(properties) == state_id


# ==================== Items ====================

@dataclass(slots=True)
class ItemType:
    """One entry of the `minecraft:item` registry.

    `components` is the raw `minecraft:*` component map from `items.json`.
    It is deliberately not interpreted into typed fields: 58 distinct
    components exist and the set is version-specific, so the registry exposes a
    generic accessor and names the handful of components the server reads.
    """

    name: str
    numeric_id: int
    components: Dict[str, Any] = field(default_factory=dict)
    block: Optional[str] = None
    """The block this item places, for block items."""

    def __repr__(self) -> str:
        return f"<ItemType {self.name} id={self.numeric_id}>"

    def component(self, name: str, default: Any = None) -> Any:
        """Read a component by id, with or without the namespace."""
        key = canonical_id(name)
        if key in self.components:
            return self.components[key]
        return self.components.get(name, default)

    def has_component(self, name: str) -> bool:
        return canonical_id(name) in self.components

    @property
    def translation_key(self) -> Optional[str]:
        item_name = self.component("minecraft:item_name")
        if isinstance(item_name, dict):
            return item_name.get("translate") or item_name.get("text")
        return None

    @property
    def max_stack_size(self) -> int:
        value = self.component("minecraft:max_stack_size")
        return int(value) if isinstance(value, (int, float)) else 64

    @property
    def max_damage(self) -> Optional[int]:
        value = self.component("minecraft:max_damage")
        return int(value) if isinstance(value, (int, float)) else None

    @property
    def item_model(self) -> Optional[str]:
        value = self.component("minecraft:item_model")
        return canonical_id(value) if isinstance(value, str) else None

    @property
    def rarity(self) -> str:
        return str(self.component("minecraft:rarity", "common"))

    @property
    def is_block_item(self) -> bool:
        return self.block is not None

    @property
    def is_damageable(self) -> bool:
        return self.max_damage is not None

    @property
    def food(self) -> Optional[Dict[str, Any]]:
        value = self.component("minecraft:food")
        return value if isinstance(value, dict) else None


@dataclass(frozen=True, slots=True)
class ItemStack:
    """An item plus a count, as recipes produce and consume them."""

    item: str
    count: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "item", canonical_id(self.item))

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "ItemStack":
        return cls(item=data["id"], count=int(data.get("count", 1)))

    def to_json(self) -> Dict[str, Any]:
        return {"id": self.item, "count": self.count}

    def __str__(self) -> str:
        return f"{self.count}x {self.item}"


# ==================== Recipes ====================

@dataclass(frozen=True, slots=True)
class Ingredient:
    """What a recipe slot accepts: one item, a tag, or a list of either.

    A `#`-prefixed entry is a tag reference and is resolved lazily through the
    tag registry, because tag membership is a plugin-visible, mutable thing.
    """

    entries: Tuple[str, ...]
    tag: Optional[str] = None

    @classmethod
    def from_json(cls, data: Any) -> "Ingredient":
        """Build from the several shapes Pumpkin's recipe data uses."""
        if isinstance(data, str):
            if data.startswith("#"):
                return cls(entries=(), tag=canonical_id(data[1:]))
            return cls(entries=(canonical_id(data),))
        if isinstance(data, list):
            items: List[str] = []
            tags: List[str] = []
            for entry in data:
                if isinstance(entry, str) and entry.startswith("#"):
                    tags.append(canonical_id(entry[1:]))
                elif isinstance(entry, str):
                    items.append(canonical_id(entry))
                elif isinstance(entry, dict) and "id" in entry:
                    items.append(canonical_id(entry["id"]))
            if len(items) == 1 and not tags:
                return cls(entries=tuple(items))
            # Vanilla's `one_of` ingredient is modelled as an item list; a tag
            # mixed into a list keeps the tag as the authoritative member.
            return cls(entries=tuple(items), tag=tags[0] if tags else None)
        if isinstance(data, dict) and "id" in data:
            return cls(entries=(canonical_id(data["id"]),))
        raise ValueError(f"cannot read recipe ingredient from {data!r}")

    @property
    def is_tag(self) -> bool:
        return self.tag is not None

    def matches(self, item: str, tag_lookup=None) -> bool:
        """Whether an item satisfies this ingredient.

        `tag_lookup(tag_id) -> set[str] | None` resolves a tag's members; when
        it is None a tag ingredient matches nothing, which is the safe default
        for a tag the registry does not know.
        """
        item = canonical_id(item)
        if self.tag is not None:
            if tag_lookup is None:
                return False
            members = tag_lookup(self.tag)
            return bool(members) and item in members
        return item in self.entries

    def items(self, tag_lookup=None) -> Tuple[str, ...]:
        """Every item this ingredient accepts, tags expanded when resolvable."""
        if self.tag is not None:
            members = tag_lookup(self.tag) if tag_lookup else None
            return tuple(sorted(members)) if members else ()
        return self.entries

    def to_json(self) -> Any:
        if self.tag is not None and not self.entries:
            return f"#{self.tag}"
        if len(self.entries) == 1:
            return self.entries[0]
        return list(self.entries)

    def __str__(self) -> str:
        if self.tag is not None and not self.entries:
            return f"#{self.tag}"
        return "|".join(self.entries)


@dataclass(slots=True)
class Recipe:
    """One entry of the recipe registry."""

    key: str
    type: str
    index: int
    result: Optional[ItemStack] = None
    ingredients: Tuple[Ingredient, ...] = ()
    category: Optional[str] = None
    group: Optional[str] = None
    pattern: Optional[Tuple[str, ...]] = None
    key_map: Optional[Dict[str, Ingredient]] = None
    cooking_time: Optional[int] = None
    experience: Optional[float] = None
    show_notification: Optional[bool] = None
    raw: Dict[str, Any] = field(default_factory=dict)
    """The untouched source record, for anything not modelled above."""

    def __repr__(self) -> str:
        return f"<Recipe {self.key} type={self.type} result={self.result}>"

    @property
    def is_cooking(self) -> bool:
        return self.type in COOKING_TYPES

    @property
    def is_crafting(self) -> bool:
        return self.type in CRAFTING_TYPES

    @property
    def is_special(self) -> bool:
        return self.type.startswith("minecraft:crafting_special_")

    @property
    def type_path(self) -> str:
        return path_of(self.type)


# Recipe type ids, in the order they appear in the 1.21.11 recipe packet.
CRAFTING_TYPES = frozenset({
    "minecraft:crafting_shaped",
    "minecraft:crafting_shapeless",
    "minecraft:crafting_transmute",
    "minecraft:crafting_decorated_pot",
})
COOKING_TYPES = frozenset({
    "minecraft:smelting",
    "minecraft:blasting",
    "minecraft:smoking",
    "minecraft:campfire_cooking",
})
SMITHING_TYPES = frozenset({
    "minecraft:smithing_transform",
    "minecraft:smithing_trim",
})
SPECIAL_TYPES = frozenset({
    "minecraft:crafting_special_armordye",
    "minecraft:crafting_special_bannerduplicate",
    "minecraft:crafting_special_bookcloning",
    "minecraft:crafting_special_firework_rocket",
    "minecraft:crafting_special_firework_star",
    "minecraft:crafting_special_firework_star_fade",
    "minecraft:crafting_special_mapcloning",
    "minecraft:crafting_special_mapextending",
    "minecraft:crafting_special_repairitem",
    "minecraft:crafting_special_shielddecoration",
    "minecraft:crafting_special_tippedarrow",
})


# ==================== Tags ====================

@dataclass(slots=True)
class Tag:
    """A named group of entries within one registry.

    Entries may include `#other_tag`, which vanilla resolves transitively.
    Ordering is preserved because the Update Tags packet sends entries in this
    order and a stable payload keeps the login deterministic.

    Mutating a tag directly goes through the owning registry's guard, so a
    post-freeze change is refused here rather than going out of step with the
    tags a connected client was told about.
    """

    registry: str
    name: str
    entries: List[str] = field(default_factory=list)
    _member_set: set = field(default_factory=set, repr=False)
    _guard: Optional[Callable[[str], None]] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.registry = canonical_id(self.registry)
        self.name = canonical_id(self.name)
        self.entries = [self._normalise(entry) for entry in self.entries]
        self._member_set = set(self.entries)

    @staticmethod
    def _normalise(entry: str) -> str:
        if entry.startswith("#"):
            return "#" + canonical_id(entry[1:])
        return canonical_id(entry)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def __contains__(self, entry: str) -> bool:
        return self._normalise(entry) in self._member_set

    def __repr__(self) -> str:
        return f"<Tag {self.name} entries={len(self.entries)}>"

    @property
    def nested_tags(self) -> Tuple[str, ...]:
        """Tag ids this tag includes, without the `#`."""
        return tuple(entry[1:] for entry in self.entries if entry.startswith("#"))

    def add(self, entry: str) -> bool:
        """Append an entry. Returns False when it was already present."""
        if self._guard is not None:
            self._guard(f"add {entry} to tag {self.name}")
        normalised = self._normalise(entry)
        if normalised in self._member_set:
            return False
        self.entries.append(normalised)
        self._member_set.add(normalised)
        return True

    def discard(self, entry: str) -> bool:
        """Remove an entry. Returns False when it was not present."""
        if self._guard is not None:
            self._guard(f"remove {entry} from tag {self.name}")
        normalised = self._normalise(entry)
        if normalised not in self._member_set:
            return False
        self.entries.remove(normalised)
        self._member_set.discard(normalised)
        return True

    def replace(self, entries: Iterable[str]) -> None:
        """Replace every entry."""
        if self._guard is not None:
            self._guard(f"replace entries of tag {self.name}")
        self.entries = [self._normalise(entry) for entry in entries]
        self._member_set = set(self.entries)

    def resolve(self, expand_tag, _seen: Optional[set] = None) -> "frozenset[str]":
        """Every member entry id, with nested tags expanded.

        `expand_tag(tag_id) -> Iterable[str]` is supplied by the tag registry.
        Cycles are impossible in vanilla data but cheap to guard against.
        """
        seen = _seen if _seen is not None else set()
        if self.name in seen:
            return frozenset()
        seen.add(self.name)

        members: set = set()
        for entry in self.entries:
            if entry.startswith("#"):
                members.update(expand_tag(entry[1:], seen))
            else:
                members.add(entry)
        return frozenset(members)

    def to_json(self) -> List[str]:
        return list(self.entries)


__all__ = [
    "BOOLEAN_VALUES",
    "BlockProperty",
    "BlockState",
    "BlockType",
    "ItemType",
    "ItemStack",
    "Ingredient",
    "Recipe",
    "Tag",
    "CRAFTING_TYPES",
    "COOKING_TYPES",
    "SMITHING_TYPES",
    "SPECIAL_TYPES",
    "DuplicateEntryError",
]
