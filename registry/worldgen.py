"""
Worldgen data - the registry face of world generation.

`assets/` has carried twelve world-generation files since the registry rework,
and until now the world read none of them: `registry/README.md` listed "worldgen
data is loaded but not used" as a known gap, terrain came from `world/noise.py`'s
own Perlin field, and the dimension table was a hand-copied dataclass in
`world/dimension.py`. This module is the bridge. It generates nothing; it turns
the raw payloads into typed, indexed views the world can query, and it proves
that the cross-references between them hold.

    from registry import get_game_registry

    worldgen = get_game_registry().worldgen
    worldgen.settings("overworld").sea_level             # 63, not a guess
    worldgen.settings("overworld").shape.min_y           # -64
    worldgen.dimension("minecraft:the_nether").height    # 256
    worldgen.biome("minecraft:plains").numeric_id        # 40
    worldgen.biome("minecraft:plains").carvers           # ('minecraft:cave', ...)
    worldgen.noise("temperature").first_octave           # -10

What the files are, and what this module does with them
-------------------------------------------------------
=================================  ==========================================
`chunk_gen_settings.json`          seven generator presets: shape, sea level,
                                   default block/fluid, surface rule, and the
                                   fifteen-slot noise router
`density_function.json`            the *resolved* router per preset - the
                                   `minecraft:overworld/continents` style
                                   references are already inlined, so a slot is
                                   a tree, not a name
`noise_parameters.json`            60 octave stacks (`firstOctave`, `amplitudes`)
`multi_noise_biome_tree.json`      the climate-parameter -> biome tree, as
                                   `{branch|leaf, parameters, ...}`
`biome.json`                       65 biomes: climate, carvers, features,
                                   spawners, effects
`carver.json`                      four configured carvers
`placed_feature.json`              258 placed features
`configured_features.json`         224 configured features
`dimension.json`                   four dimension types
`chunk_status.json`                the twelve pipeline stage names
`chunk.json`                       an orphan sample chunk - see `chunk_sample`
`chunk_status` etc.                 ...
=================================  ==========================================

Three things about the shapes are easy to get wrong, so they are asserted here
rather than discovered later:

* **The generator shape is not the dimension height.** The nether's dimension
  type is 256 blocks tall, its *terrain shape* is 128, and its logical height is
  128. `dimension()` reports the dimension, `settings().shape` the terrain.
* **`density_function.json` names its slots the way Pumpkin's struct does**
  (`finalDensity`), while `chunk_gen_settings.json` uses vanilla's names
  (`final_density`). `NOISE_ROUTER_SLOTS` is the mapping, and every preset is
  checked against it.
* **`density_function.json` is not strict JSON**: splines carry `Infinity` and
  `-Infinity` literals, which `json.load` accepts but a strict parser rejects.
  `validate()` reports any node that a future interpreter would choke on.

Validation is a list, never an exception: `validate()` returns the problems it
found, `load_registries` folds them into `GameRegistry.problems`, and
`tools/check_worldgen_data.py` prints them. The whole deep walk costs ~60ms, so
it runs on every boot rather than only in the check tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from loguru import logger

from registry.core import canonical_id

# ==================== Constants: the shape of the dump ====================

#: Vanilla noise-router slot -> the field name `density_function.json` uses for
#: it. The file is a dump of Pumpkin's parsed `BaseNoiseRouter`, whose fields are
#: camelCase; the settings file is vanilla's own JSON.
NOISE_ROUTER_SLOTS: Tuple[Tuple[str, str], ...] = (
    ("barrier", "barrierNoise"),
    ("fluid_level_floodedness", "fluidLevelFloodednessNoise"),
    ("fluid_level_spread", "fluidLevelSpreadNoise"),
    ("lava", "lavaNoise"),
    ("temperature", "temperature"),
    ("vegetation", "vegetation"),
    ("continents", "continents"),
    ("erosion", "erosion"),
    ("depth", "depth"),
    ("ridges", "ridges"),
    # Vanilla calls this slot `preliminary_surface_level`; the surface rule that
    # reads it is `find_top_surface`/`above_preliminary_surface`.
    ("preliminary_surface_level", "initialDensityWithoutJaggedness"),
    ("final_density", "finalDensity"),
    ("vein_toggle", "veinToggle"),
    ("vein_ridged", "veinRidged"),
    ("vein_gap", "veinGap"),
)

#: The presets `chunk_gen_settings.json` ships. `floating_islands` has no vanilla
#: dimension type pointing at it; it is a world-creation option.
GENERATOR_PRESETS: Tuple[str, ...] = (
    "overworld",
    "large_biomes",
    "amplified",
    "nether",
    "end",
    "caves",
    "floating_islands",
)

#: Dimension type -> the generator preset vanilla generates it with. The three
#: `*_variant` presets (large biomes, amplified, floating islands) are chosen at
#: world creation and are not reachable from a dimension name.
DIMENSION_PRESETS: Dict[str, str] = {
    "minecraft:overworld": "overworld",
    "minecraft:overworld_caves": "caves",
    "minecraft:the_nether": "nether",
    "minecraft:the_end": "end",
}

#: Density-function node classes, with the keys a consumer needs from `value`.
#: `BlendAlpha`, `BlendOffset` and `EndIslands` are unit variants and carry no
#: `value` at all.
DENSITY_FUNCTION_CLASSES: Dict[str, Tuple[str, ...]] = {
    "Constant": ("value",),
    "YClampedGradient": ("fromY", "toY", "fromValue", "toValue"),
    "BinaryOperation": ("type", "argument1", "argument2"),
    "UnaryOperation": ("type", "input"),
    "LinearOperation": ("specificType", "input"),
    "Clamp": ("input", "minValue", "maxValue"),
    "Wrapping": ("type", "wrapped"),
    "Noise": ("noise", "xzScale", "yScale"),
    "ShiftedNoise": ("shiftX", "shiftY", "shiftZ", "xzScale", "yScale", "noise"),
    "ShiftA": ("offsetNoise",),
    "ShiftB": ("offsetNoise",),
    "BlendDensity": ("input",),
    "BlendAlpha": (),
    "BlendOffset": (),
    "EndIslands": (),
    "Spline": ("spline",),
    "RangeChoice": ("input", "minInclusive", "maxExclusive", "whenInRange", "whenOutOfRange"),
    "WeirdScaledSampler": ("input", "noise", "rarityValueMapper"),
    "InterpolatedNoiseSampler": (
        "scaledXzScale", "scaledYScale", "xzFactor", "yFactor",
        "smearScaleMultiplier", "maxValue", "xzScale", "yScale",
    ),
}

BINARY_OPERATIONS: Tuple[str, ...] = ("ADD", "MUL", "MIN", "MAX")
UNARY_OPERATIONS: Tuple[str, ...] = (
    "ABS", "SQUARE", "CUBE", "HALF_NEGATIVE", "QUARTER_NEGATIVE", "SQUEEZE",
)
#: `Wrapping` is Pumpkin's wrapper enum: a memoising decorator around a node.
WRAPPING_TYPES: Tuple[str, ...] = ("Cache2D", "CacheOnce", "FlatCache", "Interpolated")
#: `LinearOperation` carries its operator under `specificType`, not `type`.
LINEAR_OPERATIONS: Tuple[str, ...] = ("ADD", "MUL")
RARITY_MAPPERS: Tuple[str, ...] = ("TYPE1", "TYPE2")

#: The seven climate parameters of a multi-noise entry, in vanilla's order.
CLIMATE_PARAMETERS: Tuple[str, ...] = (
    "temperature", "humidity", "continentalness", "erosion", "depth", "weirdness", "offset",
)

#: `chunk.json`: 16 x 16 x 384 block-state ids. One overworld column's worth, and
#: the reason this module exposes it at all is the check tool, which proves the
#: values are registry state ids. Nothing generates from it.
SAMPLE_CHUNK_WIDTH = 16
SAMPLE_CHUNK_HEIGHT = 384

#: `chunk_status.json` is the vanilla pipeline. First and last are load-bearing:
#: an empty chunk is `empty`, a finished one is `full`.
CHUNK_STATUS_COUNT = 12


# ==================== Small helpers ====================

def _canonical(name: Any) -> str:
    """Canonical id for a data-supplied name; tolerant of non-strings."""
    return canonical_id(name) if isinstance(name, str) else ""


def _bare(name: str) -> str:
    """`minecraft:plains` -> `plains`. Several files key by bare path."""
    return name.split(":", 1)[1] if ":" in name else name


def _canonical_tag(value: Any) -> str:
    """Canonical id of a `#namespace:tag` reference, without the `#`."""
    if not isinstance(value, str):
        return ""
    return canonical_id(value.lstrip("#")) if value.lstrip("#") else ""


def _id_list(value: Any) -> Tuple[str, ...]:
    """A list of ids that the data may have written as a single string.

    Vanilla allows both spellings (`"carvers": "minecraft:cave"` and
    `"carvers": ["minecraft:cave", ...]`), and iterating the string form yields
    characters - which silently turns one carver into fourteen.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        return (_canonical(value),)
    if isinstance(value, Mapping):
        return (_canonical(value),)
    return tuple(_canonical(entry) for entry in value)


def iter_dicts(node: Any) -> Iterator[Mapping[str, Any]]:
    """Every mapping in a nested JSON payload, iteratively.

    Iterative because the density-function trees are 60k nodes deep-ish and a
    recursive walk would be the only thing in this module with a stack limit.
    """
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            yield current
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)


def iter_density_nodes(node: Any) -> Iterator[Tuple[str, Mapping[str, Any]]]:
    """`(class, value)` for every density-function node in a tree."""
    for mapping in iter_dicts(node):
        node_class = mapping.get("_class")
        if node_class is None:
            continue
        value = mapping.get("value")
        yield node_class, value if isinstance(value, dict) else {}


def iter_leaves(tree: Any) -> Iterator[Tuple[str, Any]]:
    """`(biome, parameters)` for every leaf of a multi-noise tree.

    A node is `{_type: branch|leaf, parameters: [...], subTree: [...]}`; the
    entry point is a bare list, so the walk accepts both.
    """
    stack = [tree]
    while stack:
        current = stack.pop()
        if isinstance(current, list):
            stack.extend(current)
            continue
        if not isinstance(current, dict):
            continue
        if current.get("_type") == "leaf":
            yield _canonical(current.get("biome")), current.get("parameters")
            continue
        sub = current.get("subTree")
        if isinstance(sub, list):
            stack.extend(sub)
        elif isinstance(sub, dict):
            stack.append(sub)


# ==================== Typed views ====================

@dataclass(frozen=True, slots=True)
class BlockStateRef:
    """A `{Name, Properties}` pair from the data, resolved against the registry.

    `state_id` is `None` when the block or one of its property values is not in
    the registry - a plugin-free boot never hits that, and `validate()` reports
    it rather than raising, because a missing surface-rule block should not stop
    a server that can otherwise generate.
    """

    name: str
    properties: Tuple[Tuple[str, str], ...] = ()
    state_id: Optional[int] = None

    @classmethod
    def parse(cls, payload: Any, blocks) -> "BlockStateRef":
        """Build from a `{Name, Properties}` compound (or a bare name string)."""
        if isinstance(payload, str):
            name, properties = _canonical(payload), {}
        elif isinstance(payload, Mapping):
            name = _canonical(payload.get("Name", ""))
            raw = payload.get("Properties") or {}
            properties = {str(key): str(value) for key, value in raw.items()}
        else:
            return cls("", (), None)

        state = blocks.state_from_nbt(name, properties) if name else None
        return cls(
            name,
            tuple(sorted(properties.items())),
            state.id if state is not None else None,
        )

    def as_dict(self) -> Dict[str, Any]:
        """Back to the `{Name, Properties}` shape, for a caller that wants NBT."""
        payload: Dict[str, Any] = {"Name": self.name}
        if self.properties:
            payload["Properties"] = dict(self.properties)
        return payload

    def __bool__(self) -> bool:
        return bool(self.name)


@dataclass(frozen=True, slots=True)
class GenerationShape:
    """`noise` from a generator preset: where terrain lives and at what grain.

    `size_horizontal` and `size_vertical` are vanilla's cell sizes *in 4-block
    units*: 1 means a 4-block cell, 2 means 8 (see `GenerationShapeConfig`).
    """

    min_y: int
    height: int
    size_horizontal: int
    size_vertical: int

    @property
    def max_y(self) -> int:
        """One past the top block, which is how vanilla stores it."""
        return self.min_y + self.height

    @property
    def section_count(self) -> int:
        return self.height // 16

    @property
    def horizontal_cell_blocks(self) -> int:
        return self.size_horizontal * 4

    @property
    def vertical_cell_blocks(self) -> int:
        return self.size_vertical * 4

    def __repr__(self) -> str:
        return (f"<GenerationShape y={self.min_y}..{self.max_y - 1} "
                f"cells={self.horizontal_cell_blocks}x{self.vertical_cell_blocks}>")


@dataclass(frozen=True, slots=True)
class ChunkGeneratorSettings:
    """One `chunk_gen_settings.json` preset, with its router resolved."""

    preset: str
    shape: GenerationShape
    sea_level: int
    default_block: BlockStateRef
    default_fluid: BlockStateRef
    aquifers_enabled: bool
    ore_veins_enabled: bool
    legacy_random_source: bool
    disable_mob_generation: bool
    #: Vanilla slot name -> resolved density-function tree.
    noise_router: Mapping[str, Any]
    #: Vanilla slot name -> the payload as `chunk_gen_settings.json` has it
    #: (a mix of inline trees and `minecraft:...` name references).
    raw_noise_router: Mapping[str, Any]
    surface_rule: Any
    spawn_target: Tuple[Any, ...] = ()

    def router(self, slot: str) -> Any:
        """The resolved density-function tree in a router slot.

        Raises KeyError on an unknown slot rather than returning None: a typo in
        a slot name is a programming error, and a generator that silently reads
        a missing slot would produce plausible terrain from the wrong data.
        """
        try:
            return self.noise_router[slot]
        except KeyError:
            raise KeyError(
                f"generator preset {self.preset!r} has no router slot {slot!r}; "
                f"slots are {', '.join(sorted(self.noise_router))}"
            ) from None


@dataclass(frozen=True, slots=True)
class DimensionTypeData:
    """One `dimension.json` entry (the server-side view of `dimension_type`)."""

    name: str
    preset: str
    min_y: int
    height: int
    logical_height: int
    coordinate_scale: float
    has_skylight: bool
    has_ceiling: bool
    ambient_light: float
    has_fixed_time: bool
    infiniburn: str
    monster_spawn_light_level: Any
    monster_spawn_block_light_limit: int
    attributes: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def max_y(self) -> int:
        """One past the top block, matching `DimensionSettings.max_y`."""
        return self.min_y + self.height

    @property
    def section_count(self) -> int:
        return self.height // 16

    def __repr__(self) -> str:
        return (f"<DimensionTypeData {self.name} y={self.min_y}..{self.max_y - 1} "
                f"preset={self.preset}>")


@dataclass(frozen=True, slots=True)
class BiomeData:
    """One `biome.json` entry, with the ids the wire uses."""

    name: str
    #: Index in the synchronized `worldgen/biome` registry - what a chunk's
    #: biome palette holds and what `biome.json` records as `id`.
    numeric_id: Optional[int]
    temperature: float
    downfall: float
    has_precipitation: bool
    carvers: Tuple[str, ...] = ()
    #: One tuple per decoration step; empty steps are kept, because the step
    #: index is the ordering contract features are placed by.
    features: Tuple[Tuple[str, ...], ...] = ()
    spawners: Mapping[str, Any] = field(default_factory=dict)
    effects: Mapping[str, Any] = field(default_factory=dict)
    attributes: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NoiseParameterData:
    """One `noise_parameters.json` octave stack."""

    name: str
    first_octave: int
    amplitudes: Tuple[float, ...]

    @property
    def octaves(self) -> int:
        return len(self.amplitudes)


@dataclass(frozen=True, slots=True)
class CarverData:
    """One `carver.json` entry."""

    name: str
    probability: Optional[float]
    y: Any
    replaceable: str
    config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MultiNoiseSource:
    """One `multi_noise_biome_tree.json` entry (`overworld`, `nether`)."""

    name: str
    parameters: Tuple[Any, ...]
    tree: Any

    def iter_leaves(self) -> Iterator[Tuple[str, Any]]:
        return iter_leaves(self.tree)

    def biome_names(self) -> "frozenset[str]":
        return frozenset(name for name, _ in self.iter_leaves())

    def leaf_count(self) -> int:
        return sum(1 for _ in self.iter_leaves())


# ==================== The aggregate ====================

class WorldgenData:
    """Typed, cross-checked views over the twelve worldgen assets.

    Built once by `load_registries` after plugins have run and before the
    freeze, so a plugin that registers a synced biome is visible here. Every
    accessor returns an immutable view or the raw payload; nothing is copied
    except the small structures that need reshaping, so the memory the loader
    already paid for is reused.
    """

    def __init__(self, tables, blocks, synced, tags) -> None:
        self._tables = tables
        self._blocks = blocks
        self._synced = synced
        self._tags = tags

        self.settings_by_preset: Dict[str, ChunkGeneratorSettings] = {}
        self.dimensions: Dict[str, DimensionTypeData] = {}
        self.biomes: Dict[str, BiomeData] = {}
        self.noises: Dict[str, NoiseParameterData] = {}
        self.carvers: Dict[str, CarverData] = {}
        self.density_graphs: Dict[str, Mapping[str, Any]] = {}
        self.multi_noise_sources: Dict[str, MultiNoiseSource] = {}
        self.chunk_statuses: Tuple[str, ...] = ()
        self.sample_chunk: Sequence[int] = ()

        self.problems: List[str] = []
        self._deep_problems: Optional[List[str]] = None

        self._build()

    # ----- building -----

    def _table(self, name: str):
        table = self._tables.get(name) if self._tables is not None else None
        if table is None:
            self.problems.append(f"data table {name!r} is not loaded")
        return table

    def _build(self) -> None:
        self._build_density_graphs()
        self._build_settings()
        self._build_dimensions()
        self._build_biomes()
        self._build_noises()
        self._build_carvers()
        self._build_multi_noise()
        self._build_chunk_statuses()

    def _build_density_graphs(self) -> None:
        table = self._table("density_functions")
        if table is None:
            return
        data = table.data
        if not isinstance(data, Mapping):
            self.problems.append("density_function.json is not a name -> payload map")
            return
        self.density_graphs = {
            _bare(str(preset)): payload
            for preset, payload in data.items()
            if isinstance(payload, Mapping)
        }

    def _build_settings(self) -> None:
        table = self._table("chunk_gen_settings")
        if table is None:
            return
        for preset, payload in table.data.items():
            if not isinstance(payload, Mapping):
                self.problems.append(f"chunk gen settings {preset!r} is not an object")
                continue
            preset = _bare(str(preset))
            noise = payload.get("noise") or {}
            shape = GenerationShape(
                min_y=int(noise.get("min_y", 0)),
                height=int(noise.get("height", 0)),
                size_horizontal=int(noise.get("size_horizontal", 1)),
                size_vertical=int(noise.get("size_vertical", 1)),
            )
            graph = self.density_graphs.get(preset, {})
            raw_router = payload.get("noise_router") or {}
            router: Dict[str, Any] = {}
            for slot, field_name in NOISE_ROUTER_SLOTS:
                node = graph.get(field_name)
                if node is not None:
                    router[slot] = node

            self.settings_by_preset[preset] = ChunkGeneratorSettings(
                preset=preset,
                shape=shape,
                sea_level=int(payload.get("sea_level", 0)),
                default_block=BlockStateRef.parse(payload.get("default_block"), self._blocks),
                default_fluid=BlockStateRef.parse(payload.get("default_fluid"), self._blocks),
                aquifers_enabled=bool(payload.get("aquifers_enabled", False)),
                ore_veins_enabled=bool(payload.get("ore_veins_enabled", False)),
                legacy_random_source=bool(payload.get("legacy_random_source", False)),
                disable_mob_generation=bool(payload.get("disable_mob_generation", False)),
                noise_router=router,
                raw_noise_router=dict(raw_router) if isinstance(raw_router, Mapping) else {},
                surface_rule=payload.get("surface_rule"),
                spawn_target=tuple(payload.get("spawn_target") or ()),
            )

    def _build_dimensions(self) -> None:
        table = self._table("dimensions")
        if table is None:
            return
        for name, payload in table.data.items():
            if not isinstance(payload, Mapping):
                self.problems.append(f"dimension type {name!r} is not an object")
                continue
            name = _canonical(str(name))
            fixed_time = payload.get("fixed_time")
            self.dimensions[name] = DimensionTypeData(
                name=name,
                preset=DIMENSION_PRESETS.get(name, name),
                min_y=int(payload.get("min_y", 0)),
                height=int(payload.get("height", 0)),
                logical_height=int(payload.get("logical_height", payload.get("height", 0))),
                coordinate_scale=float(payload.get("coordinate_scale", 1.0)),
                has_skylight=bool(payload.get("has_skylight", False)),
                has_ceiling=bool(payload.get("has_ceiling", False)),
                ambient_light=float(payload.get("ambient_light", 0.0)),
                has_fixed_time=bool(payload.get("has_fixed_time", fixed_time is not None)),
                infiniburn=str(payload.get("infiniburn", "")),
                monster_spawn_light_level=payload.get("monster_spawn_light_level"),
                monster_spawn_block_light_limit=int(payload.get("monster_spawn_block_light_limit", 0)),
                attributes=dict(payload.get("attributes") or {}),
                raw=dict(payload),
            )

    def _build_biomes(self) -> None:
        table = self._table("biomes")
        if table is None:
            return
        for name, payload in table.data.items():
            if not isinstance(payload, Mapping):
                self.problems.append(f"biome {name!r} is not an object")
                continue
            canonical = _canonical(str(name))
            features: List[Tuple[str, ...]] = []
            for step in payload.get("features") or ():
                if isinstance(step, str):
                    features.append((_canonical(step),))
                    continue
                if not isinstance(step, Sequence):
                    features.append(())
                    continue
                features.append(tuple(
                    _canonical(entry) if isinstance(entry, str)
                    else _canonical((entry or {}).get("feature", ""))
                    for entry in step
                ))
            recorded = payload.get("id")
            self.biomes[canonical] = BiomeData(
                name=canonical,
                numeric_id=int(recorded) if isinstance(recorded, int) else self._synced_biome_id(canonical),
                temperature=float(payload.get("temperature", 0.0)),
                downfall=float(payload.get("downfall", 0.0)),
                has_precipitation=bool(payload.get("has_precipitation", False)),
                carvers=_id_list(payload.get("carvers")),
                features=tuple(features),
                spawners=dict(payload.get("spawners") or {}),
                effects=dict(payload.get("effects") or {}),
                attributes=dict(payload.get("attributes") or {}),
                raw=dict(payload),
            )

    def _synced_biome_id(self, name: str) -> Optional[int]:
        if self._synced is None:
            return None
        return self._synced.numeric_id("minecraft:worldgen/biome", name)

    def _build_noises(self) -> None:
        table = self._table("noise_parameters")
        if table is None:
            return
        for name, payload in table.data.items():
            if not isinstance(payload, Mapping):
                continue
            self.noises[_bare(str(name))] = NoiseParameterData(
                name=_bare(str(name)),
                first_octave=int(payload.get("firstOctave", 0)),
                amplitudes=tuple(float(value) for value in payload.get("amplitudes") or ()),
            )

    def _build_carvers(self) -> None:
        table = self._table("carvers")
        if table is None:
            return
        for name, payload in table.data.items():
            if not isinstance(payload, Mapping):
                continue
            config = payload.get("config") or {}
            self.carvers[_bare(str(name))] = CarverData(
                name=_bare(str(name)),
                probability=config.get("probability"),
                y=config.get("y"),
                replaceable=_canonical_tag(config.get("replaceable")),
                config=dict(payload),
            )

    def _build_multi_noise(self) -> None:
        table = self._table("multi_noise_biome_tree")
        if table is None:
            return
        for name, payload in table.data.items():
            if not isinstance(payload, Mapping):
                continue
            self.multi_noise_sources[_bare(str(name))] = MultiNoiseSource(
                name=_bare(str(name)),
                parameters=tuple(payload.get("parameters") or ()),
                tree=payload.get("subTree"),
            )

    def _build_chunk_statuses(self) -> None:
        table = self._table("chunk_statuses")
        if table is not None and isinstance(table.data, list):
            self.chunk_statuses = tuple(str(name) for name in table.data)
        sample = self._table("chunk_sample")
        if sample is not None and isinstance(sample.data, list):
            self.sample_chunk = sample.data

    # ----- accessors -----

    def settings(self, preset: str) -> Optional[ChunkGeneratorSettings]:
        """Generator settings by preset name (`overworld`, `nether`, ...)."""
        return self.settings_by_preset.get(_bare(canonical_id(preset)))

    def require_settings(self, preset: str) -> ChunkGeneratorSettings:
        settings = self.settings(preset)
        if settings is None:
            raise KeyError(
                f"no generator preset {preset!r}; loaded presets are "
                f"{', '.join(sorted(self.settings_by_preset))}"
            )
        return settings

    def settings_for_dimension(self, dimension: str) -> Optional[ChunkGeneratorSettings]:
        """The preset a dimension type generates with."""
        data = self.dimension(dimension)
        return self.settings(data.preset) if data is not None else None

    def dimension(self, name: str) -> Optional[DimensionTypeData]:
        return self.dimensions.get(canonical_id(name))

    def biome(self, name: str) -> Optional[BiomeData]:
        return self.biomes.get(canonical_id(name))

    def biome_id(self, name: str) -> Optional[int]:
        """The synchronized biome index, from the biome table first."""
        data = self.biome(name)
        if data is not None and data.numeric_id is not None:
            return data.numeric_id
        return self._synced_biome_id(canonical_id(name))

    def biome_names(self) -> List[str]:
        return sorted(self.biomes)

    def noise(self, name: str) -> Optional[NoiseParameterData]:
        return self.noises.get(_bare(canonical_id(name)))

    def carver(self, name: str) -> Optional[CarverData]:
        return self.carvers.get(_bare(canonical_id(name)))

    def density_function(self, preset: str, slot: str) -> Any:
        """One resolved router slot, without building the settings object."""
        settings = self.settings(preset)
        return None if settings is None else settings.router(slot)

    def configured_feature(self, name: str) -> Any:
        table = self._tables.get("configured_features") if self._tables else None
        return table.data.get(_bare(canonical_id(name))) if table is not None else None

    def placed_feature(self, name: str) -> Any:
        table = self._tables.get("placed_features") if self._tables else None
        return table.data.get(_bare(canonical_id(name))) if table is not None else None

    def multi_noise(self, name: str) -> Optional[MultiNoiseSource]:
        return self.multi_noise_sources.get(_bare(canonical_id(name)))

    def chunk_sample(self) -> Sequence[int]:
        """`chunk.json` - an orphan sample, kept as a fixture, not as input.

        The file is 98,304 ints (16 x 16 x 384, one 1.21.11 overworld column)
        using only three distinct values, which resolve to air, stone and
        `minecraft:water[level=0]`. It has no bedrock, deepslate, dirt or grass,
        so it is a proto-chunk from before the surface rules - and neither this
        server nor Pumpkin at its 1.21.11 commit references it from code (it
        arrived in a commit that reformatted every asset). Treat it as a
        fixture: `tools/check_worldgen_data.py` checks its length and that every
        value is a registered state id, and nothing else should read it.
        """
        return self.sample_chunk

    # ----- reporting -----

    def summary(self) -> Dict[str, Any]:
        return {
            "presets": len(self.settings_by_preset),
            "dimensions": len(self.dimensions),
            "biomes": len(self.biomes),
            "noises": len(self.noises),
            "carvers": len(self.carvers),
            "multi_noise_sources": len(self.multi_noise_sources),
            "chunk_statuses": len(self.chunk_statuses),
        }

    def describe(self) -> str:
        lines = [
            "worldgen: "
            + ", ".join(f"{name}={value}" for name, value in self.summary().items())
        ]
        for preset in sorted(self.settings_by_preset):
            settings = self.settings_by_preset[preset]
            lines.append(
                f"  {preset}: y={settings.shape.min_y}..{settings.shape.max_y - 1}, "
                f"sea={settings.sea_level}, block={settings.default_block.name}, "
                f"fluid={settings.default_fluid.name}, "
                f"router={len(settings.noise_router)}/{len(NOISE_ROUTER_SLOTS)} slots"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"<WorldgenData presets={len(self.settings_by_preset)} "
                f"biomes={len(self.biomes)} carvers={len(self.carvers)} "
                f"problems={len(self.problems)}>")

    # ==================== Validation ====================

    def validate(self, deep: bool = True) -> List[str]:
        """Every cross-reference that has to hold for generation to be sane.

        Returns a list of problems; never raises. The deep walk covers the
        density-function trees (60k nodes, ~50ms), the multi-noise trees
        (146k nodes, ~80ms), surface rules, biome carver/feature lists and the
        dimension table. Results are memoised: boot validation and a check tool
        asking again pay once.
        """
        problems = list(self.problems)
        problems.extend(self._check_settings())
        problems.extend(self._check_dimensions())
        problems.extend(self._check_biomes())
        problems.extend(self._check_carvers())
        problems.extend(self._check_statuses())

        if not deep:
            return problems

        if self._deep_problems is None:
            deep_problems: List[str] = []
            deep_problems.extend(self._check_density_graphs())
            deep_problems.extend(self._check_multi_noise())
            deep_problems.extend(self._check_surface_rules())
            deep_problems.extend(self._check_sample_chunk())
            self._deep_problems = deep_problems

        return problems + list(self._deep_problems)

    def _check_settings(self) -> List[str]:
        problems: List[str] = []
        if set(self.settings_by_preset) != set(self.density_graphs) and self.density_graphs:
            missing = sorted(set(self.settings_by_preset) - set(self.density_graphs))
            extra = sorted(set(self.density_graphs) - set(self.settings_by_preset))
            if missing:
                problems.append(f"presets with settings but no density graph: {missing}")
            if extra:
                problems.append(f"presets with a density graph but no settings: {extra}")

        for name in GENERATOR_PRESETS:
            if self.settings_by_preset and name not in self.settings_by_preset:
                problems.append(f"generator preset {name!r} is missing from chunk_gen_settings.json")

        for preset, settings in sorted(self.settings_by_preset.items()):
            shape = settings.shape
            if shape.height <= 0:
                problems.append(f"{preset}: generator shape has height {shape.height}")
            elif shape.height % 16:
                problems.append(
                    f"{preset}: generator shape height {shape.height} is not a multiple of 16"
                )
            if shape.min_y % 16:
                problems.append(
                    f"{preset}: generator shape min_y {shape.min_y} is not a multiple of 16"
                )
            if not settings.default_block.name:
                problems.append(f"{preset}: no default block")
            elif settings.default_block.state_id is None:
                problems.append(
                    f"{preset}: default block {settings.default_block.name} does not resolve "
                    f"to a registry state"
                )
            # `minecraft:air` is a legitimate default fluid (the End), and its
            # state id is 0 - so this tests for a missing resolution, not for a
            # falsy id.
            if settings.default_fluid.name and settings.default_fluid.state_id is None:
                problems.append(
                    f"{preset}: default fluid {settings.default_fluid.name} does not resolve "
                    f"to a registry state"
                )
            if settings.raw_noise_router and set(settings.raw_noise_router) != {
                slot for slot, _ in NOISE_ROUTER_SLOTS
            }:
                missing = sorted(
                    {slot for slot, _ in NOISE_ROUTER_SLOTS} - set(settings.raw_noise_router)
                )
                extra = sorted(set(settings.raw_noise_router) - {slot for slot, _ in NOISE_ROUTER_SLOTS})
                if missing:
                    problems.append(f"{preset}: noise_router is missing slots {missing}")
                if extra:
                    problems.append(f"{preset}: noise_router has unknown slots {extra}")
            if self.density_graphs and len(settings.noise_router) != len(NOISE_ROUTER_SLOTS):
                problems.append(
                    f"{preset}: resolved router has {len(settings.noise_router)} of "
                    f"{len(NOISE_ROUTER_SLOTS)} slots"
                )
        return problems

    def _check_dimensions(self) -> List[str]:
        problems: List[str] = []
        for name, dimension in sorted(self.dimensions.items()):
            if dimension.height % 16:
                problems.append(f"{name}: dimension height {dimension.height} is not a multiple of 16")
            if dimension.min_y % 16:
                problems.append(f"{name}: dimension min_y {dimension.min_y} is not a multiple of 16")
            if dimension.logical_height > dimension.height:
                problems.append(
                    f"{name}: logical height {dimension.logical_height} exceeds height "
                    f"{dimension.height}"
                )
            if self.settings_by_preset and dimension.preset not in self.settings_by_preset:
                problems.append(
                    f"{name}: no generator preset {dimension.preset!r} for this dimension"
                )
            tag = dimension.infiniburn
            if tag.startswith("#"):
                if self._tags is not None and not self._tags.has_tag("minecraft:block", tag[1:]):
                    problems.append(f"{name}: infiniburn tag {tag} is not in tags.json")

        if self.dimensions and set(self.dimensions) - set(DIMENSION_PRESETS):
            problems.append(
                f"dimension.json has dimensions with no preset mapping: "
                f"{sorted(set(self.dimensions) - set(DIMENSION_PRESETS))}"
            )
        return problems

    def _check_biomes(self) -> List[str]:
        problems: List[str] = []
        placed = self._tables.get("placed_features") if self._tables else None
        configured = self._tables.get("configured_features") if self._tables else None

        for name, biome in sorted(self.biomes.items()):
            if biome.numeric_id is None:
                problems.append(f"{name}: no synchronized worldgen/biome index")
            for carver in biome.carvers:
                if self.carvers and _bare(carver) not in self.carvers:
                    problems.append(f"{name}: carver {carver} is not in carver.json")
            if placed is not None:
                for step, entries in enumerate(biome.features):
                    for feature in entries:
                        if not feature:
                            problems.append(f"{name}: empty feature name in step {step}")
                        elif _bare(feature) not in placed.data:
                            problems.append(
                                f"{name}: step {step} feature {feature} is not in "
                                f"placed_feature.json"
                            )
            for category, spawns in (biome.spawners or {}).items():
                if not isinstance(spawns, Sequence):
                    continue
                for spawn in spawns:
                    entity = _canonical((spawn or {}).get("type", ""))
                    if entity and self._tables is not None:
                        entities = self._tables.get("entity_types")
                        if entities is not None and entities.registry is not None:
                            if entity not in entities.registry:
                                problems.append(
                                    f"{name}: spawner {category} lists unknown entity {entity}"
                                )

        if configured is not None and placed is not None:
            problems.extend(self._check_placed_features(placed, configured))
        return problems

    def _check_placed_features(self, placed, configured) -> List[str]:
        """Every placed feature must point at a configured feature.

        Only the top-level `feature` name is checked - the placement modifiers
        below it are a type tree whose implementation does not exist yet, and
        resolving them would be guessing at names the data spells out.
        """
        problems: List[str] = []
        for name, payload in placed.data.items():
            if not isinstance(payload, Mapping):
                continue
            feature = payload.get("feature")
            if isinstance(feature, str) and _bare(canonical_id(feature)) not in configured.data:
                problems.append(f"placed feature {name}: {feature} is not in configured_features.json")
        return problems

    def _check_carvers(self) -> List[str]:
        problems: List[str] = []
        for name, carver in sorted(self.carvers.items()):
            if carver.replaceable and self._tags is not None:
                if not self._tags.has_tag("minecraft:block", carver.replaceable):
                    problems.append(
                        f"carver {name}: replaceable tag {carver.replaceable} is not in tags.json"
                    )
            if carver.probability is None:
                problems.append(f"carver {name}: no probability")
        return problems

    def _check_statuses(self) -> List[str]:
        problems: List[str] = []
        if not self.chunk_statuses:
            return problems
        if len(self.chunk_statuses) != CHUNK_STATUS_COUNT:
            problems.append(
                f"chunk_status.json has {len(self.chunk_statuses)} statuses, expected "
                f"{CHUNK_STATUS_COUNT}"
            )
        if len(set(self.chunk_statuses)) != len(self.chunk_statuses):
            problems.append("chunk_status.json has duplicate status names")
        if self.chunk_statuses[0] != "empty" or self.chunk_statuses[-1] != "full":
            problems.append(
                f"chunk_status.json runs {self.chunk_statuses[0]}..{self.chunk_statuses[-1]}, "
                f"expected empty..full"
            )
        return problems

    def _check_density_graphs(self) -> List[str]:
        """Walk every router slot of every preset.

        This is the check a future interpreter depends on: a node class this
        module does not know, a missing argument, an operation name that is not
        in the enum, or a `noise` name that is not in `noise_parameters.json` all
        show up here rather than as wrong terrain.
        """
        problems: List[str] = []
        for preset in sorted(self.settings_by_preset):
            settings = self.settings_by_preset[preset]
            for slot in sorted(settings.noise_router):
                node = settings.noise_router[slot]
                for node_class, value in iter_density_nodes(node):
                    expected = DENSITY_FUNCTION_CLASSES.get(node_class)
                    if expected is None:
                        problems.append(
                            f"{preset}/{slot}: unknown density function class {node_class!r}"
                        )
                        continue
                    for key in expected:
                        if key not in value:
                            problems.append(
                                f"{preset}/{slot}: {node_class} is missing {key!r}"
                            )
                    problems.extend(
                        self._check_density_leaf(preset, slot, node_class, value)
                    )
        return problems

    def _check_density_leaf(
        self, preset: str, slot: str, node_class: str, value: Mapping[str, Any]
    ) -> List[str]:
        problems: List[str] = []
        where = f"{preset}/{slot}"

        if node_class in ("Noise", "ShiftedNoise", "WeirdScaledSampler"):
            noise = value.get("noise")
            if isinstance(noise, str) and self.noises and _bare(canonical_id(noise)) not in self.noises:
                problems.append(f"{where}: noise {noise!r} is not in noise_parameters.json")
        elif node_class in ("ShiftA", "ShiftB"):
            noise = value.get("offsetNoise")
            if isinstance(noise, str) and self.noises and _bare(canonical_id(noise)) not in self.noises:
                problems.append(f"{where}: offset noise {noise!r} is not in noise_parameters.json")
        elif node_class == "BinaryOperation":
            operator = value.get("type")
            if operator not in BINARY_OPERATIONS:
                problems.append(f"{where}: unknown binary operation {operator!r}")
        elif node_class == "UnaryOperation":
            operator = value.get("type")
            if operator not in UNARY_OPERATIONS:
                problems.append(f"{where}: unknown unary operation {operator!r}")
        elif node_class == "LinearOperation":
            operator = value.get("specificType")
            if operator not in LINEAR_OPERATIONS:
                problems.append(f"{where}: unknown linear operation {operator!r}")
        elif node_class == "Wrapping":
            wrapper = value.get("type")
            if wrapper not in WRAPPING_TYPES:
                problems.append(f"{where}: unknown wrapper {wrapper!r}")
        elif node_class == "WeirdScaledSampler":
            mapper = value.get("rarityValueMapper")
            if mapper not in RARITY_MAPPERS:
                problems.append(f"{where}: unknown rarity mapper {mapper!r}")
        elif node_class == "Constant":
            constant = value.get("value")
            if not isinstance(constant, (int, float)) or isinstance(constant, bool):
                problems.append(f"{where}: constant is {constant!r}, not a number")
        return problems

    def _check_multi_noise(self) -> List[str]:
        problems: List[str] = []
        for name, source in sorted(self.multi_noise_sources.items()):
            if len(source.parameters) != len(CLIMATE_PARAMETERS):
                problems.append(
                    f"multi-noise source {name}: {len(source.parameters)} top-level parameters, "
                    f"expected {len(CLIMATE_PARAMETERS)}"
                )
            leaves = 0
            for biome, parameters in source.iter_leaves():
                leaves += 1
                if not biome:
                    problems.append(f"multi-noise source {name}: a leaf has no biome")
                    continue
                if self.biomes and biome not in self.biomes:
                    problems.append(f"multi-noise source {name}: biome {biome} is not in biome.json")
                if not isinstance(parameters, Sequence) or len(parameters) != len(CLIMATE_PARAMETERS):
                    problems.append(
                        f"multi-noise source {name}: leaf {biome} has "
                        f"{len(parameters) if isinstance(parameters, Sequence) else 0} parameters, "
                        f"expected {len(CLIMATE_PARAMETERS)}"
                    )
            if leaves == 0:
                problems.append(f"multi-noise source {name}: no leaves")
        return problems

    def _check_surface_rules(self) -> List[str]:
        """Every block and biome a surface rule names must exist.

        Surface rules decide dirt/grass/sand/bedrock, so a name that does not
        resolve is a silently missing layer rather than a crash.
        """
        problems: List[str] = []
        for preset, settings in sorted(self.settings_by_preset.items()):
            rule = settings.surface_rule
            if rule is None:
                problems.append(f"{preset}: no surface rule")
                continue
            for mapping in iter_dicts(rule):
                state = mapping.get("result_state")
                if isinstance(state, Mapping):
                    reference = BlockStateRef.parse(state, self._blocks)
                    # `minecraft:air` is a real surface-rule result (the End and
                    # the cave presets write it), and its state id is 0.
                    if reference.name and reference.state_id is None:
                        problems.append(
                            f"{preset}/surface_rule: block {reference.name!r} does not resolve"
                        )
                biomes = mapping.get("biome_is")
                if isinstance(biomes, Sequence) and not isinstance(biomes, (str, bytes)):
                    for biome in biomes:
                        canonical = _canonical(biome)
                        if canonical and self.biomes and canonical not in self.biomes:
                            problems.append(
                                f"{preset}/surface_rule: biome {canonical} is not in biome.json"
                            )
        return problems

    def _check_sample_chunk(self) -> List[str]:
        """`chunk.json` is a fixture; this is the only thing that reads it.

        Length and value range are the whole check: if the values are global
        block state ids, every one of them must be inside the block registry.
        That is what makes the file a usable parity fixture later, and it is
        also how a wrong-length file would be caught.
        """
        problems: List[str] = []
        sample = self.sample_chunk
        if not sample:
            return problems
        expected = SAMPLE_CHUNK_WIDTH * SAMPLE_CHUNK_WIDTH * SAMPLE_CHUNK_HEIGHT
        if len(sample) != expected:
            problems.append(
                f"chunk.json has {len(sample)} entries, expected {expected} "
                f"(16 x 16 x 384, one 1.21.11 overworld column)"
            )
        total_states = self._blocks.total_states if self._blocks is not None else 0
        outside = {value for value in sample if not isinstance(value, int) or value < 0 or value >= total_states}
        if outside:
            problems.append(
                f"chunk.json has {len(outside)} value(s) outside the block registry "
                f"(0..{total_states - 1}): {sorted(outside)[:5]}"
            )
        return problems


def build_worldgen(registry) -> WorldgenData:
    """Build the worldgen views for a loaded `GameRegistry`.

    Called by `load_registries` after plugins, before the freeze, and installed
    as `registry.worldgen`.
    """
    worldgen = WorldgenData(
        registry.tables,
        registry.blocks,
        registry.synced,
        registry.tags,
    )
    logger.debug(
        f"Worldgen data: {worldgen.summary()} "
        f"({len(worldgen.problems)} structural problem(s))"
    )
    return worldgen


__all__ = [
    "BINARY_OPERATIONS",
    "BlockStateRef",
    "BiomeData",
    "CHUNK_STATUS_COUNT",
    "CLIMATE_PARAMETERS",
    "CarverData",
    "ChunkGeneratorSettings",
    "DENSITY_FUNCTION_CLASSES",
    "DIMENSION_PRESETS",
    "DimensionTypeData",
    "GENERATOR_PRESETS",
    "GenerationShape",
    "LINEAR_OPERATIONS",
    "MultiNoiseSource",
    "NOISE_ROUTER_SLOTS",
    "NoiseParameterData",
    "RARITY_MAPPERS",
    "SAMPLE_CHUNK_HEIGHT",
    "SAMPLE_CHUNK_WIDTH",
    "UNARY_OPERATIONS",
    "WRAPPING_TYPES",
    "WorldgenData",
    "build_worldgen",
    "iter_density_nodes",
    "iter_dicts",
    "iter_leaves",
]
