"""
World generator: the shared base and the backend factory.

Terrain generation has two implementations and this module is where they meet.
`[world] generator` in `config.toml` picks one:

    "vanilla"   world/vanilla/terrain.py    density functions, matches Minecraft
    "legacy"    world/legacy/generator.py   this project's own Perlin, ~30x faster

Neither lives here. What lives here is everything the two genuinely share - the
settings that describe a dimension, the terrain constants the registry fills
in, the flat generator, and the pipeline that runs a chunk through the
decoration layer in `world/feature/`. A backend subclasses `BaseGenerator` and
implements exactly one thing: `_generate_terrain`.

    from world.generator import create_generator
    generator = create_generator(GeneratorSettings(seed=1234, preset="overworld"))

The pipeline order is fixed and both backends use it unchanged:

    _generate_terrain   the backend's own job - the only abstract method
    biomes              world/feature/biomes.py
    surface             world/feature/surface.py
    _generate_ores      a hook; only the legacy backend fills it in
    carvers             world/feature/carvers.py
    features            world/feature/features.py

That order is load-bearing twice over. The three decoration passes read the
biome volume the first one wrote, and on a legacy world the ore pass has to sit
between the surface rules and the carvers - carving first would leave the veins
hanging in air, and it is where the legacy backend has always put them.

**Why ores are a hook and not a layer:** on a vanilla world the ore is already
there. `world/vanilla/features.py` places it as a `minecraft:ore_vein`
configured feature in each biome's list, so running the legacy vein pass as
well would place every vein twice. The base class therefore does nothing, and
only `LegacyWorldGenerator` overrides it.
"""

import importlib
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Type

import numpy as np
from loguru import logger

from world.block_states import BLOCKS
from world.chunk import Chunk
from world.feature import FeatureLayers


class GeneratorType(Enum):
    """Types of world generators."""
    FLAT = "flat"
    NORMAL = "normal"
    LARGE_BIOMES = "large_biomes"
    AMPLIFIED = "amplified"
    CUSTOM = "custom"


#: `GeneratorType` -> the `chunk_gen_settings.json` preset it stands for. Both
#: `normal` and `custom` land on the overworld preset; `flat` has none, because
#: vanilla's flat generator is a layer list rather than a noise router, and the
#: flat path below reads none of these values.
PRESET_FOR_GENERATOR_TYPE: Dict[GeneratorType, str] = {
    GeneratorType.NORMAL: "overworld",
    GeneratorType.LARGE_BIOMES: "large_biomes",
    GeneratorType.AMPLIFIED: "amplified",
}


@dataclass
class GeneratorSettings:
    """Configuration for world generation.

    Two kinds of field live here, and they age differently:

    * **Terrain constants** - `sea_level`, `default_block`, `default_fluid` -
      are facts about the dimension, and they are read from
      `assets/chunk_gen_settings.json` (through `registry.worldgen`) when
      `preset` names one. The dataclass defaults are the fallback for a caller
      that has no registry (a unit test, a custom dimension) and the starting
      values before the generator applies the preset.
    * **Tuning knobs** - the noise scale/octaves and the height band the legacy
      Perlin generator shapes terrain with - have no counterpart in the data.
      They describe *that* generator, not Minecraft's, and the vanilla backend
      ignores them entirely. See `world/legacy/README.md`.
    """
    generator_type: GeneratorType = GeneratorType.NORMAL
    seed: int = 0

    #: `chunk_gen_settings.json` preset to take terrain constants from.
    #: `None` keeps the values below untouched.
    preset: Optional[str] = None

    # -- legacy-only: the Perlin terrain's own shape ------------------------
    # Read by `world/legacy/generator.py` and by nothing else. The vanilla
    # backend takes its shape from the density function instead.
    noise_scale: float = 256.0
    noise_octaves: int = 4
    noise_persistence: float = 0.5
    noise_lacunarity: float = 2.0
    base_height: int = 64
    height_variance: int = 64

    # -- shared: facts about the dimension ----------------------------------
    sea_level: int = 62

    # Which blocks this dimension's terrain is made of, as global block state
    # ids. `None` means "not resolved yet": the generator fills them from the
    # preset's `default_block`/`default_fluid`, or from the registry's defaults,
    # when a world is built. They are deliberately not dataclass defaults -
    # `BLOCKS.STONE` would load 14MB of block data at import time, and a
    # `GeneratorSettings()` built before the registry finished loading would
    # freeze the registry ahead of any plugin.
    default_block: Optional[int] = None
    default_fluid: Optional[int] = None

    #: The preset's noise shape - the extent `above_bottom` / `below_top`
    #: anchors resolve against (`above_bottom: 0` is the bottom of the world).
    #: Usually the dimension's own extent, but not always: the nether's shape
    #: is 128 inside a 256-tall dimension. Chunks span the dimension's window;
    #: these are the numbers the rules mean. The generator fills them from the
    #: preset.
    world_min_y: int = 0
    world_height: int = 256

    # -- layer switches -----------------------------------------------------
    #: The legacy carver. Read only by `LegacyWorldGenerator`; the vanilla
    #: backend always uses `world/vanilla/carver.py`. `"legacy"` is this
    #: project's own wandering-tunnel carver, which is fast, connected and has
    #: nothing to do with vanilla; `"none"` carves nothing.
    cave_mode: str = "vanilla"
    #: Run the biome and surface layers after the terrain fill. They key off
    #: the preset: `chunk_gen_settings.json`, the `surface_rule` data, the
    #: biome tree. On by default because they are the difference between a
    #: grey world and a Minecraft one. Measured cost on one core, per chunk:
    #: ~140 ms for the density fill, 200-450 ms for biomes (six density trees
    #: over a quarter-resolution volume) and ~600 ms for the surface rules
    #: (one Python walk per stone cell). See `world/vanilla/README.md`,
    #: "Cost", before turning them off for a large view distance.
    generate_biomes: bool = True
    generate_surface: bool = True
    generate_features: bool = True
    generate_caves: bool = True

    # -- legacy-only: ore veins ---------------------------------------------
    generate_ores: bool = True

    # -- flat ---------------------------------------------------------------
    # `None` until resolved, for the same reason as `default_block` above.
    flat_layers: Optional[List[tuple[int, int]]] = None
    flat_height: int = 64

    def resolve_blocks(self) -> None:
        """Fill the block fields that need the registry.

        Idempotent: a preset already wrote `default_block`/`default_fluid`, so
        only the unset fields are touched. The generator calls this before
        generating anything; a caller that builds settings without a generator
        has to call it (or read `None`).
        """
        if self.default_block is None:
            self.default_block = BLOCKS.STONE
        if self.default_fluid is None:
            self.default_fluid = BLOCKS.WATER
        if self.flat_layers is None:
            self.flat_layers = [
                (1, BLOCKS.BEDROCK),   # 1 bedrock
                (2, BLOCKS.DIRT),      # 2 dirt
                (1, BLOCKS.GRASS),     # 1 grass
            ]


class BaseGenerator:
    """Everything both terrain backends share.

    A subclass implements `_generate_terrain` and nothing else is required. The
    decoration passes, the registry lookup, the flat generator and the pipeline
    order are all here, so switching `[world] generator` changes the shape of the
    stone and nothing else.

    **One lock, and what actually keeps it uncontended.** Every layer here
    caches parsed trees that carry per-chunk state - `NoiseRouter`'s
    `FlatCache` is the sharpest case, where two threads interleaving would
    evaluate one chunk against the other's caches and smear the terrain along
    Z, quietly and without an exception. The legacy terrain pass was
    re-entrant by accident (a pure function of x, z and seed), which is why
    this was never a problem before. The real defense is that no two
    concurrent workers share a generator: background workers build their own
    per (dimension, seed) - one set per worker PROCESS in the multiprocess
    work mode, one set per pool THREAD (thread-locals, see
    world/chunk_gen_worker.py) in the thread work mode - and the synchronous
    path uses the dimension's own generator. The lock stays on as the belt to
    those braces: an uncontended `threading.Lock` costs nothing measurable
    against a roughly-a-second chunk, and it makes all seven caches safe at
    once; a per-thread router would only have fixed one of them.
    """

    def __init__(self, settings: Optional[GeneratorSettings] = None, *,
                 apply_registry: bool = True):
        """
        Initialize the world generator.

        Args:
            settings: Generator settings, uses defaults if not provided
            apply_registry: Read the terrain constants for `settings.preset` out
                of the registry (sea level, default block/fluid, shape). Off is
                for tests that want the dataclass values verbatim.
        """
        if type(self) is BaseGenerator:
            raise TypeError(
                "BaseGenerator is abstract - it has no terrain of its own. Use "
                "world.generator.create_generator() to get a backend, or subclass "
                "it and implement _generate_terrain()."
            )
        self.settings = settings or GeneratorSettings()
        self._seed = self.settings.seed

        #: The registry's view of this dimension's generator, when it has one.
        self.worldgen = None
        #: `(min_y, height)` of the terrain shape, from the same preset.
        self.shape = None
        #: The four decoration passes, shared by both backends. Built from the
        #: preset and seed, neither of which `_apply_registry` changes.
        self.layers = FeatureLayers(self.settings.preset, self._seed)
        if apply_registry and self.settings.preset:
            self._apply_registry()
        # Whatever the preset did not fill comes from the registry. `None` fields
        # mean "unset", never "air", so this runs even without a preset.
        self.settings.resolve_blocks()
        self._chunk_lock = threading.Lock()

        logger.info(
            f"{type(self).__name__} initialized with seed {self._seed} "
            f"(preset {self.settings.preset!r})"
        )

    def _apply_registry(self) -> None:
        """Take the terrain constants from `assets/chunk_gen_settings.json`.

        Only the fields the data actually determines are written: sea level,
        the default block and fluid, and the terrain shape (exposed as
        `self.shape` - the fill spans the *chunk's* window, while the vanilla
        layers resolve their `above_bottom` / `below_top` anchors against the
        shape - see `world/README.md`). A missing preset leaves the dataclass
        values alone instead of failing the boot: a custom dimension
        legitimately has no preset.
        """
        from world.dimension import registry_generator_settings

        settings = registry_generator_settings(self.settings.preset)
        if settings is None:
            logger.debug(
                f"No generator preset {self.settings.preset!r} in the registry; "
                f"using the dataclass terrain constants"
            )
            return

        from world.block_states import state_id

        self.worldgen = settings
        self.shape = settings.shape
        self.settings.sea_level = settings.sea_level
        self.settings.world_min_y = settings.shape.min_y
        self.settings.world_height = settings.shape.height

        # The preset names blocks; the registry turns each name into the global
        # state id a section stores. There is nothing to fall back to any more:
        # netherrack and end stone are ordinary entries, which is what the nether
        # and the end presets always wanted to say.
        block = state_id(
            settings.default_block.name, dict(settings.default_block.properties)
        )
        self.settings.default_block = block

        fluid = settings.default_fluid
        if fluid.state_id is not None:
            self.settings.default_fluid = state_id(fluid.name, dict(fluid.properties))
        else:
            # `minecraft:air` is the end's default fluid, and air has no fluid
            # state of its own - it is block state 0.
            self.settings.default_fluid = BLOCKS.AIR

        logger.debug(
            f"Generator preset {settings.preset!r}: sea level {settings.sea_level}, "
            f"shape {settings.shape}, block "
            f"{settings.default_block.name} -> {self.settings.default_block}, "
            f"fluid {settings.default_fluid.name} -> {self.settings.default_fluid}"
        )

    # -- pipeline -----------------------------------------------------------

    def generate_chunk(self, chunk: Chunk) -> None:
        """Generate terrain for one chunk.

        The whole body is under the generator's lock: see the class docstring
        for why a generator shared between concurrent workers is not safe
        without it. Both background work modes give every worker its own
        generators, so in practice the lock is never contended.
        """
        if chunk.is_generated:
            return
        with self._chunk_lock:
            if chunk.is_generated:
                return
            self._generate_chunk_locked(chunk)
        chunk.is_generated = True
        chunk.is_dirty = True

    def _generate_chunk_locked(self, chunk: Chunk) -> None:
        """The pipeline. Split out so subclasses have one place to hook."""
        if self.settings.generator_type == GeneratorType.FLAT:
            self._generate_flat(chunk)
            chunk.update_heightmap()
            return

        self._generate_terrain(chunk)
        chunk.update_heightmap()

        # The `world/feature` layers all key off the preset: they read
        # `chunk_gen_settings.json`, `surface_rule`, the biome tree and the
        # carver data for it. A generator with no preset - a test, a custom
        # dimension, a benchmark - asked for the dataclass terrain constants
        # and gets exactly those, not a KeyError. So the three passes that
        # *need* a preset are gated on it, while the two that a backend owns
        # are not: a preset-less world still gets ore veins and legacy caves.
        vanilla = bool(self.settings.preset)
        layers = self.layers
        table = None

        if vanilla and self.settings.generate_biomes:
            layers.biomes.apply(chunk)
        if vanilla and self.settings.generate_surface:
            table = layers.table
            layers.surface.apply(chunk, table)

        if self.settings.generate_ores:
            # A no-op unless a backend overrides it. The vanilla ore arrives
            # with the feature pass below.
            self._generate_ores(chunk)

        if self.settings.generate_caves:
            self._generate_caves(chunk, vanilla)

        if vanilla and self.settings.generate_features:
            table = table if table is not None else layers.table
            layers.features.apply(chunk, table)

    def _generate_terrain(self, chunk: Chunk) -> None:
        """Fill the chunk's stone. The one thing a backend must implement."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _generate_terrain()"
        )

    def _generate_ores(self, chunk: Chunk) -> None:
        """Place ore veins. Only the legacy backend does its own.

        On a vanilla world the ore is a `minecraft:ore_vein` configured
        feature placed by `world/feature/features.py`, so this stays a no-op -
        implementing it here would double every vein.
        """

    def _generate_caves(self, chunk: Chunk, vanilla: bool) -> None:
        """Carve caves. The base carves with the vanilla carver, if it can.

        `cave_mode` is deliberately not consulted here. It is a legacy knob -
        it says which of the *legacy* carvers to prefer - and the vanilla
        backend has exactly one carver, the one in `world/vanilla/carver.py`.
        `LegacyWorldGenerator` overrides this to honour it.
        """
        if not vanilla:
            return
        self.layers.carvers.apply(
            chunk, self.layers.table, self.settings.sea_level)

    # -- shared terrain -----------------------------------------------------

    def _generate_flat(self, chunk: Chunk) -> None:
        """Generate flat terrain: the layer list placed at `flat_height`.

        Layers are placed row by row in the chunk's own window. The previous
        version filled whole *sections* per block of layer and then refilled
        every section from the last layer's own section upward with air --
        which erased the layers it had just placed. Sections start empty here,
        so "air above" costs nothing.

        Shared rather than per-backend because vanilla's flat generator is also
        a layer list, not a noise router: there is no density function to read.
        It touches no noise and builds no `NoiseRouter`, so a flat world never
        pays for one.
        """
        base_y = self.settings.flat_height
        top = chunk.min_y + chunk.height

        for thickness, block_id in self.settings.flat_layers:
            for block_y in range(base_y, base_y + thickness):
                if chunk.min_y <= block_y < top:
                    offset = block_y - chunk.min_y
                    row = chunk.sections[offset // 16].blocks.reshape(16, 16, 16)
                    row[offset % 16].fill(block_id)
            base_y += thickness

        for section in chunk.sections:
            section.recompute_non_air_count()
        chunk.is_dirty = True

    def set_seed(self, seed: int) -> None:
        """Set the world seed. A backend rebuilds whatever noise it holds."""
        self._seed = seed
        self.settings.seed = seed
        self.layers = FeatureLayers(self.settings.preset, seed)

    def generate_spawn_point(self) -> tuple[int, int, int]:
        """A spawn position above the terrain, as `(x, y, z)`.

        The live spawn path is `World.get_spawn_point`, which finds the surface
        from a real generated chunk and is therefore correct for both backends
        without help. This is the generator's own answer for a caller that wants
        one before any chunk exists; a backend that can ask its noise about the
        ground overrides it.
        """
        return (0, self.settings.sea_level + 1, 0)


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

#: `[world] generator` value -> the module that implements it. The class is
#: looked up on the module rather than stored here so that choosing a backend
#: does not import the other one - a vanilla world should not have to load the
#: legacy package to find out it does not want it.
GENERATOR_BACKENDS: Dict[str, str] = {
    "vanilla": "world.vanilla.terrain",
    "legacy": "world.legacy.generator",
}

#: `name` -> `class name` within that module.
GENERATOR_CLASSES: Dict[str, str] = {
    "vanilla": "VanillaWorldGenerator",
    "legacy": "LegacyWorldGenerator",
}

#: Third-party generators, registered by name. Holds classes, not instances.
_GENERATORS: Dict[str, Type[BaseGenerator]] = {}


def register_generator(name: str, generator_class: Type[BaseGenerator]) -> None:
    """Register a custom generator type under `name`."""
    if not (isinstance(generator_class, type) and issubclass(generator_class, BaseGenerator)):
        raise TypeError(
            f"{generator_class!r} is not a BaseGenerator subclass; a registered "
            f"backend has to implement _generate_terrain()"
        )
    _GENERATORS[name] = generator_class


def get_generator(name: str, **kwargs) -> Optional[BaseGenerator]:
    """Get a registered custom generator by name, or None."""
    generator_class = _GENERATORS.get(name)
    return generator_class(**kwargs) if generator_class else None


def create_generator(settings: Optional[GeneratorSettings] = None, *,
                     backend: Optional[str] = None) -> BaseGenerator:
    """Build the generator backend `backend` names, or the configured one.

    `backend=None` reads `[world] generator` from `config.toml` and defaults to
    `"vanilla"`. An unknown value is an error that names the legal ones rather
    than a silent fallback - a typo in a config file should not quietly give
    someone a different world than they asked for.

    Both the backend module and the config are imported *inside* this function.
    A module-level `from core.config import config` would run `core/__init__.py`
    (which imports `core.server`) during `import world`, and `core.server`
    reaches `network.client_handler`, which imports `world.chunk_streamer` back
    out of a half-initialised `world` package. Importing lazily keeps the boot
    order one-directional.
    """
    if backend is None:
        from core.config import config

        backend = config.generator_backend

    if backend in _GENERATORS:
        return _GENERATORS[backend](settings)

    try:
        module_name = GENERATOR_BACKENDS[backend]
    except KeyError:
        raise ValueError(
            f"unknown [world] generator {backend!r}; "
            f"expected one of {', '.join(sorted(GENERATOR_BACKENDS))}"
        ) from None

    module = importlib.import_module(module_name)
    generator_class = getattr(module, GENERATOR_CLASSES[backend])
    logger.debug(f"World generator backend: {backend} ({module_name}.{GENERATOR_CLASSES[backend]})")
    return generator_class(settings)


def create_default_generator(
    generator_type: str = "normal",
    seed: int = 0,
) -> BaseGenerator:
    """Create a generator with default settings, on the configured backend.

    Thin wrapper kept for callers that only care about the type and seed. The
    backend is still `[world] generator`, so this is not a way to get the
    legacy generator - import `world.legacy.generator` for that.
    """
    gen_type = GeneratorType.NORMAL

    if generator_type == "flat":
        gen_type = GeneratorType.FLAT
    elif generator_type == "large_biomes":
        gen_type = GeneratorType.LARGE_BIOMES
    elif generator_type == "amplified":
        gen_type = GeneratorType.AMPLIFIED

    settings = GeneratorSettings(
        generator_type=gen_type,
        seed=seed,
        preset=PRESET_FOR_GENERATOR_TYPE.get(gen_type),
    )

    return create_generator(settings)
