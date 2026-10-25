"""
Crotalus game registry.

One place that knows what blocks, items, recipes, tags and dynamic registries
exist in Minecraft 1.21.11 (protocol 774), loaded from the data Pumpkin ships at
its 1.21.11 commit (`assets/`, hash-verified against `42c903493`).

    from registry import init_registries, get_game_registry

    init_registries()                  # server startup, before the world loads
    blocks = get_game_registry().blocks
    state = blocks.state_of("minecraft:oak_log", {"axis": "y"})
    state.id                           # 137 - the id the client wants

Layout
------
============================  ==============================================
`registry.core`               `ResourceLocation`, `Registry`, freeze guards
`registry.entries`            entry types (blocks, states, items, recipes, tags)
`registry.blocks`             `minecraft:block`, 1,166 blocks / 29,671 states
`registry.items`              `minecraft:item`, 1,505 items + components
`registry.recipes`            all 1,470 recipes, indexed by result/type/ingredient
`registry.tags`               `tags.json`, and the Update Tags payload
`registry.synced`             the 23 synchronized registries + Registry Data
`registry.tables`             every other asset file (fluids, biomes, sounds, ...)
`registry.worldgen`           typed worldgen views: presets, dimensions, biomes,
                              carvers, router slots (see `registry.worldgen`)
`registry.plugin`             the plugin interface
`registry.loader`             asset -> `GameRegistry`
`registry.game_registry`      the aggregate object and global accessor
`registry.registry`           network adapter for the login path (legacy API)
`registry.tag_data`           tag adapter for the login path (legacy API)
============================  ==============================================

Lifecycle
---------
`init_registries()` loads everything, applies datapacks and plugins, validates,
and freezes. It must run **before** any world is generated or loaded: chunk
palettes carry block state ids resolved from these tables, and the login path
sends the synced registries out of them. After the freeze, ids cannot move any
more - a plugin may still change a value that does not renumber anything.

Writing a plugin
----------------
    from registry import RegistryPlugin, PluginPhase

    class MyPlugin(RegistryPlugin):
        name = "mypack"
        phase = PluginPhase.DEFAULT

        def on_registries(self, ctx):
            ctx.add_tag_entry("minecraft:item", "minecraft:planks", "minecraft:oak_log")
            ctx.add_recipe({...}, "mypack:oak_log_to_sticks")
            ctx.add_item("mypack:token", {"minecraft:max_stack_size": 1})

Then either pass the instance to `init_registries(plugins=[MyPlugin()])` or drop
the module into the plugins directory (`plugins/` by default), where a module
may expose `PLUGIN`, `plugins`, `register(registry)` or `setup(context)`.

Every `add_*` is append-only: entry order is the numeric id the client uses, so
inserting would renumber everything after it. `modify_*`/`replace_*` change a
value without moving an id and stay legal after the freeze.
"""

from __future__ import annotations

# Containers, identifiers and errors.
from registry.core import (
    IndexedRegistry,
    Registry,
    RegistryError,
    RegistryFrozenError,
    ResourceLocation,
    UnknownEntryError,
    UnknownRegistryError,
    ValidationError,
    canonical_id,
    path_of,
)

# Entry types.
from registry.entries import (
    BlockProperty,
    BlockState,
    BlockType,
    Ingredient,
    ItemStack,
    ItemType,
    Recipe,
    Tag,
)

# Registries.
from registry.blocks import BlockRegistry
from registry.items import ItemRegistry
from registry.recipes import RecipeRegistry
from registry.synced import SyncedEntry, SyncedRegistries
from registry.tables import DataTable, DataTables
from registry.tags import ResolutionReport, TagRegistry

# Worldgen views.
from registry.worldgen import (
    BiomeData,
    BlockStateRef,
    CarverData,
    ChunkGeneratorSettings,
    DimensionTypeData,
    GenerationShape,
    MultiNoiseSource,
    NoiseParameterData,
    WorldgenData,
    build_worldgen,
)

# Plugin interface.
from registry.plugin import (
    PluginContext,
    PluginError,
    PluginManager,
    PluginPhase,
    RegistryPlugin,
)

# Loading and the aggregate.
from registry.game_registry import (
    DEFAULT_ASSETS_DIR,
    GameRegistry,
    get_game_registry,
    init_registries,
    is_initialized,
    reset_registries,
    set_game_registry,
)
from registry.loader import load_registries, validate_registry

__all__ = [
    # core
    "IndexedRegistry",
    "Registry",
    "RegistryError",
    "RegistryFrozenError",
    "ResourceLocation",
    "UnknownEntryError",
    "UnknownRegistryError",
    "ValidationError",
    "canonical_id",
    "path_of",
    # entries
    "BlockProperty",
    "BlockState",
    "BlockType",
    "Ingredient",
    "ItemStack",
    "ItemType",
    "Recipe",
    "Tag",
    # registries
    "BlockRegistry",
    "ItemRegistry",
    "RecipeRegistry",
    "SyncedEntry",
    "SyncedRegistries",
    "DataTable",
    "DataTables",
    "ResolutionReport",
    "TagRegistry",
    # worldgen
    "BiomeData",
    "BlockStateRef",
    "CarverData",
    "ChunkGeneratorSettings",
    "DimensionTypeData",
    "GenerationShape",
    "MultiNoiseSource",
    "NoiseParameterData",
    "WorldgenData",
    "build_worldgen",
    # plugins
    "PluginContext",
    "PluginError",
    "PluginManager",
    "PluginPhase",
    "RegistryPlugin",
    # loading
    "DEFAULT_ASSETS_DIR",
    "GameRegistry",
    "get_game_registry",
    "init_registries",
    "is_initialized",
    "reset_registries",
    "set_game_registry",
    "load_registries",
    "validate_registry",
]
