# Game Registry

Minecraft 1.21.11 (protocol 774) block, item, recipe, tag and dynamic-registry
data, loaded from `assets/` before the world is generated or loaded, with an
interface for plugins to extend it.

The data in `assets/` is Pumpkin's report generated at its 1.21.11 commit
(`42c903493`); the loaded tables are:

| Table | Count |
| --- | --- |
| `minecraft:block` | 1,166 blocks, 29,671 states, 123 properties |
| `minecraft:item` | 1,505 items, 58 distinct data components |
| `minecraft:recipe` | 1,470 recipes, 22 types |
| tags | 1,131 tags in 14 registries, 10,404 entries |
| synced registries | 23 registries, 371 entries |
| data tables | 43 auxiliary tables (fluids, biomes, sounds, worldgen, entity tracking, ...) |
| worldgen views | 7 generator presets, 4 dimensions, 65 biomes, 60 noises, 4 carvers |

Load cost is ~1.2s and ~90MB, paid once at boot.

### Where each file in `assets/` actually comes from

Regenerating `assets/` from `42c903493` alone will *not* reproduce what is
there — five files are not in that commit, and two more were edited locally.
Everything below was checked by content comparison against the named commit,
not by filename:

| File | Provenance |
| --- | --- |
| the other 42 | byte-equivalent to `42c903493` |
| `packets.json` | `42c903493`, re-encoded from an ordered name list into `name -> index` — same names, same order, so every packet id is that commit's |
| `entities.json` | `42c903493`, with `loot_table` filled in on 20 entries (boats and chests); no `id` differs |
| `dimension.json`, `meta_data_type.json`, `tracked_data.json` | not in `42c903493`; content-equal to `e46459915` |
| `jukebox_song.json` | not in `42c903493` or `e46459915`; added by `142296a9b` (`feat: 1.21.9-1.21.11 support`) |
| `vein_golden_values.json` | **not Pumpkin at all** — a local golden-value fixture consumed only by `tools/check_density_functions.py` |
| `entity_tracking.json` | **local**, a side table; see below |

The last two rows are the reason this section exists. `meta_data_type.json`
and `tracked_data.json` are the two tables the entity-metadata layer reads
(the wire type id per entry, and the entry index per data key), so a
regeneration that follows the old "everything came from `42c903493`"
instruction would delete exactly the files the entity layer depends on and
leave the rest of the registry green.

`entity_tracking.json` fills a genuine hole rather than copying a table:
`client_tracking_range` / `update_interval` do not exist in any 774-era
Pumpkin commit — they first appear on the 26.3 line — so they were copied by
**entity name** (never by id, the two lists are not aligned) from
`HEAD:assets/entities.json` into a local side table. The ids in
`entities.json` still come from `42c903493`. `tools/check_entity_tracking.py`
asserts all 157 names are covered and that the values sit in the known set,
so a missing entry is a loud exit 1 rather than an entity that quietly never
despawns. `update_interval` is loaded but unused this round: with only
players in the world, movement is driven by the client's own packets.


## Lifecycle

```
start()
  ├─ init_registries()          ← must happen before ANY world access
  │    ├─ vanilla data          blocks, items, recipes, tags, synced, tables
  │    ├─ vanilla cross-check   synced view vs server-side tables
  │    ├─ datapacks             <datapacks-dir>/*/data/<ns>/...
  │    ├─ plugins               <plugins-dir>/*.py, plus init_registries(plugins=[...])
  │    ├─ worldgen views        typed presets/dimensions/biomes/carvers
  │    ├─ validation            every invariant, collected into one list
  │    └─ freeze                ids become read-only here
  └─ _init_world()              spawn generation, region loading
```

`registry/loader.py` owns that order; `core/server.py::_init_registry` calls it.
Anything that reads a registry without booting first (a tool, a test) goes
through `get_game_registry()`, which loads lazily and logs a warning.

### Why the freeze matters

A block state id is written into every chunk section palette, and a synced
registry index is written into every tag entry sent at login. Once either has
left the process, appending an entry would renumber data that is already on disk
or on a client. So `freeze()` refuses structural change (`RegistryFrozenError`),
while value-only changes stay legal:

| Legal after freeze | Refused after freeze |
| --- | --- |
| `blocks.modify_block`, `set_luminance` | `blocks.register_block` |
| `items.modify_item` | `items.register_item` |
| `recipes.replace_recipe` | `recipes.add_recipe`, `remove_recipe` |
| `synced.replace_entry` | `synced.add_entry` |
| `tags.replace_tag` (existing tag) | `tags.create_tag`, `add_entry` |

## Using it

```python
from registry import get_game_registry

registry = get_game_registry()

registry.blocks.state_of("minecraft:oak_log", {"axis": "y"}).id   # 137
registry.blocks.state(279).properties        # {'distance': '7', 'persistent': 'false', ...}
registry.blocks.find_states(axis="y")        # every y-axis state
registry.items.item("minecraft:stone").max_stack_size
registry.recipes.smelting_result("minecraft:potato").result
registry.tags.contains("minecraft:block", "minecraft:planks", "minecraft:oak_planks")
registry.tables.biomes.registry.get("minecraft:plains")
registry.synced.numeric_id("minecraft:worldgen/biome", "minecraft:plains")   # 40
```

Registry and entry ids are canonical (`minecraft:...`); bare ids are accepted on
the way in. Anything that returns a numeric id returns the index the client
uses, which is also the number a tag entry must be.

## Writing a plugin

```python
from registry import RegistryPlugin, PluginPhase

class MyPlugin(RegistryPlugin):
    name = "mypack"
    phase = PluginPhase.DEFAULT        # EARLY < DEFAULT < LATE

    def on_registries(self, ctx):
        # Data components on an existing item.
        ctx.items.modify_item("minecraft:stick", **{"minecraft:max_stack_size": 16})

        # A new server-side block. `minecraft:block` is NOT a synchronized
        # registry in 1.21.11, so a vanilla client has no id for a new block:
        # `client_block` is the vanilla state that gets drawn in its place.
        prop = ctx.add_block_property("charge", "int", min_value=0, max_value=3)
        block = ctx.add_block(
            "mypack:charger",
            properties=(prop,),
            default_values={"charge": "2"},
            item="mypack:charger",
            hardness=3.5,
            luminance=7,
            client_block="minecraft:furnace",
            client_properties={"facing": "north", "lit": "false"},
        )
        assert ctx.registry.blocks.client_state(block).block.name == "minecraft:furnace"

        # Item, recipe, tag entry, synced entry, translation.
        ctx.add_item("mypack:charger", {"minecraft:max_stack_size": 16})
        ctx.add_recipe(
            {
                "type": "minecraft:stonecutting",
                "ingredient": "minecraft:stone",
                "result": {"id": "mypack:charger", "count": 1},
            },
            "mypack:charger_from_stone",
        )
        ctx.add_tag_entry("minecraft:item", "minecraft:planks", "mypack:charger")
        ctx.add_synced_entry("minecraft:worldgen/biome", "mypack:void", {...})
        ctx.add_translation("item.mypack.charger", "Charger")
        ctx.note("mypack loaded")
```

Register it either by passing it in
(`init_registries(plugins=[MyPlugin()])`) or by dropping the module into the
plugins directory (`plugins/` by default, `server.plugins-dir` in
`config.toml`). A module may expose `PLUGIN`, `plugins`, `register(registry)` or
`setup(context)`.

Rules the API enforces, and why:

* **Append-only.** Entry order *is* the numeric id, so inserts would renumber
  everything after them. `register_item`/`add_recipe`/`add_entry` only append.
* **Registration before the freeze.** A plugin runs during `init_registries()`;
  that is the only window in which ids may be allocated.
* **Failure is isolated.** A plugin that raises is logged with its traceback and
  skipped; the server keeps booting. `strict=True` (used by the check tool and
  available to `init_registries`) turns that into a hard failure.
* **Deterministic order.** Plugins run in `(phase, registration order)`, so two
  runs of the same setup produce the same registry.

### What a plugin cannot do, and why

`minecraft:block` and `minecraft:item` are not synchronized registries in
1.21.11. The client resolves them from its own `minecraft:core` pack, so a new
block or item exists on the server but has no client-side identity; blocks are
drawn as their `client_block` stand-in, and items only work in server-side data
(drops, recipes, inventories). Showing them requires a resource pack or a modded
client.

## Datapacks

The vanilla layout works, applied before plugins:

```
<datapacks-dir>/<pack>/data/<namespace>/<registry path>/<entry>.json
<datapacks-dir>/<pack>/data/<namespace>/tags/<registry path>/<tag>.json
```

`<registry path>` is a registry id without its namespace: `worldgen/biome`,
`damage_type`, `enchantment`, `chat_type`, ... An entry whose name already
exists replaces that entry's payload in place; a new name is appended. Tags
honor `{"replace": bool, "values": [...]}`. To extend a *vanilla* tag, write it
under the `minecraft` namespace (`data/minecraft/tags/block/planks.json`).

## How the world uses it

`world/chunk.py` stores one **global block state id** per cell - the same number
the client is sent in a chunk palette, read from `assets/blocks.json` through
`registry.blocks`. `world/block_states.py` is the only module that turns a block
name into one. So:

* the chunk's own numbering *is* the registry's: there is no second table to keep
  in step, and no id space to allocate at boot;
* region files store `{Name, Properties}` (see `world/anvil.py`), so a save
  names its blocks instead of depending on the block data that wrote it;
* a plugin block gets the state ids the registry hands it when it is registered,
  which is why there is no "internal id" left to collide over.

`tools/check_registry.py` still asserts the 32 hand-checked name/state-id pairs
the old table carried, now as a pin on `assets/blocks.json`.

## Worldgen data

`registry.worldgen` is the second half of that bridge, and the one the world
generator reads:

```python
worldgen = get_game_registry().worldgen

worldgen.settings("overworld").sea_level          # 63
worldgen.settings("overworld").shape.min_y        # -64
worldgen.settings("overworld").router("final_density")   # the resolved DF tree
worldgen.settings_for_dimension("minecraft:the_nether").default_block.name  # netherrack
worldgen.dimension("minecraft:the_nether").height # 256 (its terrain shape is 128)
worldgen.biome("minecraft:plains").carvers        # ('minecraft:cave', ...)
worldgen.biome_id("minecraft:plains")             # 40, the synced index
worldgen.multi_noise("overworld").biome_names()   # the climate tree's biomes
```

The views are built once, after plugins and before the freeze, and they are
cross-checked against each other by `validate()` - every router slot resolves,
every `noise` reference is in `noise_parameters.json`, every biome's carvers and
features exist, every surface rule's block and biome resolves, every dimension's
preset exists, and every value in `chunk.json` is a registered state id. The
whole walk costs ~60ms, so it runs on every boot and its findings land in
`GameRegistry.problems`.

Three shapes in the data are worth knowing before reading the module:

* **`density_function.json` is resolved, not a registry.** It holds one 15-slot
  router per preset with the `minecraft:overworld/continents` style references
  already inlined, so a slot is a tree rather than a name. Its slot names are
  Pumpkin's struct fields (`finalDensity`); `NOISE_ROUTER_SLOTS` maps them to
  vanilla's (`final_density`).
* **It is also not strict JSON** - splines carry `Infinity`/`-Infinity`
  literals, which `json.load` accepts and a strict parser rejects.
* **The generator shape is not the dimension height.** The nether is a 256-tall
  dimension whose terrain shape is 128 and whose logical height is 128.

## Verification

```
python tools/check_registry.py          # 140 checks: data, decoding, ids, wire, plugins, datapacks
python tools/check_worldgen_data.py     # worldgen views, cross-references, world wiring
python tools/check_chunk_codec.py       # chunk section encoding still matches
python tools/check_chunk_storage.py     # chunk/region storage names its blocks
python tools/capture_registry_baseline.py   # re-capture the wire baseline
```

`tools/baselines/registry_wire_1_21_11.json` was captured from the code before
the registry rework, and `check_registry.py` compares against it: every registry
and every entry must still be sent, in the same order, with byte-identical NBT.
It also reconstructs the old tag resolution rule to prove the tag payload only
grew.

## Known gaps

* **`minecraft:point_of_interest_type` tags are not sent.** Pumpkin ships no POI
  registry, so the ids those 30 entries need cannot be derived from the data we
  have. The registry leaves that registry out of the Update Tags packet rather
  than sending it empty - an absent registry keeps the client's built-in tags,
  an empty one deletes them.
* **Items and blocks are not synchronized** (see above).
* **Recipes are not sent to the client.** The `update_recipes` packet (0x83) is
  not implemented; the registry keeps recipes in the client's order so that
  packet can be added without renumbering.
* **`recipe_remainder`, `fuels`, `spawn_egg` and `composter_increase_chance` are
  keyed by numeric item id** in Pumpkin's data and are exposed as raw tables;
  nothing consumes them yet.
* **The worldgen data is wired but not evaluated.** `registry.worldgen` hands
  the generator its preset (sea level, default block/fluid, terrain shape) and
  the world's dimensions now come from `dimension.json`, but nothing walks the
  density-function trees yet: terrain is still `world/noise.py`'s Perlin field.
  The trees, the noise parameter list and `density_function_tests.json` are
  loaded and validated for the interpreter that replaces it.
* **Biomes are not placed.** `multi_noise_biome_tree.json` and the per-biome
  carvers/features are validated and queryable, but a chunk still encodes one
  biome per section (`world/chunk_codec.py`), because chunk storage has no biome
  data yet.
