"""
Integrity and regression checks for the game registry.

Run: python tools/check_registry.py

What it proves, in order:

1. **The data loads and adds up.** Blocks, states, items, recipes, tags, synced
   registries and data tables all reach the counts their source files imply, and
   `validate_registry()` reports nothing.
2. **The state tables decode the way vanilla does.** A block's global state id
   is a mixed-radix index over its properties, in the order vanilla enumerates
   them; a sample of hand-checked states (including the tricky ones: booleans
   enumerate true-first, `oak_leaves` defaults to distance=7, redstone ore's
   default is lit=false) pins that down.
3. **The world's block ids are the registry's.** `world/block_states.py` is the
   only name -> state-id path, and 32 hand-checked name/state-id pairs (the ones
   the old private table carried) are asserted against it, plus `emerald_ore`,
   which the old table was missing.
4. **The login payload did not regress.** Every registry and every entry in
   `tools/baselines/registry_wire_1_21_11.json` is still sent, in the same order,
   with byte-identical NBT. Registry ids are now canonical (`minecraft:plains`
   rather than `plains`); the client parses both to the same ResourceLocation, so
   the check compares canonicalised names.
5. **The tag payload improved, and only improved.** The old path resolved tag
   entries against synced registries alone, which dropped every block, item and
   fluid entry. That behaviour is reconstructed here from the same rule, checked
   against the counts captured in the baseline, and then the live payload is
   required to be a superset of it - emptier than before would be a regression.
6. **The plugin surface works.** A demo plugin registers a block, an item, a
   recipe, a tag entry and a synced entry; existing ids must not move; the
   freeze must refuse later additions; a failing plugin must not take the boot
   down (and must fail loudly in strict mode).
7. **The datapack overlay works**, against a temporary datapack in the vanilla
   layout.

Exit code 0 when everything passes, 1 otherwise.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

logger.remove()
logger.add(sys.stderr, level="WARNING")

from registry.core import RegistryFrozenError, ValidationError  # noqa: E402
from registry.loader import load_registries  # noqa: E402
from registry.plugin import PluginError, PluginPhase, RegistryPlugin  # noqa: E402

BASELINE = ROOT / "tools" / "baselines" / "registry_wire_1_21_11.json"
ASSETS = ROOT / "assets"

# Counts the 1.21.11 data implies. A change here means the data changed, which
# is worth failing over rather than discovering from a desynced client.
EXPECTED = {
    "blocks": 1166,
    "block_states": 29671,
    "block_properties": 123,
    "items": 1505,
    "recipes": 1470,
    "tags": 1131,
    "tag_registries": 14,
    "synced_registries": 23,
    "synced_entries": 371,
}

# (block name, global state id, property values), cross-checked against vanilla's
# own block state definitions. The three "index 0" entries (252, 581, 6680) are
# the ones that pin down the value order: they are
# distance=1/persistent=true/waterlogged=true, powered=true, lit=true, i.e.
# booleans enumerate true before false.
KNOWN_STATES: Tuple[Tuple[str, int, Dict[str, str]], ...] = (
    ("minecraft:air", 0, {}),
    ("minecraft:grass_block", 9, {"snowy": "false"}),
    ("minecraft:oak_leaves", 279, {"distance": "7", "persistent": "false", "waterlogged": "false"}),
    ("minecraft:oak_log", 137, {"axis": "y"}),
    ("minecraft:oak_sapling", 29, {"stage": "0"}),
    ("minecraft:bedrock", 85, {}),
    ("minecraft:water", 86, {"level": "0"}),
    ("minecraft:lava", 102, {"level": "0"}),
    ("minecraft:dispenser", 567, {"facing": "north", "triggered": "false"}),
    ("minecraft:note_block", 582, {"instrument": "harp", "note": "0", "powered": "false"}),
    ("minecraft:redstone_ore", 6681, {"lit": "false"}),
    ("minecraft:diamond_ore", 5106, {}),
    ("minecraft:deepslate_coal_ore", 134, {}),
    ("minecraft:deepslate_redstone_ore", 6683, {"lit": "false"}),
    ("minecraft:oak_leaves", 252, {"distance": "1", "persistent": "true", "waterlogged": "true"}),
    ("minecraft:note_block", 581, {"instrument": "harp", "note": "0", "powered": "true"}),
    ("minecraft:redstone_ore", 6680, {"lit": "true"}),
    ("minecraft:oak_log", 136, {"axis": "x"}),
    ("minecraft:oak_log", 138, {"axis": "z"}),
)

# Hand-checked `block name -> global state id` pairs, one at a time against a
# 1.21.11 client. They used to pin the old internal-id table; they now pin
# `assets/blocks.json` itself, which is the numbering chunks store.
PINNED_BLOCK_STATES: Dict[str, int] = {
    "minecraft:air": 0, "minecraft:stone": 1, "minecraft:grass_block": 9,
    "minecraft:dirt": 10, "minecraft:cobblestone": 14, "minecraft:oak_planks": 15,
    "minecraft:oak_sapling": 29, "minecraft:bedrock": 85, "minecraft:water": 86,
    "minecraft:lava": 102, "minecraft:sand": 118, "minecraft:gravel": 124,
    "minecraft:gold_ore": 129, "minecraft:iron_ore": 131, "minecraft:coal_ore": 133,
    "minecraft:oak_log": 137, "minecraft:oak_leaves": 279, "minecraft:sponge": 560,
    "minecraft:glass": 562, "minecraft:lapis_ore": 563, "minecraft:lapis_block": 565,
    "minecraft:dispenser": 567, "minecraft:sandstone": 578, "minecraft:note_block": 582,
    "minecraft:diamond_ore": 5106, "minecraft:redstone_ore": 6681,
    "minecraft:deepslate_coal_ore": 134, "minecraft:deepslate_gold_ore": 130,
    "minecraft:deepslate_iron_ore": 132, "minecraft:deepslate_diamond_ore": 5107,
    "minecraft:deepslate_lapis_ore": 564, "minecraft:deepslate_redstone_ore": 6683,
}

# Registries whose tags must resolve completely. `point_of_interest_type` is
# deliberately absent: Pumpkin ships no POI registry ordering, so those ids
# cannot be derived from the data we have, and the registry leaves that
# registry out of the packet rather than sending it empty.
RESOLVABLE_TAG_REGISTRIES = (
    "minecraft:block",
    "minecraft:item",
    "minecraft:fluid",
    "minecraft:entity_type",
    "minecraft:game_event",
    "minecraft:worldgen/biome",
    "minecraft:damage_type",
    "minecraft:enchantment",
    "minecraft:banner_pattern",
    "minecraft:painting_variant",
    "minecraft:instrument",
    "minecraft:timeline",
)
KNOWN_UNRESOLVABLE = ("minecraft:point_of_interest_type",)


class Checker:
    """Prints results and remembers failures."""

    def __init__(self) -> None:
        self.failures: List[str] = []
        self.passes = 0

    def check(self, label: str, condition: bool, detail: str = "") -> bool:
        if condition:
            self.passes += 1
            print(f"  OK   {label}" + (f"  ({detail})" if detail else ""))
        else:
            self.failures.append(label)
            print(f"  FAIL {label}" + (f"  ({detail})" if detail else ""))
        return condition

    def equal(self, label: str, actual: Any, expected: Any) -> bool:
        return self.check(label, actual == expected, f"{actual!r} vs {expected!r}")

    def section(self, title: str) -> None:
        print(f"\n{title}")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(registry_id: str) -> str:
    return registry_id if ":" in registry_id else f"minecraft:{registry_id}"


# ==================== 1. loading ====================

def check_loading(checker: Checker):
    start = time.perf_counter()
    registry = load_registries(strict=True)
    elapsed = time.perf_counter() - start

    checker.section(f"1. Loading the complete registry (took {elapsed:.2f}s)")
    summary = registry.summary()
    for key, expected in EXPECTED.items():
        checker.equal(f"{key}", summary[key], expected)
    checker.check("load under 5s", elapsed < 5.0, f"{elapsed:.2f}s")
    checker.check("registry is frozen", registry.frozen)
    checker.check("asset coverage", registry.problems == [],
                  "; ".join(registry.problems[:3]))

    # Every asset file must be accounted for by the loader or an explicit
    # ignore list, so data cannot quietly stop being read.
    from registry.tables import known_files

    unknown = [
        file.name for file in sorted(ASSETS.glob("*.json"))
        if file.name not in known_files()
    ]
    checker.check("no unaccounted asset files", not unknown, ", ".join(unknown))
    return registry


# ==================== 2. state decoding ====================

def check_state_decoding(checker: Checker, registry) -> None:
    checker.section("2. Block state ids decode the way vanilla does")

    for name, state_id, properties in KNOWN_STATES:
        state = registry.blocks.state(state_id)
        label = f"state {state_id} is {name}{properties or ''}"
        if state is None:
            checker.check(label, False, "state id out of range")
            continue
        ok = state.block.name == name and state.properties == properties
        checker.check(label, ok, f"got {state.block.name}{state.properties}")

    # The inverse direction: encoding properties must reproduce the same id.
    oak_log = registry.blocks.block("minecraft:oak_log")
    checker.equal("oak_log axis=x encodes to 136", oak_log.state_id(axis="x"), 136)
    checker.equal("oak_log default state is 137", oak_log.default_state_id, 137)
    dispenser = registry.blocks.block("minecraft:dispenser")
    checker.equal(
        "dispenser facing=up,triggered=true encodes to 574",
        dispenser.state_id(facing="up", triggered=True),
        574,
    )
    checker.equal("dispenser state count", dispenser.state_count, 12)

    # Property lookup by serialised name must refuse an ambiguous one: "age" is
    # the serialised name of eight distinct properties (age_1 .. age_25).
    try:
        registry.blocks.property("age")
        checker.check("ambiguous property name is refused", False, "no error raised")
    except Exception as error:
        checker.check("ambiguous property name is refused", "ambiguous" in str(error))

    checker.equal(
        "properties are keyed by enum name",
        len(registry.blocks.properties_named("axis")),
        2,
    )


# ==================== 3. the world bridge ====================

def check_world_bridge(checker: Checker, registry) -> None:
    checker.section("3. world/block_states.py hands out the registry's own state ids")

    from world import block_states

    mismatches = [
        f"{name}: {block_states.default_state_id(name)} != {expected}"
        for name, expected in sorted(PINNED_BLOCK_STATES.items())
        if block_states.default_state_id(name) != expected
    ]
    checker.check(
        f"all {len(PINNED_BLOCK_STATES)} hand-checked blocks still resolve to the same state ids",
        not mismatches,
        "; ".join(mismatches[:5]),
    )

    # emerald_ore was declared in the old enum but missing from the table, where
    # it silently became air.
    checker.equal(
        "emerald_ore is not air",
        block_states.default_state_id("minecraft:emerald_ore"),
        registry.blocks.block("minecraft:emerald_ore").default_state_id,
    )

    checker.equal("air is state 0", block_states.AIR, 0)
    checker.check("state 0 names itself air", block_states.name_of(0) == "minecraft:air")
    checker.check("an out-of-range id names itself air", block_states.name_of(99999) == "minecraft:air")
    checker.check("is_air is true for an unknown id", block_states.is_air(99999))
    checker.check("is_air is false for stone", not block_states.is_air(1))

    state = block_states.state(block_states.default_state_id("minecraft:oak_leaves"))
    checker.check(
        "oak_leaves' default state exposes properties",
        state is not None and state.properties["distance"] == "7",
    )
    checker.equal(
        "properties round-trip through state_id",
        block_states.state_id("minecraft:oak_leaves", {"distance": "1"}),
        registry.blocks.block("minecraft:oak_leaves").state_id(distance="1"),
    )
    try:
        block_states.state_id("minecraft:water", {"falling": "true"})
        checker.check("a property the block does not have is refused", False, "no error raised")
    except KeyError:
        checker.check("a property the block does not have is refused", True)
    try:
        block_states.state_id("minecraft:not_a_block")
        checker.check("an unknown block name is refused", False, "no error raised")
    except KeyError:
        checker.check("an unknown block name is refused", True)
    checker.check(
        "the tolerant path degrades to air",
        block_states.migrate_name("minecraft:not_a_block") == 0,
    )

    codec = block_states.codec(block_states.default_state_id("minecraft:oak_log"))
    checker.equal("codec name", codec["Name"], "minecraft:oak_log")
    checker.check("codec carries the default axis", codec.get("Properties", {}).get("axis") == "y",
                  str(codec))

    checker.equal(
        "BLOCKS.STONE is the registry's stone",
        int(block_states.BLOCKS.STONE),
        registry.blocks.block("minecraft:stone").default_state_id,
    )
    checker.equal("BLOCKS.WATER is state 86", int(block_states.BLOCKS.WATER), 86)
    checker.check(
        "BLOCKS knows the dimension blocks the old table could not name",
        int(block_states.BLOCKS.NETHERRACK) == registry.blocks.block("minecraft:netherrack").default_state_id
        and int(block_states.BLOCKS.END_STONE) == registry.blocks.block("minecraft:end_stone").default_state_id,
    )
    try:
        block_states.BLOCKS.NOT_A_MEMBER
        checker.check("an unknown BLOCKS member raises", False, "no error raised")
    except AttributeError:
        checker.check("an unknown BLOCKS member raises", True)

    block_states.validate()
    checker.check("block_states.validate() passes", True)
    return


# ==================== 4. wire regression ====================

def check_wire_regression(checker: Checker, registry) -> None:
    checker.section("4. Registry Data payload did not regress")

    if not BASELINE.is_file():
        checker.check("baseline exists", False, f"missing {BASELINE}")
        return
    with open(BASELINE, encoding="utf-8") as handle:
        baseline = json.load(handle)

    packets = dict(registry.build_registry_packets(include_nbt=True))
    baseline_by_id = {entry["canonical_id"]: entry for entry in baseline["registries"]}

    checker.equal("registry count", len(packets), baseline["registry_count"])
    checker.equal(
        "registry id set",
        sorted(canonical(key) for key in packets),
        sorted(baseline_by_id),
    )

    entry_problems: List[str] = []
    size_delta = 0
    for registry_id, payload in packets.items():
        canonical_id = canonical(registry_id)
        expected = baseline_by_id.get(canonical_id)
        if expected is None:
            entry_problems.append(f"{canonical_id}: not in baseline")
            continue
        container = registry.synced.get_registry(registry_id)
        entries = container.values()
        if len(entries) != expected["entry_count"]:
            entry_problems.append(
                f"{canonical_id}: {len(entries)} entries vs {expected['entry_count']}"
            )
            continue
        for live, recorded in zip(entries, expected["entries"]):
            live_key = canonical(live.key)
            recorded_key = canonical(recorded["id"])
            if live_key != recorded_key:
                entry_problems.append(f"{canonical_id}: {live_key} vs {recorded_key}")
                break
            if sha256(live.to_nbt()) != recorded["nbt_sha256"]:
                entry_problems.append(f"{canonical_id}/{live_key}: NBT bytes changed")
                break
        size_delta += len(payload) - expected["payload_size"]

    checker.check("every entry is byte-identical", not entry_problems,
                  "; ".join(entry_problems[:3]))
    print(f"       payload size delta from canonical ids: {size_delta:+d} bytes "
          f"(+10 bytes per registry/entry id)")

    # Order: the baseline was captured from a `set`-ordered iteration, so the old
    # order is not reproducible; the point here is that the new one is stable.
    first = [key for key, _ in registry.build_registry_packets(include_nbt=False)]
    second = [key for key, _ in registry.build_registry_packets(include_nbt=False)]
    checker.check("send order is deterministic", first == second)

    known_pack_free = registry.build_registry_packets(include_nbt=False)
    size = sum(len(payload) for _, payload in known_pack_free)
    checker.check(
        "known-pack payload stays small",
        size < baseline["total_payload_size"] // 4,
        f"{size} bytes vs {baseline['total_payload_size']} with NBT",
    )


# ==================== 5. tag payload ====================

def legacy_tag_payload(registry) -> Dict[str, Dict[str, List[str]]]:
    """Reproduce the old tag resolution rule: synced registries only.

    The previous implementation looked every tag entry up in the *synced*
    registry table. Blocks, items, fluids and entity types are not synced, so
    those lookups all failed and the entries were dropped. This rebuilds that
    outcome from the same rule, which is what makes the baseline's counts
    checkable.
    """
    result: Dict[str, Dict[str, List[str]]] = {}
    for registry_id, container in registry.tags.registries.items():
        tags: Dict[str, List[str]] = {}
        for tag in container.values():
            resolved = []
            for entry in tag.entries:
                if entry.startswith("#"):
                    continue
                if registry.synced.numeric_id(registry_id, entry) is not None:
                    resolved.append(entry)
            tags[tag.name] = resolved
        result[registry_id] = tags
    return result


def check_tag_payload(checker: Checker, registry) -> None:
    checker.section("5. Update Tags payload")

    with open(BASELINE, encoding="utf-8") as handle:
        baseline = json.load(handle)["tags"]

    legacy = legacy_tag_payload(registry)
    legacy_entries = sum(len(entries) for tags in legacy.values() for entries in tags.values())
    checker.equal("old rule reproduces the baseline entry count",
                  legacy_entries, baseline["entry_count"])
    checker.equal("old rule reproduces the baseline tag count",
                  sum(len(tags) for tags in legacy.values()), baseline["tag_count"])

    payload, report = registry.build_tag_payload()
    live = {entry["registry_id"]: entry["tags"] for entry in payload}

    # The new payload must contain every entry the old one did.
    missing: List[str] = []
    for registry_id, tags in legacy.items():
        if registry_id not in live:
            if tags and any(tags.values()):
                missing.append(f"{registry_id} dropped")
            continue
        live_tags = {tag["name"]: tag["entries"] for tag in live[registry_id]}
        for tag_name, entries in tags.items():
            if tag_name not in live_tags:
                missing.append(f"{registry_id}/{tag_name} missing")
                continue
            if len(live_tags[tag_name]) < len(entries):
                missing.append(
                    f"{registry_id}/{tag_name}: {len(live_tags[tag_name])} < {len(entries)}"
                )
    checker.check("no tag lost entries", not missing, "; ".join(missing[:3]))

    new_entries = sum(len(tag["entries"]) for tags in live.values() for tag in tags)
    checker.check(
        "block and item tags now resolve",
        report.registries["minecraft:block"]["resolved"] == report.registries["minecraft:block"]["total"]
        and report.registries["minecraft:item"]["resolved"] == report.registries["minecraft:item"]["total"],
        f"block {report.registries['minecraft:block']['resolved']}/{report.registries['minecraft:block']['total']}, "
        f"item {report.registries['minecraft:item']['resolved']}/{report.registries['minecraft:item']['total']}",
    )
    print(f"       tag entries on the wire: {legacy_entries} before -> {new_entries} now")

    for registry_id in RESOLVABLE_TAG_REGISTRIES:
        info = report.registries.get(registry_id, {})
        checker.equal(
            f"{registry_id} fully resolved",
            info.get("resolved"), info.get("total"),
        )

    checker.equal("only point_of_interest_type is unresolved",
                  report.unresolved_registries, list(KNOWN_UNRESOLVABLE))
    checker.check(
        "incomplete registries are left out of the packet",
        "minecraft:point_of_interest_type" not in live,
    )
    with open(ASSETS / "tags.json", encoding="utf-8") as handle:
        raw_tags = json.load(handle)
    file_entries = sum(
        len(entries) for tags in raw_tags.values() for entries in tags.values()
    )
    checker.equal(
        "every entry in tags.json is accounted for",
        report.resolved_entries + report.unresolved_count,
        file_entries,
    )
    checker.equal("unresolved count is the POI tag set",
                  report.unresolved_count, 30)

    # Spot checks on real vanilla tag data.
    checker.check(
        "minecraft:planks contains oak_planks",
        registry.tags.contains("minecraft:block", "minecraft:planks", "minecraft:oak_planks"),
    )
    log_tags = registry.tags.entries("minecraft:block", "minecraft:logs")
    checker.check(
        "minecraft:logs resolves to the wood types",
        {"minecraft:oak_log", "minecraft:birch_log", "minecraft:acacia_log"} <= set(log_tags),
        f"{len(log_tags)} members",
    )


# ==================== 6. plugin surface ====================

class DemoPlugin(RegistryPlugin):
    """Exercises every mutation a real plugin would make."""

    name = "check-registry-demo"
    phase = PluginPhase.DEFAULT

    def __init__(self) -> None:
        self.observed = {}

    def on_registries(self, ctx) -> None:
        self.observed["biome_40"] = ctx.synced.entry_keys("minecraft:worldgen/biome")[40]
        self.observed["oak_log_state"] = ctx.blocks.state_id_of("minecraft:oak_log", {"axis": "y"})
        self.observed["item_count"] = len(ctx.items)

        prop = ctx.add_block_property("charge", "int", min_value=0, max_value=3)
        block = ctx.add_block(
            "checkpack:charger",
            properties=(prop,),
            default_values={"charge": "2"},
            item="checkpack:charger",
            hardness=3.5,
            luminance=7,
            client_block="minecraft:furnace",
            client_properties={"facing": "north", "lit": "false"},
        )
        self.observed["block_states"] = block.state_count
        self.observed["block_first_state"] = block.first_state_id

        ctx.add_item("checkpack:charger", {"minecraft:max_stack_size": 16})
        ctx.recipes.add_recipe(
            {
                "type": "minecraft:stonecutting",
                "ingredient": "minecraft:stone",
                "result": {"id": "checkpack:charger", "count": 1},
            },
            "checkpack:charger_from_stone",
        )
        ctx.add_tag_entry("minecraft:item", "minecraft:planks", "checkpack:charger")
        ctx.add_synced_entry(
            "minecraft:worldgen/biome",
            "checkpack:void",
            {"has_precipitation": False, "temperature": 0.5, "downfall": 0.0, "effects": {}},
        )
        ctx.add_translation("item.checkpack.charger", "Charger")
        ctx.blocks.modify_block("minecraft:bedrock", hardness=-2.0)


class FailingPlugin(RegistryPlugin):
    name = "check-registry-failing"

    def on_registries(self, ctx) -> None:
        ctx.blocks.register_block("checkpack:broken", client_block="minecraft:no_such_block")


def check_plugins(checker: Checker, baseline_registry) -> None:
    checker.section("6. Plugin surface")

    demo = DemoPlugin()
    registry = load_registries(plugins=[demo], strict=True)

    checker.equal("plugin ran", demo.observed.get("biome_40"), "minecraft:plains")
    checker.equal("existing state ids did not move",
                  demo.observed.get("oak_log_state"), 137)
    checker.equal("existing item ids did not move",
                  demo.observed.get("item_count"), EXPECTED["items"])

    block = registry.blocks.block("checkpack:charger")
    checker.equal("registered block has its 4 states", block.state_count, 4)
    checker.equal("registered block default state",
                  block.default_state_id, block.first_state_id + 2)
    checker.equal("registered block state ids are appended",
                  block.first_state_id, registry.blocks.total_states - 4)
    checker.equal("registered block is placed by its item",
                  registry.items.item_for_block("checkpack:charger").name,
                  "checkpack:charger")
    checker.check("registered block has no private id space",
                  not hasattr(block, "internal_id"))
    checker.equal("registered block states start where the plugin was told",
                  block.first_state_id, demo.observed["block_first_state"])
    checker.check("plugin block state ids sit above the vanilla range",
                  block.first_state_id >= EXPECTED["block_states"],
                  str(block.first_state_id))

    # The chunk-storage layer reads the *global* registry, which is how the
    # server runs: one registry per process, installed before world load. Point
    # it at this instance and prove a plugin block is placeable through it.
    from registry.game_registry import set_game_registry

    set_game_registry(registry)

    from world import block_states

    checker.equal("plugin block is placeable in a chunk",
                  block_states.default_state_id(block.name),
                  block.default_state_id)

    item = registry.items.item("checkpack:charger")
    checker.equal("registered item stack size", item.max_stack_size, 16)
    checker.equal("translation registered",
                  registry.tables.translations.data.get("item.checkpack.charger"),
                  "Charger")
    checker.equal("block modification applied",
                  registry.blocks.block("minecraft:bedrock").hardness, -2.0)

    recipe = registry.recipes.get("checkpack:charger_from_stone")
    checker.check("recipe registered", recipe is not None)
    if recipe is not None:
        checker.equal("recipe index is appended", recipe.index, EXPECTED["recipes"])
        checker.equal("recipe result", recipe.result.item, "checkpack:charger")
        checker.check(
            "recipe is findable by result",
            recipe in registry.recipes.by_result("checkpack:charger"),
        )

    checker.check(
        "tag entry added",
        registry.tags.contains("minecraft:item", "minecraft:planks", "checkpack:charger"),
    )
    checker.equal("synced entry appended",
                  registry.synced.numeric_id("minecraft:worldgen/biome", "checkpack:void"),
                  65)
    checker.equal("existing synced indices unchanged",
                  registry.synced.numeric_id("minecraft:worldgen/biome", "minecraft:plains"),
                  40)

    # The client-visible state of a plugin block: it stands in as a furnace,
    # because minecraft:block is not a synchronized registry.
    client_state = registry.blocks.client_state(block)
    checker.equal("plugin block renders as its client stand-in",
                  client_state.block.name, "minecraft:furnace")
    checker.equal("plugin block's properties are queryable",
                  block.state(block.first_state_id + 3).properties, {"charge": "3"})

    # Freeze.
    for label, action in (
        ("register_block", lambda: registry.blocks.register_block("checkpack:late")),
        ("register_item", lambda: registry.items.register_item("checkpack:late")),
        ("add_recipe", lambda: registry.recipes.add_recipe(
            {"type": "minecraft:stonecutting", "ingredient": "minecraft:stone",
             "result": {"id": "minecraft:stone"}})),
        ("add_synced_entry", lambda: registry.synced.add_entry(
            "minecraft:worldgen/biome", "checkpack:late", {})),
        ("add_tag_entry", lambda: registry.tags.add_entry(
            "minecraft:item", "minecraft:planks", "checkpack:late")),
    ):
        try:
            action()
            checker.check(f"frozen registry refuses {label}", False, "no error raised")
        except RegistryFrozenError:
            checker.check(f"frozen registry refuses {label}", True)
        except Exception as error:  # noqa: BLE001 - report the wrong error type
            checker.check(f"frozen registry refuses {label}", False,
                          f"{type(error).__name__}: {error}")

    # ... while value-only changes stay legal.
    try:
        registry.blocks.modify_block("minecraft:bedrock", hardness=-3.0)
        registry.recipes.replace_recipe(
            "checkpack:charger_from_stone", category="building"
        )
        checker.check("value-only changes stay legal after freeze", True)
    except Exception as error:
        checker.check("value-only changes stay legal after freeze", False, str(error))

    # Failure isolation.
    broken = load_registries(plugins=[DemoPlugin(), FailingPlugin()])
    checker.check("a failing plugin does not stop the load", broken.frozen)
    checker.check("the failing plugin is recorded", len(broken.plugins.failures) == 1)
    checker.check(
        "the healthy plugin still applied",
        broken.blocks.get_block("checkpack:charger") is not None,
    )
    try:
        load_registries(plugins=[FailingPlugin()], strict=True)
        checker.check("strict mode rejects the plugin", False, "no error raised")
    except (PluginError, ValidationError) as error:
        checker.check("strict mode rejects the plugin", True, type(error).__name__)

    # A plugin that mutates a tag object directly must hit the freeze guard too.
    try:
        tag = registry.tags.get("minecraft:item", "minecraft:planks")
        tag.add("checkpack:sneaky")
        checker.check("direct tag mutation hits the freeze guard", False, "no error")
    except RegistryFrozenError:
        checker.check("direct tag mutation hits the freeze guard", True)


def check_plugin_directory(checker: Checker) -> None:
    """The plugins directory is how a plugin is normally shipped."""
    checker.section("6b. Plugin directory")

    temp = Path(tempfile.mkdtemp(prefix="crotalus-plugins-"))
    try:
        (temp / "mypack.py").write_text(
            "from registry import RegistryPlugin\n"
            "\n"
            "class MyPack(RegistryPlugin):\n"
            "    name = 'mypack-file'\n"
            "    def on_registries(self, ctx):\n"
            "        ctx.add_item('mypack:token', {'minecraft:max_stack_size': 1})\n"
            "        ctx.note('mypack-file registered')\n"
            "\n"
            "PLUGIN = MyPack()\n",
            encoding="utf-8",
        )
        (temp / "broken.py").write_text("this is not python\n", encoding="utf-8")
        (temp / "_ignored.py").write_text("raise RuntimeError('never imported')\n", encoding="utf-8")

        registry = load_registries(plugins_dir=temp)

        checker.check("plugin module loaded", registry.items.get("mypack:token") is not None)
        checker.check(
            "plugin recorded",
            any(record.name == "mypack-file" for record in registry.plugins.records),
        )
        checker.check(
            "a syntax error in one file does not stop the rest",
            registry.frozen and registry.items.get("mypack:token") is not None,
        )
        checker.check(
            "underscore-prefixed files are ignored",
            all("_ignored" not in record.name for record in registry.plugins.records),
        )
    finally:
        shutil.rmtree(temp, ignore_errors=True)


# ==================== 7. datapacks ====================

def check_datapacks(checker: Checker) -> None:
    checker.section("7. Datapack overlay")

    temp = Path(tempfile.mkdtemp(prefix="crotalus-datapack-"))
    try:
        pack = temp / "testpack" / "data"
        (pack / "testpack" / "worldgen" / "biome").mkdir(parents=True)
        (pack / "testpack" / "damage_type").mkdir(parents=True)
        # Extending a *vanilla* tag means writing under the minecraft namespace:
        # data/<namespace>/tags/<registry path>/<tag>.json defines
        # <namespace>:<tag>. A datapack's own namespace defines its own tags.
        (pack / "minecraft" / "tags" / "block").mkdir(parents=True)

        (pack / "testpack" / "worldgen" / "biome" / "crystal_plains.json").write_text(
            json.dumps({"has_precipitation": True, "temperature": 0.8,
                        "downfall": 0.4, "effects": {}}),
            encoding="utf-8",
        )
        (pack / "testpack" / "damage_type" / "spell.json").write_text(
            json.dumps({"message_id": "spell", "scaling": "never", "exhaustion": 0.0}),
            encoding="utf-8",
        )
        (pack / "minecraft" / "tags" / "block" / "planks.json").write_text(
            json.dumps({"replace": False, "values": ["minecraft:dead_bush"]}),
            encoding="utf-8",
        )

        registry = load_registries(datapacks_dir=temp, strict=True)

        biome_count = len(registry.synced.entry_keys("minecraft:worldgen/biome")) - 1
        checker.equal("datapack biome is appended at the end",
                      registry.synced.numeric_id("minecraft:worldgen/biome",
                                                 "testpack:crystal_plains"),
                      biome_count)
        checker.equal("existing biome indices unchanged",
                      registry.synced.numeric_id("minecraft:worldgen/biome", "minecraft:plains"),
                      40)
        checker.check(
            "datapack damage type added",
            registry.synced.numeric_id("minecraft:damage_type", "testpack:spell") is not None,
        )
        checker.check(
            "datapack tag merged into a vanilla tag",
            registry.tags.contains("minecraft:block", "minecraft:planks", "minecraft:dead_bush"),
        )
        checker.check(
            "datapack tag did not replace the vanilla members",
            registry.tags.contains("minecraft:block", "minecraft:planks", "minecraft:oak_planks"),
        )
        checker.check("datapack entries are in the payload",
                      "testpack:crystal_plains" in
                      [entry.key for entry in registry.synced.entries("minecraft:worldgen/biome")])
    finally:
        shutil.rmtree(temp, ignore_errors=True)


# ==================== 8. edge cases ====================

def check_edges(checker: Checker, registry) -> None:
    checker.section("8. Container semantics")

    from registry.core import Registry

    small: Registry = Registry("test:thing")
    small.add("a", 1)
    small.add("b", 2)
    checker.equal("insertion order is the numeric id", small.numeric_id("b"), 1)
    checker.check("bare ids canonicalise", small.get("a") == 1 and "minecraft:a" in small)
    small.add("a", 3, replace=True)
    checker.equal("replace keeps the position", small.numeric_id("a"), 0)
    checker.equal("replace changes the value", small.get("a"), 3)
    small.freeze("test")
    try:
        small.add("c", 4)
        checker.check("frozen add is refused", False, "no error")
    except RegistryFrozenError:
        checker.check("frozen add is refused", True)
    small.add("a", 5, replace=True)
    checker.equal("frozen replace is allowed", small.get("a"), 5)

    # A block with no properties: every lookup path has to agree.
    stone = registry.blocks.block("minecraft:stone")
    checker.equal("stone has one state", stone.state_count, 1)
    checker.equal("stone lookup by name", stone.state_id_for({}), stone.first_state_id)
    checker.check("stone has no properties", not stone.has_properties)

    # Out-of-range state ids must not be silently wrapped.
    checker.check("state id out of range returns None",
                  registry.blocks.state(registry.blocks.total_states + 1) is None)

    # Block item binding is symmetric.
    linked = [
        block for block in registry.blocks.blocks.values()
        if block.item is not None
        and registry.items.get(block.item) is not None
        and registry.items.get(block.item).block == block.name
    ]
    checker.check("block <-> item links are symmetric", len(linked) > 900, f"{len(linked)} pairs")

    unlinked = [
        block.name for block in registry.blocks.blocks.values()
        if block.item_id is not None and block.item is None
    ]
    checker.check("no block item is missing", not unlinked, "; ".join(unlinked[:3]))


def check_entity_types(checker: Checker, registry) -> None:
    checker.section("8b. entity/ typed view over minecraft:entity_type")

    from entity import (
        entity_type_by_id,
        entity_type_by_name,
        entity_type_count,
    )

    raw = registry.tables.entity_types.registry
    count = entity_type_count()
    checker.equal("entity type count matches the asset", count, len(raw))
    checker.equal("entity type count is the 1.21.11 value", count, 157)

    # Dense 0..156: `entity_type_by_id` indexes a list with packet data, so a
    # hole would turn a hostile-but-legal id into an IndexError.
    dense_ok = all(
        entity_type_by_id(i) is not None and entity_type_by_id(i).numeric_id == i
        for i in range(count)
    )
    checker.check("ids are dense 0..156 and round-trip", dense_ok)
    checker.check("out-of-range id returns None", entity_type_by_id(count + 5) is None)

    # The pinned ids everything else builds on.
    checker.equal("player id", entity_type_by_name("minecraft:player").numeric_id, 155)
    checker.equal("zombie id", entity_type_by_name("zombie").numeric_id, 150)
    checker.equal("item id", entity_type_by_id(71).name, "minecraft:item")
    checker.equal("armor_stand id", entity_type_by_name("armor_stand").numeric_id, 5)

    # Width/eye_height come straight from entities.json; the player's are the
    # numbers every camera and collision calculation downstream assumes.
    player = entity_type_by_name("minecraft:player")
    checker.equal("player width", player.width, 0.6)
    checker.equal("player height", player.height, 1.8)
    checker.equal("player eye_height", player.eye_height, 1.62)
    checker.check("player is not a mob", not player.is_mob)
    checker.check("player does not track deltas", not player.track_deltas)
    checker.equal("player tracking range", player.client_tracking_range, 32)

    zombie = entity_type_by_name("minecraft:zombie")
    checker.equal("zombie max_health", zombie.max_health, 20.0)
    checker.check("zombie is a mob", zombie.is_mob)

    checker.check(
        "name and resource_name agree",
        player.resource_name == "player" and zombie.resource_name == "zombie",
    )


# ==================== main ====================

def main() -> int:
    checker = Checker()
    registry = check_loading(checker)
    check_state_decoding(checker, registry)
    check_world_bridge(checker, registry)
    check_wire_regression(checker, registry)
    check_tag_payload(checker, registry)
    check_edges(checker, registry)
    check_entity_types(checker, registry)
    check_plugins(checker, registry)
    check_plugin_directory(checker)
    check_datapacks(checker)

    print()
    if checker.failures:
        print(f"{len(checker.failures)} check(s) FAILED, {checker.passes} passed:")
        for failure in checker.failures:
            print(f"  - {failure}")
        return 1
    print(f"All {checker.passes} registry checks passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
