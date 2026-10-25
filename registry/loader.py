"""
Registry loader - builds a `GameRegistry` from `assets/`.

Order of operations, and why it is that order:

1. **Required files.** A missing `blocks.json` is a boot failure with a clear
   message, not a server that starts and then sends the wrong state ids.
2. **Vanilla data.** Blocks (and their properties), items, recipes, tags,
   synced registries, then every auxiliary data table. Tags load before they
   are resolved - resolution happens per login, against whatever plugins and
   datapacks added.
3. **Cross-links.** Block items are bound in both directions once both sides
   exist, rather than making either registry depend on load order.
4. **Datapacks** (`<dir>/*/data/<namespace>/...`): the vanilla layout, applied
   as an overlay on synced registries and tags.
5. **Plugins**: code, run against the complete vanilla + datapack data.
6. **Validation**: everything is checked while it is still legal to complain,
   and the problems are logged (or raised, in strict mode).
7. **Freeze**: from here on ids are read-only, and the world can be loaded.

Steps 4 and 5 both need the registry to be open, which is why the freeze is the
last thing that happens.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from loguru import logger

from registry.core import ValidationError, canonical_id
from registry.game_registry import DEFAULT_ASSETS_DIR, GameRegistry

#: Without these the registry would be silently incomplete, so they are fatal.
REQUIRED_FILES: Tuple[str, ...] = (
    "blocks.json",
    "properties.json",
    "items.json",
    "recipes.json",
    "tags.json",
    "synced_registries.json",
)

#: Synced registries whose server-side detail lives in another asset file. The
#: two views must have identical key sets, so they are cross-checked.
TYPED_SYNCED_TABLES: Dict[str, str] = {
    "minecraft:worldgen/biome": "biomes",
    "minecraft:dimension_type": "dimensions",
    "minecraft:damage_type": "damage_types",
    "minecraft:enchantment": "enchantments",
    "minecraft:jukebox_song": "jukebox_songs",
    "minecraft:chat_type": "message_types",
}


class AssetReader:
    """Cached, timed access to the `assets/` directory."""

    def __init__(self, assets_dir: Path) -> None:
        self.assets_dir = Path(assets_dir)
        self._cache: Dict[str, Any] = {}
        self.total_bytes = 0

    def path(self, filename: str) -> Path:
        return self.assets_dir / filename

    def has(self, filename: str) -> bool:
        return self.path(filename).is_file()

    def files(self) -> List[str]:
        """Every `*.json` in the assets directory, by name."""
        if not self.assets_dir.is_dir():
            return []
        return sorted(file.name for file in self.assets_dir.glob("*.json"))

    def json(self, filename: str) -> Any:
        """Read (and cache) one asset file."""
        cached = self._cache.get(filename)
        if cached is not None:
            return cached
        target = self.path(filename)
        if not target.is_file():
            raise FileNotFoundError(
                f"registry asset {filename} is missing from {self.assets_dir}"
            )
        with open(target, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self._cache[filename] = payload
        self.total_bytes += target.stat().st_size
        return payload


def load_registries(
    assets_dir: Optional[Path] = None,
    *,
    plugins: Iterable[Any] = (),
    plugins_dir: Optional[Path] = None,
    datapacks_dir: Optional[Path] = None,
    strict: bool = False,
    log_summary: bool = True,
) -> GameRegistry:
    """Load a complete, frozen `GameRegistry`.

    Args:
        assets_dir: Directory holding the Pumpkin data; defaults to `assets/`
            next to the package.
        plugins: Plugin objects or callables to apply before the freeze.
        plugins_dir: Directory of `*.py` plugin modules to import as well.
        datapacks_dir: Directory of datapacks to overlay, in name order.
        strict: Raise on validation problems and plugin failures instead of
            logging them. Used by `tools/check_registry.py`.
        log_summary: Emit the one-line summary and per-plugin outcome lines.

    Returns:
        A frozen `GameRegistry`. Use `init_registries()` to also install it as
        the process-wide instance.

    Raises:
        FileNotFoundError: A required asset file is missing.
        ValidationError: `strict=True` and something did not check out.
    """
    start = time.perf_counter()
    registry = GameRegistry(assets_dir or DEFAULT_ASSETS_DIR)
    reader = AssetReader(registry.assets_dir)

    _check_required(reader)

    timings = registry.timings

    def timed(name: str, function) -> Any:
        step_start = time.perf_counter()
        result = function()
        timings[name] = round((time.perf_counter() - step_start) * 1000, 1)
        return result

    timed("blocks", lambda: registry.blocks.load(
        reader.json("blocks.json"), reader.json("properties.json")
    ))
    timed("items", lambda: registry.items.load(
        reader.json("items.json"), reader.json("data_component.json")
    ))
    timed("recipes", lambda: registry.recipes.load(reader.json("recipes.json")))
    timed("tags", lambda: registry.tags.load(reader.json("tags.json")))
    timed("synced", lambda: registry.synced.load(reader.json("synced_registries.json")))
    timed("tables", lambda: registry.tables.load(registry.assets_dir, reader.json))

    # Vanilla purity: the synced view and the server-side detail for the same
    # registries must agree entry for entry. Checked here, before datapacks and
    # plugins, because both are *allowed* to add a synced entry without a local
    # data file - that is what `add_synced_entry` is for.
    crosscheck = timed("crosscheck", lambda: _check_typed_synced(registry))

    datapacks = 0
    if datapacks_dir is not None:
        datapacks = timed(
            "datapacks", lambda: load_datapacks(registry, Path(datapacks_dir), strict=strict)
        )

    manager = registry.plugins
    manager.strict = strict
    for plugin in plugins or ():
        manager.register(plugin)
    if plugins_dir is not None:
        manager.load_directory(Path(plugins_dir))
    context = timed("plugins", lambda: manager.apply(registry))

    # Link blocks and items last: either side may have been registered by a
    # plugin, and the link is a property of the pair, not of load order.
    timed("links", lambda: _link(registry))

    # Worldgen views: typed, indexed, cross-checked. Built here - after plugins
    # (a plugin may add a synced biome) and before the freeze - so the world
    # never has to parse or re-index a raw payload at generation time.
    registry.worldgen = timed("worldgen", lambda: _build_worldgen(registry))

    problems = timed("validate", lambda: validate_registry(registry, strict=False))
    registry.problems = crosscheck + problems

    if strict and registry.problems:
        raise ValidationError(
            f"registry validation failed with {len(registry.problems)} problem(s):\n  "
            + "\n  ".join(registry.problems[:20])
        )

    registry.freeze(
        "world generation started; ids are stored in chunks and sent to clients"
    )
    manager.notify_frozen(registry, context)

    registry.load_seconds = time.perf_counter() - start
    registry.loaded_at = time.time()

    if log_summary:
        _log_summary(registry, reader, datapacks)

    return registry


def _check_required(reader: AssetReader) -> None:
    missing = [name for name in REQUIRED_FILES if not reader.has(name)]
    if missing:
        raise FileNotFoundError(
            f"registry assets missing from {reader.assets_dir}: {', '.join(missing)}. "
            f"These ship with the server (Pumpkin's 1.21.11 data); restore them "
            f"before starting."
        )


def _link(registry: GameRegistry) -> None:
    """Bind block items in both directions.

    Runs after plugins so a plugin that registers a block and its item gets the
    link too; `BlockRegistry.link_items` sets both `block.item` and `item.block`
    from the numeric id the block data records.
    """
    registry.blocks.link_items(registry.items)
    registry.items.link_blocks(registry.blocks)


def _build_worldgen(registry: GameRegistry):
    """Build `registry.worldgen`, imported lazily to keep the loader's imports flat."""
    from registry.worldgen import build_worldgen

    return build_worldgen(registry)


def _log_summary(registry: GameRegistry, reader: AssetReader, datapacks: int) -> None:
    summary = registry.summary()
    logger.info(
        "Game registry loaded in {seconds:.2f}s: {blocks} blocks ({block_states} states), "
        "{items} items, {recipes} recipes, {tags} tags in {tag_registries} registries, "
        "{synced_registries} synced registries ({synced_entries} entries), "
        "{data_tables} data tables{datapacks}".format(
            seconds=summary["load_seconds"],
            datapacks=f", {datapacks} datapack registries" if datapacks else "",
            **summary,
        )
    )
    logger.debug(
        "Registry step timings (ms): "
        + ", ".join(f"{name}={value}" for name, value in registry.timings.items())
    )

    records = registry.plugins.records
    if records:
        logger.info(f"Registry plugins: {len(records)} registered")
        for line in registry.plugins.summary():
            if "FAILED" in line:
                logger.error(line.strip())
            else:
                logger.info(line)

    if registry.problems:
        logger.warning(
            f"Registry validation reported {len(registry.problems)} problem(s); "
            f"first: {registry.problems[0]}"
        )


# ==================== Validation ====================

def validate_registry(registry: GameRegistry, strict: bool = False) -> List[str]:
    """Run every integrity check that does not need a world.

    Covers blocks, items, recipes, tags, asset coverage and (when it has been
    built) the worldgen cross-references. The vanilla synced/typed cross-check is
    *not* included: it only holds for pristine data, and `load_registries` runs it
    before datapacks and plugins get a chance to add a synced entry legitimately.

    Returns the list of problems. In strict mode the first one is raised as a
    `ValidationError` after the whole list has been collected, so a check tool
    sees every problem in one run instead of one per fix.
    """
    problems: List[str] = []

    problems.extend(f"blocks: {problem}" for problem in registry.blocks.validate())
    problems.extend(f"items: {problem}" for problem in registry.items.validate())

    item_tags = registry.tags.item_lookup()
    problems.extend(
        f"recipes: {problem}" for problem in registry.recipes.validate(registry.items, item_tags)
    )
    problems.extend(f"tags: {problem}" for problem in registry.tags.validate())

    problems.extend(_check_asset_coverage(registry))

    if registry.worldgen is not None:
        problems.extend(f"worldgen: {problem}" for problem in registry.worldgen.validate())

    if strict and problems:
        raise ValidationError(
            f"registry validation failed with {len(problems)} problem(s):\n  "
            + "\n  ".join(problems[:20])
        )
    return problems


def _check_typed_synced(registry: GameRegistry) -> List[str]:
    """Cross-check the synced registries against their server-side tables.

    `worldgen/biome`, `dimension_type`, `damage_type`, `enchantment`,
    `jukebox_song` and `chat_type` are each described twice in `assets/`: once
    as the wire payload the client is sent, once as the richer server-side table
    used by game logic. Identical key sets are the only thing keeping those two
    views from drifting apart, so it is asserted on the pristine load.
    """
    typed = {
        registry_id: registry.tables[table].registry
        for registry_id, table in TYPED_SYNCED_TABLES.items()
        if registry.tables.get(table) is not None
    }
    return [f"synced: {problem}" for problem in registry.synced.validate(typed)]


def _check_asset_coverage(registry: GameRegistry) -> List[str]:
    """Every asset file must be loaded or explicitly ignored.

    Cheap, and it is the only guard against a data file that quietly stops being
    read when the loader changes.
    """
    from registry.tables import known_files

    accounted = known_files()
    problems = []
    for filename in registry.assets_dir.glob("*.json") if registry.assets_dir.is_dir() else ():
        if filename.name not in accounted:
            problems.append(
                f"assets: {filename.name} is not loaded by any registry module and is "
                f"not in registry.tables.IGNORED_FILES"
            )
    return problems


# ==================== Datapack overlay ====================

def load_datapacks(
    registry: GameRegistry,
    datapacks_dir: Path,
    strict: bool = False,
) -> int:
    """Overlay datapacks on the synced registries and tags.

    Layout, matching vanilla:

        <datapack>/data/<namespace>/<registry path>/<entry>.json
        <datapack>/data/<namespace>/tags/<registry path>/<tag>.json

    where `<registry path>` is a registry id without its namespace, e.g.
    `worldgen/biome`, `damage_type`, `enchantment`, or `block` for tags.

    A registry entry with a name that already exists replaces that entry's
    payload in place (its index does not move); a new name is appended. Tag
    files honor vanilla's `{"replace": bool, "values": [...]}`.

    Returns:
        The number of registries that had at least one entry applied.
    """
    if not datapacks_dir.is_dir():
        logger.debug(f"No datapack directory at {datapacks_dir}")
        return 0

    touched: set = set()
    for pack in sorted(datapacks_dir.iterdir()):
        if not pack.is_dir():
            continue
        data_dir = pack / "data"
        if not data_dir.is_dir():
            continue

        for namespace_dir in sorted(data_dir.iterdir()):
            if not namespace_dir.is_dir():
                continue
            namespace = namespace_dir.name
            touched |= _load_datapack_namespace(
                registry, namespace, namespace_dir, strict=strict
            )

    if touched:
        logger.info(
            f"Datapacks in {datapacks_dir} contributed to {len(touched)} registries: "
            f"{', '.join(sorted(touched))}"
        )
    return len(touched)


def _load_datapack_namespace(
    registry: GameRegistry,
    namespace: str,
    namespace_dir: Path,
    strict: bool,
) -> set:
    touched: set = set()

    for registry_id in list(registry.synced.registry_ids()):
        path = registry_id.split(":", 1)[1]
        entry_dir = namespace_dir / path
        if not entry_dir.is_dir():
            continue
        for file in sorted(entry_dir.glob("*.json")):
            entry_id = f"{namespace}:{file.stem}"
            try:
                payload = _read_json(file)
            except (OSError, json.JSONDecodeError) as error:
                logger.error(f"Datapack entry {file} could not be read: {error}")
                if strict:
                    raise ValidationError(f"datapack entry {file} is not valid JSON") from error
                continue
            try:
                if registry.synced.get(registry_id, entry_id) is not None:
                    registry.synced.replace_entry(registry_id, entry_id, payload)
                else:
                    registry.synced.add_entry(registry_id, entry_id, payload)
            except Exception as error:
                logger.error(f"Datapack entry {file} was rejected: {error}")
                if strict:
                    raise
                continue
            touched.add(registry_id.rstrip("/"))
            logger.debug(f"Datapack entry {registry_id}/{entry_id} applied")

    tags_dir = namespace_dir / "tags"
    if tags_dir.is_dir():
        for file in sorted(tags_dir.rglob("*.json")):
            relative = file.relative_to(tags_dir).with_suffix("")
            parts = relative.parts
            if len(parts) < 2:
                logger.warning(f"Datapack tag {file} does not name a registry; skipped")
                continue
            registry_id = canonical_id("/".join(parts[:-1]))
            tag_name = canonical_id(f"{namespace}:{parts[-1]}")
            try:
                payload = _read_json(file)
            except (OSError, json.JSONDecodeError) as error:
                logger.error(f"Datapack tag {file} could not be read: {error}")
                if strict:
                    raise ValidationError(f"datapack tag {file} is not valid JSON") from error
                continue

            values = payload.get("values", []) if isinstance(payload, dict) else payload
            if payload.get("replace") if isinstance(payload, dict) else False:
                registry.tags.replace_tag(registry_id, tag_name, values)
            else:
                registry.tags.add_entries(registry_id, tag_name, values)
            touched.add(f"{registry_id} (tags)")
            logger.debug(f"Datapack tag {registry_id}/{tag_name} applied")

    return touched


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


__all__ = [
    "AssetReader",
    "REQUIRED_FILES",
    "TYPED_SYNCED_TABLES",
    "load_registries",
    "validate_registry",
    "load_datapacks",
]
