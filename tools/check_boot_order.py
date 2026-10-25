"""
Boot-order check: the registry is complete and frozen before the world exists.

The registry has to be loaded before world generation *or* loading, because
every block state id in a chunk palette is resolved from it - a world built
against a half-loaded registry would encode ids that mean nothing. This tool
drives the real startup path (`CrotalusServer._init_registry()` then
`_init_world()`) and asserts that order, rather than trusting the call sites.

It runs in a temporary directory, so it does not touch `data/`.

Run: python tools/check_boot_order.py
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

logger.remove()
logger.add(sys.stderr, level="WARNING")

failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    marker = "OK  " if condition else "FAIL"
    print(f"  {marker} {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(label)


async def run() -> None:
    from core.server import CrotalusServer
    from registry import get_game_registry, is_initialized

    server = CrotalusServer()

    print("1. Before startup")
    check("no registry yet", not is_initialized())
    check("no world yet", server.world is None)

    print("\n2. Registry loads first")
    server._init_registry()
    registry = server.registry
    check("registry installed", registry is not None and is_initialized())
    check("registry is the global instance", registry is get_game_registry())
    check("registry is frozen", registry.frozen)
    check("no world was touched by registry loading", server.world is None)
    check(
        "summary counts",
        registry.summary()["blocks"] == 1166
        and registry.summary()["items"] == 1505
        and registry.summary()["recipes"] == 1470,
        str(registry.summary()["blocks"]),
    )

    # The world's block ids must be the registry's own state ids: there is no
    # private numbering left to translate into.
    from world import block_states

    check("oak_leaves is registry state 279", block_states.default_state_id("minecraft:oak_leaves") == 279,
          str(block_states.default_state_id("minecraft:oak_leaves")))
    check("stone is registry state 1", block_states.default_state_id("minecraft:stone") == 1)
    check("air is state 0", block_states.AIR == 0 and block_states.name_of(0) == "minecraft:air")
    check(
        "state ids fit the uint16 a section stores",
        registry.blocks.total_states <= 0x10000,
        str(registry.blocks.total_states),
    )

    print("\n3. World generation starts afterwards")
    await server._init_world()
    world = server.world
    check("world created", world is not None)
    if world is None:
        return

    spawn = world.get_spawn_point()
    check("spawn point resolved from terrain", isinstance(spawn, tuple) and spawn[1] > 0, str(spawn))
    check("spawn chunk loaded", len(world.overworld.chunk_manager.chunks) > 0,
          f"{len(world.overworld.chunk_manager.chunks)} chunks")

    chunk = next(iter(world.overworld.chunk_manager.chunks.values()))
    non_air = sum(int((section.blocks != 0).sum()) for section in chunk.sections)
    check("chunk has terrain", non_air > 1000, f"{non_air} blocks")

    # Every block in a generated chunk must be a state the registry knows.
    unknown = set()
    for section in chunk.sections:
        for value in set(section.blocks.tolist()):
            if value == 0:
                continue
            if block_states.state(int(value)) is None:
                unknown.add(int(value))
    check("every generated block id is a registry state", not unknown, str(sorted(unknown)[:5]))

    state = block_states.state(1)
    check("stone resolves through the registry", state is not None and state.block.name == "minecraft:stone")
    # Bedrock seals the bottom of the column: the `above_bottom` surface band
    # paints it into the window's bottom section (server section 0 holds world
    # Y min_y..min_y+15, i.e. -64..-49 in the overworld).
    bottom = {int(value) for value in set(chunk.sections[0].blocks.tolist())}
    check("bedrock is at the bottom of the world",
          any(block_states.name_of(value) == "minecraft:bedrock" for value in bottom),
          str(sorted(block_states.name_of(value) for value in bottom)))

    print("\n4. The login path reads the same registry")
    from registry.tag_data import get_tag_manager
    from registry.registry import get_registry_manager

    manager = get_registry_manager()
    check("registry manager shares the instance", manager.game_registry is registry)
    check("synced registries are listed", len(manager.registry_ids()) == 23)

    payload = manager.get_all_registries()
    check("payloads build", len(payload) == 23 and all(len(view) for view in payload))

    tag_payload = get_tag_manager().build_update_tags_packet_data()
    block_tags = next(
        (entry for entry in tag_payload if entry["registry_id"] == "minecraft:block"), None
    )
    check("block tags are populated", block_tags is not None
          and sum(len(tag["entries"]) for tag in block_tags["tags"]) > 4000)

    # A second explicit init must not reload (ids would change under the world).
    from registry import init_registries

    check("init_registries is idempotent", init_registries() is registry)


def main() -> int:
    workdir = tempfile.mkdtemp(prefix="crotalus-boot-")
    previous = os.getcwd()
    os.chdir(workdir)
    try:
        asyncio.run(run())
    finally:
        os.chdir(previous)
        shutil.rmtree(workdir, ignore_errors=True)

    print()
    if failures:
        print(f"{len(failures)} boot-order check(s) FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("All boot-order checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
