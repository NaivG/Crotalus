#!/usr/bin/env python
"""Capture a golden baseline of everything the registry puts on the wire.

`tools/check_registry.py` compares the live registry output against the committed
baseline, so a change to the login payload cannot pass unnoticed. The committed
file was captured from the code that ran *before* the registry rework; it is the
reference the refactor is measured against.

The comparison is only meaningful while the baseline is older than the code, so
this tool refuses to overwrite an existing baseline unless `--force` is given.
Re-capture deliberately, and say why in the commit - a baseline regenerated from
the code it is supposed to check proves nothing.

What is recorded, per registry, in send order:
  - entry count, and for every entry its id plus a SHA-256 of its NBT bytes
    (or null when the entry carries no NBT)
  - a SHA-256 of the full CRegistryData payload with NBT included

Plus: a SHA-256 of the Update Tags payload body.

Registry ids are recorded both as sent and canonicalised ("worldgen/biome" ->
"minecraft:worldgen/biome"), because the wire form of a ResourceLocation is
resolved leniently by the client and the two are the same registry to it.

Usage:
    python tools/capture_registry_baseline.py [output.json] [--force]
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT = ROOT / "tools" / "baselines" / "registry_wire_1_21_11.json"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(registry_id: str) -> str:
    """Namespace a bare registry id the way ResourceLocation.parse() does."""
    return registry_id if ":" in registry_id else f"minecraft:{registry_id}"


def capture() -> dict:
    from network.buffer import ByteBuffer
    from network.packets import CUpdateTags
    from registry.registry import RegistryDataPacketBuilder, get_registry_manager
    from registry.tag_data import get_tag_manager

    manager = get_registry_manager()
    builder = RegistryDataPacketBuilder(manager)

    registries = []
    for registry_id, packet_data in builder.build_all_packets(include_nbt=True):
        registry = manager.get_registry(registry_id)
        entries = []
        for entry in registry.entries:
            entries.append(
                {
                    "id": entry.id,
                    "nbt_sha256": sha256(entry.data) if entry.data else None,
                    "nbt_size": len(entry.data) if entry.data else 0,
                }
            )
        registries.append(
            {
                "id": registry_id,
                "canonical_id": canonical(registry_id),
                "entry_count": len(entries),
                "payload_sha256": sha256(packet_data),
                "payload_size": len(packet_data),
                "entries": entries,
            }
        )

    tag_data = get_tag_manager().build_update_tags_packet_data()
    tag_buf = ByteBuffer()
    CUpdateTags(tag_data).write(tag_buf)

    return {
        "protocol": 774,
        "minecraft": "1.21.11",
        "source": "assets/ (Pumpkin @ 42c903493)",
        "registry_count": len(registries),
        "total_payload_size": sum(r["payload_size"] for r in registries),
        "registries": registries,
        "tags": {
            "registry_count": len(tag_data),
            "tag_count": sum(len(r["tags"]) for r in tag_data),
            "entry_count": sum(
                len(t["entries"]) for r in tag_data for t in r["tags"]
            ),
            "payload_sha256": sha256(bytes(tag_buf)),
            "payload_size": len(bytes(tag_buf)),
        },
    }


def main() -> int:
    arguments = sys.argv[1:]
    force = "--force" in arguments
    remaining = [argument for argument in arguments if argument != "--force"]
    output = Path(remaining[0]) if remaining else DEFAULT_OUTPUT

    if output.is_file() and not force:
        print(
            f"Refusing to overwrite {output}.\n"
            f"That file is the reference the registry is checked against; "
            f"regenerating it from the code it checks would make "
            f"tools/check_registry.py pass by construction.\n"
            f"Pass --force if you really mean to re-capture it."
        )
        return 2

    data = capture()
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=1, sort_keys=True)
        handle.write("\n")

    print(f"Captured {data['registry_count']} registries "
          f"({data['total_payload_size']} bytes) -> {output}")
    print(f"  tags: {data['tags']['registry_count']} registries, "
          f"{data['tags']['tag_count']} tags, "
          f"{data['tags']['entry_count']} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
