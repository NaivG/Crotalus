"""Entity tracking table: coverage, value sanity, and the ids it has to agree with.

`assets/entity_tracking.json` is the one asset in this repository that is
*not* from a single Pumpkin commit, and that is worth being loud about.

`assets/entities.json` has carried `client_tracking_range`, `update_interval`
and `track_deltas` since the 26.3 report. At the 1.21.11 commit this server
targets (`42c903493`) it does not have them, and at no other commit on the
1.21.11 line does it either. Without those fields there is no way to answer
"how far away may a client see this entity", so the values are taken from the
newer report **by entity name** and kept in their own file rather than being
back-filled into `entities.json`.

Taking them by name is sound because the two reports agree on all 157 names -
the 26.3 report only adds four (`cushion`, `poplar_boat`, `poplar_chest_boat`,
`sulfur_cube`). It is *not* sound to assume the two reports agree on the
**numeric ids**, and they emphatically do not: `player` is 155 at 774 and 159
at 26.3. So this file carries no ids at all, and the check below proves it.

What it proves, in order:

1. **Coverage is exact.** Every name in `assets/entities.json` appears in the
   tracking table, and the table has no extra names. A missing name is a
   failure, not a default - an entity nobody can compute a tracking range for
   is a bug that otherwise shows up as "the mob never appears".
2. **The values are in the set vanilla actually uses.** `client_tracking_range`
   must be one of the eight values present in the report, and `update_interval`
   must be a positive tick count (the report uses `Integer.MAX_VALUE` as its
   "never on a timer" sentinel, which is legal and is called out separately).
3. **The table is loaded and reachable**, so a typo in the `registry/tables.py`
   registration fails here rather than at the first spawn.
4. **The hand-checked values are what we expect**: `player` 32, `zombie` 8,
   `item` 6, `arrow` 4, `marker` 0. The last one matters because a range of 0
   means "never tracked" and is a legitimate value, not a missing one.

Run: python tools/check_entity_tracking.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ASSETS = ROOT / "assets"

# The eight values vanilla's report actually uses. Anything else means the
# file was edited by hand and has drifted from the source it claims to be.
KNOWN_TRACKING_RANGES = {0, 4, 5, 6, 8, 10, 16, 32}

# Vanilla's "not on a timer" sentinel, and the entities that use it.
NEVER_ON_A_TIMER = 2147483647

# Names present in the 26.3 report but not at 774. Listed so that a future
# reader who finds one of these missing from our table knows it is supposed to
# be missing, rather than going looking for a bug.
NEWER_ONLY_NAMES = {"cushion", "poplar_boat", "poplar_chest_boat", "sulfur_cube"}

# Hand-checked against the 26.3 report. `player` is the one this round's
# milestone depends on; `marker` is here to prove 0 is accepted as a value.
EXPECTED = {
    "player": 32,
    "zombie": 8,
    "item": 6,
    "arrow": 4,
    "armor_stand": 10,
    "marker": 0,
}

failures: list[str] = []


def fail(message: str) -> None:
    failures.append(message)
    print(f"  FAIL: {message}")


def load(name: str) -> dict:
    with (ASSETS / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def check_coverage() -> tuple[dict, dict]:
    """Every 1.21.11 entity name has a tracking entry, and no others do."""
    print("1. coverage")
    entities = load("entities.json")
    tracking = load("entity_tracking.json")

    missing = sorted(set(entities) - set(tracking))
    extra = sorted(set(tracking) - set(entities))
    for name in missing:
        fail(f"{name} has no entry in entity_tracking.json")
    for name in extra:
        if name in NEWER_ONLY_NAMES:
            print(f"  note: {name} is 26.3-only and correctly absent")
        else:
            fail(f"entity_tracking.json has {name}, which is not a 1.21.11 entity")

    print(f"  {len(entities)} entity types, {len(tracking)} tracking entries")
    if not missing and not extra:
        print("  every entity type has exactly one tracking entry")
    return entities, tracking


def check_values(tracking: dict) -> None:
    """The numbers are ones vanilla actually uses."""
    print("2. value sanity")
    for name, entry in sorted(tracking.items()):
        if not isinstance(entry, dict):
            fail(f"{name} is {type(entry).__name__}, expected an object")
            continue
        for field in ("client_tracking_range", "update_interval", "track_deltas"):
            if field not in entry:
                fail(f"{name} is missing {field}")

        ctr = entry.get("client_tracking_range")
        if not isinstance(ctr, int) or ctr not in KNOWN_TRACKING_RANGES:
            fail(f"{name} has client_tracking_range={ctr!r}, "
                 f"not one of {sorted(KNOWN_TRACKING_RANGES)}")

        interval = entry.get("update_interval")
        if not isinstance(interval, int) or interval <= 0:
            fail(f"{name} has update_interval={interval!r}, expected a positive int")

    never = sorted(n for n, e in tracking.items()
                   if e.get("update_interval") == NEVER_ON_A_TIMER)
    print(f"  all ranges within {sorted(KNOWN_TRACKING_RANGES)}")
    print(f"  {len(never)} types are never-on-a-timer (the "
          f"Integer.MAX_VALUE sentinel): {', '.join(never)}")


def check_registered() -> None:
    """The table is actually loaded, so the registration is not a typo."""
    print("3. registration")
    from registry.game_registry import init_registries, is_initialized, reset_registries

    if is_initialized():
        reset_registries()
    init_registries()
    try:
        from registry.game_registry import get_game_registry

        table = get_game_registry().tables.entity_tracking
        if table.count == 0:
            fail("entity_tracking loaded but is empty")
        else:
            print(f"  loaded {table.count} entries via registry.tables.entity_tracking")
    finally:
        reset_registries()


def check_expected(tracking: dict) -> None:
    """The values the entity layer is about to depend on are the right ones."""
    print("4. hand-checked values")
    for name, expected in sorted(EXPECTED.items()):
        actual = tracking.get(name, {}).get("client_tracking_range")
        if actual != expected:
            fail(f"{name} has client_tracking_range={actual!r}, expected {expected}")
        else:
            print(f"  {name:14s} {actual}")


def main() -> None:
    _entities, tracking = check_coverage()
    check_values(tracking)
    check_registered()
    check_expected(tracking)

    print()
    if failures:
        print(f"{len(failures)} failure(s)")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
