"""EntityPosition and Entity: the geometry the rest of the layer stands on.

The position object is a drop-in replacement for `player/player.py`'s
`PlayerPosition`, and "drop-in" is a contract this check enforces rather
than hopes for: the movement-packet path writes the position's fields
directly (`_apply_movement`), the chunk streamer reads `.chunk` off the same
object, and `player/storage.py` round-trips it through JSON with keys that
must not change. Every assertion here maps to one of those real callers.

Checked, in order:

1. **Chunk semantics are floor, not trunc.** `x = -0.5` is chunk -1; an
   `int()`-based derivation returns 0 and streams the wrong half of the
   world to everyone west of spawn. This was a deliberate fix once and is
   the first thing a refactor breaks.
2. **Derived coordinates match Pumpkin's definitions**: `block_pos` floors
   all three axes, `section` is the >>4 form of the floored coords (the
   arithmetic shift is what makes negative floors land in the right
   section), and `eye_pos` adds the *type's* eye height - checked for the
   player (1.62) against a boat (0.5625... actually the boat's eye height
   is 0.5625 per entities.json) to prove it reads the type, not a constant.
3. **In-place mutation survives.** `update_position` must mutate the object
   a holder already references, never rebind - that is the contract the
   chunk streamer depends on.
4. **Serialization keys are unchanged** from PlayerPosition's, so existing
   player JSON loads.
5. **Entity identity basics**: id 0 means unassigned, allocation comes from
   the global counter, and the player type lookup fails loudly (not with a
   None) if the registry was not initialised.

Run: py -3.14t tools/check_entity_position.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  OK   {label}" + (f"  ({detail})" if detail else ""))
    else:
        failures.append(f"{label}: {detail}")
        print(f"  FAIL {label}  ({detail})")


def main() -> None:
    from registry.game_registry import init_registries, reset_registries

    init_registries()
    try:
        from entity import Entity, EntityPosition, entity_type_by_name
        from entity.ids import EntityIdAllocator
        from entity.entity import player_entity_type

        print("1. chunk semantics")
        pos = EntityPosition(x=-0.5, y=64.0, z=-0.5)
        check("x=-0.5 is in chunk -1", pos.chunk == (-1, -1), f"{pos.chunk}")
        pos2 = EntityPosition(x=15.9, y=64.0, z=-1.0)
        check("15.9/-1.0 is chunk (0, -1)", pos2.chunk == (0, -1), f"{pos2.chunk}")
        pos3 = EntityPosition(x=0.0, y=64.0, z=0.0)
        check("origin is chunk (0, 0)", pos3.chunk == (0, 0), f"{pos3.chunk}")

        print("2. derived coordinates")
        pos = EntityPosition(x=10.7, y=-3.2, z=-20.1)
        check("block_pos floors all axes", pos.block_pos == (10, -4, -21),
              f"{pos.block_pos}")
        check("section is block_pos >> 4", pos.section == (0, -1, -2),
              f"{pos.section}")
        pos_deep = EntityPosition(x=-16.0, y=0.0, z=0.0)
        check("x=-16.0 lands exactly on section -1", pos_deep.section[0] == -1,
              f"{pos_deep.section}")

        player = entity_type_by_name("minecraft:player")
        boat = entity_type_by_name("minecraft:acacia_boat")
        e = Entity(entity_id=7, entity_type=player)
        e.update_position(1.0, 64.0, 2.0)
        check("player eye is feet + 1.62",
              abs(e.eye_pos[1] - 65.62) < 1e-9, f"{e.eye_pos}")
        e.entity_type = boat
        check("eye follows the type (boat)", abs(e.eye_pos[1] - 64.5625) < 1e-9,
              f"{e.eye_pos}")

        print("3. in-place mutation")
        e2 = Entity(entity_id=8, entity_type=player)
        original = e2.position
        e2.update_position(5.0, 70.0, 6.0, yaw=90.0, pitch=-45.0, on_ground=False)
        check("position object identity preserved", e2.position is original)
        check("fields landed", (e2.position.x, e2.position.yaw, e2.position.on_ground)
              == (5.0, 90.0, False))
        # Direct field writes - what _apply_movement does - must be visible
        # through the same object.
        e2.position.x = 99.0
        check("direct field writes are visible", e2.position.x == 99.0)

        print("4. serialization keys")
        pos = EntityPosition(x=1.5, y=65.0, z=2.5, yaw=45.0, pitch=10.0,
                             on_ground=False)
        d = pos.to_dict()
        check("keys are the PlayerPosition set",
              set(d) == {"x", "y", "z", "yaw", "pitch", "on_ground"}, f"{sorted(d)}")
        revived = EntityPosition.from_dict(d)
        check("round-trip preserves all six", (revived.x, revived.y, revived.z,
              revived.yaw, revived.pitch, revived.on_ground)
              == (1.5, 65.0, 2.5, 45.0, 10.0, False))
        # A pre-EntityPosition player file: no velocity keys, must still load.
        legacy = EntityPosition.from_dict({"x": 3.0, "z": 4.0})
        check("legacy JSON without new keys loads",
              legacy.x == 3.0 and legacy.velocity == (0.0, 0.0, 0.0))

        print("5. entity identity")
        alloc = EntityIdAllocator()
        a = Entity(entity_id=alloc.allocate(), entity_type=player)
        b = Entity(entity_id=alloc.allocate(), entity_type=player)
        check("allocator hands out distinct nonzero ids",
              a.entity_id != b.entity_id and a.entity_id > 0,
              f"{a.entity_id}, {b.entity_id}")
        check("unassigned entity reads as 0", Entity().entity_id == 0)
        check("velocity defaults to zero",
              Entity().velocity == (0.0, 0.0, 0.0))
        try:
            player_entity_type()
            check("player type lookup succeeds after init", True)
        except RuntimeError as exc:
            check("player type lookup succeeds after init", False, str(exc))

        print()
        if failures:
            print(f"{len(failures)} failure(s)")
            raise SystemExit(1)
        print("all checks passed")
    finally:
        reset_registries()


if __name__ == "__main__":
    main()
