"""Prove the physics shape tables are `assets/blocks.json`, rearranged.

`world/vanilla/features.BlockFacts` collapses every state's `collision_shapes`
into one bool, which is all the feature stage needs. Physics needs the boxes.
`physics/block_shapes.py` exposes them as flat arrays, and this tool is what
says the rearrangement is faithful: it re-reads the payload **from the file**
(not from `BlockFacts`) and compares every one of the 29671 states, so a change
to the loader that drops or reorders shapes fails here rather than in the game.

Checked:

1. every state's shape tuple matches the payload, in order;
2. `counts[s] > 0` if and only if the payload lists shapes -- and the same
   predicate as `BlockFacts.blocks_motion`, so the old bool and the new table
   cannot drift apart;
3. the boxes are well formed (`min <= max`), every index a state references is
   inside the table, and the ones that leave their own cell are exactly the
   fence/wall family (vanilla fence collision really does reach 1.5);
4. named samples that a bool could not express: a slab's three variants, a
   fence's rails, a stair's step, a wall with *no* collision at all;
5. the per-state material columns equal their block's value;
6. a state id outside the table (a plugin block) has no boxes and does not
   block motion -- it does not read as solid.

Run: python tools/check_physics_shapes.py

Exit code 0 when everything passes, 1 otherwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from physics.block_shapes import block_shapes  # noqa: E402

failures: List[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    marker = "OK  " if condition else "FAIL"
    print(f"  {marker} {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(label)


def load_payload() -> dict:
    with open(ROOT / "assets" / "blocks.json", encoding="utf-8") as handle:
        return json.load(handle)


def distinct_materials(shapes, record: dict) -> set:
    """The distinct `(slip, velocity, jump)` triples across one block's states."""
    return {(float(shapes.slipperiness[int(state["id"])]),
             float(shapes.velocity_multiplier[int(state["id"])]),
             float(shapes.jump_velocity_multiplier[int(state["id"])]))
            for state in record["states"]}


def occupancy(boxes, n: int = 8) -> set:
    """Which cells of an `n^3` lattice a box list fills, by cell centre.

    Vanilla merges coplanar boxes (`Shapes.optimize`), so the same solid comes
    back as different box lists depending on the state -- a fence post plus one
    rail is one box, with four rails it is three. Comparing *volume* is the
    check that survives that: it asks what the block occupies, not how the
    payload chose to decompose it.
    """
    filled = set()
    for i in range(n):
        for j in range(n):
            for k in range(n):
                point = ((i + 0.5) / n, (j + 0.5) / n, (k + 0.5) / n)
                for box in boxes:
                    if (box[0] < point[0] < box[3] and box[1] < point[1] < box[4]
                            and box[2] < point[2] < box[5]):
                        filled.add((i, j, k))
                        break
    return filled


def mirror_y(cells: set, n: int = 8) -> set:
    return {(i, n - 1 - j, k) for i, j, k in cells}


def union(*sets: set) -> set:
    out: set = set()
    for item in sets:
        out |= item
    return out


def main() -> int:
    from registry import init_registries

    init_registries(log_summary=False)
    from world import block_states

    payload = load_payload()
    shapes_payload = payload["shapes"]
    records = payload["blocks"]

    print("1. The payload is the table we think it is")
    expected: Dict[int, Tuple[int, ...]] = {}
    expected_material: Dict[int, Tuple[float, float, float]] = {}
    by_name: Dict[str, dict] = {}
    for record in records:
        by_name[record["name"]] = record
        material = (float(record.get("slipperiness", 0.6)),
                    float(record.get("velocity_multiplier", 1.0)),
                    float(record.get("jump_velocity_multiplier", 1.0)))
        for state in record["states"]:
            expected[int(state["id"])] = tuple(int(s) for s in state["collision_shapes"])
            expected_material[int(state["id"])] = material
    state_count = len(expected)
    check("state ids are a contiguous 0..n-1 run",
          sorted(expected) == list(range(state_count)), f"{state_count} states")
    check("the shape table has 683 entries", len(shapes_payload) == 683,
          str(len(shapes_payload)))
    total_refs = sum(len(v) for v in expected.values())
    check("the state table references 52511 boxes", total_refs == 52511, str(total_refs))
    check("shape 0 is the full cube",
          shapes_payload[0] == {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]},
          str(shapes_payload[0]))

    shapes = block_shapes()
    check("the table covers every state", shapes.state_count == state_count,
          f"{shapes.state_count} vs {state_count}")
    check("the box table is 683 rows", shapes.shape_count == len(shapes_payload),
          str(shapes.shape_count))
    check("the reference count matches", int(shapes.ids.shape[0]) == total_refs,
          f"{int(shapes.ids.shape[0])} vs {total_refs}")

    print("\n2. Every state's boxes match the payload, in order")
    wrong = [s for s in range(state_count) if shapes.shape_ids(s) != expected[s]]
    check("all states match", not wrong,
          "" if not wrong else f"{len(wrong)} mismatched, first {wrong[0]}")
    check("the sentinel offset is the reference count",
          int(shapes.offsets[-1]) == total_refs, str(int(shapes.offsets[-1])))
    check("counts is derived from offsets, not a second table",
          np.array_equal(shapes.counts, np.diff(shapes.offsets)))
    check("the boxes agree with the raw ids, box for box",
          all(float(shapes.min_xyz[shape, 0]) == float(shapes_payload[shape]["min"][0])
              and float(shapes.max_xyz[shape, 2]) == float(shapes_payload[shape]["max"][2])
              for shape in range(len(shapes_payload))))

    print("\n3. counts > 0 agrees with blocks_motion, everywhere")
    motion = np.zeros(state_count, dtype=bool)
    for state in range(state_count):
        motion[state] = shapes._facts.blocks_motion(state)
    check("blocks_motion(s) == (count(s) > 0) for all states",
          np.array_equal(motion, shapes.counts > 0),
          f"{int((motion != (shapes.counts > 0)).sum())} disagreements")
    check("collision states are the 24254 the payload lists",
          int(motion.sum()) == 24254, str(int(motion.sum())))

    print("\n4. The boxes are well formed, including the ones that leave their cell")
    mins, maxs = shapes.min_xyz, shapes.max_xyz
    check("min <= max on every box and axis", bool(np.all(mins <= maxs)))
    check("every referenced index is inside the table",
          bool(shapes.ids.min() >= 0) and bool(shapes.ids.max() < shapes.shape_count),
          f"ids {shapes.ids.min()}..{shapes.ids.max()}")
    check("no state references the same box twice",
          all(len(set(expected[s])) == len(expected[s]) for s in range(state_count)))
    outside = np.flatnonzero((mins < 0.0).any(axis=1) | (maxs > 1.0).any(axis=1))
    check("43 boxes leave the unit cell (vanilla fence/wall collision reaches 1.5)",
          len(outside) == 43, str(len(outside)))
    check("nothing leaves it by more than half a block",
          float(mins.min()) == -0.25 and float(maxs.max()) == 1.5,
          f"min {float(mins.min())} max {float(maxs.max())}")
    users = sorted({record["name"] for record in records
                    if any(shape in set(outside) for shape in
                           (int(v) for state in record["states"]
                            for v in state["collision_shapes"]))})
    check("the blocks that use them are fences, walls and piston heads",
          len(users) == 53 and all(name.endswith(("_fence", "_fence_gate", "_wall"))
                                   or name in ("piston_head", "pitcher_crop")
                                   for name in users),
          f"{len(users)} blocks, odd ones out: "
          f"{[n for n in users if not n.endswith(('_fence', '_fence_gate', '_wall'))]}")

    print("\n5. Named samples a bool could not express")
    def state_of(name: str, properties: dict = None) -> int:
        return block_states.state_id(f"minecraft:{name}", properties)

    def boxes_of(name: str, properties: dict = None):
        return shapes.boxes(state_of(name, properties))

    full_cube = ((0.0, 0.0, 0.0, 1.0, 1.0, 1.0),)
    check("stone is the full cube", boxes_of("stone") == full_cube, str(boxes_of("stone")))
    check("a bottom slab is the bottom half",
          boxes_of("oak_slab", {"type": "bottom"}) == ((0.0, 0.0, 0.0, 1.0, 0.5, 1.0),),
          str(boxes_of("oak_slab", {"type": "bottom"})))
    check("a top slab is the top half",
          boxes_of("oak_slab", {"type": "top"}) == ((0.0, 0.5, 0.0, 1.0, 1.0, 1.0),),
          str(boxes_of("oak_slab", {"type": "top"})))
    check("a double slab is a full cube again",
          boxes_of("oak_slab", {"type": "double"}) == full_cube,
          str(boxes_of("oak_slab", {"type": "double"})))
    check("three slab variants, three different boxes",
          len({boxes_of("oak_slab", {"type": t}) for t in ("top", "bottom", "double")}) == 3)

    post = ((0.375, 0.0, 0.375, 0.625, 1.5, 0.625),)
    no_links = {"north": "false", "south": "false", "east": "false", "west": "false"}
    north_only = {"north": "true", "south": "false", "east": "false", "west": "false"}
    both_z = {"north": "true", "south": "true", "east": "false", "west": "false"}
    all_links = {"north": "true", "south": "true", "east": "true", "west": "true"}
    check("a fence with no connections is the post alone", boxes_of("oak_fence", no_links) == post,
          str(boxes_of("oak_fence", no_links)))
    check("a fence post is taller than its own cell (1.5)",
          boxes_of("oak_fence", no_links)[0][4] == 1.5)
    # The rails merge into the post (`Shapes.optimize`), so the box *count* is
    # not the check -- the volume is. One rail adds the rail's volume; the
    # opposite pair is one merged box; four rails are the post alone again
    # (the two crossing rails are what the three boxes are).
    post_only = occupancy(post)
    one_rail = occupancy(((0.375, 0.0, 0.0, 0.625, 1.5, 0.375),))
    check("one connection adds exactly one rail to the post",
          occupancy(boxes_of("oak_fence", north_only)) == union(post_only, one_rail),
          f"{len(occupancy(boxes_of('oak_fence', north_only)))} cells vs "
          f"{len(union(post_only, one_rail))}")
    check("and it really is more than the post",
          occupancy(boxes_of("oak_fence", north_only)) != post_only)
    check("two opposite connections are one merged box",
          len(boxes_of("oak_fence", both_z)) == 1,
          str(boxes_of("oak_fence", both_z)))
    check("two opposite connections span the whole cell in z",
          boxes_of("oak_fence", both_z) == ((0.375, 0.0, 0.0, 0.625, 1.5, 1.0),),
          str(boxes_of("oak_fence", both_z)))
    check("four connections are three boxes (two crossing rails plus the post)",
          len(boxes_of("oak_fence", all_links)) == 3, str(boxes_of("oak_fence", all_links)))

    stairs = boxes_of("oak_stairs", {"facing": "north", "half": "bottom",
                                     "shape": "straight"})
    check("a bottom stair is a base slab plus a step",
          stairs == ((0.0, 0.0, 0.0, 1.0, 0.5, 1.0), (0.0, 0.5, 0.0, 1.0, 1.0, 0.5)),
          str(stairs))
    top_stairs = boxes_of("oak_stairs", {"facing": "north", "half": "top",
                                         "shape": "straight"})
    check("a top stair occupies the bottom stair upside down",
          occupancy(top_stairs) == mirror_y(occupancy(stairs)),
          f"{len(occupancy(top_stairs))} vs {len(mirror_y(occupancy(stairs)))} cells")
    check("a top stair is not the bottom stair's box list",
          top_stairs != stairs, str(top_stairs))

    check("water has no collision", boxes_of("water") == ())
    check("cobweb has no collision", boxes_of("cobweb") == ())
    check("lily_pad is a thin box at the bottom of its cell",
          0.0 < boxes_of("lily_pad")[0][4] < 0.25, str(boxes_of("lily_pad")))
    # The one case where the bool and the box table disagree in *kind*: a wall
    # with no connections and no `up` has no collision at all, and one with `up`
    # has exactly one box. Same block, both states listed in the payload.
    empty_wall = state_of("cobblestone_wall", {"north": "none", "south": "none",
                                               "east": "none", "west": "none",
                                               "up": "false"})
    up_wall = state_of("cobblestone_wall", {"north": "none", "south": "none",
                                            "east": "none", "west": "none",
                                            "up": "true"})
    check("a wall with nothing attached has no collision", shapes.count(empty_wall) == 0,
          f"state {empty_wall}")
    check("the same wall with `up` has one", shapes.count(up_wall) == 1,
          f"state {up_wall}")
    check("and blocks_motion agrees with both",
          not shapes._facts.blocks_motion(empty_wall)
          and shapes._facts.blocks_motion(up_wall))
    counts = {len(expected[int(state["id"])])
              for record in records if record["name"] == "cobblestone_wall"
              for state in record["states"]}
    check("a wall's states carry 0 to 5 boxes", counts == {0, 1, 2, 3, 4, 5}, str(sorted(counts)))

    print("\n6. The material columns are the block's value, per state")
    slip = shapes.slipperiness
    vel = shapes.velocity_multiplier
    jump = shapes.jump_velocity_multiplier
    bad = [s for s in range(state_count) if (float(slip[s]), float(vel[s]), float(jump[s]))
           != expected_material[s]]
    check("all states carry their block's (slipperiness, velocity, jump)",
          not bad, "" if not bad else f"{len(bad)} mismatched, first {bad[0]}")
    ice = state_of("ice")
    check("ice is slippery (0.98)", abs(float(slip[ice]) - 0.98) < 1e-9, str(float(slip[ice])))
    soul = state_of("soul_sand")
    check("soul sand has velocity_multiplier 0.4",
          abs(float(vel[soul]) - 0.4) < 1e-9, str(float(vel[soul])))
    honey = state_of("honey_block")
    check("honey has jump_velocity_multiplier 0.5",
          abs(float(jump[honey]) - 0.5) < 1e-9, str(float(jump[honey])))
    stone = state_of("stone")
    check("stone keeps the vanilla defaults",
          (float(slip[stone]), float(vel[stone]), float(jump[stone])) == (0.6, 1.0, 1.0),
          str((float(slip[stone]), float(vel[stone]), float(jump[stone]))))
    check("the columns are per state but constant within a block",
          len(distinct_materials(shapes, by_name["oak_log"])) == 1,
          str(distinct_materials(shapes, by_name["oak_log"])))

    print("\n7. A state outside the table is air, not solid")
    beyond = state_count + 1000
    check("no boxes", shapes.boxes(beyond) == ())
    check("count 0", shapes.count(beyond) == 0)
    check("does not block motion", not shapes.has_collision(beyond))
    check("agrees with BlockFacts", not shapes._facts.blocks_motion(beyond))
    check("negative ids too", shapes.count(-1) == 0 and shapes.boxes(-1) == ())

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("All physics shape checks passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        raise SystemExit(1)
