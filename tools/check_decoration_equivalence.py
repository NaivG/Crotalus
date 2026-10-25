"""
Decoration equivalence: do the four passes still produce the same chunks, and
the same amount of work?

    python tools/check_decoration_equivalence.py [--record]

The other `check_*` tools in this directory ask whether a stage is *correct* --
`check_features.py` recomputes placement counts from the modifier definitions,
`check_surface_rules.py` checks the surface invariants and diffs against the
legacy generator. This one asks a different, narrower question, aimed at the
decoration layer specifically: **did anything move?**

Why it exists
-------------
The decoration layer is four passes (`world/feature/`: biomes -> surface ->
carvers -> features) that share one chunk buffer and mutate it in place, so a
change to any of them is invisible to a reviewer reading the diff and invisible
to a test that only checks an invariant at the end. The layer has also been
optimized for speed twice now, in both cases by replacing per-cell Python with
a table lookup or a vectorized pass -- refactors whose entire risk is "does it
compute the same answer", and which pass every property test while still
changing the output.

This caught exactly that. Rewriting `Surface.top` to index a per-state-id mask
instead of walking a column produced identical results under `check_features.py`
and `check_surface_rules.py` and changed the blocks in three chunks out of five,
because `fluids.json` lists states against the *empty* fluid, so "is this state
in the fluid table" is not the predicate `fluid_of(s) != "minecraft:empty"`.

The three things recorded, and why all three
--------------------------------------------
1. **The block and biome volumes**, hashed. The output. Anything that changes
   the world shows up here.

2. **`SurfaceStats`.** The work. `cells_visited`, `rules_evaluated` and
   `cells_changed` can all move while the final blocks stay identical -- a rule
   that is evaluated one time fewer, or stops matching one cell, can land on the
   same state. A hash alone would call that a pass.

3. **`FeatureStats`.** The same argument for the feature pass, and the more
   sensitive of the two: `features_attempted` / `features_placed` move long
   before anything visible does, because placement is driven by a seeded RNG
   whose every draw is positional.

The corpus
----------
Ten chunks at seed 0, chosen for what they exercise rather than for being
adjacent: ordinary overworld terrain, an ocean/beach shore, the badlands
region, and two frozen-ocean chunks carrying 96 and 160 icebergs. The iceberg
chunks are the reason this tool is not just "chunk (0,0)" -- `place_iceberg`
writes the column *after* the surface scan has read it, which is precisely the
ordering a "read the column once into a list" optimization can get wrong.

What is *not* covered, stated rather than implied
--------------------------------------------------
The badlands pillar (`TerrainBuilder.place_badlands_pillar`, which writes the
column *before* the surface scan reads it) never fires: a 285-chunk sweep of
the badlands region at seed 0 produced `pillars: 0` everywhere, and no chunk in
this corpus records one. That is a pre-existing gap in the surface pass, not
something this tool introduced, and it is recorded here so the absence is a
known quantity rather than a silent one.

`--record` rewrites the baseline. Only do that when the change to the
decoration layer is intended, and read the diff before committing it.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

logger.remove()

from world.chunk import Chunk, ChunkPos  # noqa: E402
from world.generator import GeneratorSettings, create_generator  # noqa: E402

BASELINE_PATH = (Path(__file__).resolve().parent / "baselines"
                 / "decoration_chunks.json")

SEED = 0
PRESET = "overworld"

#: (x, z) chunk coordinates. See "The corpus" in the module docstring.
CORPUS: List[Tuple[int, int]] = [
    (0, 0),
    (1, 0),
    (0, 1),
    (3, -2),
    (-5, 7),
    (12, -8),
    (1, -56),      # ocean / beach shore
    (-1, -61),     # badlands region
    (-162, 135),   # frozen ocean: 96 icebergs
    (-161, 136),   # frozen ocean: 160 icebergs
]

_failures = 0


def check(label: str, ok: bool, detail: str = "") -> bool:
    global _failures
    if not ok:
        _failures += 1
    print(f"  {'OK  ' if ok else 'FAIL'}  {label}" + (f"  ({detail})" if detail else ""))
    return ok


def _digest(array) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def measure(gen) -> Dict[str, dict]:
    """Generate every corpus chunk and record what the decoration layer did.

    `decorate` is wrapped rather than called directly: the feature layer is the
    thing under test, and calling it by hand here would test this tool's copy of
    its arguments instead of the layer's own.
    """
    from world.vanilla import features as vanilla_features

    captured: List[dict] = []
    real_decorate = vanilla_features.decorate

    def recording_decorate(*args, **kwargs):
        stats = real_decorate(*args, **kwargs)
        captured.append(stats.summary())
        return stats

    vanilla_features.decorate = recording_decorate
    try:
        out: Dict[str, dict] = {}
        for cx, cz in CORPUS:
            captured.clear()
            chunk = Chunk(ChunkPos(cx, cz))
            gen.generate_chunk(chunk)
            entry = {
                "blocks": _digest(chunk.blocks_view()),
                "biomes": _digest(chunk.biome_volume()),
                "surface": dict(gen.layers.surface._system.diagnostics),
            }
            if len(captured) != 1:
                check(f"chunk ({cx},{cz}) decorated once", False,
                      f"decorate called {len(captured)} times")
                continue
            entry["features"] = captured[0]
            out[f"{cx},{cz}"] = entry
        return out
    finally:
        vanilla_features.decorate = real_decorate


def report(label: str, got: dict, want: dict) -> None:
    """Every recorded field, so a mismatch says *what* moved, not just that one did."""
    for key in sorted(want):
        if key not in got:
            check(label, False, f"{key} missing from this run")
            continue
        for field in sorted(want[key]):
            expected = want[key][field]
            actual = got[key].get(field)
            if field in ("blocks", "biomes"):
                # Full digests in the report would make every line unreadable;
                # a prefix is enough to tell two chunks apart by eye.
                same = actual == expected
                detail = "" if same else f"{str(actual)[:12]} != {str(expected)[:12]}"
            else:
                same = actual == expected
                if same:
                    detail = ""
                elif isinstance(expected, dict):
                    moved = {f: (expected[f], actual.get(f)) for f in expected
                             if actual is None or actual.get(f) != expected[f]}
                    detail = ", ".join(f"{f} {a}->{b}" for f, (a, b) in moved.items())
                else:
                    detail = f"{actual} != {expected}"
            check(f"{label} {key} {field}", same, detail)


def main() -> None:
    record = "--record" in sys.argv[1:]

    gil = getattr(sys, "_is_gil_enabled", None)
    state = "enabled" if (gil and gil()) else "disabled"
    print(f"decoration equivalence  seed {SEED}  preset {PRESET}  "
          f"python {sys.version.split()[0]}  GIL {state}")
    print(f"corpus: {len(CORPUS)} chunks\n")

    gen = create_generator(GeneratorSettings(seed=SEED, preset=PRESET))
    gen.set_seed(SEED)
    # Warm the registries, the flat caches and the carver JIT outside the
    # measurement: the first chunk pays for all of it.
    gen.generate_chunk(Chunk(ChunkPos(9999, 9999)))

    got = measure(gen)
    check("every corpus chunk recorded", len(got) == len(CORPUS),
          f"{len(got)} of {len(CORPUS)}")

    if record or not BASELINE_PATH.exists():
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "_comment": (
                "Recorded by tools/check_decoration_equivalence.py --record. "
                "These are outputs of this generator, not values from a real "
                "Minecraft; they exist to catch drift in the decoration layer, "
                "so re-record only when a change there is intended and you have "
                "read the diff."
            ),
            "seed": SEED,
            "preset": PRESET,
            "chunks": got,
        }
        with open(BASELINE_PATH, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1, sort_keys=True)
        print(f"  OK    baseline recorded  "
              f"({BASELINE_PATH.relative_to(ROOT)})")
        return

    with open(BASELINE_PATH, encoding="utf-8") as handle:
        base = json.load(handle)

    if base.get("seed") != SEED or base.get("preset") != PRESET:
        check("baseline seed/preset matches", False,
              f"baseline is {base.get('preset')}/{base.get('seed')}, "
              f"this run is {PRESET}/{SEED}")
        return
    if sorted(base.get("chunks", {})) != sorted(got):
        check("baseline corpus matches", False,
              f"baseline has {sorted(base.get('chunks', {}))}")
        return

    report("chunk", got, base["chunks"])

    totals = {}
    for field in ("pillars", "icebergs"):
        totals[field] = sum(c["surface"].get(field, 0) for c in got.values())
    print(f"\n  corpus surface totals: {totals}")
    print("  (pillars stay 0 -- place_badlands_pillar never fires at this seed; "
          "see the module docstring)")

    print(f"\nRESULT: {'PASS' if _failures == 0 else f'FAIL ({_failures})'}")
    sys.exit(1 if _failures else 0)


if __name__ == "__main__":
    main()
