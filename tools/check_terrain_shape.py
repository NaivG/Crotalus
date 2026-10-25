"""Does the density-driven terrain reproduce a real vanilla chunk?

`assets/vanilla_chunks/` holds chunks dumped from an actual Minecraft server
(`source` names the Pumpkin fixture each came from). Every file carries two
98k-cell block arrays:

    `before`   the noise stage's output: stone, water and air, and nothing
               else. The file's `pass_silhouette_changes: 0` says so - the
               silhouette (which cells are solid) is already final, and the
               occupancy (which block fills them) still changes afterwards.
    `after`    the same chunk once the surface rules have dressed it.

`before` is therefore exactly what `world/vanilla/terrain.py` claims to
produce, and this compares them cell for cell - stone, water and air as block
*identities*, not just a solid mask. `check_surface_rules.py` covers `after`.

Two presets do not match, and are reported against their measured value rather
than as failures: the nether's solid bulk and the end's islands. Both are
written up in `world/DESIGN-density-preset-gaps.md`, which is the place to look
for why. This tool exists so that fixing them is visible, and so that the
overworld cannot quietly regress while they are being worked on.

Run: python tools/check_terrain_shape.py [chunk-name]
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402

from registry import init_registries  # noqa: E402
from world.block_states import name_of, state_id  # noqa: E402
from world.chunk import Chunk, ChunkPos  # noqa: E402
from world.generator import GeneratorSettings, GeneratorType, create_generator  # noqa: E402

CHUNK_DIR = ROOT / "assets" / "vanilla_chunks"

#: Minimum acceptable exact-cell agreement, per chunk. These are *ceilings on
#: the residual*, not aspirations: measured agreement on the overworld is
#: 97.4-99.6%, and the gap is a real disagreement about which cells sit within
#: a rounding error of the density threshold, not a bug to be tuned away. The
#: nether and end numbers are what the gaps document records today, so that a
#: regression there shows up as a failure rather than as a quieter number.
MIN_AGREEMENT: Dict[str, float] = {
    "overworld_0_0": 0.99,
    "overworld_13579_-2_15": 0.98,
    "overworld_7_4": 0.96,
    "overworld_badlands_-595_544": 0.99,
    "overworld_frozen_ocean_-119_183": 0.95,
    "nether_0_0": 0.45,
    "end_0_0": 0.55,
}


def load_chunk(name: str) -> dict:
    with open(CHUNK_DIR / f"{name}.json", encoding="utf-8") as handle:
        return json.load(handle)


def to_blocks(payload: dict, key: str, palette: Dict[int, int], height: int) -> np.ndarray:
    """The dump's flat array as a `(height, 16, 16)` `[y, z, x]` uint16 view.

    Same conversion `check_surface_rules.py` uses, and for the same reason:
    the dump is in `(x * height + local_y) * 16 + z` order, which is the
    transpose of the axis order every worldgen pass works in.
    """
    lookup = np.zeros(65536, dtype=np.uint16)
    for dump_id, resolved in palette.items():
        lookup[dump_id] = resolved
    raw = np.asarray(payload[key], dtype=np.uint16).reshape(16, height, 16)
    return lookup[raw].transpose(1, 2, 0).copy()


def check(name: str, payload: dict) -> bool:
    palette = {int(k): state_id(v) for k, v in payload["palette"].items()}
    min_y, height = payload["min_y"], payload["height"]
    want = to_blocks(payload, "before", palette, height)

    generator = create_generator(
        GeneratorSettings(generator_type=GeneratorType.NORMAL,
                          seed=payload["seed"], preset=payload["preset"]),
        backend="vanilla")
    chunk = Chunk(position=ChunkPos(payload["chunk_x"], payload["chunk_z"]),
                  min_y=min_y, height=height)

    # Only the terrain pass. `generate_chunk` would also run the surface
    # rules, the carvers and the features, which is what `after` holds.
    started = time.perf_counter()
    generator._generate_terrain(chunk)
    elapsed = (time.perf_counter() - started) * 1000
    got = chunk.blocks_view()

    agree = float((got == want).mean())
    floor = MIN_AGREEMENT.get(name, 0.0)
    ok = agree >= floor

    print(f"  {name:32s} {payload['preset']:9s} "
          f"{agree*100:6.2f}% (floor {floor*100:.0f}%)  {elapsed:6.0f} ms  "
          f"{'ok' if ok else 'BELOW FLOOR'}")

    if not ok:
        for state in sorted(set(want.ravel().tolist()) | set(got.ravel().tolist())):
            theirs, ours = int((want == state).sum()), int((got == state).sum())
            if theirs or ours:
                print(f"      {name_of(state):28s} vanilla {theirs:6d}   ours {ours:6d}")
    return ok


def check_legacy_unchanged() -> bool:
    """The legacy backend must produce exactly the recorded baseline output.

    `world/legacy/` was carved out of the old single `WorldGenerator`, and the
    split moved the four decoration passes into `world/feature/` and the two
    backends behind `[world] generator`. All of that is refactoring, so none of
    it may change a single block.

    The baseline file has two sections, and they are not equally strong:

    * the default section's digests were taken from the pre-split generator, so
      a match there *is* proof of byte-identity;
    * the `cave_mode_legacy` section exercises `LegacyWorldGenerator._carve_legacy`,
      which the default section never reaches (`cave_mode` defaults to
      `"vanilla"`). Its digests were recorded after the split, because the
      pre-split `.pyc` had already been recompiled by then. It is a
      change-detector, not a proof - see `_note` in the baseline file.
    """
    import hashlib

    from world.legacy.generator import LegacyWorldGenerator

    print()
    print("4. legacy backend matches its recorded baseline")
    baseline_path = ROOT / "tools" / "baselines" / "legacy_generator.json"
    if not baseline_path.exists():
        print("  SKIP: no baseline recorded")
        return True
    baseline = json.load(open(baseline_path, encoding="utf-8"))

    # (cave_mode, preset) per section: the top level is the default section.
    sections = [(None, "overworld")]
    if "cave_mode_legacy" in baseline:
        sections.append(("legacy", "overworld"))

    ok = True
    for cave_mode, preset in sections:
        entries = baseline["cave_mode_legacy"] if cave_mode else {
            k: v for k, v in baseline.items()
            if not k.startswith("_") and k != "cave_mode_legacy"
        }
        label = f"cave_mode={cave_mode or 'default'}"
        for key, want in sorted(entries.items()):
            cx, cz = (int(part) for part in key.split(","))
            generator = LegacyWorldGenerator(GeneratorSettings(
                generator_type=GeneratorType.NORMAL, seed=1234, preset=preset,
                cave_mode=cave_mode or "vanilla"))
            chunk = Chunk(position=ChunkPos(cx, cz), min_y=-64, height=384)
            generator.generate_chunk(chunk)
            got = hashlib.sha256(
                np.ascontiguousarray(chunk.blocks_view()).tobytes()).hexdigest()
            match = got == want["blocks"]
            ok &= match
            verdict = "ok" if match else "CHANGED (baseline %s)" % want["blocks"][:16]
            print(f"  {label:20s} chunk ({cx},{cz}) {got[:16]}  {verdict}")
    return ok


def main(only: Optional[str]) -> int:
    logger.remove()
    init_registries(log_summary=False)

    index = json.load(open(CHUNK_DIR / "index.json", encoding="utf-8"))
    print("1. density terrain vs vanilla 'before'")
    passed = 0
    total = 0
    for entry in index:
        if only and entry["name"] != only:
            continue
        total += 1
        passed += check(entry["name"], load_chunk(entry["name"]))

    # Determinism: the same chunk twice through the same generator. The router
    # carries per-chunk FlatCache state, so this is also the check that a
    # stale cache cannot leak from one chunk into the next.
    print()
    print("2. determinism")
    generator = create_generator(
        GeneratorSettings(generator_type=GeneratorType.NORMAL, seed=0, preset="overworld"),
        backend="vanilla")
    digests = []
    for _ in range(2):
        for cx, cz in ((0, 0), (1, 0)):
            chunk = Chunk(position=ChunkPos(cx, cz), min_y=-64, height=384)
            generator._generate_terrain(chunk)
            digests.append(np.ascontiguousarray(chunk.blocks_view()).tobytes())
    # Positions 0 and 2 are chunk (0,0); 1 and 3 are chunk (1,0).
    same_00, same_10 = digests[0] == digests[2], digests[1] == digests[3]
    differ = digests[0] != digests[1]
    print(f"  chunk (0,0) reproducible: {'ok' if same_00 else 'FAIL'}")
    print(f"  chunk (1,0) reproducible: {'ok' if same_10 else 'FAIL'}")
    print(f"  chunk (0,0) differs from (1,0): {'ok' if differ else 'FAIL'}")

    # The backend is chosen by config, so the factory has to be able to pick
    # either, and has to refuse a name it does not know rather than quietly
    # generating something else.
    print()
    print("3. factory")
    from world.legacy.generator import LegacyWorldGenerator
    from world.vanilla.terrain import VanillaWorldGenerator

    for name, expected in (("vanilla", VanillaWorldGenerator),
                           ("legacy", LegacyWorldGenerator)):
        actual = type(create_generator(GeneratorSettings(seed=1), backend=name))
        ok = actual is expected
        passed += ok
        total += 1
        print(f"  backend {name:8s} -> {actual.__name__:22s} {'ok' if ok else 'FAIL'}")

    try:
        create_generator(GeneratorSettings(seed=1), backend="banana")
        print("  unknown backend rejected: FAIL (it was accepted)")
        total += 1
    except ValueError as error:
        ok = "vanilla" in str(error) and "legacy" in str(error)
        passed += ok
        total += 1
        print(f"  unknown backend rejected: {'ok' if ok else 'FAIL (message unhelpful)'}")
        print(f"      {error}")

    passed += int(same_00) + int(same_10) + int(differ)
    total += 3

    passed += check_legacy_unchanged()
    total += 1

    print()
    print(f"RESULT: {'PASS' if passed == total else 'FAIL'} "
          f"({passed}/{total})")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
