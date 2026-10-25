# `world/feature` — the decoration layer

A generator backend decides what the terrain **is**. This package decides what
grows on it, and it is shared by every backend.

```
feature/
├── __init__.py     FeatureLayers — the four passes, assembled
├── biomes.py       -> world/vanilla/biome.py     which biome each cell is
├── surface.py      -> world/vanilla/surface.py   grass, sand, terracotta, bedrock
├── carvers.py      -> world/vanilla/carver.py    caves and canyons
└── features.py     -> world/vanilla/features.py  trees, ore, lakes, monster rooms
```

## Why it is separate

`[world] generator` switches the terrain algorithm: `world/vanilla` reads a
density function, `world/legacy` extrudes a Perlin heightmap. Neither of them
has anything to say about grass. Before this split those four passes were
private methods on the one generator class, which meant they were reachable from
the Perlin path and *only* the Perlin path - adding a second backend would have
meant either duplicating ~400 lines of layer drivers or reaching across into
another class to reuse them.

Here they are a package with a four-line API, and the pipeline order lives in
one place, `BaseGenerator._generate_chunk_locked`. Swapping the terrain
algorithm does not touch a line of this directory.

The layers only read and write a `Chunk`. They know the preset and the seed,
because every vanilla layer resolves its data through those, and they know
nothing about how the stone underneath them was made.

## Order, and why it is not alphabetical

```
biomes -> surface -> [ores] -> carvers -> features
```

The three passes after biomes all *ask the biome volume* something: the
carvers which caves a biome has, the surface rules what its top block is, the
features what to scatter. So biomes has to run first, and the other three read
what it wrote. A chunk with no biome volume silently loses all three, which is
why a broken biome source degrades to "every cell is whatever the caller set"
rather than raising.

The bracketed `ores` is the one pass that is not in this package, and its
position is load-bearing. On a legacy world it is `LegacyWorldGenerator`'s own
vein table, and it has to run *after* the surface rules (so a vein does not
land where a surface rule is about to put dirt) and *before* the carvers (so a
carver that opens the space around a vein does not leave it hanging in air).
That is why the sequence is driven by the caller rather than hidden behind a
`run()` method - a `run()` would have had to take a callback to make room.

On a vanilla world there is no such pass: the ore is a `minecraft:ore_vein`
configured feature in each biome's feature list, placed by `features.py`. See
`BaseGenerator._generate_ores`.

## Degradation

Every layer builds its vanilla counterpart lazily, once, and every one of them
is wrapped so that a missing or broken data file logs a single warning naming
what the world loses and then does nothing:

| layer | if it fails |
| --- | --- |
| biomes | chunks ship as whatever the caller set (plains) |
| surface | terrain stays bare stone, with no bedrock floor |
| carvers | no caves |
| features | nothing is decorated - no trees, no ore |

A terrain generator that died because a decoration table failed to parse would
be a poor trade, and a per-chunk retry of a 600 ms parse would be worse. Each
layer therefore latches its own `_failed` flag on the first failure.

## Cost

Measured on one core, warm, per chunk:

| pass | ms |
| --- | --- |
| biome volume (6 density slots over 1536 quart cells) | 200–450 |
| surface rules (one Python walk per stone cell) | ~600 |
| carvers | 35–55 |
| features, plains | ~40 |

This is the bulk of the vanilla backend's ~1 s per chunk, and it is the reason
`[world] generator = "legacy"` still exists. Full numbers, including the density
fill the terrain pass adds, are in `world/vanilla/README.md` under "Cost".

## Thread safety

`ChunkGeneratorThread` runs four workers against one generator per dimension, so
these four layers are entered concurrently. Their cached trees are not
re-entrant - `NoiseRouter`'s `FlatCache`, the biome tree's last-result cache and
`SurfaceSystem` all carry state between samples. `BaseGenerator` holds a lock
across the whole of `generate_chunk`, which covers all of them at once; see its
docstring for why that is a correctness requirement and not a precaution.

## Tests

```bash
python tools/check_biomes.py         # the biome pass and the tree behind it
python tools/check_surface_rules.py  # against 7 real vanilla chunks
python tools/check_carvers.py        # determinism and dispatch; no oracle
python tools/check_features.py       # determinism, dispatch, distributions
```

`check_surface_rules.py` is the one with a true oracle: it replays the stored
`before` arrays from `assets/vanilla_chunks/` through the surface pass and
compares against the stored `after`.
