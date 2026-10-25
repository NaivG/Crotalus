# `world/legacy` — the self-developed terrain generator

The half of world generation that is fast and **not** Minecraft. A chunk here is
a numba Perlin heightmap extruded into stone, with ore veins and a
wandering-tunnel carver of this project's own design. It costs about 20 ms a
chunk where the vanilla pipeline costs about a second, which is the difference
between a world you can walk around in and a loading screen.

Its replacement is `world/vanilla/terrain.py`, which builds the same shapes out
of the density functions in `assets/density_function.json` and lands within
0.4–2.6 % of a real vanilla chunk cell for cell. Choose between them with
`[world] generator` in `config.toml`.

```
legacy/
├── noise.py         numba Perlin/Simplex, and the vectorised heightmap
├── cave_carver.py   wandering tunnels + open chambers
└── generator.py     LegacyWorldGenerator: the three stages above, assembled
```

## What is legacy and what is not

`LegacyWorldGenerator` decides the shape of the stone and nothing else. Every
layer on top of it - biomes, surface rules, the vanilla carver, features - is
`world/feature/`, inherited from `BaseGenerator`, and is **identical** to what
the vanilla backend produces. So a legacy world is not a Minecraft world; it is
a Minecraft *surface* over a Perlin landscape.

Concretely, `LegacyWorldGenerator` implements three of the pipeline's stages and
inherits the rest:

| stage | here | notes |
| --- | --- | --- |
| terrain | **this package** | Perlin heightmap → stone |
| biomes | `world/feature/biomes.py` | vanilla |
| surface | `world/feature/surface.py` | vanilla, and it lays the bedrock |
| ore | **this package** | a hardcoded table of six ores |
| caves | **this package** | unless `cave_mode = "vanilla"` |
| features | `world/feature/features.py` | vanilla, and it places its own ore |

That last row is why `_generate_ores` is an overridable hook on the base class
rather than a layer: on a vanilla world the ore arrives with the feature pass,
so running both would place every vein twice.

## The knobs

In `GeneratorSettings` (all read only by this backend):

| field | default | what it does |
| --- | --- | --- |
| `noise_scale` | 256.0 | horizontal size of the noise features, in blocks |
| `noise_octaves` | 4 | how many octaves of fBm stack |
| `noise_persistence` | 0.5 | amplitude falloff per octave |
| `noise_lacunarity` | 2.0 | frequency multiplier per octave |
| `base_height` | 64 | centre of the height band |
| `height_variance` | 64 | half-height of the band |
| `generate_ores` | True | the six-ore table above |

`CaveSettings` (in `generator.py`, held as `LegacyWorldGenerator.caves`) tunes
the tunnel carver. The two worth understanding before touching anything else
are `radius` and `cave_count`, because cave density scales with
`radius^2 * cave_count`.

The nether's legacy terrain is deliberately shaped differently from the
overworld's - `world/world.py` passes `base_height=32, height_variance=32` for
it. Those two fields reach the vanilla backend too and are ignored there; its
nether shape comes from the nether's density function.

## Reaching it directly

`create_generator` loads this backend lazily and only when the config asks for
it, so nothing imports this package on a vanilla world:

```python
from world.generator import create_generator, GeneratorSettings
from world.legacy.generator import LegacyWorldGenerator

# By name, ignoring the config:
generator = create_generator(GeneratorSettings(seed=1234, preset="overworld"),
                             backend="legacy")

# Or directly, which is what the tests do:
generator = LegacyWorldGenerator(GeneratorSettings(seed=1234, preset="overworld"))
```

## Tests

```bash
python tools/check_terrain_equivalence.py   # the Perlin fill vs a plain-loop reference
python tools/check_cave_connectivity.py     # the carver's tunnels connect
python tools/check_cave_invariants.py       # carving leaves the bedrock floor alone
python tools/check_terrain_shape.py         # section 4: this backend vs its baseline
```

`check_terrain_equivalence.py` and `check_cave_invariants.py` pin themselves to
this backend explicitly; both are about Perlin-era behaviour and neither means
the same thing against a density-driven world.

`check_terrain_shape.py` section 4 holds this backend to
`tools/baselines/legacy_generator.json`, in two sections of differing strength.
The default section's digests came from the pre-split generator, so a match
there is proof of byte-identity. The `cave_mode_legacy` section covers
`_carve_legacy` — which the default never reaches, because `cave_mode` defaults
to `"vanilla"` — and was recorded after the split, so it detects changes rather
than proving the original. The `_note` field in that file says so.
