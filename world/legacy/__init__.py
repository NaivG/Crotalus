"""
`world/legacy` — the self-developed terrain generator.

This is the half of world generation that is fast and *not* Minecraft. A chunk
here is a numba Perlin heightmap extruded into stone, with ore veins and
wandering-tunnel caves of this package's own design. It exists because it costs
about 20 ms a chunk where the vanilla pipeline costs about a second, and for
years it was the only terrain there was.

Its replacement is `world/vanilla/terrain.py`, which builds the same shapes out
of the density functions in `assets/density_function.json` and lands within
0.73% of a real vanilla chunk. Pick one with `[world] generator` in
`config.toml`; `world/feature/` dresses either of them the same way.

    from world.legacy.noise import PerlinNoise
    from world.legacy.generator import LegacyWorldGenerator

Deliberately nothing is imported here. `world/generator.py` loads the backend
lazily, and a package body that pulled in its own children would make
`import world.legacy.noise` pay for `cave_carver`'s numba kernels too. Reach
for the module you want directly.

See `README.md` for what each backend is for and what the knobs mean.
"""

__all__ = [
    "LegacyWorldGenerator",
    "LegacyGeneratorSettings",
    "CaveSettings",
    "PerlinNoise",
    "SimplexNoise",
    "carve_chunk_caves",
    "carve_cheese",
]
