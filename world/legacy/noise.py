"""
Noise Generation Module

Provides fast noise sampling using numba+numpy acceleration.
Includes Perlin noise, Simplex noise, and octave noise for terrain generation.
"""

import numpy as np
from typing import Optional, Tuple
from numba import njit, prange
from numba.typed import List

# Try to import numba, fall back to pure numpy if not available
try:
    from numba import jit, vectorize
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorators
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])
    def prange(n):
        return range(n)


# Constants for noise generation
PERLIN_YWRAP = 4
PERLIN_ZWRAP = 8
PERLIN_SIZE = 4095

# Permutation table for noise
_perm: Optional[np.ndarray] = None
_perm_mod12: Optional[np.ndarray] = None


def init_perlin(seed: int) -> None:
    """Initialize the permutation table with a given seed."""
    global _perm, _perm_mod12
    
    rng = np.random.default_rng(seed)
    p = np.arange(256, dtype=np.int64)
    rng.shuffle(p)
    
    _perm = np.concatenate([p, p]).astype(np.int64)
    _perm_mod12 = _perm % 12


# Gradients for 3D noise
_GRAD3 = np.array([
    [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
    [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
    [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]
], dtype=np.float64)


@njit(cache=True)
def _fade(t: float) -> float:
    """Fade function for smooth interpolation (6t^5 - 15t^4 + 10t^3)."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@njit(cache=True)
def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation."""
    return a + t * (b - a)


@njit(cache=True)
def _grad(hash_val: int, x: float, y: float, z: float) -> float:
    """Calculate gradient."""
    h = hash_val & 15
    u = x if h < 8 else y
    v = y if h < 4 else (x if h in (12, 14) else z)
    return (u if h & 1 == 0 else -u) + (v if h & 2 == 0 else -v)


@njit(cache=True)
def perlin_noise_3d(x: float, y: float, z: float, perm: np.ndarray) -> float:
    """3D Perlin noise function."""
    X = int(np.floor(x)) & 255
    Y = int(np.floor(y)) & 255
    Z = int(np.floor(z)) & 255
    
    x -= np.floor(x)
    y -= np.floor(y)
    z -= np.floor(z)
    
    u = _fade(x)
    v = _fade(y)
    w = _fade(z)
    
    A = perm[X] + Y
    AA = perm[A] + Z
    AB = perm[A + 1] + Z
    B = perm[X + 1] + Y
    BA = perm[B] + Z
    BB = perm[B + 1] + Z
    
    return _lerp(
        _lerp(
            _lerp(_grad(perm[AA], x, y, z), _grad(perm[BA], x - 1, y, z), u),
            _lerp(_grad(perm[AB], x, y - 1, z), _grad(perm[BB], x - 1, y - 1, z), u),
            v
        ),
        _lerp(
            _lerp(_grad(perm[AA + 1], x, y, z - 1), _grad(perm[BA + 1], x - 1, y, z - 1), u),
            _lerp(_grad(perm[AB + 1], x, y - 1, z - 1), _grad(perm[BB + 1], x - 1, y - 1, z - 1), u),
            v
        ),
        w
    )


@njit(cache=True, parallel=True)
def perlin_noise_2d_array(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    perm: np.ndarray
) -> np.ndarray:
    """Generate 2D Perlin noise for an array of coordinates (fast)."""
    n = len(x_coords)
    result = np.zeros(n, dtype=np.float64)
    
    for i in prange(n):
        result[i] = perlin_noise_3d(x_coords[i], 0.0, y_coords[i], perm)
    
    return result


@njit(cache=True, parallel=True)
def perlin_noise_3d_array(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_coords: np.ndarray,
    perm: np.ndarray
) -> np.ndarray:
    """Generate 3D Perlin noise for an array of coordinates (fast)."""
    n = len(x_coords)
    result = np.zeros(n, dtype=np.float64)
    
    for i in prange(n):
        result[i] = perlin_noise_3d(x_coords[i], y_coords[i], z_coords[i], perm)
    
    return result


@njit(cache=True)
def _fast_fade(t: float) -> float:
    """Optimized fade function."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@njit(cache=True, parallel=True)
def octave_noise_2d_fast(
    x: float,
    y: float,
    octaves: int,
    persistence: float,
    lacunarity: float,
    scale: float,
    perm: np.ndarray
) -> float:
    """Fast octave noise for single point."""
    total = 0.0
    frequency = 1.0 / scale
    amplitude = 1.0
    max_value = 0.0
    
    for _ in range(octaves):
        total += perlin_noise_3d(
            x * frequency,
            0.0,
            y * frequency,
            perm
        ) * amplitude
        max_value += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    
    return total / max_value


@njit(cache=True)
def octave_noise_3d_array(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_coords: np.ndarray,
    octaves: int,
    persistence: float,
    lacunarity: float,
    scale: float,
    perm: np.ndarray
) -> np.ndarray:
    """`octave_noise_3d_fast` over a whole array of points.

    Bit-for-bit identical to calling the scalar form in a loop -- the octave
    accumulation is unchanged, only the per-call Python/dispatch overhead
    disappears. That overhead is what made the scalar form unusable for a
    whole cave volume: generation asked for 16 x 16 x 99 points, and every
    one of them was a separate call across the Python boundary.

    Deliberately not `parallel=True`, and deliberately calling the scalar
    `perlin_noise_3d` rather than `perlin_noise_3d_array`. The generator runs
    on a pool of worker threads already, so numba's threading layer would
    oversubscribe the CPU, and nesting a parallel region inside another one
    degrades to serial anyway.
    """
    n = len(x_coords)
    result = np.zeros(n, dtype=np.float64)
    frequency = 1.0 / scale
    amplitude = 1.0
    max_value = 0.0

    for _ in range(octaves):
        for i in range(n):
            result[i] += perlin_noise_3d(
                x_coords[i] * frequency,
                y_coords[i] * frequency,
                z_coords[i] * frequency,
                perm
            ) * amplitude
        max_value += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    return result / max_value


@njit(cache=True, parallel=True)
def octave_noise_3d_fast(
    x: float,
    y: float,
    z: float,
    octaves: int,
    persistence: float,
    lacunarity: float,
    scale: float,
    perm: np.ndarray
) -> float:
    """Fast octave noise for single 3D point."""
    total = 0.0
    frequency = 1.0 / scale
    amplitude = 1.0
    max_value = 0.0
    
    for _ in range(octaves):
        total += perlin_noise_3d(
            x * frequency,
            y * frequency,
            z * frequency,
            perm
        ) * amplitude
        max_value += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    
    return total / max_value


class PerlinNoise:
    """Perlin noise generator with numba acceleration."""
    
    def __init__(
        self,
        seed: int = 0,
        octaves: int = 4,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        scale: float = 256.0
    ):
        """
        Initialize Perlin noise generator.
        
        Args:
            seed: Random seed for reproducibility
            octaves: Number of noise layers
            persistence: Amplitude reduction per octave
            lacunarity: Frequency increase per octave
            scale: Base scale factor
        """
        self.seed = seed
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.scale = scale
        init_perlin(seed)
    
    def noise2d(self, x: float, y: float) -> float:
        """Get 2D noise value at coordinates."""
        return octave_noise_2d_fast(
            x, y,
            self.octaves, self.persistence, self.lacunarity,
            self.scale, _perm
        )
    
    def noise3d(self, x: float, y: float, z: float) -> float:
        """Get 3D noise value at coordinates."""
        return octave_noise_3d_fast(
            x, y, z,
            self.octaves, self.persistence, self.lacunarity,
            self.scale, _perm
        )
    
    def noise2d_array(self, x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
        """Get 2D noise for an array of coordinates (fast batch operation)."""
        return perlin_noise_2d_array(x_coords, y_coords, _perm)
    
    def noise3d_array(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        z_coords: np.ndarray
    ) -> np.ndarray:
        """Get single-octave 3D noise for an array of coordinates."""
        return perlin_noise_3d_array(x_coords, y_coords, z_coords, _perm)

    def octave_noise3d_array(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        z_coords: np.ndarray
    ) -> np.ndarray:
        """Get multi-octave 3D noise for an array of coordinates.

        The array counterpart of `noise3d`, and the one to use when sampling
        a whole volume -- it matches `noise3d` exactly, but costs one
        compiled call instead of one per point.
        """
        return octave_noise_3d_array(
            x_coords, y_coords, z_coords,
            self.octaves, self.persistence, self.lacunarity,
            self.scale, _perm
        )
    
    def get_heightmap(
        self,
        chunk_x: int,
        chunk_z: int,
        width: int = 16,
        height: int = 16,
        sea_level: int = 62
    ) -> np.ndarray:
        """
        Generate a heightmap for a chunk.
        
        Args:
            chunk_x: Chunk X coordinate
            chunk_z: Chunk Z coordinate
            width: Width of the area
            height: Height of the area
            sea_level: Sea level Y coordinate
        
        Returns:
            2D numpy array of height values
        """
        world_x = chunk_x * 16
        world_z = chunk_z * 16
        
        heightmap = np.zeros((height, width), dtype=np.float64)
        
        for i in range(height):
            for j in range(width):
                x = world_x + j
                z = world_z + i
                noise_val = self.noise2d(x, z)
                heightmap[i, j] = int((noise_val + 1) * 0.5 * 128 + sea_level)
        
        return heightmap
    
    def get_column(
        self,
        world_x: int,
        world_z: int,
        max_height: int = 256
    ) -> np.ndarray:
        """
        Generate noise column for 3D terrain.
        
        Args:
            world_x: World X coordinate
            world_z: World Z coordinate
            max_height: Maximum height
        
        Returns:
            Array of noise values at each height
        """
        column = np.zeros(max_height, dtype=np.float64)
        for y in range(max_height):
            column[y] = self.noise3d(world_x, y, world_z)
        return column


class SimplexNoise:
    """Simplex noise generator (faster for higher dimensions)."""
    
    def __init__(
        self,
        seed: int = 0,
        octaves: int = 4,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        scale: float = 256.0
    ):
        """Initialize simplex noise generator."""
        self.seed = seed
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.scale = scale
        init_perlin(seed)
        
        # Skewing factors for 2D simplex
        self.F2 = 0.5 * (np.sqrt(3.0) - 1.0)
        self.G2 = (3.0 - np.sqrt(3.0)) / 6.0
        
        # Skewing factors for 3D simplex
        self.F3 = 1.0 / 3.0
        self.G3 = 1.0 / 6.0
    
    def noise2d(self, x: float, y: float) -> float:
        """Get 2D simplex noise value."""
        # Simplified implementation using perlin as base
        return octave_noise_2d_fast(
            x, y,
            self.octaves, self.persistence, self.lacunarity,
            self.scale, _perm
        )
    
    def noise3d(self, x: float, y: float, z: float) -> float:
        """Get 3D simplex noise value."""
        return octave_noise_3d_fast(
            x, y, z,
            self.octaves, self.persistence, self.lacunarity,
            self.scale, _perm
        )

    def octave_noise3d_array(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        z_coords: np.ndarray
    ) -> np.ndarray:
        """Get multi-octave 3D noise for an array of coordinates.

        The array counterpart of `noise3d`, and the one to use when sampling
        a whole volume -- it matches `noise3d` exactly, but costs one
        compiled call instead of one per point.
        """
        return octave_noise_3d_array(
            x_coords, y_coords, z_coords,
            self.octaves, self.persistence, self.lacunarity,
            self.scale, _perm
        )
    
    def get_heightmap(
        self,
        chunk_x: int,
        chunk_z: int,
        width: int = 16,
        height: int = 16,
        sea_level: int = 62
    ) -> np.ndarray:
        """Generate heightmap for a chunk."""
        world_x = chunk_x * 16
        world_z = chunk_z * 16
        
        heightmap = np.zeros((height, width), dtype=np.float64)
        
        for i in range(height):
            for j in range(width):
                x = world_x + j
                z = world_z + i
                noise_val = self.noise2d(x, z)
                heightmap[i, j] = int((noise_val + 1) * 0.5 * 128 + sea_level)
        
        return heightmap


# Initialize with default seed
init_perlin(0)


def create_noise_generator(
    noise_type: str = "perlin",
    seed: int = 0,
    **kwargs
) -> PerlinNoise:
    """
    Factory function to create noise generators.
    
    Args:
        noise_type: Type of noise ("perlin" or "simplex")
        seed: Random seed
        **kwargs: Additional parameters passed to the generator
    
    Returns:
        Noise generator instance
    """
    if noise_type.lower() == "perlin":
        return PerlinNoise(seed=seed, **kwargs)
    elif noise_type.lower() == "simplex":
        return SimplexNoise(seed=seed, **kwargs)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")


# Optimized numpy-based operations for chunk generation
@njit(cache=True, parallel=True)
def generate_heightmap_fast(
    chunk_x: int,
    chunk_z: int,
    width: int,
    depth: int,
    scale: float,
    octaves: int,
    persistence: float,
    lacunarity: float,
    base_height: float,
    height_variance: float,
    perm: np.ndarray
) -> np.ndarray:
    """
    Highly optimized heightmap generation using numba.

    The surface sits at `base_height` plus or minus half of `height_variance`,
    so the terrain straddles the base height rather than starting at it. That
    centring is what makes `base_height` mean what it says: with the overworld
    defaults (64, 64) the surface lands in 32..96 and the player spawns at
    roughly ground level.

    Returns heightmap with shape (depth, width).
    """
    heightmap = np.zeros((depth, width), dtype=np.float64)
    world_x_base = chunk_x * 16
    world_z_base = chunk_z * 16

    for i in prange(depth):
        for j in prange(width):
            x = float(world_x_base + j)
            z = float(world_z_base + i)

            # Calculate octave noise
            total = 0.0
            frequency = 1.0 / scale
            amplitude = 1.0
            max_value = 0.0

            for _ in range(octaves):
                noise_val = perlin_noise_3d(x * frequency, 0.0, z * frequency, perm)
                total += noise_val * amplitude
                max_value += amplitude
                amplitude *= persistence
                frequency *= lacunarity

            noise_norm = total / max_value if max_value > 0 else 0.0
            height = int(noise_norm * 0.5 * height_variance + base_height)
            heightmap[i, j] = max(0, min(255, height))

    return heightmap


@njit(cache=True, parallel=True)
def generate_terrain_column_fast(
    world_x: int,
    world_z: int,
    height: int,
    scale: float,
    octaves: int,
    persistence: float,
    lacunarity: float,
    sea_level: float,
    perm: np.ndarray
) -> np.ndarray:
    """
    Generate a terrain column for 3D terrain.
    
    Args:
        world_x: World X coordinate
        world_z: World Z coordinate
        height: Maximum height
        scale: Noise scale
        octaves: Number of octaves
        persistence: Persistence value
        lacunarity: Lacunarity value
        sea_level: Sea level Y coordinate
        perm: Permutation table
    
    Returns:
        Array of block types (0 = air, 1 = stone, 2 = grass, etc.)
    """
    column = np.zeros(height, dtype=np.int32)
    
    for y in range(height):
        xf = float(world_x)
        yf = float(y)
        zf = float(world_z)
        
        # Calculate noise for terrain shape
        terrain_noise = 0.0
        frequency = 1.0 / scale
        amplitude = 1.0
        max_value = 0.0
        
        for _ in range(octaves):
            terrain_noise += perlin_noise_3d(xf * frequency, yf * frequency, zf * frequency, perm) * amplitude
            max_value += amplitude
            amplitude *= persistence
            frequency *= lacunarity
        
        terrain_noise /= max_value if max_value > 0 else 1.0
        
        # Determine block type based on height and noise
        if y < sea_level:
            column[y] = 9  # Water
        elif y == int(sea_level + terrain_noise * 20):
            column[y] = 2  # Grass
        elif y > int(sea_level + terrain_noise * 20):
            column[y] = 1  # Stone
        else:
            column[y] = 0  # Air
    
    return column


# Convenience function for quick heightmap generation
def get_chunk_heightmap(
    chunk_x: int,
    chunk_z: int,
    seed: int = 0,
    scale: float = 256.0,
    octaves: int = 4,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    base_height: float = 64.0,
    height_variance: float = 64.0
) -> np.ndarray:
    """
    Quick function to generate a chunk heightmap.

    Args:
        chunk_x: Chunk X coordinate
        chunk_z: Chunk Z coordinate
        seed: World seed
        scale: Noise scale
        octaves: Number of noise octaves
        persistence: Noise persistence
        lacunarity: Noise lacunarity
        base_height: Height the surface is centred on
        height_variance: Full peak-to-trough spread of the surface

    Returns:
        16x16 numpy array of height values
    """
    init_perlin(seed)
    return generate_heightmap_fast(
        chunk_x, chunk_z, 16, 16,
        scale, octaves, persistence, lacunarity,
        base_height, height_variance, _perm
    )
