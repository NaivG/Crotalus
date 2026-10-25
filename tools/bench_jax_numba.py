"""Compare numpy, numba, jax, jax+numba, jax+jit on the same workloads.

Three kernels exercise different shapes of work:

    K1 poly     vectorized polynomial + RMS reduction on a 1M array
    K2 field    3D distance field sqrt(x*x + y*y + z*z) over a 128^3 grid
    K3 scan     per-element branching loop on a 1M array (where numba wins)

Five variants per kernel:

    numpy        pure numpy, vectorized
    numpy+numba  @njit nopython loop over the numpy array
    jax          eager jax (still XLA-compiled internally, but no jit cache)
    jax+numba    jax computes the bulk, @njit finalises with a tight loop
    jax+jit      @jax.jit — the headline win candidate

JAX defaults to float32 and float32 SIMD; enable x64 so the comparison is
honest against numpy's float64 default.

Run: py -3.14 tools/bench_jax_numba.py [K1|K2|K3|all]
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from numba import njit


# ---------- workload inputs ----------

def make_poly_input(n: int = 1_000_000) -> np.ndarray:
    return np.linspace(-1.0, 1.0, n, dtype=np.float64)


def make_field_input(n: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    x, y, z = np.meshgrid(g, g, g, indexing="ij")
    return x, y, z


def make_scan_input(n: int = 1_000_000) -> np.ndarray:
    # ints so we can branch on sign without dtype promotion surprises
    return np.random.default_rng(0).integers(-1000, 1000, size=n, dtype=np.int64)


# ---------- kernel bodies ----------

# K1: y = a*x*x + b*x + c, then sqrt(mean(y*y))
A, B_, C = 0.1, -0.5, 3.14


def poly_numpy(x: np.ndarray) -> float:
    y = A * x * x + B_ * x + C
    return float(np.sqrt(np.mean(y * y)))


@njit(cache=True)
def _poly_loop(x: np.ndarray, out: np.ndarray) -> None:
    for i in range(x.size):
        out[i] = A * x[i] * x[i] + B_ * x[i] + C


def poly_numpy_numba(x: np.ndarray) -> float:
    out = np.empty_like(x)
    _poly_loop(x, out)
    return float(np.sqrt(np.mean(out * out)))


def poly_jax(x: jnp.ndarray) -> jnp.ndarray:
    y = A * x * x + B_ * x + C
    return jnp.sqrt(jnp.mean(y * y))


@jax.jit
def poly_jax_jit(x: jnp.ndarray) -> jnp.ndarray:
    y = A * x * x + B_ * x + C
    return jnp.sqrt(jnp.mean(y * y))


@njit(cache=True)
def _rms_loop(y: np.ndarray) -> float:
    s = 0.0
    for i in range(y.size):
        s += y[i] * y[i]
    return (s / y.size) ** 0.5


def poly_jax_numba(x: np.ndarray) -> float:
    # jax does the polynomial, numba does the manual reduction
    y_jax = A * jnp.asarray(x) * jnp.asarray(x) + B_ * jnp.asarray(x) + C
    y_np = np.asarray(y_jax)
    return _rms_loop(y_np)


# K2: d = sqrt(x*x + y*y + z*z), mean(d)
def field_numpy(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    return float(np.mean(np.sqrt(x * x + y * y + z * z)))


@njit(cache=True)
def _field_dist(x: np.ndarray, y: np.ndarray, z: np.ndarray, out: np.ndarray) -> None:
    n = x.size
    for i in range(n):
        out[i] = (x.flat[i] * x.flat[i]
                  + y.flat[i] * y.flat[i]
                  + z.flat[i] * z.flat[i]) ** 0.5


def field_numpy_numba(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    out = np.empty(x.size, dtype=np.float64)
    _field_dist(x, y, z, out)
    return float(out.mean())


def field_jax(x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.sqrt(x * x + y * y + z * z))


@jax.jit
def field_jax_jit(x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.sqrt(x * x + y * y + z * z))


@njit(cache=True)
def _mean_loop(d: np.ndarray) -> float:
    s = 0.0
    for i in range(d.size):
        s += d[i]
    return s / d.size


def field_jax_numba(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    d_jax = jnp.sqrt(jnp.asarray(x) * jnp.asarray(x)
                     + jnp.asarray(y) * jnp.asarray(y)
                     + jnp.asarray(z) * jnp.asarray(z))
    return _mean_loop(np.asarray(d_jax).ravel())


# K3: signed step — for each int x[i], if x[i] > 0 add it, else subtract it.
# then return running max so the compiler can't trivially vectorise away the branch.

@njit(cache=True)
def _signed_max_loop(x: np.ndarray) -> int:
    running = 0
    peak = 0
    for i in range(x.size):
        if x[i] > 0:
            running += x[i]
        else:
            running -= x[i]
        if running > peak:
            peak = running
    return peak


def scan_numpy(x: np.ndarray) -> int:
    # vectorised: positives add, negatives subtract, then running max via cumsum trick
    signed = np.where(x > 0, x, -x).astype(np.int64)
    running = np.cumsum(signed)
    return int(running.max())


def scan_numpy_numba(x: np.ndarray) -> int:
    return _signed_max_loop(x)


def scan_jax(x: jnp.ndarray) -> jnp.ndarray:
    signed = jnp.where(x > 0, x, -x)
    running = jnp.cumsum(signed)
    return jnp.max(running)


@jax.jit
def scan_jax_jit(x: jnp.ndarray) -> jnp.ndarray:
    signed = jnp.where(x > 0, x, -x)
    running = jnp.cumsum(signed)
    return jnp.max(running)


def scan_jax_numba(x: np.ndarray) -> int:
    # jax "computes" the signed array (small win, eager); numba does the scan
    signed = np.asarray(jnp.where(jnp.asarray(x) > 0, jnp.asarray(x), -jnp.asarray(x)))
    return _signed_max_loop(signed)


# ---------- timing harness ----------

@dataclass(frozen=True)
class Result:
    label: str
    seconds: float
    ok: bool


def time_call(label: str, fn, repeat: int = 5) -> Result:
    """Run fn once to warm (JIT compile), then time repeat times and take min."""
    try:
        fn()                  # warm: absorbs jit compile / cache build
        fn()                  # one more in case the first hit dispatched a lazy init
        best = min(_perf(fn) for _ in range(repeat))
        return Result(label, best, True)
    except Exception as exc:  # noqa: BLE001 — surface as a row, not a crash
        print(f"  {label:14s} FAILED: {type(exc).__name__}: {exc}")
        return Result(label, float("nan"), False)


def _perf(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def report(name: str, results: list[Result]) -> None:
    ok = [r for r in results if r.ok]
    if not ok:
        print(f"\n=== {name} ===  (no successful runs)")
        return
    base = min(r.seconds for r in ok)
    print(f"\n=== {name} ===")
    print(f"  {'variant':14s} {'time':>10s}  {'speedup':>9s}")
    for r in sorted(results, key=lambda r: (not r.ok, r.seconds)):
        if not r.ok:
            print(f"  {r.label:14s}        —  failed")
            continue
        ratio = base / r.seconds if r.seconds > 0 else float("inf")
        print(f"  {r.label:14s} {r.seconds * 1e3:8.2f} ms  {ratio:8.2f}x")


def run_k1() -> None:
    x_np = make_poly_input()
    x_jax = jnp.asarray(x_np)
    report("K1 poly (1M float64, vectorized + RMS)", [
        time_call("numpy",        lambda: poly_numpy(x_np)),
        time_call("numpy+numba",  lambda: poly_numpy_numba(x_np)),
        time_call("jax",          lambda: float(poly_jax(x_jax))),
        time_call("jax+numba",    lambda: poly_jax_numba(x_np)),
        time_call("jax+jit",      lambda: float(poly_jax_jit(x_jax))),
    ])


def run_k2() -> None:
    x_np, y_np, z_np = make_field_input()
    x_jax, y_jax, z_jax = jnp.asarray(x_np), jnp.asarray(y_np), jnp.asarray(z_np)
    report("K2 field (128^3 float64, sqrt + mean)", [
        time_call("numpy",        lambda: field_numpy(x_np, y_np, z_np)),
        time_call("numpy+numba",  lambda: field_numpy_numba(x_np, y_np, z_np)),
        time_call("jax",          lambda: float(field_jax(x_jax, y_jax, z_jax))),
        time_call("jax+numba",    lambda: field_jax_numba(x_np, y_np, z_np)),
        time_call("jax+jit",      lambda: float(field_jax_jit(x_jax, y_jax, z_jax))),
    ])


def run_k3() -> None:
    x_np = make_scan_input()
    x_jax = jnp.asarray(x_np)
    report("K3 scan (1M int64, signed running max)", [
        time_call("numpy",        lambda: scan_numpy(x_np)),
        time_call("numpy+numba",  lambda: scan_numpy_numba(x_np)),
        time_call("jax",          lambda: int(scan_jax(x_jax))),
        time_call("jax+numba",    lambda: scan_jax_numba(x_np)),
        time_call("jax+jit",      lambda: int(scan_jax_jit(x_jax))),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kernel", nargs="?", default="all",
                        choices=["K1", "K2", "K3", "all", "k1", "k2", "k3"])
    parser.add_argument("--repeat", type=int, default=5,
                        help="timing repeats per variant (default 5)")
    args = parser.parse_args()
    want = args.kernel.lower()
    print(f"numpy {np.__version__}  jax {jax.__version__}  x64={jax.config.read('jax_enable_x64')}")
    if want in ("k1", "all"):
        run_k1()
    if want in ("k2", "all"):
        run_k2()
    if want in ("k3", "all"):
        run_k3()


if __name__ == "__main__":
    main()
