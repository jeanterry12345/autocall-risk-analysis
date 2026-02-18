"""
Greeks calculation for autocallable products via finite difference (bump and reprice).

Central differences with common random seed for noise reduction.

Reference: Glasserman, "Monte Carlo Methods in Financial Engineering", Ch. 7.
"""

import numpy as np
from .autocall import Autocallable


def delta(
    product: Autocallable,
    bump_pct: float = 0.01,
    n_paths: int = 200_000,
    seed: int = 42,
) -> float:
    """Delta = dV/dS via central difference."""
    dS = product.S0 * bump_pct

    V_up = product.price_for_greeks(S0=product.S0 + dS, n_paths=n_paths, seed=seed)
    V_down = product.price_for_greeks(S0=product.S0 - dS, n_paths=n_paths, seed=seed)

    return (V_up - V_down) / (2 * dS)


def gamma(
    product: Autocallable,
    bump_pct: float = 0.01,
    n_paths: int = 200_000,
    seed: int = 42,
) -> float:
    """Gamma = d²V/dS² via central difference."""
    dS = product.S0 * bump_pct

    V_up = product.price_for_greeks(S0=product.S0 + dS, n_paths=n_paths, seed=seed)
    V_base = product.price_for_greeks(n_paths=n_paths, seed=seed)
    V_down = product.price_for_greeks(S0=product.S0 - dS, n_paths=n_paths, seed=seed)

    return (V_up - 2 * V_base + V_down) / (dS**2)


def vega(
    product: Autocallable,
    bump_abs: float = 0.01,
    n_paths: int = 200_000,
    seed: int = 42,
) -> float:
    """Vega = dV/dσ via central difference."""
    V_up = product.price_for_greeks(
        sigma=product.sigma + bump_abs, n_paths=n_paths, seed=seed
    )
    V_down = product.price_for_greeks(
        sigma=product.sigma - bump_abs, n_paths=n_paths, seed=seed
    )

    return (V_up - V_down) / (2 * bump_abs)


def theta(
    product: Autocallable,
    bump_days: float = 1.0,
    n_paths: int = 200_000,
    seed: int = 42,
) -> float:
    """Theta = dV/dT (daily decay)."""
    dt = bump_days / 365.0

    if product.T - dt <= 0:
        return 0.0

    V_base = product.price_for_greeks(n_paths=n_paths, seed=seed)
    V_short = product.price_for_greeks(T=product.T - dt, n_paths=n_paths, seed=seed)

    return (V_short - V_base) / bump_days


def rho(
    product: Autocallable,
    bump_abs: float = 0.001,
    n_paths: int = 200_000,
    seed: int = 42,
) -> float:
    """
    Rho = dV/dr (per 1% rate change).
    """
    V_up = product.price_for_greeks(r=product.r + bump_abs, n_paths=n_paths, seed=seed)
    V_down = product.price_for_greeks(r=product.r - bump_abs, n_paths=n_paths, seed=seed)

    return (V_up - V_down) / (2 * bump_abs)


def compute_all_greeks(
    product: Autocallable,
    n_paths: int = 200_000,
    seed: int = 42,
) -> dict:
    """Compute all Greeks for the autocallable."""
    return {
        "delta": delta(product, n_paths=n_paths, seed=seed),
        "gamma": gamma(product, n_paths=n_paths, seed=seed),
        "vega": vega(product, n_paths=n_paths, seed=seed),
        "theta": theta(product, n_paths=n_paths, seed=seed),
        "rho": rho(product, n_paths=n_paths, seed=seed),
    }


def delta_profile(
    product: Autocallable,
    spot_range: np.ndarray | None = None,
    n_paths: int = 100_000,
    seed: int = 42,
) -> dict:
    """
    Compute delta across a range of spot prices to visualize the
    discontinuity near the autocall barrier.

    Returns dict with 'spots' and 'deltas' arrays.
    """
    if spot_range is None:
        spot_range = np.linspace(product.S0 * 0.5, product.S0 * 1.5, 30)

    deltas = []
    original_S0 = product.S0

    for S in spot_range:
        product.S0 = S
        d = delta(product, n_paths=n_paths, seed=seed)
        deltas.append(d)

    product.S0 = original_S0

    return {"spots": spot_range, "deltas": np.array(deltas)}
