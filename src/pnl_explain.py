"""
PnL Explanation (Attribution) for autocallable products.

Decomposes daily P&L into Greek contributions using Taylor expansion:
    ΔP&L ≈ Δ·δS + ½Γ·(δS)² + ν·δσ + Θ·δt + unexplained

Reference: Hull Ch. 19 (Greek Letters).
"""

import numpy as np
from .autocall import Autocallable
from . import greeks as greeks_module


def pnl_attribution(
    product: Autocallable,
    dS: float,
    d_sigma: float = 0.0,
    dt_days: float = 1.0,
    n_paths: int = 200_000,
    seed: int = 42,
) -> dict:
    """
    Decompose P&L for a single day's market move.

    Parameters
    ----------
    product : Autocallable
        The product to analyze.
    dS : float
        Change in spot price (absolute).
    d_sigma : float
        Change in volatility (absolute, e.g., 0.02 = +2%).
    dt_days : float
        Time elapsed in days (default 1).
    n_paths : int
        Monte Carlo paths for pricing.
    seed : int
        Random seed.

    Returns
    -------
    dict with P&L decomposition:
        'actual_pnl': total P&L from repricing
        'delta_pnl': Δ * δS
        'gamma_pnl': ½ * Γ * (δS)²
        'vega_pnl': ν * δσ
        'theta_pnl': Θ * δt
        'unexplained': residual
        'explained_pct': percentage of P&L explained by Greeks
    """
    g = greeks_module.compute_all_greeks(product, n_paths=n_paths, seed=seed)

    V_before = product.price_for_greeks(n_paths=n_paths, seed=seed)

    new_S0 = product.S0 + dS
    new_sigma = product.sigma + d_sigma
    new_T = product.T - dt_days / 365.0

    V_after = product.price_for_greeks(
        S0=new_S0, sigma=new_sigma, T=new_T, n_paths=n_paths, seed=seed
    )

    actual_pnl = V_after - V_before

    delta_pnl = g["delta"] * dS
    gamma_pnl = 0.5 * g["gamma"] * dS**2
    vega_pnl = g["vega"] * d_sigma
    theta_pnl = g["theta"] * dt_days

    explained = delta_pnl + gamma_pnl + vega_pnl + theta_pnl
    unexplained = actual_pnl - explained

    explained_pct = (explained / actual_pnl * 100) if abs(actual_pnl) > 1e-10 else 0.0

    return {
        "actual_pnl": actual_pnl,
        "delta_pnl": delta_pnl,
        "gamma_pnl": gamma_pnl,
        "vega_pnl": vega_pnl,
        "theta_pnl": theta_pnl,
        "explained": explained,
        "unexplained": unexplained,
        "explained_pct": explained_pct,
        "greeks": g,
    }


def multi_day_pnl(
    product: Autocallable,
    spot_path: np.ndarray,
    vol_path: np.ndarray | None = None,
    n_paths: int = 100_000,
    seed: int = 42,
) -> list[dict]:
    """
    Run PnL attribution over multiple days.

    Parameters
    ----------
    spot_path : np.ndarray
        Array of daily spot prices (day 0 = current).
    vol_path : np.ndarray, optional
        Array of daily implied volatilities. If None, assumes constant vol.

    Returns
    -------
    list of daily PnL attribution dicts.
    """
    n_days = len(spot_path) - 1
    if vol_path is None:
        vol_path = np.full(len(spot_path), product.sigma)

    results = []
    current_S0 = product.S0
    current_sigma = product.sigma
    current_T = product.T

    for day in range(n_days):
        dS = spot_path[day + 1] - spot_path[day]
        d_sigma = vol_path[day + 1] - vol_path[day]

        product.S0 = spot_path[day]
        product.sigma = vol_path[day]
        product.T = current_T - day / 365.0

        ratio = spot_path[day] / current_S0
        product.autocall_barrier = product.autocall_barrier
        product.coupon_barrier = product.coupon_barrier
        product.ki_barrier = product.ki_barrier

        attr = pnl_attribution(
            product, dS=dS, d_sigma=d_sigma, dt_days=1.0,
            n_paths=n_paths, seed=seed + day,
        )
        attr["day"] = day + 1
        attr["spot"] = spot_path[day + 1]
        results.append(attr)

    product.S0 = current_S0
    product.sigma = current_sigma
    product.T = current_T

    return results
