"""
Stress testing for autocallable products.

Historical and hypothetical scenarios measuring impact on product value.

Reference: Basel III stress testing framework; FRTB guidelines.
"""

import numpy as np
from .autocall import Autocallable


HISTORICAL_SCENARIOS = {
    "Lehman 2008": {"spot_shock": -0.35, "vol_shock": 0.30},
    "COVID Mars 2020": {"spot_shock": -0.30, "vol_shock": 0.40},
    "Crise Euro 2011": {"spot_shock": -0.20, "vol_shock": 0.15},
    "Flash Crash 2010": {"spot_shock": -0.10, "vol_shock": 0.20},
    "Correction Fev 2018": {"spot_shock": -0.12, "vol_shock": 0.25},
    "SG Kerviel 2008": {"spot_shock": -0.07, "vol_shock": 0.10},
}

HYPOTHETICAL_SCENARIOS = {
    "Marche stable (+5%)": {"spot_shock": 0.05, "vol_shock": -0.02},
    "Correction moderee (-10%)": {"spot_shock": -0.10, "vol_shock": 0.10},
    "Crash severe (-25%)": {"spot_shock": -0.25, "vol_shock": 0.25},
    "Crash extreme (-40%)": {"spot_shock": -0.40, "vol_shock": 0.35},
    "Vol spike seul": {"spot_shock": 0.0, "vol_shock": 0.20},
    "Vol collapse": {"spot_shock": 0.05, "vol_shock": -0.10},
}


def stress_test(
    product: Autocallable,
    spot_shock: float,
    vol_shock: float,
    n_paths: int = 100_000,
    seed: int = 42,
) -> dict:
    """
    Apply a stress scenario and measure impact.

    Parameters
    ----------
    spot_shock : float
        Relative spot change (e.g., -0.30 = -30%).
    vol_shock : float
        Absolute vol change (e.g., 0.20 = +20 vol points).

    Returns
    -------
    dict with base price, stressed price, P&L impact, and risk metrics.
    """
    base = product.price(n_paths=n_paths, seed=seed)

    new_S0 = product.S0 * (1 + spot_shock)
    new_sigma = max(product.sigma + vol_shock, 0.01)

    stressed = product.price_for_greeks(
        S0=new_S0, sigma=new_sigma, n_paths=n_paths, seed=seed
    )

    stressed_full = Autocallable(
        S0=new_S0,
        autocall_barrier=product.autocall_barrier / product.S0,
        coupon_barrier=product.coupon_barrier / product.S0,
        ki_barrier=product.ki_barrier / product.S0,
        coupon_rate=product.coupon_rate,
        n_observations=product.n_observations,
        T=product.T,
        r=product.r,
        sigma=new_sigma,
        notional=product.notional,
    )
    stressed_detail = stressed_full.price(n_paths=n_paths, seed=seed)

    pnl_impact = stressed - base["price"]
    pnl_pct = pnl_impact / product.notional * 100

    return {
        "base_price": base["price"],
        "stressed_price": stressed,
        "pnl_impact": pnl_impact,
        "pnl_pct": pnl_pct,
        "ki_prob_base": base["ki_probability"],
        "ki_prob_stressed": stressed_detail["ki_probability"],
        "autocall_prob_base": base["total_autocall_prob"],
        "autocall_prob_stressed": stressed_detail["total_autocall_prob"],
    }


def run_all_scenarios(
    product: Autocallable,
    n_paths: int = 100_000,
    seed: int = 42,
) -> dict:
    """Run all historical and hypothetical stress scenarios."""
    results = {}

    for name, params in HISTORICAL_SCENARIOS.items():
        results[name] = stress_test(product, **params, n_paths=n_paths, seed=seed)
        results[name]["type"] = "historical"

    for name, params in HYPOTHETICAL_SCENARIOS.items():
        results[name] = stress_test(product, **params, n_paths=n_paths, seed=seed)
        results[name]["type"] = "hypothetical"

    return results


def find_worst_scenario(results: dict) -> tuple[str, dict]:
    """Identify the worst-case scenario by P&L impact."""
    worst_name = min(results, key=lambda k: results[k]["pnl_impact"])
    return worst_name, results[worst_name]
