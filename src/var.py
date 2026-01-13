"""
Value at Risk (VaR) and backtesting for autocallable products.

Reference: Hull Ch. 22 (Value at Risk); Basel III/FRTB internal models approach.
"""

import numpy as np
from scipy import stats
from .autocall import Autocallable


def var_historical(
    product: Autocallable,
    historical_returns: np.ndarray,
    confidence: float = 0.99,
    horizon: int = 1,
    n_paths: int = 50_000,
    seed: int = 42,
) -> dict:
    """
    Historical VaR: reprice product under each historical return scenario.

    Parameters
    ----------
    historical_returns : np.ndarray
        Array of historical daily log returns.
    confidence : float
        Confidence level (e.g., 0.99).
    horizon : int
        Holding period in days.

    Returns
    -------
    dict with VaR, CVaR, and P&L distribution statistics.
    """
    if horizon > 1:
        n_windows = len(historical_returns) - horizon + 1
        multi_day_returns = np.array([
            np.sum(historical_returns[i:i + horizon])
            for i in range(n_windows)
        ])
        returns_used = multi_day_returns
    else:
        returns_used = historical_returns

    base_price = product.price_for_greeks(n_paths=n_paths, seed=seed)

    pnl_scenarios = []
    for ret in returns_used:
        new_S0 = product.S0 * np.exp(ret)
        new_price = product.price_for_greeks(S0=new_S0, n_paths=n_paths, seed=seed)
        pnl_scenarios.append(new_price - base_price)

    pnl_scenarios = np.array(pnl_scenarios)

    alpha = 1 - confidence
    var = -np.percentile(pnl_scenarios, alpha * 100)
    cvar = -np.mean(pnl_scenarios[pnl_scenarios <= -var])

    return {
        "var": var,
        "cvar": cvar,
        "confidence": confidence,
        "horizon": horizon,
        "n_scenarios": len(returns_used),
        "mean_pnl": np.mean(pnl_scenarios),
        "std_pnl": np.std(pnl_scenarios),
        "min_pnl": np.min(pnl_scenarios),
        "max_pnl": np.max(pnl_scenarios),
        "pnl_distribution": pnl_scenarios,
    }


def var_parametric(
    product: Autocallable,
    daily_vol: float,
    confidence: float = 0.99,
    horizon: int = 1,
    n_paths: int = 50_000,
    seed: int = 42,
) -> dict:
    """
    Parametric (delta-normal) VaR using Greeks.

    VaR ≈ |Δ| * S * σ * z_α * √(horizon)
    """
    from . import greeks as greeks_module

    g = greeks_module.compute_all_greeks(product, n_paths=n_paths, seed=seed)

    z_alpha = stats.norm.ppf(confidence)
    dollar_delta = abs(g["delta"]) * product.S0
    linear_var = dollar_delta * daily_vol * z_alpha * np.sqrt(horizon)

    gamma_adjustment = 0.5 * abs(g["gamma"]) * (product.S0 * daily_vol)**2
    var_with_gamma = linear_var + gamma_adjustment

    return {
        "var_linear": linear_var,
        "var_with_gamma": var_with_gamma,
        "confidence": confidence,
        "horizon": horizon,
        "delta": g["delta"],
        "gamma": g["gamma"],
        "z_alpha": z_alpha,
    }


def kupiec_test(
    violations: np.ndarray,
    confidence: float = 0.99,
) -> dict:
    """
    Kupiec POF test for VaR backtesting.

    H0: observed violation rate = expected violation rate (1 - confidence).
    """
    n = len(violations)
    x = np.sum(violations)

    if x == 0 or x == n:
        return {
            "test_statistic": 0.0,
            "p_value": 1.0,
            "reject_H0": False,
            "n_observations": n,
            "n_violations": int(x),
            "violation_rate": x / n,
            "expected_rate": 1 - confidence,
        }

    p0 = 1 - confidence
    p_obs = x / n

    lr = 2 * (
        x * np.log(p_obs / p0)
        + (n - x) * np.log((1 - p_obs) / (1 - p0))
    )

    p_value = 1 - stats.chi2.cdf(lr, df=1)

    return {
        "test_statistic": lr,
        "p_value": p_value,
        "reject_H0": p_value < 0.05,
        "n_observations": n,
        "n_violations": int(x),
        "violation_rate": p_obs,
        "expected_rate": p0,
    }
