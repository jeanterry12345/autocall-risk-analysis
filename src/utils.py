"""
Utility functions for autocallable structured product analysis.

Reference: Hull, "Options, Futures, and Other Derivatives", Ch. 14.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .local_vol import VolModel


def simulate_gbm_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate GBM paths using vectorized log-normal dynamics.

    S(t+dt) = S(t) * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)

    Parameters
    ----------
    S0 : float
        Initial spot price.
    r : float
        Risk-free rate (annualized).
    sigma : float
        Volatility (annualized).
    T : float
        Time to maturity in years.
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of simulated paths.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (n_steps + 1, n_paths) with S0 at index 0.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    Z = rng.standard_normal((n_steps, n_paths))
    log_returns = drift + diffusion * Z

    log_paths = np.zeros((n_steps + 1, n_paths))
    log_paths[0] = np.log(S0)
    log_paths[1:] = np.cumsum(log_returns, axis=0) + np.log(S0)

    return np.exp(log_paths)


def simulate_paths_local_vol(
    S0: float,
    r: float,
    vol_model: VolModel,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate paths under a general volatility model (local vol, term structure, etc.).

    Uses Euler-Maruyama discretization in log-space:
        ln S(t+dt) = ln S(t) + (r - 0.5*sigma(S,t)^2)*dt + sigma(S,t)*sqrt(dt)*Z

    The vol_model.vol_vectorized(S_array, t) method is called once per timestep
    for all paths simultaneously (vectorized for performance).

    Parameters
    ----------
    S0 : float
        Initial spot price.
    r : float
        Risk-free rate.
    vol_model : VolModel
        Object with vol(S, t) and optionally vol_vectorized(S_array, t).
    T : float
        Time to maturity.
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of paths.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Shape (n_steps + 1, n_paths), same as simulate_gbm_paths.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    paths = np.zeros((n_steps + 1, n_paths))
    paths[0] = S0

    # Check if model supports vectorized evaluation
    has_vectorized = hasattr(vol_model, "vol_vectorized")

    for step in range(n_steps):
        t = step * dt
        S_current = paths[step]

        # Get volatilities for all paths at this timestep
        if has_vectorized:
            sigmas = vol_model.vol_vectorized(S_current, t)
        else:
            sigmas = np.array([vol_model.vol(S, t) for S in S_current])

        Z = rng.standard_normal(n_paths)
        drift = (r - 0.5 * sigmas**2) * dt
        diffusion = sigmas * np.sqrt(dt) * Z

        paths[step + 1] = S_current * np.exp(drift + diffusion)

    return paths


def simulate_gbm_paths_antithetic(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Antithetic variates for variance reduction.

    Generates n_paths/2 original paths and n_paths/2 antithetic paths.
    Total output: n_paths paths.
    """
    half = n_paths // 2

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    Z = rng.standard_normal((n_steps, half))
    Z_anti = np.concatenate([Z, -Z], axis=1)

    log_returns = drift + diffusion * Z_anti

    log_paths = np.zeros((n_steps + 1, n_paths))
    log_paths[0] = np.log(S0)
    log_paths[1:] = np.cumsum(log_returns, axis=0) + np.log(S0)

    return np.exp(log_paths)
