"""
Utility functions for autocallable structured product analysis.

Reference: Hull, "Options, Futures, and Other Derivatives", Ch. 14.
"""

import numpy as np


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
