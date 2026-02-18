"""
Local volatility model derived from implied volatility surface via Dupire's formula.

Dupire (1994):
    sigma_loc^2(K, T) = dw/dT / D(k, w, dw/dk, d^2w/dk^2)

where w is total implied variance, k = ln(K/F), and D is the Dupire denominator
computed from SVI analytical derivatives.

The strike-direction derivatives are computed analytically from the SVI
parameterization (smooth, no numerical noise); the maturity-direction derivative
uses finite differences between adjacent calibrated slices.

Reference:
- Dupire, B. "Pricing with a Smile" (Risk, 1994).
- Gatheral, J. "The Volatility Surface" (Wiley, 2006), Ch. 2 & 5.
"""

import logging
from typing import Optional, Protocol, runtime_checkable

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .vol_surface import (
    ImpliedVolSurface,
    svi_total_variance,
    svi_dw_dk,
    svi_d2w_dk2,
)

logger = logging.getLogger(__name__)


# =====================================================================
# 1. VolModel Protocol (Strategy Pattern)
# =====================================================================

@runtime_checkable
class VolModel(Protocol):
    """Protocol for volatility models used in MC simulation."""

    def vol(self, S: float, t: float) -> float:
        """Return instantaneous volatility at spot S and time t."""
        ...

    @property
    def model_name(self) -> str:
        ...


class ConstantVol:
    """Constant volatility (backwards compatible with existing GBM code)."""

    def __init__(self, sigma: float):
        self._sigma = sigma

    def vol(self, S: float, t: float) -> float:
        return self._sigma

    @property
    def model_name(self) -> str:
        return f"Constant(σ={self._sigma:.2%})"

    def shift(self, delta: float) -> "ConstantVol":
        return ConstantVol(self._sigma + delta)


class TermStructureVol:
    """
    Deterministic ATM term structure: sigma(t) interpolated from calibrated vols.

    Ignores moneyness (no smile). Piecewise-constant between maturities.
    """

    def __init__(self, maturities: np.ndarray, atm_vols: np.ndarray):
        self._maturities = np.asarray(maturities)
        self._atm_vols = np.asarray(atm_vols)

    def vol(self, S: float, t: float) -> float:
        return float(np.interp(t, self._maturities, self._atm_vols))

    @property
    def model_name(self) -> str:
        return "TermStructure(ATM)"

    def shift(self, delta: float) -> "TermStructureVol":
        return TermStructureVol(self._maturities, self._atm_vols + delta)

    @classmethod
    def from_surface(cls, surface: ImpliedVolSurface) -> "TermStructureVol":
        """Build term structure from ATM vols of a calibrated surface."""
        mats = surface.maturities
        atm_vols = np.array([surface.atm_vol(T) for T in mats])
        return cls(mats, atm_vols)


# =====================================================================
# 2. Dupire Local Volatility
# =====================================================================

def _dupire_denominator(k: float, w: float, dw_dk: float, d2w_dk2: float) -> float:
    """
    Dupire denominator in variance form:

    D = 1 - (k/w) * dw/dk + (1/4) * (-1/4 - 1/w + k^2/w^2) * (dw/dk)^2 + (1/2) * d2w/dk2

    Returns D (must be > 0 for valid local vol).
    """
    if w < 1e-12:
        return 1.0
    term1 = 1.0 - (k / w) * dw_dk
    term2 = 0.25 * (-0.25 - 1.0 / w + k**2 / w**2) * dw_dk**2
    term3 = 0.5 * d2w_dk2
    return term1 + term2 + term3


class LocalVol:
    """
    Local volatility model from SVI-calibrated implied vol surface.

    Computes sigma_loc(S, t) via Dupire formula with:
    - Analytical SVI derivatives along strike (dw/dk, d^2w/dk^2)
    - Finite differences along maturity (dw/dT)
    - Pre-computed grid with bilinear interpolation for MC performance

    Parameters
    ----------
    surface : ImpliedVolSurface
        Calibrated implied vol surface.
    spot_ref : float
        Reference spot (S0 at calibration time).
    r : float
        Risk-free rate.
    vol_floor : float
        Minimum local vol (prevents zero/negative from numerics).
    vol_cap : float
        Maximum local vol (prevents explosion near boundaries).
    """

    def __init__(
        self,
        surface: ImpliedVolSurface,
        spot_ref: float,
        r: float,
        vol_floor: float = 0.05,
        vol_cap: float = 1.5,
    ):
        self._surface = surface
        self._spot_ref = spot_ref
        self._r = r
        self._vol_floor = vol_floor
        self._vol_cap = vol_cap
        self._interpolator: Optional[RegularGridInterpolator] = None
        self._S_grid: Optional[np.ndarray] = None
        self._t_grid: Optional[np.ndarray] = None

    def _compute_local_vol_at(self, S: float, t: float) -> float:
        """
        Compute local vol at a single (S, t) point via Dupire formula.

        Uses SVI analytical derivatives for strike direction,
        finite differences for maturity direction.
        """
        t = max(t, 1e-4)
        F = self._spot_ref * np.exp(self._r * t)
        K = S  # In Dupire, K is the "strike" dimension of the surface
        k = np.log(K / F)

        # Get SVI params at this maturity
        params = self._surface._get_svi_params_at_T(t)
        w = svi_total_variance(np.array([k]), params)[0]
        dw_dk_val = svi_dw_dk(k, params)
        d2w_dk2_val = svi_d2w_dk2(k, params)

        # dw/dT via finite difference
        dt = 0.01
        t_up = min(t + dt, float(self._surface.maturities[-1]))
        t_down = max(t - dt, float(self._surface.maturities[0]))
        actual_dt = t_up - t_down
        if actual_dt < 1e-6:
            # Can't compute dw/dT, return ATM vol
            return float(np.clip(
                np.sqrt(max(w / t, 0)), self._vol_floor, self._vol_cap
            ))

        params_up = self._surface._get_svi_params_at_T(t_up)
        params_down = self._surface._get_svi_params_at_T(t_down)
        w_up = svi_total_variance(np.array([k]), params_up)[0]
        w_down = svi_total_variance(np.array([k]), params_down)[0]
        dw_dT = (w_up - w_down) / actual_dt

        # Calendar spread condition: dw/dT must be >= 0
        dw_dT = max(dw_dT, 1e-8)

        # Dupire formula
        D = _dupire_denominator(k, w, dw_dk_val, d2w_dk2_val)
        D = max(D, 1e-6)  # Prevent division by zero

        local_var = dw_dT / D
        local_vol = np.sqrt(max(local_var, 0))

        return float(np.clip(local_vol, self._vol_floor, self._vol_cap))

    def precompute_grid(
        self,
        S_range: tuple = (0.4, 1.8),
        T_range: Optional[tuple] = None,
        n_S: int = 200,
        n_T: int = 100,
    ) -> None:
        """
        Pre-compute local vol on (S, t) grid for fast MC interpolation.

        S_range is specified as fraction of spot_ref.
        """
        if T_range is None:
            T_min = float(self._surface.maturities[0])
            T_max = float(self._surface.maturities[-1])
            T_range = (max(T_min, 0.01), T_max)

        S_vals = np.linspace(
            self._spot_ref * S_range[0],
            self._spot_ref * S_range[1],
            n_S,
        )
        t_vals = np.linspace(T_range[0], T_range[1], n_T)

        grid = np.zeros((n_T, n_S))
        for i, t in enumerate(t_vals):
            for j, S in enumerate(S_vals):
                grid[i, j] = self._compute_local_vol_at(S, t)

        self._S_grid = S_vals
        self._t_grid = t_vals
        self._interpolator = RegularGridInterpolator(
            (t_vals, S_vals), grid,
            method="linear", bounds_error=False,
            fill_value=None,  # extrapolate
        )

        logger.info(
            "Local vol grid precomputed: %d×%d, S=[%.0f, %.0f], T=[%.3f, %.3f]",
            n_T, n_S, S_vals[0], S_vals[-1], t_vals[0], t_vals[-1],
        )

    def vol(self, S: float, t: float) -> float:
        """
        Return local vol at (S, t).

        Uses precomputed grid if available, otherwise computes directly.
        """
        if self._interpolator is not None:
            val = float(self._interpolator(np.array([[t, S]]))[0])
            return np.clip(val, self._vol_floor, self._vol_cap)
        return self._compute_local_vol_at(S, t)

    def vol_vectorized(self, S_array: np.ndarray, t: float) -> np.ndarray:
        """
        Vectorized local vol for an array of spot values at time t.

        Critical for MC performance: called once per timestep for all paths.
        """
        if self._interpolator is not None:
            points = np.column_stack([
                np.full(len(S_array), t),
                S_array,
            ])
            vals = self._interpolator(points)
            return np.clip(vals, self._vol_floor, self._vol_cap)

        # Fallback: loop (slow)
        return np.array([self._compute_local_vol_at(S, t) for S in S_array])

    @property
    def model_name(self) -> str:
        return "LocalVol(Dupire-SVI)"

    def shift(self, delta: float) -> "LocalVol":
        """Parallel shift: shift the underlying implied vol surface."""
        shifted_surface = self._surface.shift(delta)
        lv = LocalVol(
            shifted_surface, self._spot_ref, self._r,
            self._vol_floor, self._vol_cap,
        )
        if self._interpolator is not None:
            # Re-precompute with same grid specs
            lv.precompute_grid(
                S_range=(
                    float(self._S_grid[0] / self._spot_ref),
                    float(self._S_grid[-1] / self._spot_ref),
                ),
                T_range=(float(self._t_grid[0]), float(self._t_grid[-1])),
                n_S=len(self._S_grid),
                n_T=len(self._t_grid),
            )
        return lv


# =====================================================================
# 3. Factory: build LocalVol from market data
# =====================================================================

def build_local_vol_from_market(
    use_sample: bool = False,
    precompute: bool = True,
) -> tuple:
    """
    Build LocalVol model from market data end-to-end.

    Returns
    -------
    tuple (local_vol, surface, spot_spy, r)
        local_vol : LocalVol model
        surface : ImpliedVolSurface (for visualization)
        spot_spy : float (SPY spot for reference)
        r : float (risk-free rate)
    """
    from .vol_surface import build_vol_surface_from_market
    from .market_data import load_sample_data, fetch_spot, fetch_risk_free_rate

    if use_sample:
        data = load_sample_data()
        spot = data["spot_spy"]
        r = data["risk_free_rate"]
    else:
        spot = fetch_spot("SPY")
        r = fetch_risk_free_rate()

    surface = build_vol_surface_from_market(use_sample=use_sample)

    lv = LocalVol(surface, spot_ref=spot, r=r)
    if precompute:
        lv.precompute_grid()

    return lv, surface, spot, r
