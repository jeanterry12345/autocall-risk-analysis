"""Tests for local volatility model and integration with autocallable pricing."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.local_vol import ConstantVol, TermStructureVol, LocalVol
from src.vol_surface import ImpliedVolSurface
from src.utils import simulate_paths_local_vol
from src.autocall import Autocallable


class TestConstantVol:
    def test_returns_same_value(self):
        cv = ConstantVol(0.25)
        assert cv.vol(100, 0.5) == 0.25
        assert cv.vol(50, 1.0) == 0.25
        assert cv.vol(200, 0.01) == 0.25

    def test_model_name(self):
        cv = ConstantVol(0.20)
        assert "20" in cv.model_name

    def test_shift(self):
        cv = ConstantVol(0.25)
        shifted = cv.shift(0.05)
        assert abs(shifted.vol(100, 1.0) - 0.30) < 1e-10


class TestTermStructureVol:
    def test_interpolation(self):
        mats = np.array([0.25, 0.5, 1.0, 2.0])
        vols = np.array([0.30, 0.25, 0.22, 0.20])
        ts = TermStructureVol(mats, vols)

        assert abs(ts.vol(100, 0.25) - 0.30) < 1e-10
        assert abs(ts.vol(100, 2.0) - 0.20) < 1e-10
        # Interpolated value between 0.5 and 1.0
        v = ts.vol(100, 0.75)
        assert 0.22 < v < 0.25

    def test_ignores_spot(self):
        """Term structure only depends on t, not S."""
        ts = TermStructureVol(np.array([1.0]), np.array([0.20]))
        assert ts.vol(50, 1.0) == ts.vol(200, 1.0)

    def test_shift(self):
        ts = TermStructureVol(np.array([1.0]), np.array([0.20]))
        shifted = ts.shift(0.05)
        assert abs(shifted.vol(100, 1.0) - 0.25) < 1e-10


class TestLocalVol:
    @pytest.fixture
    def mock_surface(self):
        calibrations = {
            0.5: {"params": (0.02, 0.10, -0.3, 0.0, 0.1), "rmse": 0.001, "success": True, "n_points": 10},
            1.0: {"params": (0.04, 0.08, -0.4, 0.0, 0.15), "rmse": 0.001, "success": True, "n_points": 10},
            2.0: {"params": (0.08, 0.06, -0.5, 0.0, 0.2), "rmse": 0.001, "success": True, "n_points": 10},
        }
        return ImpliedVolSurface(spot=100, r=0.03, svi_calibrations=calibrations)

    def test_local_vol_positive(self, mock_surface):
        lv = LocalVol(mock_surface, spot_ref=100, r=0.03)
        for S in [70, 90, 100, 110, 130]:
            for t in [0.5, 1.0, 1.5]:
                v = lv.vol(S, t)
                assert v > 0, f"Negative local vol at S={S}, t={t}"
                assert v < 2.0, f"Local vol too high at S={S}, t={t}"

    def test_local_vol_bounded(self, mock_surface):
        lv = LocalVol(mock_surface, spot_ref=100, r=0.03, vol_floor=0.05, vol_cap=1.5)
        v = lv.vol(100, 1.0)
        assert 0.05 <= v <= 1.5

    def test_precompute_grid(self, mock_surface):
        lv = LocalVol(mock_surface, spot_ref=100, r=0.03)
        lv.precompute_grid(n_S=50, n_T=20)
        assert lv._interpolator is not None

        v = lv.vol(100, 1.0)
        assert v > 0

    def test_vectorized(self, mock_surface):
        lv = LocalVol(mock_surface, spot_ref=100, r=0.03)
        lv.precompute_grid(n_S=50, n_T=20)

        S_arr = np.array([80, 90, 100, 110, 120], dtype=float)
        vols = lv.vol_vectorized(S_arr, 1.0)
        assert len(vols) == 5
        assert all(v > 0 for v in vols)


class TestLocalVolPaths:
    def test_paths_shape(self):
        cv = ConstantVol(0.25)
        paths = simulate_paths_local_vol(100, 0.03, cv, 1.0, 100, 1000, seed=42)
        assert paths.shape == (101, 1000)

    def test_paths_start_at_S0(self):
        cv = ConstantVol(0.25)
        paths = simulate_paths_local_vol(100, 0.03, cv, 1.0, 100, 1000, seed=42)
        assert np.allclose(paths[0], 100.0)

    def test_paths_positive(self):
        cv = ConstantVol(0.25)
        paths = simulate_paths_local_vol(100, 0.03, cv, 1.0, 100, 1000, seed=42)
        assert np.all(paths > 0)

    def test_constant_vol_drift(self):
        """Mean of log(S_T/S_0) should be approximately (r - sigma^2/2)*T."""
        S0, r, sigma, T = 100, 0.05, 0.20, 1.0
        cv = ConstantVol(sigma)
        paths = simulate_paths_local_vol(S0, r, cv, T, 252, 100_000, seed=42)
        log_returns = np.log(paths[-1] / S0)
        expected_drift = (r - 0.5 * sigma**2) * T
        assert abs(np.mean(log_returns) - expected_drift) < 0.01


class TestAutocallIntegration:
    def test_backwards_compatible(self):
        """Autocallable without vol_model matches original behavior."""
        ac = Autocallable(S0=100, autocall_barrier=1.0, coupon_barrier=0.8,
                          ki_barrier=0.6, coupon_rate=0.05, n_observations=8,
                          T=2.0, r=0.03, sigma=0.25)
        r = ac.price(n_paths=50_000, seed=42)
        assert r["price"] > 0
        assert 0 < r["total_autocall_prob"] < 1.0

    def test_constant_vol_model_similar(self):
        """ConstantVol wrapper should give similar results to no vol_model."""
        kwargs = dict(S0=100, autocall_barrier=1.0, coupon_barrier=0.8,
                      ki_barrier=0.6, coupon_rate=0.05, n_observations=8,
                      T=2.0, r=0.03, sigma=0.25)

        ac_orig = Autocallable(**kwargs)
        ac_wrap = Autocallable(**kwargs, vol_model=ConstantVol(0.25))

        r_orig = ac_orig.price(n_paths=100_000, seed=42)
        r_wrap = ac_wrap.price(n_paths=100_000, seed=42)

        # Not identical (different simulation paths), but similar
        assert abs(r_orig["price"] - r_wrap["price"]) < 3.0
        assert abs(r_orig["total_autocall_prob"] - r_wrap["total_autocall_prob"]) < 0.05

    def test_vol_model_in_description(self):
        ac = Autocallable(S0=100, autocall_barrier=1.0, coupon_barrier=0.8,
                          ki_barrier=0.6, coupon_rate=0.05, n_observations=8,
                          T=2.0, r=0.03, sigma=0.25,
                          vol_model=ConstantVol(0.25))
        assert "Constant" in ac.description()
