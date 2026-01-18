"""Tests for autocallable pricing engine."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.autocall import Autocallable


@pytest.fixture
def standard_autocall():
    """Standard Phoenix Autocallable for testing."""
    return Autocallable(
        S0=100,
        autocall_barrier=1.0,
        coupon_barrier=0.8,
        ki_barrier=0.6,
        coupon_rate=0.05,
        n_observations=8,
        T=2.0,
        r=0.03,
        sigma=0.25,
    )


class TestAutocallValidation:
    def test_negative_S0_raises(self):
        with pytest.raises(ValueError, match="S0 must be positive"):
            Autocallable(S0=-100, autocall_barrier=1.0, coupon_barrier=0.8,
                         ki_barrier=0.6, coupon_rate=0.05, n_observations=8,
                         T=2.0, r=0.03, sigma=0.25)

    def test_invalid_barrier_order_raises(self):
        with pytest.raises(ValueError, match="Barriers must satisfy"):
            Autocallable(S0=100, autocall_barrier=0.5, coupon_barrier=0.8,
                         ki_barrier=0.6, coupon_rate=0.05, n_observations=8,
                         T=2.0, r=0.03, sigma=0.25)

    def test_negative_vol_raises(self):
        with pytest.raises(ValueError, match="Volatility must be positive"):
            Autocallable(S0=100, autocall_barrier=1.0, coupon_barrier=0.8,
                         ki_barrier=0.6, coupon_rate=0.05, n_observations=8,
                         T=2.0, r=0.03, sigma=-0.1)


class TestAutocallPricing:
    def test_price_is_positive(self, standard_autocall):
        result = standard_autocall.price(n_paths=50_000, seed=42)
        assert result["price"] > 0

    def test_price_bounded_by_notional_plus_coupons(self, standard_autocall):
        """Price should not exceed notional + all coupons (undiscounted)."""
        result = standard_autocall.price(n_paths=50_000, seed=42)
        max_payoff = 100 * (1 + 0.05 * 8)
        assert result["price"] < max_payoff

    def test_price_convergence(self, standard_autocall):
        """Price with more paths should have smaller std error."""
        r1 = standard_autocall.price(n_paths=10_000, seed=42)
        r2 = standard_autocall.price(n_paths=100_000, seed=42)
        assert r2["std_error"] < r1["std_error"]

    def test_reproducibility(self, standard_autocall):
        """Same seed should give same price."""
        r1 = standard_autocall.price(n_paths=50_000, seed=123)
        r2 = standard_autocall.price(n_paths=50_000, seed=123)
        assert abs(r1["price"] - r2["price"]) < 1e-10

    def test_autocall_probabilities_sum_leq_one(self, standard_autocall):
        result = standard_autocall.price(n_paths=50_000, seed=42)
        assert result["total_autocall_prob"] <= 1.0 + 1e-10

    def test_expected_life_within_bounds(self, standard_autocall):
        result = standard_autocall.price(n_paths=50_000, seed=42)
        assert 0 < result["expected_life"] <= standard_autocall.T

    def test_ki_probability_between_0_and_1(self, standard_autocall):
        result = standard_autocall.price(n_paths=50_000, seed=42)
        assert 0 <= result["ki_probability"] <= 1.0


class TestAutocallSensitivity:
    def test_higher_vol_increases_ki_probability(self):
        """Higher volatility should increase knock-in probability."""
        low_vol = Autocallable(S0=100, autocall_barrier=1.0, coupon_barrier=0.8,
                               ki_barrier=0.6, coupon_rate=0.05, n_observations=8,
                               T=2.0, r=0.03, sigma=0.15)
        high_vol = Autocallable(S0=100, autocall_barrier=1.0, coupon_barrier=0.8,
                                ki_barrier=0.6, coupon_rate=0.05, n_observations=8,
                                T=2.0, r=0.03, sigma=0.40)

        r_low = low_vol.price(n_paths=50_000, seed=42)
        r_high = high_vol.price(n_paths=50_000, seed=42)
        assert r_high["ki_probability"] > r_low["ki_probability"]

    def test_higher_spot_increases_autocall_prob(self):
        """Higher spot relative to fixed barrier should increase autocall probability."""
        ac = Autocallable(S0=100, autocall_barrier=1.0, coupon_barrier=0.8,
                          ki_barrier=0.6, coupon_rate=0.05, n_observations=8,
                          T=2.0, r=0.03, sigma=0.25)
        r_low = ac.price(n_paths=50_000, seed=42)

        ac.S0 = 110  # spot up, barriers stay at 100/80/60
        r_high = ac.price(n_paths=50_000, seed=42)
        ac.S0 = 100

        assert r_high["total_autocall_prob"] > r_low["total_autocall_prob"]


class TestGreeks:
    def test_delta_finite_and_reasonable(self, standard_autocall):
        """Delta should be finite and within reasonable bounds.
        Note: autocall delta can be negative at ATM because early autocall
        means fewer coupons collected (less value for investor)."""
        from src.greeks import delta
        d = delta(standard_autocall, n_paths=100_000, seed=42)
        assert np.isfinite(d)
        assert abs(d) < 5.0  # reasonable bound for notional=100

    def test_vega_finite(self, standard_autocall):
        from src.greeks import vega
        v = vega(standard_autocall, n_paths=100_000, seed=42)
        assert np.isfinite(v)


class TestPnLExplain:
    def test_pnl_attribution_explained_pct(self, standard_autocall):
        """PnL should be mostly explained by Greeks for small moves."""
        from src.pnl_explain import pnl_attribution
        result = pnl_attribution(
            standard_autocall, dS=1.0, d_sigma=0.0, dt_days=1.0,
            n_paths=100_000, seed=42,
        )
        assert abs(result["explained_pct"]) > 50


class TestStressTesting:
    def test_crash_reduces_value(self, standard_autocall):
        from src.stress_testing import stress_test
        result = stress_test(
            standard_autocall, spot_shock=-0.30, vol_shock=0.20,
            n_paths=50_000, seed=42,
        )
        assert result["pnl_impact"] < 0

    def test_positive_market_increases_value(self, standard_autocall):
        from src.stress_testing import stress_test
        result = stress_test(
            standard_autocall, spot_shock=0.10, vol_shock=-0.05,
            n_paths=50_000, seed=42,
        )
        assert result["pnl_impact"] > 0
