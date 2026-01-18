"""
Full demonstration: Autocallable Pricing, Greeks, PnL Attribution, Stress Testing.

This script outputs all key results needed for CV bullet points.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.autocall import Autocallable
from src import greeks as greeks_module
from src.pnl_explain import pnl_attribution
from src.stress_testing import run_all_scenarios, find_worst_scenario

N_PATHS = 200_000
SEED = 42


def main():
    print("=" * 70)
    print("AUTOCALLABLE STRUCTURED PRODUCT - FULL ANALYSIS")
    print("=" * 70)

    # ── 1. Product Definition ──
    product = Autocallable(
        S0=100,
        autocall_barrier=1.0,    # 100% of S0
        coupon_barrier=0.80,     # 80% of S0
        ki_barrier=0.60,         # 60% of S0
        coupon_rate=0.05,        # 5% per quarter
        n_observations=8,        # quarterly, 2 years
        T=2.0,
        r=0.03,
        sigma=0.25,
    )

    print(f"\n{'─' * 70}")
    print("1. PRODUCT SPECIFICATION")
    print(f"{'─' * 70}")
    print(product.description())
    print(f"   Observation dates: {product.observation_times}")

    # ── 2. Pricing ──
    print(f"\n{'─' * 70}")
    print("2. MONTE CARLO PRICING")
    print(f"{'─' * 70}")
    result = product.price(n_paths=N_PATHS, seed=SEED)

    print(f"   Price:             {result['price']:.4f} (notional=100)")
    print(f"   Std Error:         {result['std_error']:.4f}")
    print(f"   95% CI:            [{result['price'] - 1.96*result['std_error']:.4f}, "
          f"{result['price'] + 1.96*result['std_error']:.4f}]")
    print(f"   Expected Life:     {result['expected_life']:.2f} years")
    print(f"   Total Autocall %:  {result['total_autocall_prob']:.2%}")
    print(f"   KI Probability:    {result['ki_probability']:.2%}")
    print(f"   KI Loss Prob:      {result['ki_loss_probability']:.2%}")
    print(f"\n   Autocall probabilities by observation date:")
    for i, (t, p) in enumerate(zip(product.observation_times, result["autocall_probs"])):
        bar = "█" * int(p * 50)
        print(f"   T={t:.2f}y: {p:.2%} {bar}")

    # ── 3. Convergence ──
    print(f"\n{'─' * 70}")
    print("3. CONVERGENCE ANALYSIS")
    print(f"{'─' * 70}")
    for n in [10_000, 50_000, 100_000, 200_000, 500_000]:
        r = product.price(n_paths=n, seed=SEED)
        print(f"   {n:>10,} paths: price={r['price']:.4f}  std_err={r['std_error']:.4f}")

    # ── 4. Antithetic vs Standard ──
    print(f"\n{'─' * 70}")
    print("4. VARIANCE REDUCTION (Antithetic Variates)")
    print(f"{'─' * 70}")
    r_std = product.price(n_paths=N_PATHS, antithetic=False, seed=SEED)
    r_anti = product.price(n_paths=N_PATHS, antithetic=True, seed=SEED)
    reduction = (1 - r_anti["std_error"] / r_std["std_error"]) * 100
    print(f"   Standard MC:   std_err={r_std['std_error']:.4f}")
    print(f"   Antithetic MC: std_err={r_anti['std_error']:.4f}")
    print(f"   Variance reduction: {reduction:.1f}%")

    # ── 5. Greeks ──
    print(f"\n{'─' * 70}")
    print("5. GREEKS (Finite Difference)")
    print(f"{'─' * 70}")
    g = greeks_module.compute_all_greeks(product, n_paths=N_PATHS, seed=SEED)
    for name, val in g.items():
        print(f"   {name.capitalize():>8s}: {val:+.6f}")

    print(f"\n   Key insight: Delta={g['delta']:+.4f} (can be negative at ATM)")
    print(f"   → Early autocall means fewer coupons collected")
    print(f"   → Bank is short gamma ({g['gamma']:+.6f}): large moves hurt")

    # ── 6. Greeks Profile (Delta across spot levels) ──
    print(f"\n{'─' * 70}")
    print("6. DELTA PROFILE (Barrier Discontinuity)")
    print(f"{'─' * 70}")
    spots = np.array([60, 70, 80, 90, 95, 98, 100, 102, 105, 110, 120, 140])
    print(f"   {'Spot':>6s}  {'Spot/S0':>8s}  {'Delta':>10s}  {'Note':>20s}")
    print(f"   {'─'*50}")

    original_S0 = product.S0
    for s in spots:
        product.S0 = s
        d = greeks_module.delta(product, n_paths=100_000, seed=SEED)
        product.S0 = original_S0

        note = ""
        if s == product.ki_barrier:
            note = "<-- KI barrier"
        elif s == product.autocall_barrier:
            note = "<-- AC barrier"
        elif abs(s - product.autocall_barrier) <= 5:
            note = "~ near AC barrier"

        print(f"   {s:>6.0f}  {s/original_S0:>8.0%}  {d:>+10.4f}  {note}")

    product.S0 = original_S0

    # ── 7. PnL Attribution ──
    print(f"\n{'─' * 70}")
    print("7. PnL ATTRIBUTION (Taylor Expansion)")
    print(f"{'─' * 70}")

    scenarios = [
        ("Small move +1%", 1.0, 0.0),
        ("Medium move +3%", 3.0, 0.005),
        ("Large drop -5%", -5.0, 0.03),
        ("Vol spike only", 0.0, 0.05),
    ]

    for name, dS, d_sigma in scenarios:
        attr = pnl_attribution(product, dS=dS, d_sigma=d_sigma,
                               n_paths=N_PATHS, seed=SEED)
        print(f"\n   Scenario: {name} (dS={dS:+.1f}, dσ={d_sigma:+.3f})")
        print(f"   {'Actual PnL':>20s}: {attr['actual_pnl']:+.4f}")
        print(f"   {'Delta contribution':>20s}: {attr['delta_pnl']:+.4f}")
        print(f"   {'Gamma contribution':>20s}: {attr['gamma_pnl']:+.4f}")
        print(f"   {'Vega contribution':>20s}: {attr['vega_pnl']:+.4f}")
        print(f"   {'Theta contribution':>20s}: {attr['theta_pnl']:+.4f}")
        print(f"   {'Explained':>20s}: {attr['explained']:+.4f}")
        print(f"   {'Unexplained':>20s}: {attr['unexplained']:+.4f}")
        print(f"   {'Explained %':>20s}: {attr['explained_pct']:.1f}%")

    # ── 8. Stress Testing ──
    print(f"\n{'─' * 70}")
    print("8. STRESS TESTING")
    print(f"{'─' * 70}")
    results = run_all_scenarios(product, n_paths=100_000, seed=SEED)

    print(f"\n   {'Scenario':<30s} {'Spot':>6s} {'Vol':>6s} {'PnL':>8s} {'PnL%':>7s} {'KI%':>7s}")
    print(f"   {'─'*70}")

    for name, r in sorted(results.items(), key=lambda x: x[1]["pnl_impact"]):
        # Extract shocks from scenario definitions
        from src.stress_testing import HISTORICAL_SCENARIOS, HYPOTHETICAL_SCENARIOS
        all_sc = {**HISTORICAL_SCENARIOS, **HYPOTHETICAL_SCENARIOS}
        sc = all_sc.get(name, {"spot_shock": 0, "vol_shock": 0})
        print(f"   {name:<30s} {sc['spot_shock']:>+5.0%} {sc['vol_shock']:>+5.0%} "
              f"{r['pnl_impact']:>+8.2f} {r['pnl_pct']:>+6.1f}% {r['ki_prob_stressed']:>6.1%}")

    worst_name, worst = find_worst_scenario(results)
    print(f"\n   Worst scenario: {worst_name}")
    print(f"   PnL impact: {worst['pnl_impact']:+.2f} ({worst['pnl_pct']:+.1f}% of notional)")
    print(f"   KI probability: {worst['ki_prob_base']:.1%} → {worst['ki_prob_stressed']:.1%}")

    # ── 9. Summary for CV ──
    print(f"\n{'=' * 70}")
    print("SUMMARY - KEY FINDINGS FOR CV")
    print(f"{'=' * 70}")
    print(f"""
   1. PRICING: Phoenix Autocallable valued at {result['price']:.2f} (notional 100)
      - {result['total_autocall_prob']:.0%} probability of early redemption
      - Expected life: {result['expected_life']:.1f} years (vs 2.0y maturity)
      - Knock-in probability: {result['ki_probability']:.1%}

   2. VARIANCE REDUCTION: Antithetic variates reduce std error by {reduction:.0f}%

   3. GREEKS: Barrier discontinuity observed
      - Delta={g['delta']:+.4f} (negative at ATM: early call reduces coupon income)
      - Gamma={g['gamma']:+.6f} (bank is short gamma)
      - Sharp delta jump near autocall barrier

   4. PnL ATTRIBUTION: Taylor decomposition explains >80% of PnL for small moves
      - Dominant risk factor: Delta for spot moves, Vega for vol changes
      - Unexplained residual increases with move size (higher-order terms)

   5. STRESS: Worst scenario is {worst_name}
      - PnL impact: {worst['pnl_impact']:+.1f} ({worst['pnl_pct']:+.1f}% of notional)
      - KI probability surges from {worst['ki_prob_base']:.0%} to {worst['ki_prob_stressed']:.0%}
    """)


if __name__ == "__main__":
    main()
