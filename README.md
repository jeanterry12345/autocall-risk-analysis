# Autocall Risk Analysis

Pricing, risk analysis and P&L attribution for Phoenix Autocallable structured products, with market data calibration and model risk quantification.

## Features

- **Monte Carlo Pricing** : Phoenix Autocallable with autocall barrier, coupon mechanism, and knock-in put
- **Greeks** : Delta, Gamma, Vega, Theta, Rho via finite difference (bump and reprice)
- **P&L Attribution** : Taylor expansion decomposition (Delta, Gamma, Vega, Theta contributions)
- **Stress Testing** : Historical scenarios (Lehman 2008, COVID 2020) and hypothetical shocks
- **VaR** : Historical and parametric (delta-normal) with Kupiec backtesting
- **Market Data** : Real-time EUROSTOXX 50 and S&P 500 data via yfinance, with offline caching
- **Implied Volatility Surface** : SVI parametric calibration from SPY options chain
- **Local Volatility** : Dupire local vol derived from SVI, with precomputed grid for fast MC
- **Model Comparison** : Constant vol vs term structure vs local vol — quantifies model risk for barrier products

## Structure

```
src/
├── autocall.py          # Product definition and MC pricing engine
├── greeks.py            # Greeks via finite difference
├── pnl_explain.py       # P&L decomposition
├── stress_testing.py    # Stress scenarios
├── var.py               # VaR and backtesting
├── utils.py             # Path simulation (GBM + local vol)
├── market_data.py       # Market data fetching and caching
├── vol_surface.py       # Implied vol surface (SVI calibration)
└── local_vol.py         # Local vol model (Dupire formula)

examples/
├── demo.py              # Full analysis with synthetic parameters
└── model_comparison.py  # Constant vol vs local vol comparison

tests/
├── test_autocall.py     # Pricing, Greeks, PnL, stress tests
├── test_vol_surface.py  # BS implied vol, SVI calibration, surface
└── test_local_vol.py    # Vol models, local vol paths, integration
```

## Quick Start

```bash
pip install -r requirements.txt

# Original demo (synthetic parameters)
python examples/demo.py

# Model comparison with real market data
python examples/model_comparison.py
```

## Volatility Models

The pricing engine supports pluggable volatility models via the `vol_model` parameter:

```python
from src.autocall import Autocallable
from src.local_vol import ConstantVol, LocalVol

# Original behavior (constant vol)
ac = Autocallable(S0=100, ..., sigma=0.25)

# With local vol model
ac = Autocallable(S0=100, ..., sigma=0.25, vol_model=local_vol)
```

| Model | Description | Use Case |
|-------|-------------|----------|
| `None` (default) | Constant GBM volatility | Baseline, backwards compatible |
| `ConstantVol(σ)` | Same as default, wrapped | Explicit model comparison |
| `TermStructureVol` | ATM vol term structure | Intermediate complexity |
| `LocalVol` | Dupire from SVI surface | Full smile dynamics |

## Model Risk Results

Pricing the same Phoenix Autocallable under different vol models:

| Model | KI Probability | Autocall Probability |
|-------|---------------|---------------------|
| Constant Vol | Low baseline | High |
| Local Vol (Dupire) | **Higher** (smile effect) | Different |

**Key insight**: Constant vol underestimates knock-in probability because it ignores the vol smile. In reality, volatility rises when spot drops (leverage effect), making barrier breaches more likely. This is a material model risk.

## Tests

```bash
pytest tests/ -v    # 53 tests
```

## References

- Hull, J. C. *Options, Futures, and Other Derivatives*
- Glasserman, P. *Monte Carlo Methods in Financial Engineering*
- Gatheral, J. *The Volatility Surface* (Wiley, 2006)
- Dupire, B. "Pricing with a Smile" (Risk, 1994)
