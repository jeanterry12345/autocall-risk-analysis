# Autocall Risk Analysis

Pricing, risk analysis and P&L attribution for Phoenix Autocallable structured products.

## Features

- **Monte Carlo Pricing** : Phoenix Autocallable with autocall barrier, coupon mechanism, and knock-in put
- **Greeks** : Delta, Gamma, Vega, Theta, Rho via finite difference (bump and reprice)
- **P&L Attribution** : Taylor expansion decomposition (Delta, Gamma, Vega, Theta contributions)
- **Stress Testing** : Historical scenarios (Lehman 2008, COVID 2020) and hypothetical shocks
- **VaR** : Historical and parametric (delta-normal) with Kupiec backtesting

## Structure

```
src/
├── autocall.py          # Product definition and MC pricing engine
├── greeks.py            # Greeks via finite difference
├── pnl_explain.py       # P&L decomposition
├── stress_testing.py    # Stress scenarios
├── var.py               # VaR and backtesting
└── utils.py             # GBM path simulation
```

## Usage

```bash
pip install -r requirements.txt
python examples/demo.py
```

## Tests

```bash
pytest tests/ -v
```

## References

- Hull, J. C. *Options, Futures, and Other Derivatives*
- Glasserman, P. *Monte Carlo Methods in Financial Engineering*
