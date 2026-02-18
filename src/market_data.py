"""
Market data fetcher with caching for EUROSTOXX 50, SPX options, and risk-free rates.

Data sources:
- EUROSTOXX 50 spot & history: yfinance (^STOXX50E)
- SPX options chain: yfinance (SPY ETF) for vol surface construction
- Risk-free rate: 3-month T-bill (^IRX) via yfinance, fallback to ECB deposit rate

Reference: Gatheral, "The Volatility Surface" (Wiley, 2006), Ch. 1.
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache configuration
# ---------------------------------------------------------------------------
CACHE_DIR = Path(__file__).parent.parent / ".cache" / "market_data"
DEFAULT_CACHE_TTL_HOURS = 24


def _cache_key(name: str, **kwargs) -> str:
    """Generate deterministic cache key from name + parameters."""
    raw = f"{name}_{json.dumps(kwargs, sort_keys=True)}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.parquet"


def _meta_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.meta.json"


def _load_cache(key: str, ttl_hours: int = DEFAULT_CACHE_TTL_HOURS) -> Optional[pd.DataFrame]:
    """Load cached DataFrame if exists and not expired."""
    cache_file = _cache_path(key)
    meta_file = _meta_path(key)

    if not cache_file.exists() or not meta_file.exists():
        return None

    with open(meta_file) as f:
        meta = json.load(f)

    age_hours = (time.time() - meta["timestamp"]) / 3600
    if age_hours > ttl_hours:
        logger.debug("Cache expired for key=%s (age=%.1fh)", key, age_hours)
        return None

    logger.debug("Cache hit for key=%s (age=%.1fh)", key, age_hours)
    return pd.read_parquet(cache_file)


def _save_cache(key: str, data: pd.DataFrame) -> None:
    """Save DataFrame to cache as parquet with timestamp metadata."""
    cache_file = _cache_path(key)
    meta_file = _meta_path(key)

    data.to_parquet(cache_file)
    with open(meta_file, "w") as f:
        json.dump({"timestamp": time.time(), "key": key}, f)

    logger.debug("Cached key=%s -> %s", key, cache_file)


# ---------------------------------------------------------------------------
# Spot & historical data
# ---------------------------------------------------------------------------
def fetch_spot(ticker: str = "^STOXX50E") -> float:
    """
    Fetch current spot price.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker (e.g., "^STOXX50E", "^GSPC", "SPY").

    Returns
    -------
    float
        Most recent closing price.
    """
    tk = yf.Ticker(ticker)
    hist = tk.history(period="5d")
    if hist.empty:
        raise RuntimeError(f"No data returned for ticker={ticker}")
    return float(hist["Close"].iloc[-1])


def fetch_historical_prices(
    ticker: str = "^STOXX50E",
    period: str = "10y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data with caching.

    Returns
    -------
    pd.DataFrame
        Columns: Open, High, Low, Close, Volume. DatetimeIndex.
    """
    key = _cache_key("hist", ticker=ticker, period=period, interval=interval)
    cached = _load_cache(key)
    if cached is not None:
        return cached

    tk = yf.Ticker(ticker)
    df = tk.history(period=period, interval=interval)
    if df.empty:
        raise RuntimeError(f"No historical data for ticker={ticker}")

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    _save_cache(key, df)
    return df


def compute_log_returns(prices: pd.DataFrame, column: str = "Close") -> np.ndarray:
    """
    Compute daily log returns from a price series.

    Parameters
    ----------
    prices : pd.DataFrame
        Must contain the specified column.
    column : str
        Price column name.

    Returns
    -------
    np.ndarray
        Log returns array of length len(prices) - 1.
    """
    p = prices[column].values
    return np.diff(np.log(p))


def compute_realized_vol(
    returns: np.ndarray,
    window: int = 252,
    annualize: bool = True,
) -> float:
    """
    Compute annualized realized volatility from log returns.

    Uses the most recent `window` observations (default 252 = 1 year).

    Parameters
    ----------
    returns : np.ndarray
        Daily log returns.
    window : int
        Number of trailing observations to use.
    annualize : bool
        If True, multiply by sqrt(252).

    Returns
    -------
    float
        Realized volatility.
    """
    data = returns[-window:] if len(returns) > window else returns
    vol = np.std(data, ddof=1)
    if annualize:
        vol *= np.sqrt(252)
    return float(vol)


# ---------------------------------------------------------------------------
# Options chain
# ---------------------------------------------------------------------------
def fetch_options_chain(
    ticker: str = "SPY",
    min_days_to_expiry: int = 30,
    max_days_to_expiry: int = 730,
    min_open_interest: int = 10,
    moneyness_range: tuple = (0.80, 1.20),
) -> pd.DataFrame:
    """
    Fetch filtered options chain for vol surface construction.

    Uses OTM options for each strike: puts for K < spot, calls for K >= spot.
    OTM options have negligible American early exercise premium and better
    liquidity than their ITM counterparts, giving cleaner implied vols.

    Liquidity filter: accepts options with openInterest >= threshold OR
    volume > 0 (at least one trade today). This handles after-hours data
    where yfinance often reports OI=0 despite active contracts.

    Price selection: uses mid_price = (bid+ask)/2 when available;
    falls back to lastPrice when bid/ask are zero (e.g., after hours).

    Parameters
    ----------
    ticker : str
        Ticker for options data.
    min_days_to_expiry : int
        Minimum days to expiry filter (default 30; very short-dated
        options are noisy due to pin risk and microstructure effects).
    max_days_to_expiry : int
        Maximum days to expiry filter.
    min_open_interest : int
        Minimum open interest filter. Options also pass if volume > 0
        (handles after-hours data where OI is unavailable).
    moneyness_range : tuple
        (min, max) as fraction of spot. E.g., (0.80, 1.20) keeps
        strikes between 80% and 120% of spot.

    Returns
    -------
    pd.DataFrame
        Columns: strike, expiry, price, bid, ask, lastPrice, volume,
        openInterest, days_to_expiry, T, option_type.
    """
    key = _cache_key(
        "opts_v3", ticker=ticker, min_dte=min_days_to_expiry,
        max_dte=max_days_to_expiry, min_oi=min_open_interest,
        mr=moneyness_range,
    )
    cached = _load_cache(key)
    if cached is not None:
        return cached

    tk = yf.Ticker(ticker)
    expirations = tk.options
    if not expirations:
        raise RuntimeError(f"No options expirations for ticker={ticker}")

    # Get spot for moneyness filtering
    spot = float(tk.history(period="5d")["Close"].iloc[-1])
    strike_lo = spot * moneyness_range[0]
    strike_hi = spot * moneyness_range[1]

    today = pd.Timestamp.now().normalize()
    all_rows = []

    for exp_str in expirations:
        exp_date = pd.Timestamp(exp_str)
        dte = (exp_date - today).days
        if dte < min_days_to_expiry or dte > max_days_to_expiry:
            continue

        chain = tk.option_chain(exp_str)

        def _liquidity_filter(df):
            """Accept if OI >= threshold OR volume > 0 (at least traded today)."""
            vol_col = df["volume"].fillna(0)
            oi_col = df["openInterest"].fillna(0)
            return (oi_col >= min_open_interest) | (vol_col > 0)

        # OTM puts: K < spot (left wing of smile)
        puts = chain.puts
        if not puts.empty:
            puts = puts[
                (puts["strike"] >= strike_lo)
                & (puts["strike"] < spot)
                & _liquidity_filter(puts)
            ].copy()
            if not puts.empty:
                puts["option_type"] = "put"

        # OTM calls: K >= spot (right wing + ATM)
        calls = chain.calls
        if not calls.empty:
            calls = calls[
                (calls["strike"] >= spot)
                & (calls["strike"] <= strike_hi)
                & _liquidity_filter(calls)
            ].copy()
            if not calls.empty:
                calls["option_type"] = "call"

        # Combine OTM from both sides
        parts = []
        if not puts.empty and "option_type" in puts.columns:
            parts.append(puts)
        if not calls.empty and "option_type" in calls.columns:
            parts.append(calls)
        if not parts:
            continue

        opts = pd.concat(parts, ignore_index=True)
        opts["expiry"] = exp_date
        opts["days_to_expiry"] = dte
        opts["T"] = dte / 365.0

        # Price: prefer mid_price, fallback to lastPrice
        has_quotes = (opts["bid"] > 0) & (opts["ask"] > 0)
        opts["price"] = np.where(
            has_quotes,
            (opts["bid"] + opts["ask"]) / 2.0,
            opts["lastPrice"],
        )

        all_rows.append(opts[[
            "strike", "expiry", "price", "bid", "ask", "lastPrice",
            "volume", "openInterest", "days_to_expiry", "T", "option_type",
        ]])

    if not all_rows:
        raise RuntimeError(f"No options data after filtering for ticker={ticker}")

    result = pd.concat(all_rows, ignore_index=True)

    # Remove options with zero or negative price
    result = result[result["price"] > 0].reset_index(drop=True)

    _save_cache(key, result)
    return result


# ---------------------------------------------------------------------------
# Risk-free rate
# ---------------------------------------------------------------------------
def fetch_risk_free_rate() -> float:
    """
    Fetch risk-free rate proxy.

    Uses 13-week T-bill rate (^IRX) from Yahoo Finance.
    Fallback: ECB deposit facility rate (3.0% as of early 2026).

    Returns
    -------
    float
        Annualized risk-free rate (e.g., 0.04 for 4%).
    """
    try:
        tk = yf.Ticker("^IRX")
        hist = tk.history(period="5d")
        if not hist.empty:
            rate = float(hist["Close"].iloc[-1]) / 100.0
            if 0.0 <= rate <= 0.15:
                return rate
    except Exception:
        pass

    logger.warning("^IRX unavailable, using ECB deposit rate fallback (3.0%)")
    return 0.03


# ---------------------------------------------------------------------------
# Offline fallback — sample data
# ---------------------------------------------------------------------------
SAMPLE_DIR = Path(__file__).parent.parent / "data" / "sample"


def save_sample_data(
    spot_stoxx: float,
    risk_free_rate: float,
    historical_returns: np.ndarray,
    realized_vol: float,
    options_chain: pd.DataFrame,
    spot_spx: float,
    spot_spy: Optional[float] = None,
) -> None:
    """
    Save a snapshot of market data for offline / reproducible analysis.

    Parameters
    ----------
    spot_stoxx : float
        EUROSTOXX 50 spot price.
    risk_free_rate : float
        Risk-free rate.
    historical_returns : np.ndarray
        EUROSTOXX 50 daily log returns.
    realized_vol : float
        Annualized realized vol.
    options_chain : pd.DataFrame
        SPY options chain.
    spot_spx : float
        S&P 500 index spot price (~6800).
    spot_spy : float, optional
        SPY ETF spot price (~680). If None, estimated as spot_spx / 10.
    """
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    meta = {
        "spot_stoxx": spot_stoxx,
        "spot_spx": spot_spx,
        "spot_spy": spot_spy if spot_spy is not None else spot_spx / 10.0,
        "risk_free_rate": risk_free_rate,
        "realized_vol": realized_vol,
        "timestamp": time.time(),
    }
    with open(SAMPLE_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    np.save(SAMPLE_DIR / "historical_returns.npy", historical_returns)
    options_chain.to_parquet(SAMPLE_DIR / "options_chain.parquet")

    logger.info("Sample data saved to %s", SAMPLE_DIR)


def load_sample_data() -> dict:
    """
    Load pre-saved sample data for offline / reproducible analysis.

    Returns
    -------
    dict with keys:
        'spot_stoxx': float
        'spot_spx': float
        'risk_free_rate': float
        'historical_returns': np.ndarray
        'realized_vol': float
        'options_chain': pd.DataFrame
    """
    meta_file = SAMPLE_DIR / "meta.json"
    if not meta_file.exists():
        raise FileNotFoundError(
            f"No sample data found at {SAMPLE_DIR}. "
            "Run save_sample_data() first to create a snapshot."
        )

    with open(meta_file) as f:
        meta = json.load(f)

    returns = np.load(SAMPLE_DIR / "historical_returns.npy")
    options = pd.read_parquet(SAMPLE_DIR / "options_chain.parquet")

    return {
        "spot_stoxx": meta["spot_stoxx"],
        "spot_spx": meta["spot_spx"],
        "spot_spy": meta.get("spot_spy", meta["spot_spx"] / 10.0),
        "risk_free_rate": meta["risk_free_rate"],
        "realized_vol": meta["realized_vol"],
        "historical_returns": returns,
        "options_chain": options,
    }


# ---------------------------------------------------------------------------
# Convenience: fetch all data needed for the project
# ---------------------------------------------------------------------------
def fetch_all_market_data(save_snapshot: bool = True) -> dict:
    """
    Fetch all market data needed for autocall analysis.

    Fetches EUROSTOXX 50 + SPX data, computes returns and vol,
    optionally saves a snapshot for offline use.

    Parameters
    ----------
    save_snapshot : bool
        If True, save data to data/sample/ for offline fallback.

    Returns
    -------
    dict with keys:
        'spot_stoxx': float (EUROSTOXX 50 spot)
        'spot_spx': float (S&P 500 spot)
        'risk_free_rate': float
        'historical_prices_stoxx': pd.DataFrame
        'historical_returns_stoxx': np.ndarray
        'realized_vol_stoxx': float
        'options_chain_spx': pd.DataFrame
    """
    logger.info("Fetching EUROSTOXX 50 data...")
    spot_stoxx = fetch_spot("^STOXX50E")
    hist_stoxx = fetch_historical_prices("^STOXX50E", period="10y")
    returns_stoxx = compute_log_returns(hist_stoxx)
    rvol_stoxx = compute_realized_vol(returns_stoxx)

    logger.info("Fetching S&P 500 / SPY data...")
    spot_spx = fetch_spot("^GSPC")
    spot_spy = fetch_spot("SPY")
    options_spx = fetch_options_chain("SPY")

    logger.info("Fetching risk-free rate...")
    rfr = fetch_risk_free_rate()

    result = {
        "spot_stoxx": spot_stoxx,
        "spot_spx": spot_spx,
        "spot_spy": spot_spy,
        "risk_free_rate": rfr,
        "historical_prices_stoxx": hist_stoxx,
        "historical_returns_stoxx": returns_stoxx,
        "realized_vol_stoxx": rvol_stoxx,
        "options_chain_spx": options_spx,
    }

    if save_snapshot:
        save_sample_data(
            spot_stoxx=spot_stoxx,
            risk_free_rate=rfr,
            historical_returns=returns_stoxx,
            realized_vol=rvol_stoxx,
            options_chain=options_spx,
            spot_spx=spot_spx,
            spot_spy=spot_spy,
        )

    logger.info(
        "Market data ready: STOXX50E=%.2f, SPX=%.2f, r=%.2f%%, σ_realized=%.2f%%",
        spot_stoxx, spot_spx, rfr * 100, rvol_stoxx * 100,
    )

    return result
