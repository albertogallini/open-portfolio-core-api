"""
data.py  –  Universe & market-data layer for the factor risk model.

Responsibilities
----------------
* Load index constituents (S&P 500 or user-supplied list).
* Batch-download and cache adjusted close + volume for the full universe.
* Fetch & cache quarterly income statement, cash-flow, balance sheet via
  yfinance  (extends your existing financial_cache / cashflow_cache globals).
* Expose clean helpers consumed by FactorBuilder.
"""
from __future__ import annotations

import datetime
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ── Global caches ─────────────────────────────────────────────────────────────
# Keyed by ticker only, holding the RAW (unfiltered-by-date) yfinance frames.
# The point-in-time / report-lag filter is re-applied on every call in
# get_quarterly_fundamentals() so the result never depends on call order.
financial_cache:      Dict = {}
cashflow_cache:       Dict = {}
balance_sheet_cache:  Dict = {}

# Bulk panel caches – set once per calibration run
_price_panel:   Optional[pd.DataFrame] = None   # date × ticker, adj close
_volume_panel:  Optional[pd.DataFrame] = None   # date × ticker, volume
_sector_cache:  Dict[str, str]         = {}      # ticker → GICS sector string
_shares_cache:  Dict[str, float]       = {}      # ticker → shares outstanding


# ── Universe ──────────────────────────────────────────────────────────────────

def load_sp500_tickers() -> List[str]:
    """Scrape current S&P 500 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tbl = pd.read_html(
        url,
        header=0,
        storage_options={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
    )[0]
    return tbl["Symbol"].str.replace(".", "-", regex=False).tolist()


# ── Bulk price / volume panel ─────────────────────────────────────────────────

def load_price_volume_panel(
    tickers: List[str],
    start: datetime.date,
    end: datetime.date,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download daily adjusted close and volume for the entire universe at once.

    Returns
    -------
    prices  : pd.DataFrame  shape [dates × tickers]
    volumes : pd.DataFrame  shape [dates × tickers]
    """
    global _price_panel, _volume_panel

    logger.info("Downloading price panel for %d tickers …", len(tickers))
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if len(tickers) == 1:
        prices  = raw[["Close"]].rename(columns={"Close": tickers[0]})
        volumes = raw[["Volume"]].rename(columns={"Volume": tickers[0]})
    else:
        prices  = raw.xs("Close",  axis=1, level=1)
        volumes = raw.xs("Volume", axis=1, level=1)

    prices.index  = pd.to_datetime(prices.index).date
    volumes.index = pd.to_datetime(volumes.index).date

    _price_panel  = prices.sort_index()
    _volume_panel = volumes.sort_index()

    logger.info(
        "Price panel loaded: %d tickers × %d trading days",
        _price_panel.shape[1], _price_panel.shape[0],
    )
    return _price_panel, _volume_panel


def get_price_panel() -> pd.DataFrame:
    if _price_panel is None:
        raise RuntimeError("Call load_price_volume_panel() first.")
    return _price_panel


def get_volume_panel() -> pd.DataFrame:
    if _volume_panel is None:
        raise RuntimeError("Call load_price_volume_panel() first.")
    return _volume_panel


# ── Quarterly fundamental snapshot ───────────────────────────────────────────

def get_quarterly_fundamentals(
    ticker_symbol: str,
    target_date: datetime.date,
    report_lag_days: int = 45,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Return (income_stmt, cashflow, balance_sheet) DataFrames containing only
    statements that would have been PUBLISHED by target_date, i.e. with
    period-end date <= target_date - report_lag_days. A quarter-end date is
    not a publication date: 10-Qs land ~30-45 days later, 10-Ks up to ~90.

    Each DataFrame is sorted descending by period-end date, capped at 8 quarters.
    Raw (unfiltered) yfinance frames are cached per ticker so this filter is
    reapplied identically on every call, independent of call order.
    """
    if ticker_symbol not in financial_cache:
        try:
            ticker = yf.Ticker(ticker_symbol)

            def _clean(df: pd.DataFrame) -> pd.DataFrame:
                df = df.T.copy()
                df.index = pd.to_datetime(df.index).date
                return df.sort_index(ascending=False)

            financial_cache[ticker_symbol]     = _clean(ticker.quarterly_income_stmt)
            cashflow_cache[ticker_symbol]      = _clean(ticker.quarterly_cashflow)
            balance_sheet_cache[ticker_symbol] = _clean(ticker.quarterly_balance_sheet)

        except Exception as exc:
            logger.debug("Fundamental fetch failed for %s: %s", ticker_symbol, exc)
            financial_cache[ticker_symbol]     = None
            cashflow_cache[ticker_symbol]      = None
            balance_sheet_cache[ticker_symbol] = None

    as_of = target_date - datetime.timedelta(days=report_lag_days)

    def _as_of(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is None:
            return None
        return df[df.index <= as_of].head(8)

    return (
        _as_of(financial_cache[ticker_symbol]),
        _as_of(cashflow_cache[ticker_symbol]),
        _as_of(balance_sheet_cache[ticker_symbol]),
    )


# ── Fundamental aggregation helpers ──────────────────────────────────────────

def get_ttm(df: Optional[pd.DataFrame], field: str) -> Optional[float]:
    """
    Trailing-twelve-month sum of a field. yfinance often exposes only
    ~5-8 quarters of history, so fewer than 4 available quarters is common
    (especially early in a name's history); summing just 2-3 quarters and
    calling it "TTM" understates it by up to half, so short series are
    scaled up to a 4-quarter-equivalent instead of summed as-is.
    """
    if df is None or field not in df.columns:
        return None
    series = df[field].dropna().head(4)
    n = len(series)
    if n < 2:
        return None
    return float(series.sum()) * (4.0 / n)


def get_latest_bs(df: Optional[pd.DataFrame], field: str) -> Optional[float]:
    """Return the most-recent balance-sheet value for a field."""
    if df is None or field not in df.columns:
        return None
    vals = df[field].dropna().head(1)
    return float(vals.iloc[0]) if len(vals) else None


def first_valid_ttm(df: Optional[pd.DataFrame], fields: List[str]) -> Optional[float]:
    """Try a list of field-name aliases and return the first non-None TTM sum."""
    for f in fields:
        val = get_ttm(df, f)
        if val is not None:
            return val
    return None


def first_valid_bs(df: Optional[pd.DataFrame], fields: List[str]) -> Optional[float]:
    for f in fields:
        val = get_latest_bs(df, f)
        if val is not None:
            return val
    return None


# ── Sector & shares ───────────────────────────────────────────────────────────

def get_sector(ticker_symbol: str) -> str:
    if ticker_symbol in _sector_cache:
        return _sector_cache[ticker_symbol]
    try:
        info   = yf.Ticker(ticker_symbol).info
        sector = (info.get("sector") or "Unknown").strip() or "Unknown"
    except Exception:
        sector = "Unknown"
    _sector_cache[ticker_symbol] = sector
    return sector


def get_shares_outstanding(ticker_symbol: str) -> Optional[float]:
    """
    Current shares outstanding, applied to every historical target_date the
    caller uses it for. yfinance does not expose historical share counts, so
    buybacks/issuance/splits between then and now slightly skew the `size`
    factor and market-cap-based ratios (value_ep, value_bp) for older dates.
    Acceptable for now; not point-in-time correct.
    """
    if ticker_symbol in _shares_cache:
        v = _shares_cache[ticker_symbol]
        return None if np.isnan(v) else v
    try:
        info   = yf.Ticker(ticker_symbol).info
        shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
        val    = float(shares) if shares else np.nan
    except Exception:
        val = np.nan
    _shares_cache[ticker_symbol] = val
    return None if np.isnan(val) else val


def prefetch_sector_and_shares(tickers: List[str]) -> None:
    """Warm the sector / shares caches before the main calibration loop."""
    logger.info("Pre-fetching sector & share data for %d tickers …", len(tickers))
    for t in tickers:
        if t not in _sector_cache:
            get_sector(t)
        if t not in _shares_cache:
            get_shares_outstanding(t)
    logger.info("Prefetch complete.")
