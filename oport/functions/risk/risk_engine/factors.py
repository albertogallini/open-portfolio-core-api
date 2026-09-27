"""
factors.py  –  FactorBuilder: constructs the N × K exposure matrix.

Factor catalogue
----------------
Style (continuous, cross-sectionally z-scored after winsorisation)
  value_ep      Earnings Yield   = TTM Net Income / Market Cap
  value_bp      Book Yield       = Book Equity   / Market Cap
  momentum      Price Momentum   = ret(t-252 → t-21)
  quality_roe   ROE              = TTM Net Income / Book Equity
  quality_gm    Gross Margin     = TTM Gross Profit / TTM Revenue
  size          log(Market Cap)
  low_vol       –Realised Vol    (positive loading → lower volatility)

Technical (continuous, same normalisation)
  short_rev     –ret(t-21 → t)   (positive loading → lower reversal drag)
  liquidity     log(avg 20-day dollar volume)

Sector (binary dummies, one sector dropped as reference)
  sector_<X>    one-hot per GICS sector
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from .config import ModelConfig
from .data import (
    get_quarterly_fundamentals,
    first_valid_ttm,
    first_valid_bs,
    get_sector,
    get_shares_outstanding,
)

logger = logging.getLogger(__name__)

# yfinance field-name aliases (vary across tickers / API versions)
_EQUITY_FIELDS      = ["Stockholders Equity", "Total Stockholder Equity",
                        "Common Stock Equity", "Total Equity Gross Minority Interest"]
_NET_INCOME_FIELDS  = ["Net Income", "Net Income Common Stockholders",
                        "Net Income Including Noncontrolling Interests"]
_REVENUE_FIELDS     = ["Total Revenue"]
_GROSS_PROFIT_FIELDS= ["Gross Profit"]


class FactorBuilder:
    """
    Builds the N × K exposure matrix for a given calibration date.

    Parameters
    ----------
    config  : ModelConfig
    prices  : pd.DataFrame  shape [dates × tickers], adjusted close
    volumes : pd.DataFrame  shape [dates × tickers], share volume
    """

    def __init__(
        self,
        config: ModelConfig,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
    ):
        self.cfg     = config
        self.prices  = prices
        self.volumes = volumes

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def build(
        self,
        tickers: List[str],
        target_date,
        prefetch_fundamentals: bool = False,
    ) -> pd.DataFrame:
        """
        Return a DataFrame of shape [valid_tickers × all_factors] for target_date.

        Columns: style factors + technical factors + sector dummies.
        Stocks with insufficient price history are dropped silently.

        Parameters
        ----------
        prefetch_fundamentals : if True, fetches fundamentals in one pass before
                                computing exposures (reduces repeated yfinance calls).
        """
        # ── 1. Filter to stocks with sufficient price history ────────────────
        px_hist = (
            self.prices
            .loc[self.prices.index <= target_date, :]
            .loc[:, self.prices.columns.isin(tickers)]
        )
        obs_count = px_hist.notna().sum()
        valid_tickers = obs_count[obs_count >= self.cfg.min_price_history].index.tolist()

        if not valid_tickers:
            logger.warning("No stocks with sufficient price history at %s", target_date)
            return pd.DataFrame()

        # ── 2. Build raw exposure rows ───────────────────────────────────────
        records: Dict[str, Dict] = {}
        for ticker in valid_tickers:
            row = self._compute_row(ticker, target_date, px_hist)
            if row is not None:
                records[ticker] = row

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(records, orient="index")

        # ── 3. Sector dummies ────────────────────────────────────────────────
        df = self._add_sector_dummies(df)

        # ── 4. Winsorise + cross-sectional z-score continuous factors ────────
        cont_cols = [c for c in self.cfg.continuous_factor_names if c in df.columns]
        df[cont_cols] = df[cont_cols].apply(self._winsorise_zscore, axis=0).fillna(0.0)

        logger.debug(
            "Built exposure matrix at %s: %d stocks × %d factors",
            target_date, df.shape[0], df.shape[1],
        )
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # Per-stock row computation
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_row(
        self,
        ticker: str,
        target_date,
        px_hist: pd.DataFrame,
    ) -> Optional[Dict]:
        prices = px_hist[ticker].dropna()
        if len(prices) < self.cfg.min_price_history:
            return None

        p_today  = float(prices.iloc[-1])
        vol_hist = (
            self.volumes
            .loc[self.volumes.index <= target_date, ticker]
            .dropna()
        )

        row: Dict = {}

        # ── Price-based factors ──────────────────────────────────────────────
        row["momentum"]  = self._momentum(prices)
        row["low_vol"]   = self._low_vol(prices)
        row["short_rev"] = self._short_rev(prices)
        row["liquidity"] = self._liquidity(prices, vol_hist)

        # ── Market cap (anchors both size and value factors) ─────────────────
        shares  = get_shares_outstanding(ticker)
        mkt_cap = (p_today * shares) if shares else None
        row["size"] = np.log(mkt_cap) if (mkt_cap and mkt_cap > 0) else np.nan

        # ── Fundamental factors ──────────────────────────────────────────────
        inc, _cf, bs = get_quarterly_fundamentals(
            ticker, target_date, report_lag_days=self.cfg.report_lag_days
        )

        net_income   = first_valid_ttm(inc, _NET_INCOME_FIELDS)
        revenue      = first_valid_ttm(inc, _REVENUE_FIELDS)
        gross_profit = first_valid_ttm(inc, _GROSS_PROFIT_FIELDS)
        book_equity  = first_valid_bs(bs,  _EQUITY_FIELDS)

        def safe_ratio(num, den) -> float:
            if num is None or den is None or den == 0:
                return np.nan
            return num / den

        row["value_ep"]    = safe_ratio(net_income,   mkt_cap)
        row["value_bp"]    = safe_ratio(book_equity,  mkt_cap)
        row["quality_roe"] = safe_ratio(net_income,   book_equity)
        row["quality_gm"]  = safe_ratio(gross_profit, revenue)

        return row

    # ─────────────────────────────────────────────────────────────────────────
    # Individual factor helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _momentum(self, prices: pd.Series) -> float:
        lw, sk = self.cfg.momentum_long_window, self.cfg.momentum_skip_window
        if len(prices) < lw:
            return np.nan
        p_end   = float(prices.iloc[-(sk + 1)])
        p_start = float(prices.iloc[-lw])
        if p_start <= 0:
            return np.nan
        return p_end / p_start - 1.0

    def _low_vol(self, prices: pd.Series) -> float:
        w = self.cfg.vol_window
        if len(prices) < w + 1:
            return np.nan
        rets = prices.iloc[-(w + 1):].pct_change().dropna()
        vol  = float(rets.std() * np.sqrt(252))
        return -vol  # flip sign: positive loading → lower vol

    def _short_rev(self, prices: pd.Series) -> float:
        w = self.cfg.short_rev_window
        if len(prices) < w + 1:
            return np.nan
        ret = float(prices.iloc[-1]) / float(prices.iloc[-(w + 1)]) - 1.0
        return -ret   # flip sign: positive loading → less reversal drag

    def _liquidity(self, prices: pd.Series, volumes: pd.Series) -> float:
        w = self.cfg.liquidity_window
        common = prices.index.intersection(volumes.index)
        if len(common) < w:
            return np.nan
        dv = (prices.loc[common] * volumes.loc[common]).dropna()
        if len(dv) < w:
            return np.nan
        avg_dv = float(dv.iloc[-w:].mean())
        return np.log(avg_dv) if avg_dv > 0 else np.nan

    # ─────────────────────────────────────────────────────────────────────────
    # Sector dummies
    # ─────────────────────────────────────────────────────────────────────────

    def _add_sector_dummies(self, df: pd.DataFrame) -> pd.DataFrame:
        sectors = pd.Series(
            {t: get_sector(t) for t in df.index},
            name="sector",
        )
        # drop_first=True removes one reference sector → avoids perfect multicollinearity
        dummies = pd.get_dummies(sectors, prefix="sector", drop_first=True).astype(float)
        return df.join(dummies, how="left")

    # ─────────────────────────────────────────────────────────────────────────
    # Normalisation
    # ─────────────────────────────────────────────────────────────────────────

    def _winsorise_zscore(self, col: pd.Series) -> pd.Series:
        """Winsorise at ±N·σ, then cross-sectionally z-score (handles NaN)."""
        valid = col.dropna()
        if len(valid) < 5:
            return pd.Series(np.nan, index=col.index)
        mu, sigma = valid.mean(), valid.std()
        if sigma == 0 or np.isnan(sigma):
            return pd.Series(0.0, index=col.index)
        clipped = col.clip(mu - self.cfg.winsor_sigma * sigma,
                           mu + self.cfg.winsor_sigma * sigma)
        mu2  = clipped.mean()
        std2 = clipped.std()
        if std2 == 0:
            return pd.Series(0.0, index=col.index)
        return (clipped - mu2) / std2
