"""
config.py  –  All tunable parameters for the Barra-style factor risk model.

Defaults replicate a simplified MSCI Barra USE4 / Axioma WW4 setup.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    # ── Lookback windows (trading days) ─────────────────────────────────────
    momentum_long_window:  int = 252   # 12-month price return (long leg)
    momentum_skip_window:  int = 21    # skip last month to avoid micro-reversal
    short_rev_window:      int = 21    # short-term reversal look-back
    vol_window:            int = 252   # realised-vol estimation window
    liquidity_window:      int = 20    # avg daily dollar-volume window
    min_price_history:     int = 252   # drop stocks with fewer price observations

    # ── Cross-sectional regression ───────────────────────────────────────────
    wls_scheme: str    = "sqrt_mktcap"  # "sqrt_mktcap" | "equal" | "mktcap"
    winsor_sigma: float = 3.0           # winsorise continuous exposures at ±N·σ
    min_stocks_in_regression: int = 50  # skip date if universe too thin

    # ── Fundamentals reporting lag ───────────────────────────────────────────
    # A quarter-end date is not when the market learns the numbers: 10-Qs
    # come out ~30-45 days later, 10-Ks up to ~90. Filtering fundamentals on
    # quarter-end date alone lets the model see earnings before the market
    # did. report_lag_days is subtracted from target_date before filtering.
    report_lag_days: int = 45

    # ── EWMA half-lives (trading days) ───────────────────────────────────────
    factor_cov_halflife: int = 63   # factor return covariance  (~3 months)
    idio_var_halflife:   int = 21   # idiosyncratic variance    (~1 month)

    # ── Style factors ────────────────────────────────────────────────────────
    style_factors: List[str] = field(default_factory=lambda: [
        "value_ep",     # Earnings Yield   = TTM Net Income / Market Cap
        "value_bp",     # Book Yield       = Book Equity   / Market Cap
        "momentum",     # Price Momentum   = ret(t-252 → t-21)
        "quality_roe",  # ROE              = TTM Net Income / Book Equity
        "quality_gm",   # Gross Margin     = TTM Gross Profit / TTM Revenue
        "size",         # log(Market Cap)
        "low_vol",      # –Realised Vol    (positive loading → lower vol)
    ])

    # ── Technical factors ────────────────────────────────────────────────────
    technical_factors: List[str] = field(default_factory=lambda: [
        "short_rev",    # –ret(t-21 → t)  (positive loading → lower reversal drag)
        "liquidity",    # log(avg 20d dollar volume)
    ])

    @property
    def continuous_factor_names(self) -> List[str]:
        """Style + technical (sector dummies are appended dynamically)."""
        return self.style_factors + self.technical_factors
