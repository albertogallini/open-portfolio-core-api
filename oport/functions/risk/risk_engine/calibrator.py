"""
calibrator.py  –  DailyCalibrator: orchestrates the full pipeline.

Pipeline per trading day
------------------------
  1. Compute daily returns from price panel.
  2. Build exposure matrix  X_t  [N × K]  via FactorBuilder.
  3. Run WLS cross-sectional regression  →  factor returns  f_t, residuals ε_t.
  4. Update EWMA covariance via FactorRiskModel.ingest().
  5. Persist RegressionResult and exposures for downstream queries.

Outputs
-------
  model.factor_returns                    pd.DataFrame  [dates × K]
  model.get_factor_covariance()           pd.DataFrame  [K × K]
  model.get_idio_variance()               pd.Series     [tickers]
  model.get_risk_decomposer(exposures)    RiskDecomposer
  calibrator.exposures_history            dict[date → pd.DataFrame]
"""
from __future__ import annotations

import datetime
import logging
from typing import Dict, List, Optional

import pandas as pd

from .config import ModelConfig
from .data import (
    load_sp500_tickers,
    load_price_volume_panel,
    prefetch_sector_and_shares,
    get_price_panel,
    get_volume_panel,
)
from .factors import FactorBuilder
from .model import (
    FactorRiskModel,
    RegressionResult,
    run_cross_sectional_regression,
)

logger = logging.getLogger(__name__)


class DailyCalibrator:
    """
    Runs the end-to-end Barra-style factor risk model calibration
    over a range of trading dates.

    Parameters
    ----------
    cfg      : ModelConfig
    tickers  : list of ticker symbols (your index universe).
               Pass None to auto-load S&P 500 from Wikipedia.
    """

    def __init__(
        self,
        cfg:     ModelConfig,
        tickers: Optional[List[str]] = None,
    ):
        self.cfg     = cfg
        self.tickers = tickers
        self.model   = FactorRiskModel(cfg)

        # Latest exposure matrix (updated each day)
        self.latest_exposures: Optional[pd.DataFrame] = None

        # Full history of exposure matrices keyed by date
        self.exposures_history: Dict[object, pd.DataFrame] = {}

        self._prices:  Optional[pd.DataFrame] = None
        self._volumes: Optional[pd.DataFrame] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Data loading
    # ─────────────────────────────────────────────────────────────────────────

    def load_data(
        self,
        start: datetime.date,
        end:   datetime.date,
    ) -> None:
        """
        Download price/volume panel and pre-fetch sector/shares metadata.

        Call this once before run().  Adds an extra 252-day warm-up buffer
        so that momentum and vol factors have sufficient history from day 1.
        """
        if self.tickers is None:
            logger.info("Loading S&P 500 universe from Wikipedia …")
            self.tickers = load_sp500_tickers()

        # Extend start by a full year for factor warm-up
        data_start = start - datetime.timedelta(days=int(365 * 1.5))

        self._prices, self._volumes = load_price_volume_panel(
            self.tickers, data_start, end
        )
        prefetch_sector_and_shares(self.tickers)

    # ─────────────────────────────────────────────────────────────────────────
    # Main calibration loop
    # ─────────────────────────────────────────────────────────────────────────

    def run(
        self,
        start: datetime.date,
        end:   datetime.date,
        store_exposures: bool = False,
    ) -> FactorRiskModel:
        """
        Calibrate the model daily from start to end (inclusive).

        Parameters
        ----------
        store_exposures : keep every day's exposure matrix in exposures_history.
                          Memory-intensive for large universes; False by default.

        Returns
        -------
        The updated FactorRiskModel (also accessible via self.model).
        """
        if self._prices is None:
            self.load_data(start, end)

        builder     = FactorBuilder(self.cfg, self._prices, self._volumes)
        trading_days = self._get_trading_days(start, end)

        logger.info(
            "Starting calibration: %s → %s (%d trading days, %d tickers)",
            start, end, len(trading_days), len(self.tickers),
        )

        for i, date in enumerate(trading_days):
            try:
                result = self._calibrate_one_day(date, builder)
                if result is not None:
                    self.model.ingest(result)
                    self.latest_exposures = result.exposures
                    if store_exposures:
                        self.exposures_history[date] = result.exposures.copy()

                if (i + 1) % 20 == 0:
                    logger.info(
                        "  [%d/%d] %s  –  R²=%.3f  factors=%d  stocks=%d",
                        i + 1, len(trading_days), date,
                        result.r_squared if result else float("nan"),
                        len(result.factor_returns) if result else 0,
                        len(result.residuals) if result else 0,
                    )

            except Exception as exc:
                logger.error("Error on %s: %s", date, exc, exc_info=True)
                continue

        logger.info(
            "Calibration complete. %d days processed.",
            self.model.n_days_calibrated,
        )
        return self.model

    # ─────────────────────────────────────────────────────────────────────────
    # Single-day calibration
    # ─────────────────────────────────────────────────────────────────────────

    def _calibrate_one_day(
        self,
        date,
        builder: FactorBuilder,
    ) -> Optional[RegressionResult]:

        # ── 1. Daily returns  r_t ────────────────────────────────────────────
        px = self._prices
        prev_dates = px.index[px.index < date]
        if len(prev_dates) == 0:
            return None
        prev_date = prev_dates[-1]

        p_today = px.loc[date]
        p_prev  = px.loc[prev_date]

        returns = (p_today / p_prev - 1.0).dropna()
        returns = returns[returns.between(-0.5, 0.5)]  # remove splits / data errors

        if len(returns) < self.cfg.min_stocks_in_regression:
            return None

        # ── 2. Exposure matrix  X_t ──────────────────────────────────────────
        exposures = builder.build(
            tickers     = returns.index.tolist(),
            target_date = date,
        )
        if exposures.empty:
            return None

        # ── 3. Cross-sectional WLS regression ────────────────────────────────
        result = run_cross_sectional_regression(
            date      = date,
            returns   = returns,
            exposures = exposures,
            cfg       = self.cfg,
        )
        
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_trading_days(
        self, start: datetime.date, end: datetime.date
    ) -> List:
        """Return sorted list of dates in the price panel within [start, end]."""
        idx = self._prices.index
        return sorted(d for d in idx if start <= d <= end)

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience query methods
    # ─────────────────────────────────────────────────────────────────────────

    def get_risk_decomposer(self):
        """
        Return a RiskDecomposer using the most recently calibrated exposures.
        Use this to query stock / portfolio risk.
        """
        if self.latest_exposures is None:
            raise RuntimeError("Run calibration first.")
        return self.model.get_risk_decomposer(self.latest_exposures)

    def report(self) -> None:
        """Print a concise summary of the calibrated model."""
        print("=" * 60)
        print(f"Factor Risk Model  |  {self.model.n_days_calibrated} trading days calibrated")
        print(f"Last date          :  {self.model.last_date}")
        print(f"Universe size      :  {len(self.tickers)} tickers")
        print()
        print("─── Factor Return Summary (annualised) ─────────────────────")
        print(self.model.summary().round(3).to_string())
        print()
        print("─── Average Cross-Sectional R² ─────────────────────────────")
        print(f"  {self.model.r_squared_history.mean():.4f}")
        print("=" * 60)
