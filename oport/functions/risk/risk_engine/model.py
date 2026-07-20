"""
model.py  –  Core model: WLS regression, EWMA covariance, risk decomposition.

Classes
-------
RegressionResult   dataclass holding the output of one day's cross-sectional WLS.
FactorRiskModel    stateful model that accumulates factor returns and
                   maintains live EWMA covariance matrices.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import ModelConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RegressionResult:
    """Output of a single day's cross-sectional WLS estimation."""
    date:              object                  # calibration date
    factor_returns:    pd.Series               # f_t  [K]
    residuals:         pd.Series               # ε_t  [N] – indexed by ticker
    exposures:         pd.DataFrame            # X_t  [N × K]  (copy, for archiving)
    r_squared:         float = np.nan
    factor_t_stats:    pd.Series = field(default_factory=pd.Series)


# ─────────────────────────────────────────────────────────────────────────────
# WLS cross-sectional regression
# ─────────────────────────────────────────────────────────────────────────────

def _wls_regression(
    returns:   pd.Series,       # r_t  [N]
    exposures: pd.DataFrame,    # X_t  [N × K]  (may contain NaN in fundamentals)
    scheme:    str = "sqrt_mktcap",
    size_col:  str = "size",
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Weighted Least Squares:   r = X f + ε
                              f̂ = (X'WX)⁻¹ X'Wr

    Stocks with any NaN in X or r are dropped.

    Returns
    -------
    f_hat     : np.ndarray  [K]      – factor returns
    residuals : np.ndarray  [N_used] – idiosyncratic returns (aligned to idx)
    r2        : float                – weighted R²
    idx       : np.ndarray  [N_used] – integer positions used (into exposures.index)
    """
    # ── Align returns and exposures ──────────────────────────────────────────
    common = returns.index.intersection(exposures.index)
    r = returns.loc[common].values.astype(float)
    X = exposures.loc[common].values.astype(float)

    # Drop rows with any NaN
    mask = np.isfinite(r) & np.all(np.isfinite(X), axis=1)
    r, X = r[mask], X[mask]
    idx  = np.where(mask)[0]

    if len(r) < exposures.shape[1] + 5:
        raise ValueError(f"Only {len(r)} complete rows – too few for {X.shape[1]} factors.")

    # ── Regression weights ───────────────────────────────────────────────────
    if scheme == "equal":
        w = np.ones(len(r))
    elif scheme == "sqrt_mktcap" and size_col in exposures.columns:
        # size = log(mktcap), so mktcap = exp(size); weight = mktcap^0.5
        size_vals = exposures.loc[common].iloc[mask][size_col].values.astype(float)
        w = np.where(np.isfinite(size_vals), np.exp(0.5 * size_vals), 1.0)
    else:
        w = np.ones(len(r))

    w = np.maximum(w, 1e-12)      # guard against zeros
    sqrt_w = np.sqrt(w)

    # ── WLS via transformed OLS ──────────────────────────────────────────────
    X_w = sqrt_w[:, None] * X
    r_w = sqrt_w * r
    f_hat, _resid, _rank, _sv = np.linalg.lstsq(X_w, r_w, rcond=None)

    # ── Residuals (in original, unweighted space) ────────────────────────────
    residuals = r - X @ f_hat

    # ── Weighted R² ─────────────────────────────────────────────────────────
    ss_res = float(w @ residuals**2)
    ss_tot = float(w @ (r - np.average(r, weights=w))**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return f_hat, residuals, r2, idx


def run_cross_sectional_regression(
    date,
    returns:   pd.Series,
    exposures: pd.DataFrame,
    cfg:       ModelConfig,
) -> Optional[RegressionResult]:
    """
    Wrapper that runs WLS and packages results into a RegressionResult.

    Returns None if the universe is too small or regression fails.
    """
    if len(returns) < cfg.min_stocks_in_regression:
        logger.warning("Universe too small at %s (%d stocks) – skipping.", date, len(returns))
        return None

    try:
        f_hat, resid_vals, r2, used_idx = _wls_regression(
            returns, exposures,
            scheme   = cfg.wls_scheme,
            size_col = "size",
        )
    except ValueError as e:
        logger.warning("Regression failed at %s: %s", date, e)
        return None

    common        = returns.index.intersection(exposures.index)
    used_tickers  = common[np.array(
        [i for i, m in enumerate(
            np.isfinite(returns.loc[common].values) &
            np.all(np.isfinite(exposures.loc[common].values), axis=1)
        ) if m]
    )]

    factor_returns = pd.Series(f_hat, index=exposures.columns, name=date)
    residuals      = pd.Series(resid_vals, index=used_tickers[:len(resid_vals)], name=date)

    # ── Newey-West-style t-stats (homoskedastic approximation) ───────────────
    common_full   = returns.index.intersection(exposures.index)
    X_used        = exposures.loc[common_full].dropna().values.astype(float)
    var_matrix    = np.diag(np.linalg.pinv(X_used.T @ X_used)) * (resid_vals**2).mean()
    se            = np.sqrt(np.maximum(var_matrix, 0.0))
    t_stats       = pd.Series(f_hat / np.maximum(se, 1e-12), index=exposures.columns, name=date)

    return RegressionResult(
        date           = date,
        factor_returns = factor_returns,
        residuals      = residuals,
        exposures      = exposures.copy(),
        r_squared      = r2,
        factor_t_stats = t_stats,
    )


# ─────────────────────────────────────────────────────────────────────────────
# EWMA covariance estimator
# ─────────────────────────────────────────────────────────────────────────────

def _halflife_to_lambda(halflife: int) -> float:
    return 0.5 ** (1.0 / halflife)


class EWMACovarianceEstimator:
    """
    Maintains an Exponentially Weighted Moving Average (EWMA) covariance
    matrix for factor returns and EWMA variance for idiosyncratic returns.

    All updates are online (one observation at a time) to support streaming
    daily calibration without storing the full history.
    """

    def __init__(self, cfg: ModelConfig):
        self.cfg     = cfg
        self.lam_f   = _halflife_to_lambda(cfg.factor_cov_halflife)   # factor cov λ
        self.lam_d   = _halflife_to_lambda(cfg.idio_var_halflife)      # idio var λ

        # Initialised lazily on first update
        self._F: Optional[np.ndarray]         = None   # [K × K] factor covariance
        self._D: Optional[Dict[str, float]]   = None   # ticker → idio variance
        self._factor_names: Optional[List[str]] = None
        self._n_updates: int = 0

    # ── Public update ─────────────────────────────────────────────────────────

    def update(self, result: RegressionResult) -> None:
        """Ingest one day's RegressionResult and update both EWMA matrices."""
        f = result.factor_returns
        e = result.residuals

        # ── Initialise factor covariance ─────────────────────────────────────
        if self._F is None:
            K = len(f)
            self._factor_names = list(f.index)
            self._F = np.diag(f.values**2)          # initialise with today's outer product
        else:
            # Align: handle factor universe changes gracefully
            f_aligned = f.reindex(self._factor_names).fillna(0.0).values
            outer     = np.outer(f_aligned, f_aligned)
            self._F   = self.lam_f * self._F + (1 - self.lam_f) * outer

        # ── Initialise / update idio variance ────────────────────────────────
        if self._D is None:
            self._D = {}
        for ticker, eps in e.items():
            if not np.isfinite(eps):
                continue
            eps2 = eps ** 2
            if ticker in self._D:
                self._D[ticker] = self.lam_d * self._D[ticker] + (1 - self.lam_d) * eps2
            else:
                self._D[ticker] = eps2

        self._n_updates += 1

    # ── Public getters ────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._F is not None and self._n_updates > 0

    def get_factor_covariance(self) -> pd.DataFrame:
        """Return the K × K factor covariance matrix F."""
        if not self.is_ready:
            raise RuntimeError("No updates yet.")
        return pd.DataFrame(
            self._F,
            index=self._factor_names,
            columns=self._factor_names,
        )

    def get_idio_variance(self) -> pd.Series:
        """Return per-ticker idiosyncratic variance D (diagonal of D matrix)."""
        if not self.is_ready:
            raise RuntimeError("No updates yet.")
        return pd.Series(self._D, name="idio_var")

    def get_factor_names(self) -> List[str]:
        return list(self._factor_names) if self._factor_names else []


# ─────────────────────────────────────────────────────────────────────────────
# Risk decomposition
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StockRisk:
    ticker:        str
    factor_var:    float   # x_i' F x_i
    idio_var:      float   # D_ii
    total_var:     float   # factor_var + idio_var
    factor_vol:    float   # sqrt(factor_var)   in annual % if returns were daily
    idio_vol:      float   # sqrt(idio_var)
    total_vol:     float   # sqrt(total_var)
    factor_share:  float   # factor_var / total_var


@dataclass
class PortfolioRisk:
    factor_var:       float
    idio_var:         float
    total_var:        float
    factor_vol:       float
    idio_vol:         float
    total_vol:        float
    factor_share:     float
    factor_contributions: pd.Series   # per-factor contribution to total var


def _annualise_vol(daily_var: float, scale: float = 252.0) -> float:
    return float(np.sqrt(max(daily_var, 0.0) * scale) * 100.0)   # in %


class RiskDecomposer:
    """
    Decomposes risk using the factor model:
        Σ = X F Xᵀ + D
    """

    def __init__(
        self,
        factor_cov:   pd.DataFrame,     # F  [K × K]
        idio_var:     pd.Series,         # D_ii  indexed by ticker
        latest_exposures: pd.DataFrame,  # X  [N × K], most recent date
    ):
        self.F = factor_cov
        self.D = idio_var
        self.X = latest_exposures.reindex(columns=factor_cov.columns).fillna(0.0)

    def stock_risk(self, ticker: str, scale: float = 252.0) -> Optional[StockRisk]:
        if ticker not in self.X.index:
            logger.debug("Ticker %s not in exposure matrix.", ticker)
            return None

        x     = self.X.loc[ticker].values
        F_arr = self.F.values
        f_var = float(x @ F_arr @ x)
        d_var = float(self.D.get(ticker, np.nan))
        if not np.isfinite(d_var):
            return None
        t_var = f_var + d_var

        return StockRisk(
            ticker       = ticker,
            factor_var   = f_var,
            idio_var     = d_var,
            total_var    = t_var,
            factor_vol   = _annualise_vol(f_var,  scale),
            idio_vol     = _annualise_vol(d_var,  scale),
            total_vol    = _annualise_vol(t_var,  scale),
            factor_share = f_var / t_var if t_var > 0 else np.nan,
        )

    def portfolio_risk(
        self,
        weights: pd.Series,   # ticker → portfolio weight (sum to 1)
        scale:   float = 252.0,
    ) -> PortfolioRisk:
        """
        Full portfolio variance decomposition.

        Total var = w' X F X' w  +  w' D w
        Factor contribution of factor k = f_k² * (w' x_k)²
        """
        # Align
        tickers = weights.index.intersection(self.X.index)
        w = weights.loc[tickers].values.astype(float)
        X = self.X.loc[tickers].values
        F = self.F.values

        # Factor component
        factor_exposures = X.T @ w          # [K]  – portfolio factor exposure vector
        f_var = float(factor_exposures @ F @ factor_exposures)

        # Idio component
        d_vals = self.D.reindex(tickers).fillna(0.0).values
        d_var  = float(w**2 @ d_vals)

        t_var  = f_var + d_var

        # Per-factor contribution: Δf_k = 2 * (F @ factor_exposures)[k] * factor_exposures[k]
        # (marginal contribution × exposure, summing to total factor var)
        f_contrib_vals = factor_exposures * (F @ factor_exposures)
        factor_contributions = pd.Series(
            f_contrib_vals,
            index=self.F.index,
            name="factor_var_contribution",
        )

        return PortfolioRisk(
            factor_var            = f_var,
            idio_var              = d_var,
            total_var             = t_var,
            factor_vol            = _annualise_vol(f_var,  scale),
            idio_vol              = _annualise_vol(d_var,  scale),
            total_vol             = _annualise_vol(t_var,  scale),
            factor_share          = f_var / t_var if t_var > 0 else np.nan,
            factor_contributions  = factor_contributions,
        )

    def all_stocks_risk(self, scale: float = 252.0) -> pd.DataFrame:
        """Return a DataFrame with risk decomposition for every stock in the model."""
        rows = []
        for ticker in self.X.index:
            sr = self.stock_risk(ticker, scale=scale)
            if sr is not None:
                rows.append({
                    "ticker":       sr.ticker,
                    "total_vol":    sr.total_vol,
                    "factor_vol":   sr.factor_vol,
                    "idio_vol":     sr.idio_vol,
                    "factor_share": sr.factor_share,
                    "factor_var":   sr.factor_var,
                    "idio_var":     sr.idio_var,
                })
        return pd.DataFrame(rows).set_index("ticker").sort_values("total_vol", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# FactorRiskModel – top-level object
# ─────────────────────────────────────────────────────────────────────────────

class FactorRiskModel:
    """
    Stateful Barra-style factor risk model.

    Usage
    -----
    model = FactorRiskModel(cfg)
    model.ingest(regression_result)   # called by DailyCalibrator each day

    # After sufficient history:
    model.get_risk_decomposer(latest_exposures)   → RiskDecomposer
    model.factor_returns                          → pd.DataFrame  [dates × factors]
    model.get_factor_covariance()                 → pd.DataFrame  [K × K]
    model.get_idio_variance()                     → pd.Series     [tickers]
    """

    def __init__(self, cfg: ModelConfig):
        self.cfg        = cfg
        self._cov_est   = EWMACovarianceEstimator(cfg)
        self._results:  List[RegressionResult] = []

    def ingest(self, result: RegressionResult) -> None:
        """Update the model with one day's regression result."""
        self._cov_est.update(result)
        self._results.append(result)

    # ── Outputs ───────────────────────────────────────────────────────────────

    @property
    def factor_returns(self) -> pd.DataFrame:
        """Time series of factor returns: [dates × factors]."""
        if not self._results:
            return pd.DataFrame()
        return pd.DataFrame(
            [r.factor_returns for r in self._results],
        )

    @property
    def factor_t_stats(self) -> pd.DataFrame:
        """Time series of factor t-stats: [dates × factors]."""
        if not self._results:
            return pd.DataFrame()
        return pd.DataFrame([r.factor_t_stats for r in self._results])

    @property
    def r_squared_history(self) -> pd.Series:
        return pd.Series(
            {r.date: r.r_squared for r in self._results},
            name="r_squared",
        )

    def get_factor_covariance(self) -> pd.DataFrame:
        """F matrix: EWMA covariance of factor returns [K × K]."""
        return self._cov_est.get_factor_covariance()

    def get_idio_variance(self) -> pd.Series:
        """EWMA idiosyncratic variance per ticker."""
        return self._cov_est.get_idio_variance()

    def get_risk_decomposer(
        self, latest_exposures: pd.DataFrame
    ) -> RiskDecomposer:
        """Build a RiskDecomposer from the current covariance state."""
        if not self._cov_est.is_ready:
            raise RuntimeError("Model has not been updated yet.")
        return RiskDecomposer(
            factor_cov        = self.get_factor_covariance(),
            idio_var          = self.get_idio_variance(),
            latest_exposures  = latest_exposures,
        )

    @property
    def n_days_calibrated(self) -> int:
        return len(self._results)

    @property
    def last_date(self):
        return self._results[-1].date if self._results else None

    def summary(self) -> pd.DataFrame:
        """Annualised factor return statistics over the calibration history."""
        fr = self.factor_returns
        if fr.empty:
            return pd.DataFrame()
        stats = pd.DataFrame({
            "mean_ann_%":    fr.mean() * 252 * 100,
            "vol_ann_%":     fr.std()  * np.sqrt(252) * 100,
            "sharpe":        fr.mean() / fr.std() * np.sqrt(252),
            "min":           fr.min(),
            "max":           fr.max(),
            "avg_t_stat":    self.factor_t_stats.mean(),
        })
        return stats.sort_values("sharpe", ascending=False)
