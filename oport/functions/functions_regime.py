"""
functions_regime.py
====================
Wrapper functions for the Wasserstein-HMM regime engine and the
regime-conditioned portfolio construction step.

Same shape as functions_risk.py: one initializer that returns a
calibrator + fitted model, a handful of thin getters that each return a
DataFrame, and one "construction" call that does the actual stock
picking. Everything downstream (oport_api.py) treats this exactly like
functions_risk.py -- same Config/pickle/error_collector pattern.
"""

import datetime
import numpy as np
import pandas as pd

# Import the regime engine package (co-located next to risk_engine)
from .regime.regime_engine import (
    RegimeModelConfig,
    RegimeCalibrator,
    PortfolioConstructor,
)


def _select_regime_factors(factor_returns: pd.DataFrame, regime_factor_names=None) -> pd.DataFrame:
    """
    model.factor_returns carries EVERY column the WLS regression solved
    for: the continuous style/technical factors AND one-hot sector
    dummies (see config.py's style_factors/technical_factors + the
    sector_<X> columns FactorBuilder appends). Feeding all of that into
    a rolling Gaussian HMM (~20 dims off a 252-day window) is
    numerically fragile, so by default we restrict regime inference to
    the continuous factors only -- sector dummies stay in play later,
    at the exposure-mapping step, just not for detecting the regime
    itself. Pass regime_factor_names explicitly to override.
    """
    if regime_factor_names is not None:
        cols = [c for c in regime_factor_names if c in factor_returns.columns]
    else:
        cols = [c for c in factor_returns.columns if not c.startswith("sector_")]
    if not cols:
        raise ValueError("No usable factor columns found for regime inference.")
    return factor_returns[cols]


def initialize_regime_model(factor_returns: pd.DataFrame, start_date=None, end_date=None,
                             regime_factor_names=None, **config_kwargs):
    """
    Initialize the RegimeCalibrator and run the rolling Wasserstein-HMM
    estimation over the given factor-return history.

    `factor_returns` is expected to be the SAME factor-return time series
    produced by the already-calibrated Barra-style risk model
    (functions_risk.get_factor_returns(model)) -- this is what ties
    regime inference to the risk model's own factor structure instead of
    a separate macro data source. By default only the continuous
    style/technical columns are used for the HMM itself (see
    _select_regime_factors); pass regime_factor_names to change that.
    """
    fr = _select_regime_factors(factor_returns, regime_factor_names)

    if start_date is None:
        start_date = fr.index.min()
    if end_date is None:
        end_date = fr.index.max()

    cfg_params = dict(
        min_regimes=2,
        max_regimes=4,
        estimation_window=252,
        refit_frequency=21,
        order_selection_holdout=21,
    )
    cfg_params.update(config_kwargs)
    cfg = RegimeModelConfig(**cfg_params)

    cal = RegimeCalibrator(cfg)
    cal.load_factor_returns(fr)
    cal.run(start=start_date, end=end_date)
    return cal, cfg


def get_regime_summary(cal: RegimeCalibrator) -> pd.DataFrame:
    """
    1. Per-regime summary (persistence, expected dwell time, sample share).
    Corresponds to: cal.summary()
    """
    return cal.summary()


def get_current_regime(cal: RegimeCalibrator) -> pd.DataFrame:
    """
    2. Snapshot of the currently active regime.
    """
    rid = cal.current_regime_
    # Confidence is the one-step-ahead PREDICTED probability of regime `rid`
    # at T+1 (pi_{T+1} = p_T . A), not the filtered probability at T -- the
    # filtered probability describes where we believe we ARE, which is a
    # different (and less relevant, for anything forward-looking) question
    # than where we expect to BE next.
    conf = float(cal.predicted_next_proba().get(rid, 0.0))
    row = {
        "regime_id": rid,
        "confidence": conf,
        "expected_dwell_days": cal.expected_dwell_time(rid),
        "as_of": cal.regime_label_.index[-1],
    }
    return pd.DataFrame([row]).set_index("regime_id")


def get_regime_probabilities(cal: RegimeCalibrator, last_n_days=None) -> pd.DataFrame:
    """
    3. Filtered (causal) regime probability time series.
    Corresponds to: cal.regime_prob_
    """
    probs = cal.regime_prob_
    if last_n_days is not None:
        return probs.tail(last_n_days)
    return probs


def get_transition_matrix(cal: RegimeCalibrator) -> pd.DataFrame:
    """
    4. Regime transition matrix (persistent-ID space).
    Corresponds to: cal.transition_matrix_
    """
    return cal.transition_matrix_


def get_expected_dwell_times(cal: RegimeCalibrator) -> pd.DataFrame:
    """
    5. Expected sojourn time (trading days) for every tracked regime.
    """
    rows = [
        {"regime_id": rid, "expected_dwell_days": cal.expected_dwell_time(rid)}
        for rid in cal.transition_matrix_.index
    ]
    return pd.DataFrame(rows).set_index("regime_id")


def get_regime_conditional_factor_moments(cal: RegimeCalibrator, regime=None):
    """
    6. Regime-conditional factor mean vector + covariance matrix.
    Corresponds to: cal.conditional_moments(regime)
    Returns (mean: Series, cov: DataFrame).
    """
    return cal.conditional_moments(regime)


def get_latest_exposures(risk_cal) -> pd.DataFrame:
    """
    The risk calibrator's most recently calibrated cross-sectional
    exposure matrix (stocks x [style + technical + sector-dummy
    factors]) -- calibrator.py updates `self.latest_exposures` on every
    `_calibrate_one_day()` call inside `run()`.
    """
    if risk_cal.latest_exposures is None:
        raise RuntimeError(
            "risk_cal.latest_exposures is None -- run risk calibration "
            "(cal.run(...)) before constructing a regime-aware portfolio."
        )
    return risk_cal.latest_exposures


def get_latest_idio_var(risk_cal, tickers) -> pd.Series:
    """
    Per-stock idiosyncratic variance, straight from
    FactorRiskModel.get_idio_variance() (calibrator.py's own EWMA idio
    variance store, half-life = cfg.idio_var_halflife).
    """
    idio = risk_cal.model.get_idio_variance()
    return idio.reindex(tickers)


def construct_regime_portfolio(
    risk_cal,
    regime_cal: RegimeCalibrator,
    universe,
    current_weights: pd.Series = None,
    target_n: int = None,
    max_weight: float = None,
    transaction_cost_bps: float = None,
    risk_aversion: float = None,
    regime: int = None,
    cfg: RegimeModelConfig = None,
) -> pd.DataFrame:
    """
    7. Regime-aware, transaction-cost-aware stock picking.

    Given a universe (list of tickers -- an index's constituents or a
    portfolio's holdings), maps the current (or a specified) regime's
    conditional factor moments onto each stock via its factor exposures
    from the already-calibrated Barra-style risk model, then solves a
    transaction-cost-aware mean-variance problem to produce target
    weights, together with a suggested holding period per name equal to
    the regime's expected dwell time.

    Returns a DataFrame indexed by ticker with columns:
        weight, prev_weight, trade, expected_return_regime,
        regime_id, regime_confidence, expected_hold_days
    """
    from . import functions_risk

    # display_regime is what we report (id, confidence, dwell time). The
    # FORECAST used for expected returns is separate: when the caller does
    # not pin a specific regime, conditional_moments(None) blends every
    # regime's moments by the predicted next-day belief rather than
    # hard-picking display_regime -- see RegimeCalibrator.conditional_moments.
    display_regime = regime if regime is not None else regime_cal.current_regime_
    regime_confidence = float(regime_cal.predicted_next_proba().get(display_regime, 0.0))
    expected_hold_days = regime_cal.expected_dwell_time(display_regime)
    mu_f_regime, _ = regime_cal.conditional_moments(regime)

    exposures = get_latest_exposures(risk_cal)
    factor_cov = functions_risk.get_factor_covariance(risk_cal.model)   # full K factors, incl. sector dummies
    idio_var = get_latest_idio_var(risk_cal, universe)

    # Regime only speaks to the continuous style/technical factors (see
    # _select_regime_factors). Sector-dummy factors get a flat 0 prior
    # instead of their unconditional historical mean: that mean is a 30-40%
    # annualised bull-market average acting as the market/beta term, and
    # handing every stock that as a baseline expected return is pure
    # extrapolation, not a regime call. The regime-sensitive factors are
    # shrunk toward their OWN unconditional mean by regime_tilt_alpha,
    # rather than fully trusting a single regime-conditional estimate.
    pcfg = cfg or RegimeModelConfig()
    alpha = pcfg.regime_tilt_alpha
    full_hist_mean = functions_risk.get_factor_returns(risk_cal.model).mean()
    mu_f = pd.Series(0.0, index=full_hist_mean.index)
    mu_uncond_regime = full_hist_mean.reindex(mu_f_regime.index)
    mu_f.loc[mu_f_regime.index] = mu_uncond_regime + alpha * (mu_f_regime - mu_uncond_regime)

    pc = PortfolioConstructor(pcfg)
    return pc.construct(
        exposures=exposures,
        factor_cov=factor_cov,
        idio_var=idio_var,
        regime_factor_mean=mu_f,
        regime_id=display_regime,
        regime_confidence=regime_confidence,
        expected_hold_days=expected_hold_days,
        current_weights=current_weights,
        universe=universe,
        target_n=target_n,
        max_weight=max_weight,
        transaction_cost_bps=transaction_cost_bps,
        risk_aversion=risk_aversion,
    )


def update_regime_model_daily(cal: RegimeCalibrator, factor_returns_incremental: pd.DataFrame):
    """
    8. Incremental daily update (live use-case).
    Appends the newest day(s) of factor returns and re-runs the walk-
    forward pass (refit only triggers internally at refit_frequency, so
    this is cheap on non-refit days).
    """
    combined = pd.concat([cal.factor_returns_, factor_returns_incremental])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    cal.load_factor_returns(combined)
    cal.run()
    return cal
