"""
functions_risk.py
=================
Wrapper functions for the factor risk model outputs from risk_engine.
"""

import sys
import datetime
from typing import Optional
import numpy as np
import pandas as pd

# Import the newly copied local risk_engine package
from .risk.risk_engine import ModelConfig, DailyCalibrator

def initialize_risk_model(tickers=None, start_date=None, end_date=None, **config_kwargs):
    """
    Initialize the DailyCalibrator and run the model over a historical window.
    """
    if start_date is None:
        start_date = datetime.date(2022, 1, 1)
    if end_date is None:
        end_date = datetime.date(2024, 12, 31)
        
    cfg_params = dict(
        factor_cov_halflife=63,
        idio_var_halflife=21,
        wls_scheme="sqrt_mktcap",
        winsor_sigma=3.0,
        min_stocks_in_regression=20,
    )
    cfg_params.update(config_kwargs)
    cfg = ModelConfig(**cfg_params)
    
    cal = DailyCalibrator(cfg, tickers=tickers)
    cal.load_data(start=start_date, end=end_date)
    model = cal.run(start=start_date, end=end_date)
    return cal, model

def get_calibration_summary(model):
    """
    1. Summary DataFrame.
    Corresponds to: model.summary()
    """
    return model.summary()

def get_factor_returns(model, last_n_days=None):
    """
    2. Factor returns time series.
    Corresponds to: model.factor_returns
    """
    fr = model.factor_returns
    if last_n_days is not None:
        return fr.tail(last_n_days)
    return fr

def get_cumulative_factor_returns(model):
    """
    3. Cumulative factor performance.
    Corresponds to: (1 + fr).cumprod() - 1
    """
    fr = model.factor_returns
    cum_fr = (1 + fr).cumprod() - 1
    return cum_fr.iloc[-1].sort_values(ascending=False).to_frame(name="cumulative_return")

def get_factor_significance(model):
    """
    4. Factor t-stats (signal of factor significance).
    Corresponds to solving for avg |t| and % days |t| > 2.
    """
    avg_t = model.factor_t_stats.mean()
    pct_sig = (model.factor_t_stats.abs() > 2.0).mean() * 100
    return pd.DataFrame({"avg_|t|": avg_t, "pct_sig_%": pct_sig})

def get_factor_covariance(model):
    """
    5. Factor covariance matrix.
    Corresponds to: model.get_factor_covariance()
    """
    return model.get_factor_covariance()

def get_factor_correlation(model, annualize=True):
    """
    5b. Factor correlation matrix.
    """
    F = model.get_factor_covariance()
    scale = 252 if annualize else 1
    D_std = np.sqrt(np.diag(F.values) * scale)
    corr = F.values / np.outer(D_std, D_std)
    return pd.DataFrame(corr, index=F.index, columns=F.columns)

def get_all_stocks_risk(cal, scale=252):
    """
    6. Stock-level risk decomposition.
    Corresponds to: decomp.all_stocks_risk(scale=252)
    """
    decomp = cal.get_risk_decomposer()
    return decomp.all_stocks_risk(scale=scale)

def get_single_stock_risk(cal, ticker):
    """
    6b. Single stock drill-down returning a DataFrame.
    """
    decomp = cal.get_risk_decomposer()
    sr = decomp.stock_risk(ticker)
    if sr is None:
        return pd.DataFrame()
    return pd.DataFrame([{
        "ticker":       sr.ticker,
        "total_vol":    sr.total_vol,
        "factor_vol":   sr.factor_vol,
        "idio_vol":     sr.idio_vol,
        "factor_share": sr.factor_share,
        "factor_var":   sr.factor_var,
        "idio_var":     sr.idio_var,
    }]).set_index("ticker")

def get_portfolio_risk(
    cal,
    weights,
    confidence: float = 0.95,
    horizon_days: int = 1,
    portfolio_mv: float = None,
):
    """
    7. Portfolio risk decomposition + ex-ante parametric VaR, all in one DataFrame.

    decomp.portfolio_risk() returns:
      - total_var / factor_var / idio_var : RAW DAILY variance (decimal, un-annualized)
      - total_vol / factor_vol / idio_vol : ANNUALIZED vol, in PERCENTAGE POINTS
        (i.e. total_vol = 100 * sqrt(total_var * scale))

    OUTPUT CONVENTION: all vol/sigma/VaR fields below are annualized (or
    horizon-scaled) PERCENTAGE POINTS (e.g. 4.307 == 4.307%, not 0.04307).
    Var fields are in matching squared-percent units, so the identity
        total_vol_out ** 2 == total_var_out
    still holds on the returned values.

    NOTE: this function expects `weights` on a 0-100 percentage scale
    (it divides by 100.0 internally, since decomp.portfolio_risk() itself
    expects fractional weights summing to 1). The caller in oport_api.py
    must pass weights * 100.0 when the aggregated weights are fractional
    (sum ~= 1).

    VaR columns:
      - sigma_daily   : daily 1-sigma, in percentage points
      - sigma_horizon : sigma scaled to *horizon_days* via sqrt-of-time, in percentage points
      - VaR_pct       : parametric VaR, in percentage points of portfolio value
      - VaR_value     : VaR in currency units (only when portfolio_mv is provided)

    Parameters
    ----------
    confidence   : 0.95 or 0.99
    horizon_days : number of trading days for the VaR horizon
    portfolio_mv : market value of the portfolio (optional); when given,
                   VaR_value is included in the output.
    """
    Z_SCORES = {0.95: 1.6449, 0.99: 2.3263}
    if confidence not in Z_SCORES:
        raise ValueError(f"Unsupported confidence level: {confidence}. Use 0.95 or 0.99.")
    z = Z_SCORES[confidence]

    ANNUALIZATION_SCALE = 252.0

    print("--- get_portfolio_risk Input Weights ---")
    print(weights)
    print("----------------------------------------")

    decomp = cal.get_risk_decomposer()
    pr = decomp.portfolio_risk(weights / 100.0)

    # var: raw daily -> annualized decimal
    total_var_annual  = pr.total_var  * ANNUALIZATION_SCALE
    factor_var_annual = pr.factor_var * ANNUALIZATION_SCALE
    idio_var_annual   = pr.idio_var   * ANNUALIZATION_SCALE

    # vol: already annualized, but in percentage points -> decimal fraction
    total_vol_annual  = pr.total_vol  / 100.0
    factor_vol_annual = pr.factor_vol / 100.0
    idio_vol_annual   = pr.idio_vol   / 100.0

    sigma_daily   = total_vol_annual / np.sqrt(ANNUALIZATION_SCALE)
    sigma_horizon = sigma_daily * np.sqrt(horizon_days)
    var_pct       = z * sigma_horizon

    print("--- get_portfolio_risk Computed Values (internal, decimal fraction) ---")
    print(f"confidence: {confidence}")
    print(f"horizon_days: {horizon_days}")
    print(f"z: {z}")
    print(f"pr.total_var (raw daily): {pr.total_var}")
    print(f"pr.factor_var (raw daily): {pr.factor_var}")
    print(f"pr.idio_var (raw daily): {pr.idio_var}")
    print(f"pr.total_vol (annualized, pct points): {pr.total_vol}")
    print(f"pr.factor_vol (annualized, pct points): {pr.factor_vol}")
    print(f"pr.idio_vol (annualized, pct points): {pr.idio_vol}")
    print(f"total_var_annual (decimal): {total_var_annual}")
    print(f"total_vol_annual (decimal): {total_vol_annual}")
    print(f"CHECK total_vol_annual**2 vs total_var_annual: {total_vol_annual**2} vs {total_var_annual}")
    print(f"sigma_daily (decimal): {sigma_daily}")
    print(f"sigma_horizon (decimal): {sigma_horizon}")
    print(f"var_pct (decimal): {var_pct}")
    print(f"portfolio_mv: {portfolio_mv}")
    print("------------------------------------------")

    # --- Convert everything to percentage-point display units ---
    # vol/sigma/VaR: linear in the underlying quantity -> * 100
    # var: quadratic in the underlying quantity -> * 100**2, to preserve
    #      the identity total_vol_pct ** 2 == total_var_pct
    row = {
        "total_vol":      total_vol_annual  * 100.0,
        "factor_vol":     factor_vol_annual * 100.0,
        "idio_vol":       idio_vol_annual   * 100.0,
        "total_var":      total_var_annual  * 100.0**2,
        "factor_var":     factor_var_annual * 100.0**2,
        "idio_var":       idio_var_annual   * 100.0**2,
        "factor_share":   pr.factor_share,          # already a 0-1 ratio, left as-is
        "confidence":     confidence,
        "horizon_days":   horizon_days,
        "sigma_daily":    sigma_daily   * 100.0,
        "sigma_horizon":  sigma_horizon * 100.0,
        "VaR_pct":        var_pct       * 100.0,
    }
    if portfolio_mv is not None:
        # VaR_value stays in currency units, unaffected by the % display change
        row["VaR_value"] = var_pct * portfolio_mv

    return pd.DataFrame([row], index=["portfolio"])
    
def get_portfolio_factor_contributions(cal, weights):
    """
    7b. Portfolio factor contributions returning a DataFrame.
    """
    decomp = cal.get_risk_decomposer()
    pr = decomp.portfolio_risk(weights / 100.0)
    return pr.factor_contributions.to_frame(name="contribution")

def get_security_risk_contributions(
    cal,
    weights: pd.Series,
    confidence: float = 0.95,
    horizon_days: int = 1,
    portfolio_mv: Optional[float] = None,
) -> pd.DataFrame:
    """
    Euler risk-contribution decomposition: one row per portfolio holding,
    plus a 'portfolio' summary row identical to get_portfolio_risk output.

    Security rows expose:
      weight        -- portfolio weight (0-100 scale)
      total_vol     -- annualised standalone vol of the stock (%)
      factor_vol    -- factor component of standalone vol (%)
      idio_vol      -- idio component of standalone vol (%)
      factor_share  -- factor_var / total_var of the standalone stock (0-1)
      RC_vol        -- Euler vol contribution to portfolio total_vol (%),
                       security rows sum to portfolio total_vol
      RC_pct        -- % share of portfolio variance, security rows sum to 100

    Portfolio row additionally exposes total_var, factor_var, idio_var,
    confidence, horizon_days, sigma_daily, sigma_horizon, VaR_pct (and
    VaR_value when portfolio_mv is provided) — identical to get_portfolio_risk.

    Tickers absent from the risk model exposure matrix are silently excluded;
    the portfolio row reflects only the covered subset.
    weights must be on the same 0-100 percentage scale as get_portfolio_risk.
    """
    Z_SCORES = {0.95: 1.6449, 0.99: 2.3263}
    if confidence not in Z_SCORES:
        raise ValueError(f"Unsupported confidence level: {confidence}. Use 0.95 or 0.99.")
    z = Z_SCORES[confidence]
    ANNUALIZATION_SCALE = 252.0

    decomp = cal.get_risk_decomposer()
    w_frac = weights / 100.0

    tickers = w_frac.index.intersection(decomp.X.index)
    if tickers.empty:
        raise ValueError("None of the portfolio tickers are present in the risk model.")

    w_arr   = w_frac.loc[tickers].values.astype(float)
    X_arr   = decomp.X.loc[tickers].values.astype(float)   # [N x K]
    F_arr   = decomp.F.values.astype(float)                 # [K x K]
    D_arr   = decomp.D.reindex(tickers).fillna(0.0).values.astype(float)  # [N]

    # (Sigma @ w) decomposed into factor and idio parts
    fe             = X_arr.T @ w_arr          # [K] portfolio factor exposures
    F_fe           = F_arr @ fe               # [K]
    factor_sigma_w = X_arr @ F_fe             # [N] factor part of (Sigma @ w)
    idio_sigma_w   = D_arr * w_arr            # [N] idio part  of (Sigma @ w)
    sigma_w        = factor_sigma_w + idio_sigma_w

    t_var_daily    = float(w_arr @ sigma_w)
    sigma_p_daily  = float(np.sqrt(max(t_var_daily, 0.0)))

    # Euler contributions (variance share and vol contribution)
    RC_pct = (w_arr * sigma_w / t_var_daily * 100.0) if t_var_daily > 0 else np.zeros(len(tickers))
    RC_vol = (w_arr * sigma_w / sigma_p_daily * np.sqrt(ANNUALIZATION_SCALE) * 100.0) if sigma_p_daily > 0 else np.zeros(len(tickers))

    # Portfolio-level summary via decomp.portfolio_risk (consistent with get_portfolio_risk)
    pr = decomp.portfolio_risk(w_frac.loc[tickers])

    total_vol_annual  = pr.total_vol  / 100.0
    factor_vol_annual = pr.factor_vol / 100.0
    idio_vol_annual   = pr.idio_vol   / 100.0
    total_var_annual  = pr.total_var  * ANNUALIZATION_SCALE
    factor_var_annual = pr.factor_var * ANNUALIZATION_SCALE
    idio_var_annual   = pr.idio_var   * ANNUALIZATION_SCALE

    sigma_daily_dec   = total_vol_annual / np.sqrt(ANNUALIZATION_SCALE)
    sigma_horizon_dec = sigma_daily_dec * np.sqrt(horizon_days)
    var_pct_dec       = z * sigma_horizon_dec

    # Security rows
    rows = []
    for i, ticker in enumerate(tickers):
        sr = decomp.stock_risk(ticker, scale=ANNUALIZATION_SCALE)
        row = {
            "weight":        w_arr[i] * 100.0,
            "total_vol":     sr.total_vol    if sr is not None else np.nan,
            "factor_vol":    sr.factor_vol   if sr is not None else np.nan,
            "idio_vol":      sr.idio_vol     if sr is not None else np.nan,
            "factor_share":  sr.factor_share if sr is not None else np.nan,
            "RC_vol":        RC_vol[i],
            "RC_pct":        RC_pct[i],
            "total_var":     np.nan,
            "factor_var":    np.nan,
            "idio_var":      np.nan,
            "confidence":    np.nan,
            "horizon_days":  np.nan,
            "sigma_daily":   np.nan,
            "sigma_horizon": np.nan,
            "VaR_pct":       np.nan,
        }
        if portfolio_mv is not None:
            row["VaR_value"] = np.nan
        rows.append((ticker, row))

    # Portfolio summary row — mirrors get_portfolio_risk column layout
    portfolio_row = {
        "weight":        100.0,
        "total_vol":     total_vol_annual  * 100.0,
        "factor_vol":    factor_vol_annual * 100.0,
        "idio_vol":      idio_vol_annual   * 100.0,
        "factor_share":  pr.factor_share,
        "RC_vol":        total_vol_annual  * 100.0,
        "RC_pct":        100.0,
        "total_var":     total_var_annual  * 100.0 ** 2,
        "factor_var":    factor_var_annual * 100.0 ** 2,
        "idio_var":      idio_var_annual   * 100.0 ** 2,
        "confidence":    confidence,
        "horizon_days":  horizon_days,
        "sigma_daily":   sigma_daily_dec   * 100.0,
        "sigma_horizon": sigma_horizon_dec * 100.0,
        "VaR_pct":       var_pct_dec       * 100.0,
    }
    if portfolio_mv is not None:
        portfolio_row["VaR_value"] = var_pct_dec * portfolio_mv

    index = [t for t, _ in rows] + ["portfolio"]
    data  = [r for _, r in rows] + [portfolio_row]
    return pd.DataFrame(data, index=index)


def update_model_daily(cal, date):
    """
    8. Incremental daily update (live use-case).
    Corresponds to running calibrator for a single day.
    """
    return cal.run(start=date, end=date)


