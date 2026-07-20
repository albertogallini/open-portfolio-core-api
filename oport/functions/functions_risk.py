"""
functions_risk.py
=================
Wrapper functions for the factor risk model outputs from risk_engine.
"""

import sys
import datetime
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

def update_model_daily(cal, date):
    """
    8. Incremental daily update (live use-case).
    Corresponds to running calibrator for a single day.
    """
    return cal.run(start=date, end=date)


