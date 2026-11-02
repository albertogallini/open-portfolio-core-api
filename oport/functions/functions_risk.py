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
from .linking import carino_linking_return_split
from oport import fields


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


def get_portfolio_factor_attribution(
    cal,
    df_h_agg: pd.DataFrame,
    port_series: Optional[dict] = None,
    start_date: Optional[datetime.date] = None,
    end_date: Optional[datetime.date] = None,
    include_factor_returns: bool = False,
) -> pd.DataFrame:
    """
    Historical factor-based return attribution for a portfolio.

    For each common trading date, computes:
      - Portfolio factor exposure: beta_{p, k} = sum_i (w_i * X_{i, k})
      - Factor return component: c_{k, t} = beta_{p, k} * f_{k, t}
      - Total factor return: sum_k c_{k, t}
      - Portfolio holdings return: R_{hlds, t}
      - Excess return: excess_return_t = R_{hlds, t} - sum_k c_{k, t}

    And links each component over time using Cariño linking (acc_ prefix):
      - acc_<factor>: Cariño linked factor return component
      - acc_factor_return: sum of Cariño linked factor return components
      - acc_return_hlds: compounded portfolio holdings return (R - 1)
      - acc_excess_return: Cariño linked excess return component (acc_return_hlds - acc_factor_return)

    Invariants satisfied on every single date:
      1. sum_k (factor_k) + excess_return == return_hlds
      2. sum_k (acc_factor_k) + acc_excess_return == acc_return_hlds
    """
    model = getattr(cal, "model", cal)
    results = getattr(model, "_results", [])
    if not results:
        raise ValueError("The risk model contains no calibrated daily results.")

    def _to_date(d):
        if isinstance(d, datetime.date):
            return d
        return pd.to_datetime(d).date()

    results_by_date = {_to_date(r.date): r for r in results}

    if fields.FIELD_DATE not in df_h_agg.columns:
        raise ValueError(f"Expected '{fields.FIELD_DATE}' column in portfolio data.")

    port_dates_unique = sorted({_to_date(d) for d in df_h_agg[fields.FIELD_DATE].unique()})

    common_dates = [d for d in port_dates_unique if d in results_by_date]
    if start_date is not None:
        dt_s = _to_date(start_date)
        common_dates = [d for d in common_dates if d >= dt_s]
    if end_date is not None:
        dt_e = _to_date(end_date)
        common_dates = [d for d in common_dates if d <= dt_e]

    if not common_dates:
        raise ValueError("No overlapping dates found between portfolio holdings and risk model calibration.")

    port_ret_map = {}
    if port_series and "date" in port_series and "return_hlds" in port_series:
        for p_dt, p_r in zip(port_series["date"], port_series["return_hlds"]):
            port_ret_map[_to_date(p_dt)] = p_r

    sample_res = results_by_date[common_dates[0]]
    factor_names = sample_res.factor_returns.index.tolist()

    rows = []
    for idx, dt in enumerate(common_dates):
        res = results_by_date[dt]
        f_returns = res.factor_returns
        X = res.exposures

        day_hlds = df_h_agg[df_h_agg[fields.FIELD_DATE].apply(_to_date) == dt]

        w_series = None
        if fields.FIELD_EVAL_POSITION_BASIS_WEIGHTS in day_hlds.columns:
            w_series = day_hlds.groupby(fields.FIELD_TICKER)[fields.FIELD_EVAL_POSITION_BASIS_WEIGHTS].sum()
            if w_series.isna().all() or w_series.sum() == 0:
                w_series = None

        if w_series is None and fields.FIELD_EVAL_MKTVALUE_WEIGHTS_HLDS in day_hlds.columns:
            w_series = day_hlds.groupby(fields.FIELD_TICKER)[fields.FIELD_EVAL_MKTVALUE_WEIGHTS_HLDS].sum() / 100.0

        if w_series is None:
            tickers = day_hlds[fields.FIELD_TICKER].unique()
            w_series = pd.Series(1.0 / len(tickers) if len(tickers) > 0 else 1.0, index=tickers)

        # Check portfolio return
        r_val = port_ret_map.get(dt, None)
        if r_val is None or pd.isna(r_val):
            if fields.FIELD_EVAL_RETURN_HLDS in day_hlds.columns:
                r_hlds_vals = day_hlds[fields.FIELD_EVAL_RETURN_HLDS]
                if r_hlds_vals.isna().all():
                    r_val = 1.0
                else:
                    if (
                        fields.FIELD_EVAL_POSITION_BASIS_WEIGHTS in day_hlds.columns
                        and not day_hlds[fields.FIELD_EVAL_POSITION_BASIS_WEIGHTS].isna().all()
                    ):
                        w_vals = day_hlds[fields.FIELD_EVAL_POSITION_BASIS_WEIGHTS]
                    else:
                        w_vals = day_hlds[fields.FIELD_EVAL_MKTVALUE_WEIGHTS_HLDS] / 100.0
                    w_sum = w_vals.sum()
                    if w_sum > 0:
                        r_val = float(((r_hlds_vals - 1.0) * w_vals).sum() / w_sum + 1.0)
                    else:
                        r_val = 1.0
            else:
                r_val = 1.0

        # Inception / first day has no return
        is_inception = (idx == 0) and (pd.isna(r_val) or r_val == 1.0)

        if is_inception:
            hlds_ret = 0.0
            factor_comps = np.zeros(len(factor_names))
            tot_factor_ret = 0.0
            excess_ret = 0.0
        else:
            hlds_ret = float(r_val - 1.0)
            covered = w_series.index.intersection(X.index)
            if not covered.empty:
                w_vec = w_series.loc[covered].values.astype(float)
                X_mat = X.loc[covered, factor_names].values.astype(float)
                beta_p = X_mat.T @ w_vec
            else:
                beta_p = np.zeros(len(factor_names))

            factor_comps = beta_p * f_returns.loc[factor_names].values.astype(float)
            tot_factor_ret = float(np.sum(factor_comps))
            excess_ret = hlds_ret - tot_factor_ret

        row = {
            "date": str(dt),
            "return_hlds": hlds_ret,
            "holdings_return": hlds_ret,
            "factor_return": tot_factor_ret,
            "excess_return": excess_ret,
        }
        for k_idx, f_name in enumerate(factor_names):
            row[f_name] = float(factor_comps[k_idx])
            if include_factor_returns:
                row[f_name + "_return"] = float(f_returns.loc[f_name])
        rows.append(row)

    df_attr = pd.DataFrame(rows)

    # Multi-period Cariño linking
    n = len(df_attr)
    slice_r_all = 1.0 + df_attr["return_hlds"].values

    acc_dict = {
        col: np.zeros(n)
        for col in factor_names + ["return_hlds", "holdings_return", "factor_return", "excess_return"]
    }

    for i in range(1, n):
        slice_r = slice_r_all[: i + 1]
        R_i = float(np.prod(slice_r))
        acc_dict["return_hlds"][i] = R_i - 1.0
        acc_dict["holdings_return"][i] = R_i - 1.0

        for f in factor_names:
            c_slice = 1.0 + df_attr[f].values[: i + 1]
            acc_dict[f][i] = carino_linking_return_split(c_slice, slice_r, R_i) - 1.0

        c_tot = 1.0 + df_attr["factor_return"].values[: i + 1]
        acc_dict["factor_return"][i] = carino_linking_return_split(c_tot, slice_r, R_i) - 1.0

        c_exc = 1.0 + df_attr["excess_return"].values[: i + 1]
        acc_dict["excess_return"][i] = carino_linking_return_split(c_exc, slice_r, R_i) - 1.0

    for col in acc_dict:
        df_attr["acc_" + col] = acc_dict[col]

    df_attr.set_index("date", drop=False, inplace=True)
    return df_attr


get_factor_attribution = get_portfolio_factor_attribution



