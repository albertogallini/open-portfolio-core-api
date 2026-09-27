"""
oport_api_regime_additions.py
==============================
Paste-in additions for oport_api.py to add regime-aware portfolio
construction alongside the existing RISK_CALIBRATION / FACTOR_COVARIANCE
/ PORTFOLIO_RISK endpoints. Follows the exact same conventions already
used in that file:

  - one module-level constant + ApiNames entry per operation
  - a `<thing>_logic(...)` function with the same param signature shape
    (source / filesystem_folder / bucket_name / access_key_id /
    secret_access_key / s3_host / s3_port / ftp_host / ftp_user /
    ftp_pass, plus operation-specific args)
  - const_and_utils.Config(...) + config.validate()
  - pickled model persistence via const_and_utils.write_pickle /
    read_pickle, alongside the existing RISK_MODEL_FILE
  - errors funneled through error_collector.get_collector()
  - a bare DataFrame return (None on error, matching every other
    *_logic function in this file)

--------------------------------------------------------------------
1) Add to the imports block:
--------------------------------------------------------------------

from .functions import (
    functions_characteristics,
    functions_pattribution,
    functions_performance,
    functions_performance_daily,
    functions_positions,
    functions_aggregation,
    functions_tree,
    functions_risk,
    functions_regime,          # <-- NEW
)

--------------------------------------------------------------------
2) Add to the operation-name constants block:
--------------------------------------------------------------------

REGIME_CALIBRATION = "REGIME_CALIBRATION"
REGIME_SUMMARY = "REGIME_SUMMARY"
PORTFOLIO_CONSTRUCTION = "PORTFOLIO_CONSTRUCTION"

# API Names
ApiNames = {
    ...
    RISK_CALIBRATION: "RISK CALIBRATION",
    FACTOR_COVARIANCE: "FACTOR COVARIANCE",
    PORTFOLIO_RISK: "PORTFOLIO RISK",
    REGIME_CALIBRATION: "REGIME CALIBRATION",          # <-- NEW
    REGIME_SUMMARY: "REGIME SUMMARY",                   # <-- NEW
    PORTFOLIO_CONSTRUCTION: "PORTFOLIO CONSTRUCTION",   # <-- NEW
}

RISK_MODEL_FILE = "risk_model.pkl"
REGIME_MODEL_FILE = "regime_model.pkl"        # <-- NEW

--------------------------------------------------------------------
3) Add the three logic functions below anywhere after
   get_portfolio_risk_logic (they depend on RISK_MODEL_FILE existing).
--------------------------------------------------------------------
"""

from typing import Optional
import os
import pandas as pd

# (these names already exist in oport_api.py's own namespace: const_and_utils,
#  fields, error_collector, functions_risk, functions_performance_daily,
#  functions_aggregation, ApiNames, RISK_MODEL_FILE, REGIME_MODEL_FILE,
#  REGIME_CALIBRATION, REGIME_SUMMARY, PORTFOLIO_CONSTRUCTION)


def regime_calibration_logic(
    source: str,
    filesystem_folder: Optional[str],
    bucket_name: Optional[str],
    access_key_id: Optional[str],
    secret_access_key: Optional[str],
    s3_host: Optional[str],
    s3_port: Optional[str],
    ftp_host: Optional[str],
    ftp_user: Optional[str],
    ftp_pass: Optional[str],
    start_date: str = "",
    end_date: str = "",
    min_regimes: int = 2,
    max_regimes: int = 4,
    estimation_window: Optional[int] = None,
    refit_frequency: int = 21,
) -> pd.DataFrame:
    """
    Calibrate the Wasserstein-HMM regime model off the factor-return
    time series of an ALREADY-CALIBRATED risk model (run
    risk_calibration_logic first -- this reads the same RISK_MODEL_FILE
    pickle that risk_calibration_logic writes).

    estimation_window: leave as None (the default) to derive a value that
    actually satisfies min_params_multiplier for the real factor count and
    max_regimes -- a fixed 252 default silently violated that guard for
    any realistic (~9-dimensional) continuous factor set, since the guard
    only depends on estimation_window/max_regimes/min_params_multiplier,
    not on how much history is loaded.
    """
    err_collector = error_collector.get_collector()

    if filesystem_folder is not None:
        filesystem_folder = filesystem_folder.replace("|", "/")
    else:
        filesystem_folder = ""

    try:
        from datetime import datetime

        config = const_and_utils.Config(
            source=source,
            filesystem_folder=filesystem_folder,
            output_file_name="",  # not used for loading the common model
            bucket_name=bucket_name,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            s3_host=s3_host,
            s3_port=s3_port,
            ftp_host=ftp_host,
            ftp_user=ftp_user,
            ftp_pass=ftp_pass,
        )
        config.validate()

        # 1. Load the already-calibrated risk model
        risk_model_path = (
            filesystem_folder + "/" + RISK_MODEL_FILE if filesystem_folder else RISK_MODEL_FILE
        )
        risk_cal = const_and_utils.read_pickle(config, risk_model_path)
        if risk_cal is None:
            raise ValueError(
                "No calibrated risk model found. Please run risk_calibration first."
            )

        # 2. Pull its factor-return history and optionally window it
        fr = functions_risk.get_factor_returns(risk_cal.model)
        if len(start_date) != 0:
            try:
                dt_start = datetime.strptime(start_date, "%m-%d-%Y")
            except ValueError:
                dt_start = datetime.strptime(start_date, "%Y-%m-%d")
            fr = fr[fr.index >= dt_start]
        if len(end_date) != 0:
            try:
                dt_end = datetime.strptime(end_date, "%m-%d-%Y")
            except ValueError:
                dt_end = datetime.strptime(end_date, "%Y-%m-%d")
            fr = fr[fr.index <= dt_end]

        # 3. Fit the regime model. estimation_window is only forwarded when
        #    the caller explicitly set it -- otherwise initialize_regime_model
        #    derives a value self-consistent with min_params_multiplier/
        #    max_regimes for the real factor count (see its docstring).
        regime_kwargs = dict(
            min_regimes=min_regimes,
            max_regimes=max_regimes,
            refit_frequency=refit_frequency,
        )
        if estimation_window is not None:
            regime_kwargs["estimation_window"] = estimation_window
        regime_cal, regime_cfg = functions_regime.initialize_regime_model(
            factor_returns=fr,
            **regime_kwargs,
        )

        # 4. Persist it, same pattern as the risk model pickle
        regime_model_path = (
            filesystem_folder + "/" + REGIME_MODEL_FILE if filesystem_folder else REGIME_MODEL_FILE
        )
        const_and_utils.write_pickle(config, regime_model_path, regime_cal)

        print(regime_cal.report())

        # 5. Return summary
        return functions_regime.get_regime_summary(regime_cal)

    except Exception as err_msg:
        err_collector.record_error(
            operation_name=ApiNames[REGIME_CALIBRATION], error_details=str(err_msg)
        )

    return None


def get_regime_summary_logic(
    source: str,
    filesystem_folder: Optional[str],
    bucket_name: Optional[str],
    access_key_id: Optional[str],
    secret_access_key: Optional[str],
    s3_host: Optional[str],
    s3_port: Optional[str],
    ftp_host: Optional[str],
    ftp_user: Optional[str],
    ftp_pass: Optional[str],
) -> pd.DataFrame:
    """Current regime snapshot + probabilities/transition matrix, all
    stitched into one DataFrame the way get_portfolio_risk does."""
    err_collector = error_collector.get_collector()

    if filesystem_folder is not None:
        filesystem_folder = filesystem_folder.replace("|", "/")
    else:
        filesystem_folder = ""

    try:
        config = const_and_utils.Config(
            source=source,
            filesystem_folder=filesystem_folder,
            output_file_name="",
            bucket_name=bucket_name,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            s3_host=s3_host,
            s3_port=s3_port,
            ftp_host=ftp_host,
            ftp_user=ftp_user,
            ftp_pass=ftp_pass,
        )
        config.validate()

        regime_model_path = (
            filesystem_folder + "/" + REGIME_MODEL_FILE if filesystem_folder else REGIME_MODEL_FILE
        )
        regime_cal = const_and_utils.read_pickle(config, regime_model_path)
        if regime_cal is None:
            raise ValueError(
                "No calibrated regime model found. Please run regime_calibration first."
            )

        current = functions_regime.get_current_regime(regime_cal)
        summary = functions_regime.get_regime_summary(regime_cal)
        return current.join(summary, rsuffix="_summary", how="left")

    except Exception as err_msg:
        err_collector.record_error(
            operation_name=ApiNames[REGIME_SUMMARY], error_details=str(err_msg)
        )

    return None


def construct_portfolio_logic(
    universe_name: str,
    source: str,
    filesystem_folder: Optional[str],
    bucket_name: Optional[str],
    access_key_id: Optional[str],
    secret_access_key: Optional[str],
    s3_host: Optional[str],
    s3_port: Optional[str],
    ftp_host: Optional[str],
    ftp_user: Optional[str],
    ftp_pass: Optional[str],
    input_date: Optional[str] = None,
    target_n: Optional[int] = None,
    max_weight: Optional[float] = None,
    transaction_cost_bps: Optional[float] = None,
    risk_aversion: Optional[float] = None,
) -> pd.DataFrame:
    """
    Regime-aware stock picking over a universe (an index's constituents
    OR an existing portfolio's holdings -- `universe_name` is resolved
    the same way portfolio_name/calibration_universe already are
    elsewhere in this file: via the "<name>_DAILY_HOLDINGS" file(s)).

    Requires both RISK_MODEL_FILE and REGIME_MODEL_FILE to already
    exist (run risk_calibration then regime_calibration first).

    Output columns: weight, prev_weight, trade, expected_return_regime,
    regime_id, regime_confidence, expected_hold_days -- one row per
    selected ticker, indexed by ticker.
    """
    err_collector = error_collector.get_collector()

    if filesystem_folder is not None:
        filesystem_folder = filesystem_folder.replace("|", "/")
    else:
        filesystem_folder = ""

    try:
        from datetime import datetime, timedelta

        config = const_and_utils.Config(
            source=source,
            filesystem_folder=filesystem_folder,
            output_file_name=universe_name,
            bucket_name=bucket_name,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            s3_host=s3_host,
            s3_port=s3_port,
            ftp_host=ftp_host,
            ftp_user=ftp_user,
            ftp_pass=ftp_pass,
        )
        config.validate()

        # 1. Load the calibrated risk + regime models
        risk_model_path = (
            filesystem_folder + "/" + RISK_MODEL_FILE if filesystem_folder else RISK_MODEL_FILE
        )
        risk_cal = const_and_utils.read_pickle(config, risk_model_path)
        if risk_cal is None:
            raise ValueError("No calibrated risk model found. Please run risk_calibration first.")

        regime_model_path = (
            filesystem_folder + "/" + REGIME_MODEL_FILE if filesystem_folder else REGIME_MODEL_FILE
        )
        regime_cal = const_and_utils.read_pickle(config, regime_model_path)
        if regime_cal is None:
            raise ValueError("No calibrated regime model found. Please run regime_calibration first.")

        # 2. Resolve the universe + current weights (same holdings-file
        #    pattern as risk_calibration_logic / get_portfolio_risk_logic).
        #    `universe_name` can itself be an index (e.g. an S&P/Nasdaq
        #    universe carries its own market-cap weights in its holdings
        #    file) -- so "current weights" here doubles as "start the
        #    optimizer from the index" when there is no separate portfolio.
        #    Only when NO holdings snapshot exists at all (anywhere at or
        #    before t_date) does this fall back to a 100%-cash start.
        universe_prefix = universe_name + "_" + const_and_utils.DAILY_HOLDINGS_PREFIX
        holdings_files = const_and_utils.list_files(
            config=config, input_folder=filesystem_folder, prefix=universe_prefix,
        )

        current_weights = None
        universe_tickers = []
        if holdings_files:
            if input_date:
                t_date = pd.to_datetime(input_date).date()
            else:
                t_date = datetime.now().date()
            # A tight +/-10-day window silently produced a cash start
            # whenever the run date didn't line up with a recent holdings
            # snapshot (e.g. running against an index whose file is only
            # refreshed periodically). Look back far enough to find the
            # most recent snapshot ON OR BEFORE t_date instead -- that IS
            # the index's (or portfolio's) current weights as of t_date --
            # and only treat it as missing if there is truly nothing.
            t_date_start = t_date - timedelta(days=400)

            df_perf = functions_performance_daily.compute_performance_daily(
                config=config,
                prefix=universe_prefix,
                start_date=t_date_start,
                end_date=t_date,
                holdings_files=holdings_files,
            )
            df_valid = df_perf.dropna(subset=["quantity", "Close"])
            valid_dates = [d for d in df_valid[fields.FIELD_DATE].unique() if d <= t_date]
            if valid_dates:
                snapshot_date = max(valid_dates)
                staleness_days = (t_date - snapshot_date).days
                if staleness_days > 10:
                    print(
                        "Warning: construct_portfolio_logic: no {} holdings snapshot "
                        "within 10 days of {} -- using the most recent one available, "
                        "from {} ({} days stale).".format(
                            universe_name, t_date, snapshot_date, staleness_days
                        )
                    )
                df_agg, _, _ = functions_aggregation.aggregate(
                    portfolio_daily_fields=df_perf, start_date=snapshot_date, end_date=snapshot_date,
                )
                weight_col = fields.FIELD_EVAL_MKTVALUE_WEIGHTS_HLDS
                if not df_agg.empty and weight_col in df_agg.columns:
                    current_weights = df_agg.groupby(fields.FIELD_TICKER)[weight_col].sum() / 100.0
                    universe_tickers = list(current_weights.index)

        if not universe_tickers:
            # Truly no holdings/index snapshot on file at all -- fall back
            # to every ticker the risk model was calibrated on, starting
            # from cash (full entry cost). This is now the genuine
            # last-resort path, not the common case.
            print(
                "Warning: construct_portfolio_logic: no {} holdings snapshot found "
                "on or before {} -- starting from 100% cash (turnover = 1.0) over "
                "the full risk-model universe instead of the index/portfolio's own "
                "weights.".format(
                    universe_name, t_date if holdings_files else input_date
                )
            )
            exposures = functions_regime.get_latest_exposures(risk_cal)
            universe_tickers = list(exposures.index)

        # 3. Construct the regime-conditioned portfolio
        result = functions_regime.construct_regime_portfolio(
            risk_cal=risk_cal,
            regime_cal=regime_cal,
            universe=universe_tickers,
            current_weights=current_weights,
            target_n=target_n,
            max_weight=max_weight,
            transaction_cost_bps=transaction_cost_bps,
            risk_aversion=risk_aversion,
        )
        return result

    except Exception as err_msg:
        err_collector.record_error(
            operation_name=ApiNames[PORTFOLIO_CONSTRUCTION], error_details=str(err_msg)
        )

    return None
