import unittest
import datetime
import warnings
import logging
import sys
import pandas as pd
import numpy as np

# Configure logging to print to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    stream=sys.stdout
)

# Import the risk wrapper functions
from oport.functions.functions_risk import (
    initialize_risk_model,
    get_calibration_summary,
    get_factor_returns,
    get_cumulative_factor_returns,
    get_factor_significance,
    get_factor_covariance,
    get_factor_correlation,
    get_all_stocks_risk,
    get_single_stock_risk,
    get_portfolio_risk,
    get_portfolio_factor_contributions,
    get_security_risk_contributions,
    update_model_daily
)

MY_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM",
    "JNJ",  "V",    "PG",    "XOM",  "UNH",  "HD",   "MA",   "BAC",
    "ABBV", "PFE",  "AVGO",  "COST", "WMT",  "MRK",  "LLY",  "CVX",
    "ACN",  "ORCL", "TMO",   "MCD",  "CSCO", "ABT",
]

TRAIN_START = datetime.date(2022, 1, 1)
TRAIN_END   = datetime.date(2024, 12, 31)

class TestRiskEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Suppress DeprecationWarnings
        warnings.simplefilter("ignore", DeprecationWarning)
        
        # Initialize the model once for all tests to save time on data loading.
        # We mimic step 1 of main.py.
        print("Initializing risk model...")
        cls.cal, cls.model = initialize_risk_model(
            tickers=MY_UNIVERSE, 
            start_date=TRAIN_START, 
            end_date=TRAIN_END
        )

    def test_initialize_risk_model(self):
        """Test that model initializes successfully with history."""
        self.assertIsNotNone(self.cal)
        self.assertIsNotNone(self.model)
        self.assertTrue(self.model.n_days_calibrated > 0)

    def test_get_calibration_summary(self):
        """Test Output 1 logic: calibration summary."""
        summary_df = get_calibration_summary(self.model)
        self.assertIsInstance(summary_df, pd.DataFrame)
        self.assertFalse(summary_df.empty)
        self.assertIn("mean_ann_%", summary_df.columns)
        self.assertIn("sharpe", summary_df.columns)

    def test_get_factor_returns(self):
        """Test Output 3 logic: Factor returns time series."""
        returns_df = get_factor_returns(self.model, last_n_days=5)
        self.assertIsInstance(returns_df, pd.DataFrame)
        self.assertFalse(returns_df.empty)
        self.assertLessEqual(len(returns_df), 5)

    def test_get_cumulative_factor_returns(self):
        """Test Output 3b logic: Cumulative factor performance."""
        cum_returns_df = get_cumulative_factor_returns(self.model)
        self.assertIsInstance(cum_returns_df, pd.DataFrame)
        self.assertFalse(cum_returns_df.empty)
        self.assertIn("cumulative_return", cum_returns_df.columns)

    def test_get_factor_significance(self):
        """Test Output 4 logic: Factor t-stats."""
        sig_df = get_factor_significance(self.model)
        self.assertIsInstance(sig_df, pd.DataFrame)
        self.assertFalse(sig_df.empty)
        self.assertIn("avg_|t|", sig_df.columns)
        self.assertIn("pct_sig_%", sig_df.columns)

    def test_get_factor_covariance(self):
        """Test Output 5 logic: Factor covariance matrix."""
        cov_df = get_factor_covariance(self.model)
        self.assertIsInstance(cov_df, pd.DataFrame)
        self.assertFalse(cov_df.empty)
        self.assertEqual(cov_df.shape[0], cov_df.shape[1])

    def test_get_factor_correlation(self):
        """Test Output 5b logic: Factor correlation matrix."""
        corr_df = get_factor_correlation(self.model, annualize=True)
        self.assertIsInstance(corr_df, pd.DataFrame)
        self.assertFalse(corr_df.empty)
        self.assertEqual(corr_df.shape[0], corr_df.shape[1])

    def test_get_all_stocks_risk(self):
        """Test Output 6 logic: Stock-level risk decomposition for all stocks."""
        risk_df = get_all_stocks_risk(self.cal, scale=252)
        self.assertIsInstance(risk_df, pd.DataFrame)
        self.assertFalse(risk_df.empty)
        self.assertIn("total_vol", risk_df.columns)
        self.assertIn("factor_vol", risk_df.columns)

    def test_get_single_stock_risk(self):
        """Test Output 6b logic: Single stock drill-down risk."""
        stock_df = get_single_stock_risk(self.cal, "AAPL")
        self.assertIsInstance(stock_df, pd.DataFrame)
        self.assertFalse(stock_df.empty)
        self.assertIn("AAPL", stock_df.index)
        self.assertIn("total_vol", stock_df.columns)

    def test_get_portfolio_risk(self):
        """Test Output 7 logic: Portfolio risk decomposition."""
        ew_weights = pd.Series(1.0 / len(MY_UNIVERSE), index=MY_UNIVERSE)
        
        port_risk_df = get_portfolio_risk(self.cal, ew_weights)
        self.assertIsInstance(port_risk_df, pd.DataFrame)
        self.assertFalse(port_risk_df.empty)
        self.assertIn("portfolio", port_risk_df.index)
        self.assertIn("total_vol", port_risk_df.columns)
        
        port_contrib_df = get_portfolio_factor_contributions(self.cal, ew_weights)
        self.assertIsInstance(port_contrib_df, pd.DataFrame)
        self.assertFalse(port_contrib_df.empty)
        self.assertIn("contribution", port_contrib_df.columns)

    def test_get_security_risk_contributions(self):
        """Euler risk-contribution decomposition: structure, invariants, and portfolio-row consistency."""
        ew_weights = pd.Series(1.0 / len(MY_UNIVERSE) * 100.0, index=MY_UNIVERSE)

        tree_df = get_security_risk_contributions(self.cal, ew_weights)
        self.assertIsInstance(tree_df, pd.DataFrame)
        self.assertFalse(tree_df.empty)

        # Portfolio summary row must be present
        self.assertIn("portfolio", tree_df.index)

        # Required columns present on all rows
        for col in ("weight", "total_vol", "factor_vol", "idio_vol", "factor_share", "RC_vol", "RC_pct"):
            self.assertIn(col, tree_df.columns)

        # Security rows: standalone vols must be positive
        security_rows = tree_df.drop(index="portfolio")
        self.assertTrue((security_rows["total_vol"] > 0).all(), "All standalone total_vols should be positive")
        self.assertTrue((security_rows["factor_vol"] >= 0).all())
        self.assertTrue((security_rows["idio_vol"] >= 0).all())

        # Euler invariant: RC_pct sums to ~100 across security rows
        rc_pct_sum = security_rows["RC_pct"].sum()
        self.assertAlmostEqual(rc_pct_sum, 100.0, places=3, msg="RC_pct should sum to 100")

        # Euler invariant: RC_vol sums to portfolio total_vol
        portfolio_total_vol = tree_df.loc["portfolio", "total_vol"]
        rc_vol_sum = security_rows["RC_vol"].sum()
        self.assertAlmostEqual(rc_vol_sum, portfolio_total_vol, places=3,
                               msg="RC_vol should sum to portfolio total_vol")

        # Portfolio row total_vol matches get_portfolio_risk output
        port_risk_df = get_portfolio_risk(self.cal, ew_weights)
        self.assertAlmostEqual(
            tree_df.loc["portfolio", "total_vol"],
            port_risk_df.loc["portfolio", "total_vol"],
            places=3,
            msg="Tree-risk portfolio total_vol must match get_portfolio_risk",
        )
        self.assertAlmostEqual(
            tree_df.loc["portfolio", "VaR_pct"],
            port_risk_df.loc["portfolio", "VaR_pct"],
            places=3,
            msg="Tree-risk portfolio VaR_pct must match get_portfolio_risk",
        )

        # VaR_value included when portfolio_mv supplied
        tree_mv = get_security_risk_contributions(self.cal, ew_weights, portfolio_mv=1_000_000)
        self.assertIn("VaR_value", tree_mv.columns)
        self.assertFalse(pd.isna(tree_mv.loc["portfolio", "VaR_value"]))

        print("Security risk contributions (top 5 by RC_pct):")
        print(security_rows.nlargest(5, "RC_pct")[["weight", "total_vol", "RC_vol", "RC_pct"]])

    def test_update_model_daily(self):
        """Test Output 8 logic: Incremental daily update on existing calibrated model."""
        # Re-run for the last train date to test the API safely within unit tests
        updated_model = update_model_daily(self.cal, date=TRAIN_END)
        self.assertIsNotNone(updated_model)

    def test_no_lookahead_leakage_in_regression_exposures(self):
        """
        A1 regression test: the exposure matrix used in day t's cross-
        sectional regression must be built as of t-1. Perturbing t's price
        for one stock must NOT change that exposure matrix -- price-based
        factors (short_rev, size, value_ep/bp, liquidity, low_vol) all read
        price/volume history up to and including their target_date, so
        building them at t would put the day's own return inside the
        regressors.
        """
        from oport.functions.risk.risk_engine.factors import FactorBuilder

        cal = self.cal
        dates = cal._prices.index
        date = dates[len(dates) // 2]
        prev_date = dates[dates < date][-1]

        tickers = cal._prices.columns[cal._prices.loc[date].notna()].tolist()
        builder = FactorBuilder(cal.cfg, cal._prices, cal._volumes)

        exposures_before = builder.build(tickers=tickers, target_date=prev_date)

        perturbed_prices = cal._prices.copy()
        shocked_ticker = tickers[0]
        perturbed_prices.loc[date, shocked_ticker] *= 1.25  # +25% shock ON day t only
        builder_perturbed = FactorBuilder(cal.cfg, perturbed_prices, cal._volumes)

        # Exposures built as of t-1 (the fixed behaviour) must be untouched
        # by a shock injected at t.
        exposures_after = builder_perturbed.build(tickers=tickers, target_date=prev_date)
        pd.testing.assert_frame_equal(exposures_before, exposures_after)

        # Positive control: exposures built as of t itself (the old, leaky
        # target_date) DO pick up the shock -- confirms this test would
        # have caught the original bug.
        exposures_date_before = builder.build(tickers=tickers, target_date=date)
        exposures_date_after = builder_perturbed.build(tickers=tickers, target_date=date)
        self.assertFalse(exposures_date_before.equals(exposures_date_after))


class TestFundamentalsPointInTime(unittest.TestCase):
    """A2/A3/A4: fundamentals must not leak unpublished filings, must be
    filtered identically regardless of call order, and TTM sums from
    partial history must not be understated."""

    def test_ttm_scales_partial_quarters(self):
        from oport.functions.risk.risk_engine.data import get_ttm

        df4 = pd.DataFrame({"NI": [10.0, 10.0, 10.0, 10.0]})
        self.assertAlmostEqual(get_ttm(df4, "NI"), 40.0)

        # Only 2 of 4 quarters available: summing as-is would understate
        # TTM by half. Scale to a 4-quarter-equivalent instead.
        df2 = pd.DataFrame({"NI": [10.0, 10.0]})
        self.assertAlmostEqual(get_ttm(df2, "NI"), 40.0)

        # A single quarter is too little to extrapolate from.
        df1 = pd.DataFrame({"NI": [10.0]})
        self.assertIsNone(get_ttm(df1, "NI"))

    def test_fundamentals_cache_keyed_by_ticker_not_quarter(self):
        """A3: the cache holds RAW frames keyed by ticker alone. A
        (ticker, quarter) key would lock in whichever call's target_date
        filter ran first for the rest of that quarter."""
        from oport.functions.risk.risk_engine.data import (
            get_quarterly_fundamentals, financial_cache,
        )
        ticker = "AAPL"
        get_quarterly_fundamentals(ticker, datetime.date.today())
        self.assertIn(ticker, financial_cache)
        self.assertNotIsInstance(next(iter(financial_cache.keys())), tuple)

    def test_point_in_time_filtering_is_call_order_independent(self):
        """A2/A3: a target_date strictly between two known quarter-ends
        must only see the earlier one; a big report_lag must push a
        quarter back out of view even after its period-end date has
        passed. Dates are derived from AAPL's OWN live quarters so this
        doesn't hardcode calendar assumptions about what yfinance returns."""
        from oport.functions.risk.risk_engine.data import (
            get_quarterly_fundamentals, financial_cache,
        )
        ticker = "AAPL"
        get_quarterly_fundamentals(ticker, datetime.date.today(), report_lag_days=0)
        raw = financial_cache.get(ticker)
        if raw is None or len(raw) < 2:
            self.skipTest("Not enough live AAPL quarterly history to run this check.")

        quarter_ends = sorted(raw.index)
        early_q, late_q = quarter_ends[-2], quarter_ends[-1]
        mid_date = early_q + (late_q - early_q) / 2

        inc_mid, _, _ = get_quarterly_fundamentals(ticker, mid_date, report_lag_days=0)
        self.assertIn(early_q, inc_mid.index)
        self.assertNotIn(late_q, inc_mid.index)

        after_late = late_q + datetime.timedelta(days=10)
        inc_lagged, _, _ = get_quarterly_fundamentals(ticker, after_late, report_lag_days=90)
        self.assertNotIn(late_q, inc_lagged.index,
                          "A quarter should not be visible before it clears report_lag_days.")

        inc_no_lag, _, _ = get_quarterly_fundamentals(ticker, after_late, report_lag_days=0)
        self.assertIn(late_q, inc_no_lag.index)


class TestNegativeBookEquityGuard(unittest.TestCase):
    """LBO-style balance sheets (e.g. DELL, whose buyback/debt history left
    common equity around -$1.4B..-$2.8B) carry structurally negative book
    equity. net_income / negative_book_equity flips ROE's sign into
    something that reads as a loss-maker for a profitable company -- an
    artifact, not a real quality signal. Guard must treat negative book
    equity as missing (NaN) for ROE, the same convention Fama-French use
    when excluding negative-book-equity names from HML/ROE construction."""

    def test_negative_book_equity_yields_nan_roe_not_a_sign_flip(self):
        import oport.functions.risk.risk_engine.factors as factors_mod
        from oport.functions.risk.risk_engine.config import ModelConfig

        idx = pd.bdate_range("2024-01-01", periods=260)
        prices = pd.DataFrame({"DELL": np.linspace(50.0, 60.0, len(idx))}, index=idx)
        volumes = pd.DataFrame({"DELL": [1_000_000] * len(idx)}, index=idx)

        income_df = pd.DataFrame(
            {"Net Income": [1e9, 1e9, 1e9, 1e9]},
            index=[datetime.date(2024, 1, 1) - datetime.timedelta(days=90 * i) for i in range(4)],
        )
        bs_df = pd.DataFrame(
            {"Common Stock Equity": [-2e9]},
            index=[datetime.date(2024, 1, 1)],
        )

        orig_fund   = factors_mod.get_quarterly_fundamentals
        orig_shares = factors_mod.get_shares_outstanding
        factors_mod.get_quarterly_fundamentals = lambda *a, **k: (income_df, None, bs_df)
        factors_mod.get_shares_outstanding = lambda ticker: 1e8  # -> positive mkt_cap
        try:
            builder = factors_mod.FactorBuilder(ModelConfig(), prices, volumes)
            row = builder._compute_row("DELL", idx[-1], prices)
        finally:
            factors_mod.get_quarterly_fundamentals = orig_fund
            factors_mod.get_shares_outstanding = orig_shares

        self.assertIsNotNone(row)
        self.assertTrue(
            np.isnan(row["quality_roe"]),
            "Negative book equity must yield NaN quality_roe, not a sign-flipped ratio "
            f"(got {row['quality_roe']})",
        )
        # value_ep (net_income / mkt_cap) doesn't divide by book equity, so
        # a profitable company must still show a positive earnings yield.
        self.assertGreater(row["value_ep"], 0.0)
        # value_bp is left as a real (if extreme) negative book yield --
        # winsorization handles that at the cross-sectional step, unlike
        # quality_roe's sign-flip which is uninterpretable at any scale.
        self.assertLess(row["value_bp"], 0.0)


class TestFactorSummarySanityBounds(unittest.TestCase):
    """D2: FactorRiskModel.summary() must warn on implausible factors
    (|Sharpe| > 4 or avg |t| > 1.5) rather than reporting them silently --
    exactly the numbers a look-ahead leak like A1 would have produced."""

    def test_implausible_sharpe_triggers_warning(self):
        from oport.functions.risk.risk_engine.model import FactorRiskModel

        stats = pd.DataFrame({
            "sharpe": [-12.5, 0.8],
            "avg_t_stat": [0.9, 0.5],
        }, index=["short_rev", "momentum"])
        with self.assertLogs("oport.functions.risk.risk_engine.model", level="WARNING") as cm:
            FactorRiskModel._warn_on_implausible_factors(stats)
        self.assertTrue(any("short_rev" in msg for msg in cm.output))

    def test_plausible_factors_do_not_warn(self):
        from oport.functions.risk.risk_engine.model import FactorRiskModel

        records = []
        handler = logging.Handler()
        handler.emit = records.append
        logger_ = logging.getLogger("oport.functions.risk.risk_engine.model")
        logger_.addHandler(handler)
        try:
            stats = pd.DataFrame({
                "sharpe": [0.5, -0.3],
                "avg_t_stat": [1.0, 0.8],
            }, index=["momentum", "value_ep"])
            FactorRiskModel._warn_on_implausible_factors(stats)
        finally:
            logger_.removeHandler(handler)
        self.assertEqual(len(records), 0)


if __name__ == "__main__":
    import os
    current_directory = os.getcwd()
    print("Current working dir: {}".format(current_directory))
    unittest.main()
