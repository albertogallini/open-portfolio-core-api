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

    def test_update_model_daily(self):
        """Test Output 8 logic: Incremental daily update on existing calibrated model."""
        # Re-run for the last train date to test the API safely within unit tests
        updated_model = update_model_daily(self.cal, date=TRAIN_END)
        self.assertIsNotNone(updated_model)

if __name__ == "__main__":
    import os
    current_directory = os.getcwd()
    print("Current working dir: {}".format(current_directory))
    unittest.main()
