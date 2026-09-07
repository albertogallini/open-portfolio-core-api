import unittest
import pandas as pd
import os
import shutil
from datetime import datetime
from oport.oport_api import (
    risk_calibration_logic,
    get_factor_covariance_logic,
    get_portfolio_risk_logic,
    get_portfolio_tree_risk_logic,
    RISK_MODEL_FILE
)
from oport import fields, error_collector

class TestRiskApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Base data directory (relative path)
        cls.test_data_dir = "tests/test_attribution_md/test_portfolios"
        cls.universe_dir = os.path.join(cls.test_data_dir, "indexes")
        
        # Test targets
        cls.calibration_universe = "nasdaq_index"
        cls.portfolio_name = "test_e2e_stocksplits"
        
        # Calibration date range (matching stocksplits dataset part)
        cls.start_date = "2021-10-01"
        cls.end_date = "2021-11-05"
        cls.input_date = "2021-11-05"  # Friday
        cls.monday_date = "2021-11-08" # Monday (should fallback to Nov 5)
        cls.source = "filesystem"
        
        # Placeholder for model path
        cls.model_path_universe = os.path.join(cls.universe_dir, RISK_MODEL_FILE)
        cls.model_path_portfolio = os.path.join(cls.test_data_dir, RISK_MODEL_FILE)

    def test_01_calibration(self):
        print("\n=== Testing Risk Calibration ===")
        df_summary = risk_calibration_logic(
            calibration_universe=self.calibration_universe,
            start_date=self.start_date,
            end_date=self.end_date,
            source=self.source,
            filesystem_folder=self.universe_dir,
            bucket_name=None,
            access_key_id=None,
            secret_access_key=None,
            s3_host=None,
            s3_port=None,
            ftp_host=None,
            ftp_user=None,
            ftp_pass=None
        )
        if df_summary is None:
            collector = error_collector.get_collector()
            print("ERRORS found in calibration:", collector.get_errors())
        
        self.assertIsInstance(df_summary, pd.DataFrame)
        self.assertFalse(df_summary.empty, "Calibration summary is empty")
        print("Calibration successful. Summary head:")
        print(df_summary.head())
        self.assertTrue(os.path.exists(self.model_path_universe), "Serialized model not found")

    def test_02_factor_covariance(self):
        print("\n=== Testing Factor Covariance ===")
        # Model was saved in universe_dir during test_01
        df_cov = get_factor_covariance_logic(
            source=self.source,
            filesystem_folder=self.universe_dir,
            bucket_name=None,
            access_key_id=None,
            secret_access_key=None,
            s3_host=None,
            s3_port=None,
            ftp_host=None,
            ftp_user=None,
            ftp_pass=None
        )
        self.assertIsInstance(df_cov, pd.DataFrame)
        self.assertFalse(df_cov.empty, "Factor covariance matrix is empty")
        print(f"Factor covariance matrix shape: {df_cov.shape}")
        print("Columns:", df_cov.columns.tolist())

    def test_03_portfolio_risk(self):
        print("\n=== Testing Portfolio Risk ===")
        # For this test, we need the model and portfolio holdings in the same folder 
        # (as per current API architecture which uses one 'filesystem_folder' for everything).
        # We copy the model from universe_dir (where it was created) to test_data_dir.
        if os.path.exists(self.model_path_universe):
             shutil.copy2(self.model_path_universe, self.model_path_portfolio)
        
        df_risk = get_portfolio_risk_logic(
            portfolio_name=self.portfolio_name,
            source=self.source,
            filesystem_folder=self.test_data_dir,
            bucket_name=None,
            access_key_id=None,
            secret_access_key=None,
            s3_host=None,
            s3_port=None,
            ftp_host=None,
            ftp_user=None,
            ftp_pass=None,
            input_date=self.input_date
        )
        
        self.assertIsInstance(df_risk, pd.DataFrame)
        self.assertFalse(df_risk.empty, "Portfolio risk decomposition is empty")
        print("Portfolio Risk Decomposition head:")
        print(df_risk.head())

    def test_04_portfolio_risk_weekend_fallback(self):
        print("\n=== Testing Portfolio Risk Weekend Fallback (Monday fallback to Friday) ===")
        # Passing Monday Oct 13, 2025. 
        # The logic should calculate performance from Friday Oct 10 to Friday Oct 10 (since t_date=Monday, prev_date=Friday)
        # Wait, if t_date=Monday, prev_date = Monday - 1 day = Sunday -> fallback to Friday.
        # So it computes performance between Friday and Monday? 
        # Actually, in get_portfolio_risk_logic:
        # t_date = Monday
        # prev_date = Sunday -> fallback -> Friday.
        # So it computes performance between Friday and Monday.
        
        df_risk = get_portfolio_risk_logic(
            portfolio_name=self.portfolio_name,
            source=self.source,
            filesystem_folder=self.test_data_dir,
            bucket_name=None,
            access_key_id=None,
            secret_access_key=None,
            s3_host=None,
            s3_port=None,
            ftp_host=None,
            ftp_user=None,
            ftp_pass=None,
            input_date=self.monday_date
        )
        
        self.assertIsInstance(df_risk, pd.DataFrame)
        self.assertFalse(df_risk.empty, "Portfolio risk decomposition (fallback) is empty")
        print("Portfolio Risk Decomposition (Fallback) successful.")

    def test_05_portfolio_tree_risk(self):
        print("\n=== Testing Portfolio Tree Risk (Euler decomposition) ===")
        df_tree = get_portfolio_tree_risk_logic(
            portfolio_name=self.portfolio_name,
            source=self.source,
            filesystem_folder=self.test_data_dir,
            bucket_name=None,
            access_key_id=None,
            secret_access_key=None,
            s3_host=None,
            s3_port=None,
            ftp_host=None,
            ftp_user=None,
            ftp_pass=None,
            input_date=self.input_date,
        )

        if df_tree is None:
            collector = error_collector.get_collector()
            print("ERRORS:", collector.get_errors())

        self.assertIsInstance(df_tree, pd.DataFrame)
        self.assertFalse(df_tree.empty, "Portfolio tree risk is empty")
        self.assertIn("portfolio", df_tree.index)

        # Required columns
        for col in ("weight", "total_vol", "factor_vol", "idio_vol", "RC_vol", "RC_pct", "VaR_pct"):
            self.assertIn(col, df_tree.columns, f"Missing column: {col}")

        security_rows = df_tree.drop(index="portfolio")
        self.assertFalse(security_rows.empty, "No security rows returned")

        # Euler invariants
        self.assertAlmostEqual(security_rows["RC_pct"].sum(), 100.0, places=1,
                               msg="RC_pct must sum to 100")
        portfolio_vol = df_tree.loc["portfolio", "total_vol"]
        self.assertAlmostEqual(security_rows["RC_vol"].sum(), portfolio_vol, places=1,
                               msg="RC_vol must sum to portfolio total_vol")

        # Portfolio row consistent with get_portfolio_risk_logic
        df_risk = get_portfolio_risk_logic(
            portfolio_name=self.portfolio_name,
            source=self.source,
            filesystem_folder=self.test_data_dir,
            bucket_name=None,
            access_key_id=None,
            secret_access_key=None,
            s3_host=None,
            s3_port=None,
            ftp_host=None,
            ftp_user=None,
            ftp_pass=None,
            input_date=self.input_date,
        )
        self.assertAlmostEqual(
            df_tree.loc["portfolio", "total_vol"],
            df_risk.loc["portfolio", "total_vol"],
            places=2,
            msg="Tree risk portfolio vol must match get_portfolio_risk",
        )

        print("Portfolio Tree Risk (security rows):")
        print(security_rows[["weight", "total_vol", "RC_vol", "RC_pct"]].to_string())
        print("Portfolio row:")
        print(df_tree.loc[["portfolio"]].to_string())

    def test_06_portfolio_tree_risk_weekend_fallback(self):
        print("\n=== Testing Portfolio Tree Risk Weekend Fallback ===")
        df_tree = get_portfolio_tree_risk_logic(
            portfolio_name=self.portfolio_name,
            source=self.source,
            filesystem_folder=self.test_data_dir,
            bucket_name=None,
            access_key_id=None,
            secret_access_key=None,
            s3_host=None,
            s3_port=None,
            ftp_host=None,
            ftp_user=None,
            ftp_pass=None,
            input_date=self.monday_date,
        )
        self.assertIsInstance(df_tree, pd.DataFrame)
        self.assertFalse(df_tree.empty, "Portfolio tree risk (fallback) is empty")
        self.assertIn("portfolio", df_tree.index)
        print("Portfolio Tree Risk (Fallback) successful.")

    @classmethod
    def tearDownClass(cls):
        # Cleanup
        #for path in [cls.model_path_universe, cls.model_path_portfolio]:
        #    if os.path.exists(path):
        #        os.remove(path)
        pass

if __name__ == "__main__":
    unittest.main()
