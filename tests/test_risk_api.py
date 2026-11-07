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
    get_factor_attribution_logic,
    get_portfolio_factor_attribution_logic,
    RISK_MODEL_FILE,
)
from oport import fields, error_collector
from oport.router import get_factor_attribution, get_portfolio_factor_attribution

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

    def test_07_factor_attribution(self):
        print("\n=== Testing Factor-Based Attribution ===")
        df_attr = get_factor_attribution_logic(
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
            start_date=self.start_date,
            end_date=self.end_date,
        )

        if df_attr is None:
            collector = error_collector.get_collector()
            print("ERRORS:", collector.get_errors())

        self.assertIsInstance(df_attr, pd.DataFrame)
        self.assertFalse(df_attr.empty, "Factor attribution dataframe is empty")

        # Required core columns
        for col in [
            "date",
            "return_hlds",
            "holdings_return",
            "factor_return",
            "excess_return",
            "acc_return_hlds",
            "acc_holdings_return",
            "acc_factor_return",
            "acc_excess_return",
        ]:
            self.assertIn(col, df_attr.columns, f"Missing required column: {col}")

        # Identify factor columns (columns without 'acc_' and not in core meta columns)
        meta_cols = {
            "date",
            "return_hlds",
            "holdings_return",
            "factor_return",
            "excess_return",
            "acc_return_hlds",
            "acc_holdings_return",
            "acc_factor_return",
            "acc_excess_return",
        }
        factor_cols = [c for c in df_attr.columns if not c.startswith("acc_") and c not in meta_cols]
        self.assertTrue(len(factor_cols) > 0, "No factor columns found in attribution output")

        # Every factor must have an acc_ counterpart
        for f in factor_cols:
            self.assertIn(f"acc_{f}", df_attr.columns, f"Missing accumulated column for factor {f}")

        # Check mathematical invariants across all rows
        for idx, row in df_attr.iterrows():
            # 1. Daily sum of factor components == factor_return
            daily_factor_sum = sum(row[f] for f in factor_cols)
            self.assertAlmostEqual(
                daily_factor_sum,
                row["factor_return"],
                places=6,
                msg=f"Row {idx} daily factor sum != factor_return",
            )

            # 2. Daily factor_return + excess_return == return_hlds
            self.assertAlmostEqual(
                row["factor_return"] + row["excess_return"],
                row["return_hlds"],
                places=6,
                msg=f"Row {idx} daily factor_return + excess_return != return_hlds",
            )

            # 3. Accumulated sum of acc_factor components == acc_factor_return
            acc_factor_sum = sum(row[f"acc_{f}"] for f in factor_cols)
            self.assertAlmostEqual(
                acc_factor_sum,
                row["acc_factor_return"],
                places=6,
                msg=f"Row {idx} acc factor sum != acc_factor_return",
            )

            # 4. Accumulated acc_factor_return + acc_excess_return == acc_return_hlds
            self.assertAlmostEqual(
                row["acc_factor_return"] + row["acc_excess_return"],
                row["acc_return_hlds"],
                places=6,
                msg=f"Row {idx} acc_factor_return + acc_excess_return != acc_return_hlds",
            )

        # 5. Check alias consistency
        pd.testing.assert_series_equal(df_attr["return_hlds"], df_attr["holdings_return"], check_names=False)
        pd.testing.assert_series_equal(df_attr["acc_return_hlds"], df_attr["acc_holdings_return"], check_names=False)

        # 6. Verify get_portfolio_factor_attribution_logic alias gives identical result
        df_attr_alias = get_portfolio_factor_attribution_logic(
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
            start_date=self.start_date,
            end_date=self.end_date,
        )
        pd.testing.assert_frame_equal(df_attr, df_attr_alias)

        print(f"Factor attribution returned {len(df_attr)} daily observations with {len(factor_cols)} factors.")
        print("Attribution tail (selected columns):")
        display_cols = ["date", "return_hlds", "acc_return_hlds", "factor_return", "acc_factor_return", "excess_return", "acc_excess_return"]
        print(df_attr[display_cols].tail(3).to_string())

    def test_08_factor_attribution_subwindow(self):
        print("\n=== Testing Factor Attribution Subwindow ===")
        sub_start = "2021-10-15"
        sub_end = "2021-10-29"
        df_attr = get_factor_attribution_logic(
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
            start_date=sub_start,
            end_date=sub_end,
        )
        self.assertIsInstance(df_attr, pd.DataFrame)
        self.assertFalse(df_attr.empty, "Subwindow factor attribution is empty")
        dates = df_attr["date"].tolist()
        self.assertGreaterEqual(dates[0], sub_start)
        self.assertLessEqual(dates[-1], sub_end)
        print(f"Subwindow from {dates[0]} to {dates[-1]} ({len(df_attr)} days) successful.")

    def test_09_factor_attribution_router_command(self):
        print("\n=== Testing Factor Attribution OpenBB Router Commands ===")
        obb_res1 = get_factor_attribution(
            portfolio_name=self.portfolio_name,
            source=self.source,
            filesystem_folder=self.test_data_dir,
            start_date=self.start_date,
            end_date=self.end_date,
        )
        self.assertIsNotNone(obb_res1)
        self.assertIsNotNone(obb_res1.results)
        self.assertEqual(len(obb_res1.results), 25)

        obb_res2 = get_portfolio_factor_attribution(
            portfolio_name=self.portfolio_name,
            source=self.source,
            filesystem_folder=self.test_data_dir,
            start_date=self.start_date,
            end_date=self.end_date,
        )
        self.assertIsNotNone(obb_res2)
        self.assertIsNotNone(obb_res2.results)
        self.assertEqual(len(obb_res1.results), len(obb_res2.results))

        # Check sample fields on the first result dictionary
        first_row = obb_res1.results[0]
        self.assertIn("date", first_row)
        self.assertIn("return_hlds", first_row)
        self.assertIn("factor_return", first_row)
        self.assertIn("excess_return", first_row)
        self.assertIn("acc_return_hlds", first_row)
        self.assertIn("acc_factor_return", first_row)
        self.assertIn("acc_excess_return", first_row)
        print("Router commands get_factor_attribution and get_portfolio_factor_attribution verified successfully.")

    @classmethod
    def tearDownClass(cls):
        # Cleanup
        #for path in [cls.model_path_universe, cls.model_path_portfolio]:
        #    if os.path.exists(path):
        #        os.remove(path)
        pass

if __name__ == "__main__":
    unittest.main()
