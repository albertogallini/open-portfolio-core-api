import unittest
import pandas as pd
import os
from datetime import datetime
from oport.oport_api import get_portfolio_risk_logic
from oport import error_collector

class TestRiskFallback(unittest.TestCase):
    def setUp(self):
        # Path relative to project root
        self.test_data_dir = "tests/test_attribution_md/test_portfolios"
        self.portfolio_name = "test_e2e_stocksplits"
        self.source = "filesystem"
        # 2021-11-15 is after the last date in the file (2021-11-10)
        self.future_date = "2021-11-15"

    def test_risk_date_fallback(self):
        print(f"\n=== Testing Portfolio Risk Fallback for date {self.future_date} ===")
        
        # This should fallback to 2021-11-10
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
            input_date=self.future_date
        )
        
        if df_risk is not None:
             print(f"Risk columns: {df_risk.columns.tolist()}")
             
        self.assertIsInstance(df_risk, pd.DataFrame)
        self.assertFalse(df_risk.empty, "Portfolio risk decomposition is empty")
        print("Portfolio Risk Decomposition successful with fallback.")
        
    def test_risk_no_valid_weight_error(self):
        print(f"\n=== Testing Portfolio Risk Error for non-existent portfolio/range ===")
        # 2020-01-01 is before any data in the file (starts 2020-05-06)
        old_date = "2020-01-01"
        
        error_collector.clear_collector()
        collector = error_collector.get_collector()

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
            input_date=old_date
        )
        
        self.assertIsNone(df_risk)
        err_data = collector.get_errors()
        errors_list = err_data.get('errors', []) if err_data else []
        self.assertTrue(any("The portfolio does not have valid weight on the specified date or before." in str(e) for e in errors_list))
        print("Caught and verified expected error in error collector.")

if __name__ == "__main__":
    unittest.main()
