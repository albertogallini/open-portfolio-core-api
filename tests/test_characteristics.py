import unittest
import warnings

from oport.const_and_utils import Config


class TestLinking(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Suppress DeprecationWarnings
        warnings.simplefilter("ignore", DeprecationWarning)

    def test_char_indicator(self):
        import pandas as pd
        from oport.functions.functions_characteristics import get_char_indicator

        # Define the stock ticker and target date
        ticker_symbol = "AAPL"  # Example: Apple Inc.
        target_date = pd.to_datetime("2024-10-02").date()

        from unittest.mock import patch, MagicMock
        import pandas as pd
        import numpy as np
        
        # Mock data creation
        dates = [
            pd.to_datetime("2024-09-30").date(),
            pd.to_datetime("2024-06-30").date(),
            pd.to_datetime("2024-03-31").date(),
            pd.to_datetime("2023-12-31").date(),
            pd.to_datetime("2023-09-30").date()
        ]
        print(f"DEBUG TEST: Dates defined: {dates}")
        columns_fin = ['Diluted EPS', 'Net Income', 'Basic Average Shares', 'EBITDA', 'Total Revenue']
        columns_cf = ['Operating Cash Flow', 'Free Cash Flow']
        
        # Create dummy dataframes
        # existing code expects Ticker.quarterly_income_stmt (untransposed originally)
        # The function transposes it: ticker.quarterly_income_stmt.T
        # So we should provide it in a way that .T works and gives us what we expect (rows as dates)
        # Actually expected: columns are dates, index are metrics
        
        data_fin = np.random.rand(len(columns_fin), len(dates)) * 100
        df_fin = pd.DataFrame(data_fin, index=columns_fin, columns=dates)
        
        data_cf = np.random.rand(len(columns_cf), len(dates)) * 100
        df_cf = pd.DataFrame(data_cf, index=columns_cf, columns=dates)
        
        # Mock Ticker
        # We patch 'oport.functions.functions_characteristics.yf' because patching 'yfinance.Ticker' 
        # wasn't effective (likely due to how it's imported or used).
        # By patching the 'yf' name in the module under test, we catch any access to it.
        with patch("oport.functions.functions_characteristics.yf") as MockYF:
            # MockYF is the module mock. MockYF.Ticker should be our Ticker class mock.
            # MockYF.Ticker() returns the instance.
            MockTicker = MockYF.Ticker
            mock_instance = MockTicker.return_value
            
            # Setup the instance attributes
            mock_instance.quarterly_income_stmt = df_fin
            mock_instance.quarterly_cashflow = df_cf
            
            # Mock history
            dates_hist = pd.date_range(end=target_date, periods=10)
            df_hist = pd.DataFrame({
                "Close": np.random.rand(len(dates_hist)) * 100,
                "Dividends": np.zeros(len(dates_hist))
            }, index=dates_hist)
            mock_instance.history.return_value = df_hist
            
            # Mock Estimates
            # Create DataFrames for estimates
            ee_df = pd.DataFrame({"avg": [1.50, 1.60, 6.00, 7.00]}, index=["0q", "+1q", "0y", "+1y"])
            re_df = pd.DataFrame({"avg": [100.0, 110.0, 400.0, 450.0]}, index=["0q", "+1q", "0y", "+1y"])
            ge_df = pd.DataFrame({"Stock": [0.05, 0.06, 0.10, 0.12, 0.15]}, index=["0q", "+1q", "0y", "+1y", "LTG"])
            
            mock_instance.earnings_estimate = ee_df
            mock_instance.revenue_estimate = re_df
            mock_instance.growth_estimates = ge_df

            # Call function
            from oport.fields import (
                FIELD_EVAL_EBITDA_TTM, 
                FIELD_EVAL_FREE_CASH_FLOW_TTM, 
                FIELD_EVAL_TOTAL_REVENUE_TTM,
                FIELD_EVAL_EBITDA_YOY,
                FIELD_EVAL_FREE_CASH_FLOW_YOY,
                FIELD_EVAL_EST_EARNINGS_0Q,
                FIELD_EVAL_EST_GROWTH_5Y
            )
            
            char_indicators = [
                FIELD_EVAL_EBITDA_TTM, 
                FIELD_EVAL_FREE_CASH_FLOW_TTM, 
                FIELD_EVAL_TOTAL_REVENUE_TTM,
                FIELD_EVAL_EBITDA_YOY,
                FIELD_EVAL_FREE_CASH_FLOW_YOY,
                FIELD_EVAL_EST_EARNINGS_0Q,
                FIELD_EVAL_EST_GROWTH_5Y
            ]
            
            d = get_char_indicator(ticker_symbol, target_date, char_indicators)
            d2 = get_char_indicator(ticker_symbol, target_date, char_indicators)
        
        print(d)
        print(d2)
        # We can't strictly compare d and d2 if we don't clear cache and mocks change, 
        # but here d2 SHOULD come from cache populated by d1 run.
        self.assertEqual(d, d2)
        
        # Verify new indicators are present
        
        self.assertIn(FIELD_EVAL_EBITDA_TTM, d)
        self.assertIn(FIELD_EVAL_FREE_CASH_FLOW_TTM, d)
        self.assertIn(FIELD_EVAL_TOTAL_REVENUE_TTM, d)
        
        # Check YoY
        # Mock data (from previous code block replacement in test) was random, so exact value check difficult,
        # but existence check is good.
        self.assertIn(FIELD_EVAL_EBITDA_YOY, d)
        self.assertIn(FIELD_EVAL_FREE_CASH_FLOW_YOY, d)
        
        # Check Estimates
        self.assertIn(FIELD_EVAL_EST_EARNINGS_0Q, d)
        self.assertIn(FIELD_EVAL_EST_GROWTH_5Y, d)
        
        print(f"EBITDA TTM: {d.get(FIELD_EVAL_EBITDA_TTM)}")
        print(f"FCF TTM: {d.get(FIELD_EVAL_FREE_CASH_FLOW_TTM)}")
        print(f"EBITDA YoY: {d.get(FIELD_EVAL_EBITDA_YOY)}")
        print(f"Est Earnings 0Q: {d.get(FIELD_EVAL_EST_EARNINGS_0Q)}")


if __name__ == "__main__":
    import os

    current_directory = os.getcwd()
    print("Current working dir: {}".format(current_directory))
    unittest.main()
