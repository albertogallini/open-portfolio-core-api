import unittest
from oport.functions import functions_performance, functions_aggregation
from oport.const_and_utils import (
    Config,
    list_files,
    read_csv,
    TRANSACTION_FILE_TICKERS,
    DAILY_HOLDINGS_PREFIX,
)
from oport.fields import TRANSACTION_FIELD_TR_DATE, FIELD_EVAL_RETURN
import warnings
from datetime import date


class TestLinking(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Suppress DeprecationWarnings
        warnings.simplefilter("ignore", DeprecationWarning)

    def test_load_holdings_portf_vs_index_no_classification(self):
        from oport.openbb import obb

        response = obb.oport.holdings(
            portfolio_name="test",
            benchmark_name="oex_index",
            classification="",
            start_date="08-01-2024",
            end_date="10-31-2024",
            filesystem_folder=".|tests|test_holdings|",
            filesystem_folder_benchmark=".|tests|test_portfolios|indexes|",
            source="filesystem",
        )

        r = obb.oport.portfolio_totals(
            portfolio_name="test",
            benchmark_name="oex_index",
            filesystem_folder=".|tests|test_holdings|",
            filesystem_folder_benchmark=".|tests|test_portfolios|indexes|",
            source="filesystem",
            start_date="08-01-2024",
            end_date="10-28-2024",
        )

    def test_load_holdings_index_vs_index_no_classification(self):
        from oport.openbb import obb

        response = obb.oport.holdings(
            portfolio_name="oex_index",
            benchmark_name="oex_index",
            classification="",
            start_date="08-01-2024",
            end_date="10-31-2024",
            filesystem_folder=".|tests|test_portfolios|indexes|",
            filesystem_folder_benchmark=".|tests|test_portfolios|indexes|",
            source="filesystem",
        )

        r = obb.oport.portfolio_totals(
            portfolio_name="oex_index",
            benchmark_name="oex_index",
            filesystem_folder=".|tests|test_portfolios|indexes|",
            filesystem_folder_benchmark=".|tests|test_portfolios|indexes|",
            source="filesystem",
            start_date="08-01-2024",
            end_date="10-28-2024",
        )

    def test_load_holdings_index_vs_index_no_classification_one_field(self):
        from oport.openbb import obb

        response = obb.oport.holdings(
            portfolio_name="test",
            benchmark_name="oex_index",
            classification="",
            start_date="10-01-2024",
            end_date="10-31-2024",
            filesystem_folder=".|tests|test_holdings|",
            filesystem_folder_benchmark=".|tests|test_portfolios|indexes|",
            source="filesystem",
        )

        r = obb.oport.portfolio_totals(
            portfolio_name="test",
            benchmark_name="oex_index",
            filesystem_folder=".|tests|test_holdings|",
            filesystem_folder_benchmark=".|tests|test_portfolios|indexes|",
            plot_field=FIELD_EVAL_RETURN,
            source="filesystem",
            start_date="10-01-2024",
            end_date="10-28-2024",
        )


if __name__ == "__main__":
    import os

    current_directory = os.getcwd()
    print("Current working dir: {}".format(current_directory))
    unittest.main()
