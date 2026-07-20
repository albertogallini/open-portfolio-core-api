import unittest
from oport.functions import functions_performance_daily, functions_aggregation
from oport.const_and_utils import (
    Config,
    list_files,
    read_csv,
    TRANSACTION_FILE_TICKERS,
    DAILY_HOLDINGS_PREFIX,
)
from oport.fields import TRANSACTION_FIELD_TR_DATE
import warnings
from datetime import date


class TestHoldings(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Suppress DeprecationWarnings
        warnings.simplefilter("ignore", DeprecationWarning)

    def test_load_holdings_with_aggregation(self):

        output_folder = ".|tests|test_portfolios|test_e2e_stocksplits"
        config = Config(
            source="filesystem",
            filesystem_folder=output_folder.replace("|", "/"),
            output_file_name="test_e2e_stocksplits",
            bucket_name=None,
            access_key_id=None,
            secret_access_key=None,
            s3_host=None,
            s3_port=None,
            ftp_host=None,
            ftp_user=None,
            ftp_pass=None,
        )
        config.validate()

        start_date = date(2024, 1, 1)
        end_date = date(2024, 3, 31)

        transactions = read_csv(
            config=config,
            file_path=config.filesystem_folder
            + "/"
            + config.output_file_name
            + "_"
            + TRANSACTION_FILE_TICKERS,
            start_date=start_date,
            end_date=end_date,
            date_field=TRANSACTION_FIELD_TR_DATE,
        )

        prefix = config.output_file_name + "_" + DAILY_HOLDINGS_PREFIX
        files = list_files(
            config=config, input_folder=config.filesystem_folder, prefix=prefix
        )

        hlds = functions_performance_daily.performance_daily_mfiles(
            config=config,
            transactions=transactions,
            prefix=prefix,
            files=files,
            start_date=start_date,
            end_date=end_date,
        )

        hlds, portfolio, buckets = functions_aggregation.aggregate(
            portfolio_daily_fields=hlds,
            classification="ccy|securityType".split("|"),
            start_date=start_date,
            end_date=end_date,
            time_aggregation=False,
        )
        self.assertTrue(len(portfolio["date"]) == 91)
        self.assertTrue(len(buckets.items()) == 4)

    def test_load_holdings_index(self):
        from oport.openbb import obb

        output_folder = ".|tests|test_portfolios|indexes|"
        config = Config(
            source="filesystem",
            filesystem_folder=output_folder.replace("|", "/"),
            output_file_name="oex_idx",
            bucket_name=None,
            access_key_id=None,
            secret_access_key=None,
            s3_host=None,
            s3_port=None,
            ftp_host=None,
            ftp_user=None,
            ftp_pass=None,
        )

        obb.oport.holdings(
            portfolio_name=config.output_file_name,
            classification="ccy|securityType",
            start_date="10-01-2024",
            end_date="10-28-2024",
            characteristics="Diluted_EPS|Basic_Average_Shares|P-E",
            filesystem_folder=config.filesystem_folder,
            source="filesystem",
        )

        r = obb.oport.portfolio_totals(
            portfolio_name=config.output_file_name,
            filesystem_folder=config.filesystem_folder,
            source=config.source,
            start_date="10-01-2024",
            end_date="10-28-2024",
        )
        self.assertTrue(len(r.results) == 20)

    def test_load_holdings_index_vs_index(self):
        from oport.openbb import obb

        output_folder = ".|tests|test_portfolios|indexes|"
        config = Config(
            source="filesystem",
            filesystem_folder=output_folder.replace("|", "/"),
            output_file_name="oex_idx",
            bucket_name=None,
            access_key_id=None,
            secret_access_key=None,
            s3_host=None,
            s3_port=None,
            ftp_host=None,
            ftp_user=None,
            ftp_pass=None,
        )

        obb.oport.holdings(
            portfolio_name=config.output_file_name,
            benchmark_name="oex_idx",
            classification="ccy|securityType",
            start_date="10-01-2024",
            end_date="10-28-2024",
            characteristics="Diluted_EPS|Basic_Average_Shares|P-E",
            filesystem_folder=config.filesystem_folder,
            source="filesystem",
        )

        r = obb.oport.portfolio_totals(
            portfolio_name=config.output_file_name,
            filesystem_folder=config.filesystem_folder,
            source=config.source,
            start_date="10-01-2024",
            end_date="10-28-2024",
        )
        self.assertTrue(len(r.results) == 20)

    def test_load_holdings_portf_vs_index(self):
        from oport.openbb import obb

        response = obb.oport.holdings(
            portfolio_name="test",
            benchmark_name="oex_idx",
            classification="ccy|securityType",
            start_date="10-01-2024",
            end_date="10-31-2024",
            characteristics="Diluted_EPS|Basic_Average_Shares|P-E",
            filesystem_folder=".|tests|test_holdings|",
            filesystem_folder_benchmark=".|tests|test_portfolios|indexes|",
            source="filesystem",
        )
        self.assertTrue(len(response.results) == 2892)

    def test_load_holdings_portf_vs_index_no_classification(self):
        from oport.openbb import obb

        response = obb.oport.holdings(
            portfolio_name="test",
            benchmark_name="oex_idx",
            classification="",
            start_date="10-01-2024",
            end_date="10-31-2024",
            characteristics="Diluted_EPS|Basic_Average_Shares|P-E",
            filesystem_folder=".|tests|test_holdings|",
            filesystem_folder_benchmark=".|tests|test_portfolios|indexes|",
            source="filesystem",
        )

        self.assertTrue(len(response.results) == 2768)


if __name__ == "__main__":
    import os

    current_directory = os.getcwd()
    print("Current working dir: {}".format(current_directory))
    unittest.main()
