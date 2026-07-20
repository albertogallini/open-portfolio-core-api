import unittest
from oport.functions import functions_performance, functions_aggregation
from oport import fields
from oport.const_and_utils import Config, list_files, read_csv, PORTFOLIO_HOLDINGS_DAILY
import warnings
from datetime import date


class TestLinking(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Suppress DeprecationWarnings
        warnings.simplefilter("ignore", DeprecationWarning)

    def test_read_csv_file(self):

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

        transactions = read_csv(
            config=config,
            file_path=config.filesystem_folder
            + "/"
            + config.output_file_name
            + "_"
            + PORTFOLIO_HOLDINGS_DAILY,
        )
        transactions_s = read_csv(
            config=config,
            file_path=config.filesystem_folder
            + "/"
            + config.output_file_name
            + "_"
            + PORTFOLIO_HOLDINGS_DAILY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 10),
            date_field=fields.FIELD_DATE,
        )

        print("{},{}".format(transactions.shape[0], transactions_s.shape[0]))
        self.assertTrue(transactions.shape[0] > transactions_s.shape[0])

    def test_read_csv_ftp(self):

        output_folder = "."
        config = Config(
            source="ftp",
            filesystem_folder=output_folder.replace("|", "/"),
            output_file_name="test_e2e_totals",
            bucket_name=None,
            access_key_id=None,
            secret_access_key=None,
            s3_host=None,
            s3_port=None,
            ftp_host="imhere.x10hosting.com",
            ftp_user="alberto",
            ftp_pass="OportOport!",
        )
        config.validate()

        transactions = read_csv(
            config=config,
            file_path=config.filesystem_folder
            + "/"
            + config.output_file_name
            + "_"
            + PORTFOLIO_HOLDINGS_DAILY,
        )
        transactions_s = read_csv(
            config=config,
            file_path=config.filesystem_folder
            + "/"
            + config.output_file_name
            + "_"
            + PORTFOLIO_HOLDINGS_DAILY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 10),
            date_field=fields.FIELD_DATE,
        )

        print("{},{}".format(transactions.shape[0], transactions_s.shape[0]))
        self.assertTrue(transactions.shape[0] > transactions_s.shape[0])


if __name__ == "__main__":
    import os

    current_directory = os.getcwd()
    print("Current working dir: {}".format(current_directory))
    unittest.main()
