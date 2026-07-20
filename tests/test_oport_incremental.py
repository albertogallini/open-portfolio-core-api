import unittest

from oport.openbb import obb
from oport.const_and_utils import *
from pandas.testing import assert_frame_equal
from oport.fields import *
import warnings


class TestOportIncrmentality(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Suppress DeprecationWarnings
        warnings.simplefilter("ignore", DeprecationWarning)

    def test_impute_positions(self):
        output_folder = ".|tests|test_portfolios|"
        config = Config(
            source="filesystem",
            filesystem_folder=output_folder.replace("|", "/"),
            output_file_name="test_incremental",
            bucket_name=None,
            access_key_id=None,
            secret_access_key=None,
            s3_host=None,
            s3_port=None,
            ftp_host=None,
            ftp_user=None,
            ftp_pass=None,
        )

        obb.oport.impute_positions(
            source="filesystem",
            transaction_file="t1.xls",
            transactions_field_mapping="transactions_field.json",
            filesystem_folder=output_folder,
            output_file_name=config.output_file_name,
            portfolio_base_ccy="EUR",
        )

        obb.oport.impute_positions(
            source="filesystem",
            transaction_file="t2.xls",
            transactions_field_mapping="transactions_field.json",
            filesystem_folder=output_folder,
            output_file_name=config.output_file_name,
            portfolio_base_ccy="EUR",
            append=True,
        )

        df_tr_incr = read_csv(
            config=config,
            file_path=config.filesystem_folder
            + "/"
            + config.output_file_name
            + "_"
            + TRANSACTION_FILE_TICKERS,
            date_field=TRANSACTION_FIELD_TR_DATE,
        )

        df_hlds_incr = read_csv(
            config=config,
            file_path=config.filesystem_folder
            + "/"
            + config.output_file_name
            + "_"
            + PORTFOLIO_HOLDINGS,
        )

        obb.oport.impute_positions(
            source="filesystem",
            transaction_file="transactions_full.xls",
            transactions_field_mapping="transactions_field.json",
            filesystem_folder=output_folder,
            output_file_name="test",
            portfolio_base_ccy="EUR",
        )

        df_tr_full = read_csv(
            config=config,
            file_path=config.filesystem_folder
            + "/"
            + "test"
            + "_"
            + TRANSACTION_FILE_TICKERS,
            date_field=TRANSACTION_FIELD_TR_DATE,
        )
        df_hlds_full = read_csv(
            config=config,
            file_path=config.filesystem_folder
            + "/"
            + "test"
            + "_"
            + PORTFOLIO_HOLDINGS,
        )

        trfl = [
            TRANSACTION_FIELD_ISIN,
            FIELD_TICKER,
            TRANSACTION_FIELD_TR_DATE,
            TRANSACTION_FIELD_BUY_SELL,
            TRANSACTION_FIELD_QUANTITIY,
            TRANSACTION_FIELD_PRICE,
            TRANSACTION_FIELD_CCY,
        ]

        assert_frame_equal(
            df_tr_incr.sort_values(trfl)[trfl].reset_index(drop=True),
            df_tr_full.sort_values(trfl)[trfl].reset_index(drop=True),
        )

        hldsfl = [
            FIELD_TICKER,
            FIELD_ISIN,
            FIELD_SECURITY_TYPE,
            FIELD_MARKET_SECTOR,
            FIELD_DATE,
            FIELD_EVAL_QUANTITY,
            FIELD_CCY,
        ]

        assert_frame_equal(
            df_hlds_incr.sort_values(hldsfl)[hldsfl].reset_index(drop=True),
            df_hlds_full.sort_values(hldsfl)[hldsfl].reset_index(drop=True),
        )


if __name__ == "__main__":

    import os

    current_directory = os.getcwd()
    print("Current working dir: {}".format(current_directory))

    test_directory = current_directory + "/tests/test_portfolios/"
    for file in os.listdir(test_directory):
        if file.endswith(".csv"):
            os.remove(os.path.join(test_directory, file))

    unittest.main()
    exit()
