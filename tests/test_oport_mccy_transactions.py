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

        trfl = [
            TRANSACTION_FIELD_ISIN,
            TRANSACTION_FIELD_TR_DATE,
            TRANSACTION_FIELD_BUY_SELL,
            TRANSACTION_FIELD_QUANTITIY,
            TRANSACTION_FIELD_PRICE,
            TRANSACTION_FIELD_CCY,
        ]

        config = Config(
            source="filesystem",
            filesystem_folder=output_folder.replace("|", "/"),
            output_file_name="test_mccy",
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
            transaction_file="tr_mccy.xls",
            transactions_field_mapping="transactions_field.json",
            filesystem_folder=output_folder,
            output_file_name=config.output_file_name,
            portfolio_base_ccy="EUR",
            append=False,
        )

        df_tr_isin = read_csv(
            config=config,
            file_path=config.filesystem_folder
            + "/"
            + config.output_file_name
            + "_"
            + TRANSACTION_FILE_TICKERS,
            date_field=TRANSACTION_FIELD_TR_DATE,
        )

        # print(df_tr_isin.sort_values(trfl)[trfl].reset_index(drop=True))

        obb.oport.holdings(
            portfolio_name=config.output_file_name,
            filesystem_folder=config.filesystem_folder,
            source="filesystem",
        )

        config = Config(
            source="filesystem",
            filesystem_folder=output_folder.replace("|", "/"),
            output_file_name="test_mccy_figi",
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
            transaction_file="tr_figiid.xls",
            transactions_field_mapping="transactions_field.json",
            filesystem_folder=output_folder,
            output_file_name=config.output_file_name,
            portfolio_base_ccy="EUR",
            append=False,
        )

        df_tr_bbg = read_csv(
            config=config,
            file_path=config.filesystem_folder
            + "/"
            + config.output_file_name
            + "_"
            + TRANSACTION_FILE_TICKERS,
            date_field=TRANSACTION_FIELD_TR_DATE,
        )

        # print(df_tr_bbg.sort_values(trfl)[trfl].reset_index(drop=True))

        obb.oport.holdings(
            portfolio_name=config.output_file_name,
            filesystem_folder=config.filesystem_folder,
            source="filesystem",
        )

        df_tr_bbg = df_tr_bbg[
            df_tr_bbg[FIELD_ISIN] != "IT0005599474"
        ]  # let's remove BONDS as by ISIN cannot be resolved

        assert_frame_equal(
            df_tr_isin.sort_values(trfl)[trfl].reset_index(drop=True),
            df_tr_bbg.sort_values(trfl)[trfl].reset_index(drop=True),
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
