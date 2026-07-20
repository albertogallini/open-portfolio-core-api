from oport.openbb import obb


import unittest

from oport.const_and_utils import *
from pandas.testing import assert_frame_equal
from oport.fields import *
import warnings


class TestImputePositionsFields(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Suppress DeprecationWarnings
        warnings.simplefilter("ignore", DeprecationWarning)

    def test_portfolio_totals_e2e(self):

        print("TEST E2E - FILESYSTEM")
        output_folder = ".|tests|test_portfolios|"
        config = Config(
            source="filesystem",
            filesystem_folder=output_folder.replace("|", "/"),
            output_file_name="test_stocksplits",
            bucket_name=None,
            access_key_id=None,
            secret_access_key=None,
            s3_host=None,
            s3_port=None,
            ftp_host=None,
            ftp_user=None,
            ftp_pass=None,
        )

        r = obb.oport.impute_positions(
            source="filesystem",
            transaction_file="transactions_full.xls",
            transactions_field_mapping="transactions_field.json",
            filesystem_folder=output_folder,
            output_file_name=config.output_file_name,
            portfolio_base_ccy="EUR",
            select_exchange=False,
            additional_fields="sector|industry|market",
        )

        self.assertTrue("sector" in r.results[0])
        self.assertTrue("industry" in r.results[0])
        self.assertTrue("market" in r.results[0])


if __name__ == "__main__":

    import os

    current_directory = os.getcwd()
    print("Current working dir: {}".format(current_directory))

    unittest.main()
    exit()
