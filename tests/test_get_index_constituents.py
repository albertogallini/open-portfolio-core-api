import unittest

from oport.openbb import obb
from oport.const_and_utils import *
from oport.fields import *
from oport.oport_api import get_index_constituents
from oport.router import create_index
import warnings

from openbb_core.app.utils import (
    basemodel_to_df,
)


class TestIndexConstituents(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Suppress DeprecationWarnings
        warnings.simplefilter("ignore", DeprecationWarning)

    def test_get_index_constituents(self):

        output_folder = ".|tests|test_portfolios|indexes|"

        config = Config(
            source="filesystem",
            filesystem_folder=output_folder.replace("|", "/"),
            output_file_name="nasdaq_index_test",
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

        index_dataframe = get_index_constituents(
            index_ticker="NDX",
            period="7d",
            source="filesystem",
            additional_fields="sector|industry|market",
            output_file_name=config.output_file_name,
            filesystem_folder=config.filesystem_folder,
            bucket_name=None,
            access_key_id=None,
            secret_access_key=None,
            s3_host=None,
            s3_port=None,
            ftp_host=None,
            ftp_user=None,
            ftp_pass=None,
        )

        assert index_dataframe is not None
        self.assertTrue(index_dataframe.shape[0] > 0)
        self.assertTrue("ticker" in index_dataframe.columns)
        self.assertTrue("sector" in index_dataframe.columns)

    def test_get_index_constituents_api(self):

        output_folder = ".|tests|test_portfolios|indexes|"

        config = Config(
            source="filesystem",
            filesystem_folder=output_folder.replace("|", "/"),
            output_file_name="nasdaq_index_test2",
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

        index = create_index(
            index_ticker="NDX",
            period="7d",
            source="filesystem",
            additional_fields="sector|industry|market",
            output_file_name=config.output_file_name,
            filesystem_folder=config.filesystem_folder,
            bucket_name=None,
            access_key_id=None,
            secret_access_key=None,
            s3_host=None,
            s3_port=None,
            ftp_host=None,
            ftp_user=None,
            ftp_pass=None,
        )

        assert index.results is not None
        df = pd.DataFrame(index.results)
        self.assertTrue(df.shape[0] > 0)
        self.assertTrue("ticker" in df.columns)
        self.assertTrue("sector" in df.columns)
        self.assertTrue("market" in df.columns)


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
