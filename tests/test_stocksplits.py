from oport.openbb import obb


import unittest

from oport.const_and_utils import *
from pandas.testing import assert_frame_equal
from oport.fields import *
import warnings


class TestOportE2E(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Suppress DeprecationWarnings
        warnings.simplefilter("ignore", DeprecationWarning)


    def test_portfolio_stocksplit_baseccyoverride_e2e(self):

        print("TEST E2E - FILESYSTEM")
        output_folder = ".|tests|test_portfolios|"
        config = Config(
            source="filesystem",
            filesystem_folder=output_folder.replace("|", "/"),
            output_file_name="test_base_ccy",
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
            transaction_file="stocksplits.xls",
            transactions_field_mapping="transactions_field.json",
            filesystem_folder=output_folder,
            output_file_name=config.output_file_name,
            portfolio_base_ccy="USD",
            select_exchange=False,
        )

        r =obb.oport.holdings(
            portfolio_name=config.output_file_name,
            classification="ccy|securityType",
            start_date="01-01-2020",
            end_date="01-28-2024",
            filesystem_folder=config.filesystem_folder,
            source="filesystem",
        )

        assert (r.results[1][FIELD_EVAL_CCY_RETURN] != 0)
        assert (r.results[2][FIELD_EVAL_CCY_RETURN] != 0)
        assert (r.results[3][FIELD_EVAL_CCY_RETURN] == 0)
        assert (r.results[4][FIELD_EVAL_CCY_RETURN] == 0)        
    

    def test_portfolio_totals_stocksplit_e2e(self):

        print("TEST E2E - FILESYSTEM")
        output_folder = ".|tests|test_portfolios|"
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

        obb.oport.impute_positions(
            source="filesystem",
            transaction_file="stocksplits.xls",
            transactions_field_mapping="transactions_field.json",
            filesystem_folder=output_folder,
            output_file_name=config.output_file_name,
            portfolio_base_ccy="EUR",
            select_exchange=False,
        )

        obb.oport.holdings(
            portfolio_name=config.output_file_name,
            classification="ccy|securityType",
            start_date="01-01-2020",
            end_date="08-28-2024",
            filesystem_folder=config.filesystem_folder,
            source="filesystem",
        )

        r = obb.oport.portfolio_totals(
            portfolio_name=config.output_file_name,
            filesystem_folder=config.filesystem_folder,
            source="filesystem",
            start_date="01-01-2020",
            end_date="08-28-2024",
            classification="ccy|securityType",
        )

        acc_fs = [
            FIELD_EVAL_ACC_T_PNL,
            FIELD_EVAL_ACC_T_RETURN,
            FIELD_EVAL_ACC_T_CCY_RETURN,
        ]
        for f in acc_fs:
            assert "portfolio_"+f in r.results[0]

    def test_performance_fields(self):

        print("TEST E2E - FILESYSTEM")
        output_folder = ".|tests|test_portfolios|"
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

        obb.oport.impute_positions(
            source="filesystem",
            transaction_file="stocksplits.xls",
            transactions_field_mapping="transactions_field.json",
            filesystem_folder=output_folder,
            output_file_name=config.output_file_name,
            portfolio_base_ccy="EUR",
            select_exchange=False,
        )

        r = obb.oport.holdings(
            portfolio_name=config.output_file_name,
            classification="ccy|securityType",
            start_date="01-01-2024",
            end_date="08-28-2024",
            performance_risk="mean|var_99|skewness|beta|var_95|std_dev|1d_ret|1w_ret|1Q_ret|1Y_ret",
            filesystem_folder=config.filesystem_folder,
            source="filesystem",
        )
        perf_fs = [
            FIELD_EVAL_1D_RETURN,
            FIELD_EVAL_1Q_RETURN,
            FIELD_EVAL_1W_RETURN,
            FIELD_EVAL_1Y_RETURN,
            FIELD_EVAL_STD_DEV,
            FIELD_EVAL_VAR_95,
        ]
        for f in perf_fs:
            assert f in r.results[0]
        assert len(r.results) == 1141


if __name__ == "__main__":

    import os

    current_directory = os.getcwd()
    print("Current working dir: {}".format(current_directory))

    unittest.main()
    exit()
