import unittest

import numpy as np
import pandas as pd
from oport import const_and_utils
from oport.functions import (
    functions_aggregation,
    functions_performance_daily,
    functions_tree,
    functions_pattribution,
)
from oport.const_and_utils import Config
from oport.functions.functions_pattribution import is_root
from oport.fields import (
    ATTRIBUTION_INPUT_DAILY_FIELDS,
    BASIC_HOLDINGS_FIELDS,
    BASIC_POSITON_DESCR_FIELDS,
    FIELD_EVAL_ACC_T_ALLOCATION,
    FIELD_EVAL_ACC_T_CCY_RETURN,
    FIELD_EVAL_ACC_T_CTR_TO_RETURN,
    FIELD_EVAL_ACC_T_LEVERAGE,
    FIELD_EVAL_ACC_T_RETURN,
    FIELD_EVAL_ACC_T_RETURN_HLDS_LCL_CCY,
    FIELD_EVAL_ACC_T_RETURN_LCL_CCY,
    FIELD_EVAL_ACC_T_SELECTION,
    FIELD_EVAL_ACC_T_TRANSACTION_RETURN_LCL_CCY,
    FIELD_EVAL_CCY_RETURN,
    FIELD_EVAL_CTR_TO_RETURN,
    FIELD_EVAL_RETURN,
    FIELD_EVAL_TRANSACTION_RETURN_LCL_CCY,
    FIELD_TREE_POSITION_ID,
    FIELD_TREE_POSITION_ID_PVSB,
    FIELD_EVAL_ALLOCATION,
    FIELD_EVAL_SELECTION,
    FIELD_EVAL_LEVERAGE,
    FIELD_EVAL_RETURN_HLDS_LCL_CCY,
    TRANSACTION_FIELD_TR_DATE,
    TREE_ACTIVE_SUFFIX,
    FIELD_DATE,
    TREE_BENCHMARK_SUFFIX,
)
import warnings
from datetime import date

from oport.oport_api import performance_attribution_logic


class TestPerformanceAttrbution(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Suppress DeprecationWarnings
        warnings.simplefilter("ignore", DeprecationWarning)

    def test_load_holdings_index_vs_index(self):
        from oport.openbb import obb

        output_folder = ".|tests|test_attribution|"
        output_folder = None

        r = obb.oport.performance_attribution(
            portfolio_name="nasdaq_index",
            benchmark_name="nasdaq_index",
            filesystem_folder=".|tests|test_portfolios|indexes",
            filesystem_folder_benchmark=".|tests|test_portfolios|indexes",
            classification="ccy|securityType",
            source="filesystem",
            start_date="10-10-2025",
            end_date="10-20-2025",
        )

        for f in [FIELD_EVAL_ALLOCATION, FIELD_EVAL_SELECTION, FIELD_EVAL_LEVERAGE]:
            self.assertTrue(f in r.results[0])

    def test_load_holdings_index_vs_index_calcs(self):
        # %%
        portfolio_name = "dec2024"
        benchmark_name = "oex_index"
        classification = "ccy|securityType"

        filesystem_folder = ".|tests|test_attribution|".replace("|", "/")
        config = Config(
            source="filesystem",
            filesystem_folder=filesystem_folder,
            output_file_name=portfolio_name,
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

        config_benchmark = Config(
            source="filesystem",
            filesystem_folder=filesystem_folder,
            output_file_name=benchmark_name,
            bucket_name=None,
            access_key_id=None,
            secret_access_key=None,
            s3_host=None,
            s3_port=None,
            ftp_host=None,
            ftp_user=None,
            ftp_pass=None,
        )

        config_benchmark.validate()

        start_date = date(2024, 10, 2)
        end_date = date(2024, 10, 4)

        ########################    Load portfolio
        portfolio_holdings_prefix = (
            portfolio_name + "_" + const_and_utils.DAILY_HOLDINGS_PREFIX
        )
        portfolio_holdings_files = const_and_utils.list_files(
            config=config,
            input_folder=filesystem_folder,
            prefix=portfolio_holdings_prefix,
        )
        portfolio_transactions = const_and_utils.read_csv(
            config=config,
            file_path=config.filesystem_folder
            + "/"
            + portfolio_name
            + "_"
            + const_and_utils.TRANSACTION_FILE_TICKERS,
            start_date=start_date,
            end_date=end_date,
            date_field=TRANSACTION_FIELD_TR_DATE,
        )

        ########################    Load benchmark
        benchmark_holdings_prefix = (
            benchmark_name + "_" + const_and_utils.DAILY_HOLDINGS_PREFIX
        )
        benchmark_holdings_files = const_and_utils.list_files(
            config=config_benchmark,
            input_folder=filesystem_folder,
            prefix=benchmark_holdings_prefix,
        )
        benchmark_transactions = const_and_utils.read_csv(
            config=config,
            file_path=config_benchmark.filesystem_folder
            + "/"
            + benchmark_name
            + "_"
            + const_and_utils.TRANSACTION_FILE_TICKERS,
            start_date=start_date,
            end_date=end_date,
            date_field=TRANSACTION_FIELD_TR_DATE,
        )

        ########################   get basic holdings snapshot data and basic daily performance figures
        performance_daily = functions_performance_daily.compute_performance_daily(
            config=config,
            prefix=portfolio_holdings_prefix,
            transactions=portfolio_transactions,
            start_date=start_date,
            end_date=end_date,
            holdings_files=portfolio_holdings_files,
        )
        benchmark_performance_daily = (
            functions_performance_daily.compute_performance_daily(
                config=config_benchmark,
                prefix=benchmark_holdings_prefix,
                transactions=benchmark_transactions,
                start_date=start_date,
                end_date=end_date,
                holdings_files=benchmark_holdings_files,
            )
        )

        start_date, end_date = (
            functions_aggregation.resolve_time_portf_vs_bench_time_frame(
                performance_daily, benchmark_performance_daily, start_date, end_date
            )
        )

        ########################   aggregate at bucket/top level
        portfolio = None
        portfolio_buckets = None
        benchmark = None
        benchmark_buckets = None
        print("Bucketing by {} ... ".format(classification))
        classification = classification.split("|")
        performance_daily, portfolio, portfolio_buckets = (
            functions_aggregation.aggregate(
                portfolio_daily_fields=performance_daily,
                start_date=start_date,
                end_date=end_date,
                classification=classification,
                time_aggregation=False,
                metrics=BASIC_HOLDINGS_FIELDS + ATTRIBUTION_INPUT_DAILY_FIELDS,
                performance_attribution=True,
            )
        )

        benchmark_performance_daily, benchmark, benchmark_buckets = (
            functions_aggregation.aggregate(
                portfolio_daily_fields=benchmark_performance_daily,
                start_date=start_date,
                end_date=end_date,
                classification=classification,
                time_aggregation=False,
                metrics=BASIC_HOLDINGS_FIELDS + ATTRIBUTION_INPUT_DAILY_FIELDS,
                performance_attribution=True,
            )
        )

        ########################    Merge Buckets and positions rows to make a complete tree
        tree = functions_tree.merge_buckets(
            portfolio=portfolio,
            portfolio_buckets=portfolio_buckets,
            portfolio_hlds=performance_daily,
            benchmark=benchmark,
            benchmark_buckets=benchmark_buckets,
            benchmark_hlds=benchmark_performance_daily,
            indexing=True,
        )

        ATTRIBUTION_DAILY_FIELDS_BENCH = [
            f"{field}_bench"
            for field in ATTRIBUTION_INPUT_DAILY_FIELDS + [FIELD_TREE_POSITION_ID]
        ]
        tree = tree[
            [FIELD_TREE_POSITION_ID]
            + [FIELD_TREE_POSITION_ID_PVSB]
            + BASIC_POSITON_DESCR_FIELDS
            + ATTRIBUTION_INPUT_DAILY_FIELDS
            + ATTRIBUTION_DAILY_FIELDS_BENCH
        ]

        tree = functions_pattribution.compute_performance_attribution(
            tree=tree,
            start_date=start_date,
            end_date=end_date,
            time_aggregation=True,
            verbose=True,
        )

        # format the daily returns and contribution to %
        tree = functions_aggregation.adjust_returns_dataframe(tree)

        self.assertTrue(len(tree) > 0)

        tree_1003 = tree[tree[FIELD_DATE] == date(2024, 10, 3)]
        root_1003 = tree_1003[
            tree_1003[FIELD_TREE_POSITION_ID_PVSB].apply(lambda x: is_root(x))
        ]
        tree_1004 = tree[tree[FIELD_DATE] == date(2024, 10, 4)]
        root_1004 = tree_1004[
            tree_1004[FIELD_TREE_POSITION_ID_PVSB].apply(lambda x: is_root(x))
        ]

        print("DEBUG info: ")
        print(
            "{} -> {}".format(
                FIELD_EVAL_ALLOCATION, root_1003[FIELD_EVAL_ALLOCATION].values
            )
        )
        print(
            "{} -> {}".format(
                FIELD_EVAL_SELECTION, root_1003[FIELD_EVAL_SELECTION].values
            )
        )
        print(
            "{} -> {}".format(
                FIELD_EVAL_LEVERAGE, root_1003[FIELD_EVAL_LEVERAGE].values
            )
        )
        print(
            "{} -> {}".format(
                FIELD_EVAL_CCY_RETURN,
                root_1003[FIELD_EVAL_CCY_RETURN + TREE_ACTIVE_SUFFIX].values,
            )
        )
        print(
            "{} -> {}".format(
                FIELD_EVAL_TRANSACTION_RETURN_LCL_CCY,
                root_1003[
                    FIELD_EVAL_TRANSACTION_RETURN_LCL_CCY + TREE_ACTIVE_SUFFIX
                ].values,
            )
        )
        print(
            "{} -> {} = {} - {} ".format(
                FIELD_EVAL_RETURN,
                root_1003[FIELD_EVAL_RETURN + TREE_ACTIVE_SUFFIX].values,
                root_1003[FIELD_EVAL_RETURN].values,
                root_1003[FIELD_EVAL_RETURN + TREE_BENCHMARK_SUFFIX].values,
            )
        )
        print(
            "{} -> {}".format(
                FIELD_EVAL_RETURN_HLDS_LCL_CCY,
                root_1003[FIELD_EVAL_RETURN_HLDS_LCL_CCY + TREE_ACTIVE_SUFFIX].values,
            )
        )
        print(
            "{} -> {}".format(
                FIELD_EVAL_CTR_TO_RETURN,
                root_1003[FIELD_EVAL_CTR_TO_RETURN + TREE_ACTIVE_SUFFIX].values,
            )
        )
        print("----------")
        print("----------")
        print(
            "{} -> {}".format(
                FIELD_EVAL_ALLOCATION, root_1004[FIELD_EVAL_ALLOCATION].values
            )
        )
        print(
            "{} -> {}".format(
                FIELD_EVAL_SELECTION, root_1004[FIELD_EVAL_SELECTION].values
            )
        )
        print(
            "{} -> {}".format(
                FIELD_EVAL_LEVERAGE, root_1004[FIELD_EVAL_LEVERAGE].values
            )
        )
        print(
            "{} -> {}".format(
                FIELD_EVAL_CCY_RETURN,
                root_1004[FIELD_EVAL_CCY_RETURN + TREE_ACTIVE_SUFFIX].values,
            )
        )
        print(
            "{} -> {}".format(
                FIELD_EVAL_TRANSACTION_RETURN_LCL_CCY,
                root_1004[
                    FIELD_EVAL_TRANSACTION_RETURN_LCL_CCY + TREE_ACTIVE_SUFFIX
                ].values,
            )
        )
        print(
            "{} -> {} = {} - {} ".format(
                FIELD_EVAL_RETURN,
                root_1004[FIELD_EVAL_RETURN + TREE_ACTIVE_SUFFIX].values,
                root_1004[FIELD_EVAL_RETURN].values,
                root_1004[FIELD_EVAL_RETURN + TREE_BENCHMARK_SUFFIX].values,
            )
        )
        print(
            "{} -> {}".format(
                FIELD_EVAL_RETURN_HLDS_LCL_CCY,
                root_1004[FIELD_EVAL_RETURN_HLDS_LCL_CCY + TREE_ACTIVE_SUFFIX].values,
            )
        )
        print(
            "{} -> {}".format(
                FIELD_EVAL_CTR_TO_RETURN,
                root_1004[FIELD_EVAL_CTR_TO_RETURN + TREE_ACTIVE_SUFFIX].values,
            )
        )
        print("----------")
        print(
            "{} -> {}".format(
                FIELD_EVAL_ALLOCATION, root_1004[FIELD_EVAL_ACC_T_ALLOCATION].values
            )
        )
        print(
            "{} -> {}".format(
                FIELD_EVAL_SELECTION, root_1004[FIELD_EVAL_ACC_T_SELECTION].values
            )
        )
        print(
            "{} -> {}".format(
                FIELD_EVAL_LEVERAGE, root_1004[FIELD_EVAL_ACC_T_LEVERAGE].values
            )
        )
        print(
            "{} -> {}".format(
                FIELD_EVAL_ACC_T_CCY_RETURN,
                root_1004[FIELD_EVAL_ACC_T_CCY_RETURN + TREE_ACTIVE_SUFFIX].values,
            )
        )
        print(
            "{} -> {}".format(
                FIELD_EVAL_TRANSACTION_RETURN_LCL_CCY,
                root_1004[
                    FIELD_EVAL_ACC_T_TRANSACTION_RETURN_LCL_CCY + TREE_ACTIVE_SUFFIX
                ].values,
            )
        )
        print(
            "{} -> {}".format(
                FIELD_EVAL_ACC_T_RETURN,
                root_1004[FIELD_EVAL_ACC_T_RETURN + TREE_ACTIVE_SUFFIX].values,
            )
        )
        print(
            "{} -> {}".format(
                FIELD_EVAL_ACC_T_RETURN_HLDS_LCL_CCY,
                root_1004[
                    FIELD_EVAL_ACC_T_RETURN_HLDS_LCL_CCY + TREE_ACTIVE_SUFFIX
                ].values,
            )
        )
        print(
            "{} -> {}".format(
                FIELD_EVAL_CTR_TO_RETURN,
                root_1004[FIELD_EVAL_ACC_T_CTR_TO_RETURN + TREE_ACTIVE_SUFFIX].values,
            )
        )

        self.assertTrue(
            np.isclose(
                (
                    root_1003[FIELD_EVAL_ALLOCATION]
                    + root_1003[FIELD_EVAL_SELECTION]
                    + root_1003[FIELD_EVAL_LEVERAGE]
                ),
                root_1003[FIELD_EVAL_RETURN_HLDS_LCL_CCY + TREE_ACTIVE_SUFFIX],
                rtol=0,  # relative tolerance
                atol=0.00000001,  # absolute tolerance (your desired precision)
            ).all()
        )

        self.assertTrue(
            np.isclose(
                (
                    root_1003[FIELD_EVAL_ALLOCATION]
                    + root_1003[FIELD_EVAL_SELECTION]
                    + root_1003[FIELD_EVAL_LEVERAGE]
                    + root_1003[FIELD_EVAL_CCY_RETURN + TREE_ACTIVE_SUFFIX]
                    + root_1003[
                        FIELD_EVAL_TRANSACTION_RETURN_LCL_CCY + TREE_ACTIVE_SUFFIX
                    ]
                ),
                root_1003[FIELD_EVAL_RETURN + TREE_ACTIVE_SUFFIX],
                rtol=0,  # relative tolerance
                atol=0.00000001,  # absolute tolerance (your desired precision)
            ).all()
        )

        self.assertTrue(
            np.isclose(
                (
                    root_1003[FIELD_EVAL_ACC_T_RETURN_LCL_CCY + TREE_ACTIVE_SUFFIX]
                    + root_1003[FIELD_EVAL_ACC_T_CCY_RETURN + TREE_ACTIVE_SUFFIX]
                ),
                root_1003[FIELD_EVAL_ACC_T_RETURN + TREE_ACTIVE_SUFFIX],
                rtol=0,  # relative tolerance
                atol=0.00000001,  # absolute tolerance (your desired precision)
            ).all()
        )

        self.assertTrue(
            np.isclose(
                (
                    root_1004[FIELD_EVAL_ALLOCATION]
                    + root_1004[FIELD_EVAL_SELECTION]
                    + root_1004[FIELD_EVAL_LEVERAGE]
                ),
                root_1004[FIELD_EVAL_RETURN_HLDS_LCL_CCY + TREE_ACTIVE_SUFFIX],
                rtol=0,  # relative tolerance
                atol=0.00000001,  # absolute tolerance (your desired precision)
            ).all()
        )

        self.assertTrue(
            np.isclose(
                (
                    root_1004[FIELD_EVAL_ALLOCATION]
                    + root_1004[FIELD_EVAL_SELECTION]
                    + root_1004[FIELD_EVAL_LEVERAGE]
                    + root_1004[FIELD_EVAL_CCY_RETURN + TREE_ACTIVE_SUFFIX]
                    + root_1004[
                        FIELD_EVAL_TRANSACTION_RETURN_LCL_CCY + TREE_ACTIVE_SUFFIX
                    ]
                ),
                root_1004[FIELD_EVAL_RETURN + TREE_ACTIVE_SUFFIX],
                rtol=0,  # relative tolerance
                atol=0.00000001,  # absolute tolerance (your desired precision)
            ).all()
        )

        self.assertTrue(
            np.isclose(
                (
                    root_1004[FIELD_EVAL_ACC_T_ALLOCATION]
                    + root_1004[FIELD_EVAL_ACC_T_SELECTION]
                    + root_1004[FIELD_EVAL_ACC_T_LEVERAGE]
                ),
                root_1004[FIELD_EVAL_ACC_T_RETURN_HLDS_LCL_CCY + TREE_ACTIVE_SUFFIX],
                rtol=0,  # relative tolerance
                atol=0.00000001,  # absolute tolerance (your desired precision)
            ).all()
        )

        self.assertTrue(
            np.isclose(
                (
                    root_1004[FIELD_EVAL_ACC_T_ALLOCATION]
                    + root_1004[FIELD_EVAL_ACC_T_SELECTION]
                    + root_1004[FIELD_EVAL_ACC_T_LEVERAGE]
                    + root_1004[FIELD_EVAL_ACC_T_CCY_RETURN + TREE_ACTIVE_SUFFIX]
                    + root_1004[
                        FIELD_EVAL_ACC_T_TRANSACTION_RETURN_LCL_CCY + TREE_ACTIVE_SUFFIX
                    ]
                ),
                root_1004[FIELD_EVAL_ACC_T_RETURN + TREE_ACTIVE_SUFFIX],
                rtol=0,  # relative tolerance
                atol=0.00000001,  # absolute tolerance (your desired precision)
            ).all()
        )

    def test_complex_attibution(self):

        output_folder = ".|tests|test_attribution_md|test_portfolios"

        tree = performance_attribution_logic(
            portfolio_name="test",
            benchmark_name="nasdaq",
            filesystem_folder=output_folder,
            filesystem_folder_benchmark=output_folder,
            classification="sector",
            trend_analysis=True,
            source="filesystem",
            s3_host=None,
            s3_port=None,
            bucket_name="test",
            bucket_name_benchmark="test",
            access_key_id="09YsmK4pUhi4xmEO1aDV",
            secret_access_key="neOjRLLE1fuO4PV7S46vTahkIjN30WXu09dQ8Ow7",
            start_date="10-6-2025",
            end_date="10-10-2025",
            ftp_host=None,
            ftp_user=None,
            ftp_pass=None,
        )

        first_date = True
        for d in tree[FIELD_DATE].unique():
            if first_date:
                first_date = False
                continue

            tree_at_date = tree[tree[FIELD_DATE] == d]
            root_at_date = tree_at_date[
                tree_at_date[FIELD_TREE_POSITION_ID_PVSB].apply(lambda x: is_root(x))
            ]
            print("DEBUG info: ")
            print(
                "{} -> {}".format(
                    FIELD_EVAL_ALLOCATION, root_at_date[FIELD_EVAL_ALLOCATION].values
                )
            )
            print(
                "{} -> {}".format(
                    FIELD_EVAL_SELECTION, root_at_date[FIELD_EVAL_SELECTION].values
                )
            )
            print(
                "{} -> {}".format(
                    FIELD_EVAL_LEVERAGE, root_at_date[FIELD_EVAL_LEVERAGE].values
                )
            )
            print(
                "{} -> {}".format(
                    FIELD_EVAL_CCY_RETURN,
                    root_at_date[FIELD_EVAL_CCY_RETURN + TREE_ACTIVE_SUFFIX].values,
                )
            )
            print(
                "{} -> {}".format(
                    FIELD_EVAL_TRANSACTION_RETURN_LCL_CCY,
                    root_at_date[
                        FIELD_EVAL_TRANSACTION_RETURN_LCL_CCY + TREE_ACTIVE_SUFFIX
                    ].values,
                )
            )
            print(
                "{} -> {} = {} - {} ".format(
                    FIELD_EVAL_RETURN,
                    root_at_date[FIELD_EVAL_RETURN + TREE_ACTIVE_SUFFIX].values,
                    root_at_date[FIELD_EVAL_RETURN].values,
                    root_at_date[FIELD_EVAL_RETURN + TREE_BENCHMARK_SUFFIX].values,
                )
            )
            print(
                "{} -> {}".format(
                    FIELD_EVAL_RETURN_HLDS_LCL_CCY,
                    root_at_date[
                        FIELD_EVAL_RETURN_HLDS_LCL_CCY + TREE_ACTIVE_SUFFIX
                    ].values,
                )
            )
            print(
                "{} -> {}".format(
                    FIELD_EVAL_CTR_TO_RETURN,
                    root_at_date[FIELD_EVAL_CTR_TO_RETURN + TREE_ACTIVE_SUFFIX].values,
                )
            )

            self.assertTrue(tree is not None)

            self.assertTrue(
                np.isclose(
                    (
                        root_at_date[FIELD_EVAL_ALLOCATION]
                        + root_at_date[FIELD_EVAL_SELECTION]
                        + root_at_date[FIELD_EVAL_LEVERAGE]
                        + root_at_date[FIELD_EVAL_CCY_RETURN + TREE_ACTIVE_SUFFIX]
                        + root_at_date[
                            FIELD_EVAL_TRANSACTION_RETURN_LCL_CCY + TREE_ACTIVE_SUFFIX
                        ]
                    ),
                    root_at_date[FIELD_EVAL_RETURN + TREE_ACTIVE_SUFFIX],
                    rtol=0,  # relative tolerance
                    atol=0.00000001,  # absolute tolerance (your desired precision)
                ).all()
            )

            self.assertTrue(
                np.isclose(
                    (
                        root_at_date[FIELD_EVAL_ACC_T_ALLOCATION]
                        + root_at_date[FIELD_EVAL_ACC_T_SELECTION]
                        + root_at_date[FIELD_EVAL_ACC_T_LEVERAGE]
                        + root_at_date[FIELD_EVAL_ACC_T_CCY_RETURN + TREE_ACTIVE_SUFFIX]
                        + root_at_date[
                            FIELD_EVAL_ACC_T_TRANSACTION_RETURN_LCL_CCY
                            + TREE_ACTIVE_SUFFIX
                        ]
                    ),
                    root_at_date[FIELD_EVAL_ACC_T_RETURN + TREE_ACTIVE_SUFFIX],
                    rtol=0,  # relative tolerance
                    atol=0.00000001,  # absolute tolerance (your desired precision)
                ).all()
            )


if __name__ == "__main__":
    import os

    current_directory = os.getcwd()
    print("Current working dir: {}".format(current_directory))
    unittest.main()

# %%
