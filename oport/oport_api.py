"""open_port router command example with POST and GET methods sharing encapsulated logic."""

from typing import Optional, Tuple
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd
import re
import os

from openbb_core.app.model.obbject import OBBject

from .functions import (
    functions_characteristics,
    functions_pattribution,
    functions_performance,
    functions_performance_daily,
    functions_positions,
    functions_aggregation,
    functions_tree,
    functions_risk,
    functions_regime,    
)
from oport import const_and_utils
from oport import fields
from oport import error_collector


IMPUTE_POSITIONS = "IMPUTE_POSITIONS"
HOLDINGS = "HOLDINGS"
PERFORMANCE_ATTRIBUTION = "PERFORMANCE_ATTRIBUTION"
PORTFOLIO_TOTALS = "PORTFOLIO_TOTALS"
CREATE_INDEX_CONSTITUENTS = "CREATE_INDEX_CONSTITUENTS"
RISK_CALIBRATION = "RISK_CALIBRATION"
FACTOR_COVARIANCE = "FACTOR_COVARIANCE"
PORTFOLIO_RISK = "PORTFOLIO_RISK"
REGIME_CALIBRATION = "REGIME_CALIBRATION"
REGIME_SUMMARY = "REGIME_SUMMARY"
PORTFOLIO_CONSTRUCTION = "PORTFOLIO_CONSTRUCTION"
PORTFOLIO_TREE_RISK = "PORTFOLIO_TREE_RISK"

# API Names
ApiNames = {
    IMPUTE_POSITIONS: "IMPUTE POSITIONS",
    HOLDINGS: "HOLDINGS",
    PERFORMANCE_ATTRIBUTION: "PERFORMANCE ATTRIBUTION",
    PORTFOLIO_TOTALS: "PORTFOLIO TOTALS",
    CREATE_INDEX_CONSTITUENTS: "CREATE INDEX CONSTITUENTS",
    RISK_CALIBRATION: "RISK CALIBRATION",
    FACTOR_COVARIANCE: "FACTOR COVARIANCE",
    PORTFOLIO_RISK: "PORTFOLIO RISK",
    REGIME_CALIBRATION: "REGIME CALIBRATION",
    REGIME_SUMMARY: "REGIME SUMMARY",
    PORTFOLIO_CONSTRUCTION: "PORTFOLIO CONSTRUCTION",
    PORTFOLIO_TREE_RISK: "PORTFOLIO TREE RISK",
}

RISK_MODEL_FILE = "risk_model.pkl"
REGIME_MODEL_FILE = "regime_model.pkl"  

# Encapsulated logic for impute_positions
def impute_positions_logic(
    output_file_name: str,
    portfolio_base_ccy: str,
    transaction_file: str,
    transactions_field_mapping: str,
    append: bool,
    select_exchange: bool,
    additional_fields: Optional[str],
    source: str,
    filesystem_folder: Optional[str],
    bucket_name: Optional[str],
    access_key_id: Optional[str],
    secret_access_key: Optional[str],
    s3_host: Optional[str],
    s3_port: Optional[str],
    ftp_host: Optional[str],
    ftp_user: Optional[str],
    ftp_pass: Optional[str],
) -> pd.DataFrame:
    err_collector = error_collector.get_collector()

    if filesystem_folder is not None:
        filesystem_folder = filesystem_folder.replace("|", "/")
    else:
        filesystem_folder = ""

    try:
        import os
        import json

        current_directory = os.getcwd()

        config = const_and_utils.Config(
            source=source,
            filesystem_folder=filesystem_folder,
            output_file_name=output_file_name,
            bucket_name=bucket_name,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            s3_host=s3_host,
            s3_port=s3_port,
            ftp_host=ftp_host,
            ftp_user=ftp_user,
            ftp_pass=ftp_pass,
        )
        config.validate()

        fields_config = const_and_utils.read_json(
            config=config,
            file_path=filesystem_folder + "/" + transactions_field_mapping,
        )

        # portfolio_base_ccy override if specified in the transaction feild mapping file
        if (
            "portfolio_base_ccy" in fields_config and
            fields_config["portfolio_base_ccy"] is not None and
            output_file_name in fields_config["portfolio_base_ccy"]
        ):
            portfolio_base_ccy = fields_config["portfolio_base_ccy"][output_file_name]


        print(
            "Fetching portfolio {} transaction\nOutput folder: {}\nOutput portfolio name: {}\nWith transaction field mapping: {}".format(
                transaction_file,
                filesystem_folder,
                output_file_name,
                transactions_field_mapping,
            )
        )

        mode = "append" if append else ""
        functions_positions.generate_transaction_file(
            config=config,
            transaction_file_path=filesystem_folder + "/" + transaction_file,
            fields_config=fields_config,
            mode=mode,
            select_exchange=select_exchange,
        )
        functions_positions.generate_holdings(config=config)
        functions_positions.unroll_daily_holdings(config=config)
        r = functions_positions.fetch_portfolio_prices(
            config=config,
            mode="online",
            base_currency=portfolio_base_ccy,
            additional_fields=additional_fields,
        )

        return r

    except Exception as err_msg:
        err_collector.record_error(
            operation_name=ApiNames[IMPUTE_POSITIONS], error_details=str(err_msg)
        )

    return None


# Encapsulated logic for holdings
def holdings_logic(
    portfolio_name: str,
    start_date: str,
    end_date: str,
    classification: str,
    characteristics: str,
    performance_risk: str,
    source: str,
    benchmark_name: Optional[str],
    filesystem_folder: Optional[str],
    filesystem_folder_benchmark: Optional[str],
    bucket_name: Optional[str],
    bucket_name_benchmark: Optional[str],
    access_key_id: Optional[str],
    secret_access_key: Optional[str],
    s3_host: Optional[str],
    s3_port: Optional[str],
    ftp_host: Optional[str],
    ftp_user: Optional[str],
    ftp_pass: Optional[str],
) -> pd.DataFrame:
    err_collector = error_collector.get_collector()

    if filesystem_folder is not None:
        filesystem_folder = filesystem_folder.replace("|", "/")
    else:
        filesystem_folder = ""

    if filesystem_folder_benchmark is not None:
        filesystem_folder_benchmark = filesystem_folder_benchmark.replace("|", "/")
    else:
        filesystem_folder_benchmark = ""

    if classification == "":
        classification = None
    else:
        classification = classification.split("|")

    try:
        from datetime import datetime
        import pandas as pd

        config = const_and_utils.Config(
            source=source,
            filesystem_folder=filesystem_folder,
            output_file_name=portfolio_name,
            bucket_name=bucket_name,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            s3_host=s3_host,
            s3_port=s3_port,
            ftp_host=ftp_host,
            ftp_user=ftp_user,
            ftp_pass=ftp_pass,
        )
        config.validate()
        config_benchmark = const_and_utils.Config(
            source=source,
            filesystem_folder=filesystem_folder_benchmark,
            output_file_name=benchmark_name,
            bucket_name=bucket_name_benchmark,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            s3_host=s3_host,
            s3_port=s3_port,
            ftp_host=ftp_host,
            ftp_user=ftp_user,
            ftp_pass=ftp_pass,
        )
        if len(benchmark_name) > 0:
            config_benchmark.validate()

        print(
            "holdings portfolio {} = holdings benchmark {} - input_folders: portfolio -> {}, benchmark -> {}".format(
                portfolio_name,
                benchmark_name,
                filesystem_folder,
                filesystem_folder_benchmark,
            )
        )

        if len(start_date) != 0 and len(end_date) != 0:
            try:
                start_date = datetime.strptime(start_date, "%m-%d-%Y").date()
            except ValueError:
                start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
            try:
                end_date = datetime.strptime(end_date, "%m-%d-%Y").date()
            except ValueError:
                end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        else:
            start_date = None
            end_date = None

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
            date_field=fields.TRANSACTION_FIELD_TR_DATE,
        )

        benchmark_holdings_prefix = (
            benchmark_name + "_" + const_and_utils.DAILY_HOLDINGS_PREFIX
        )
        benchmark_holdings_files = const_and_utils.list_files(
            config=config_benchmark,
            input_folder=filesystem_folder_benchmark,
            prefix=benchmark_holdings_prefix,
        )
        benchmark_transactions = const_and_utils.read_csv(
            config=config_benchmark,
            file_path=config_benchmark.filesystem_folder
            + "/"
            + benchmark_name
            + "_"
            + const_and_utils.TRANSACTION_FILE_TICKERS,
            start_date=start_date,
            end_date=end_date,
            date_field=fields.TRANSACTION_FIELD_TR_DATE,
        )

        portfolio_hlds = functions_performance_daily.compute_performance_daily(
            config=config,
            prefix=portfolio_holdings_prefix,
            transactions=portfolio_transactions,
            start_date=start_date,
            end_date=end_date,
            holdings_files=portfolio_holdings_files,
            classification=classification,
        )
        benchmark_hlds = functions_performance_daily.compute_performance_daily(
            config=config_benchmark,
            prefix=benchmark_holdings_prefix,
            transactions=benchmark_transactions,
            start_date=start_date,
            end_date=end_date,
            holdings_files=benchmark_holdings_files,
            classification=classification,
        )

        if characteristics:
            characteristics = characteristics.split("|")
            if len(characteristics) > 0:
                print(
                    "Computing characteristics indicators: {}".format(characteristics)
                )
                portfolio_hlds = functions_characteristics.compute_characteristics(
                    hlds=portfolio_hlds, char_fields=characteristics
                )
                benchmark_hlds = functions_characteristics.compute_characteristics(
                    hlds=benchmark_hlds, char_fields=characteristics
                )

        portfolio = None
        portfolio_buckets = None
        benchmark = None
        benchmark_buckets = None
        print("Bucketing by {} ... ".format(classification))

        portfolio_hlds, portfolio, portfolio_buckets = functions_aggregation.aggregate(
            portfolio_daily_fields=portfolio_hlds,
            classification=classification,
            start_date=start_date,
            end_date=end_date,
            time_aggregation=False,
            metrics=portfolio_hlds.columns.tolist(),
        )

        benchmark_hlds, benchmark, benchmark_buckets = functions_aggregation.aggregate(
            portfolio_daily_fields=benchmark_hlds,
            classification=classification,
            start_date=start_date,
            end_date=end_date,
            time_aggregation=False,
            metrics=benchmark_hlds.columns.tolist(),
        )

        if performance_risk:
            performance_risk = performance_risk.split("|")
            if len(performance_risk) > 0:
                print(
                    "Computing performance and ex-post risk indicators: {}".format(
                        performance_risk
                    )
                )
                portfolio_hlds, portfolio, portfolio_buckets = (
                    functions_performance.compute_performance_indicators(
                        portfolio_hlds,
                        portfolio_buckets,
                        portfolio,
                        benchmark,
                        performance_risk,
                    )
                )

        response = functions_tree.merge_buckets(
            portfolio=portfolio,
            portfolio_buckets=portfolio_buckets,
            portfolio_hlds=portfolio_hlds,
            benchmark=benchmark,
            benchmark_buckets=benchmark_buckets,
            benchmark_hlds=benchmark_hlds,
        )

        response = functions_aggregation.adjust_returns_dataframe(response)

        print("Done!")
        return response

    except Exception as err_msg:
        err_collector.record_error(
            operation_name=ApiNames[HOLDINGS], error_details=str(err_msg)
        )

    return None


# Encapsulated logic for portfolio_totals
def portfolio_totals_logic(
    portfolio_name: str,
    plot_field: Optional[str],
    start_date: str,
    end_date: str,
    classification: str,
    source: str,
    benchmark_name: Optional[str],
    filesystem_folder: Optional[str],
    filesystem_folder_benchmark: Optional[str],
    bucket_name: Optional[str],
    bucket_name_benchmark: Optional[str],
    access_key_id: Optional[str],
    secret_access_key: Optional[str],
    s3_host: Optional[str],
    s3_port: Optional[str],
    ftp_host: Optional[str],
    ftp_user: Optional[str],
    ftp_pass: Optional[str],
) -> Tuple[dict, dict, dict, dict]:

    err_collector = error_collector.get_collector()

    if filesystem_folder is not None:
        filesystem_folder = filesystem_folder.replace("|", "/")
    else:
        filesystem_folder = ""

    if filesystem_folder_benchmark is not None:
        filesystem_folder_benchmark = filesystem_folder_benchmark.replace("|", "/")
    else:
        filesystem_folder_benchmark = ""

    if classification == "":
        classification = None
    else:
        classification = classification.split("|")

    try:
        import os
        from datetime import datetime
        import pandas as pd

        config = const_and_utils.Config(
            source=source,
            filesystem_folder=filesystem_folder,
            output_file_name=portfolio_name,
            bucket_name=bucket_name,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            s3_host=s3_host,
            s3_port=s3_port,
            ftp_host=ftp_host,
            ftp_user=ftp_user,
            ftp_pass=ftp_pass,
        )
        config.validate()

        config_benchmark = const_and_utils.Config(
            source=source,
            filesystem_folder=filesystem_folder_benchmark,
            output_file_name=benchmark_name,
            bucket_name=bucket_name_benchmark,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            s3_host=s3_host,
            s3_port=s3_port,
            ftp_host=ftp_host,
            ftp_user=ftp_user,
            ftp_pass=ftp_pass,
        )

        config_benchmark.validate()
        full_time_frame = False

        if len(start_date) != 0 and len(end_date) != 0:
            try:
                start_date = datetime.strptime(start_date, "%m-%d-%Y").date()
            except ValueError:
                start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
            try:
                end_date = datetime.strptime(end_date, "%m-%d-%Y").date()
            except ValueError:
                end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
            if start_date >= end_date:
                err_msg = "Time frame not valid start-date:{}  end-date:{}".format(
                    start_date, end_date
                )
                raise Exception(err_msg)

        else:
            full_time_frame = True
            start_date = None
            end_date = None

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
            date_field=fields.TRANSACTION_FIELD_TR_DATE,
        )

        benchmark_holdings_prefix = (
            benchmark_name + "_" + const_and_utils.DAILY_HOLDINGS_PREFIX
        )
        benchmark_holdings_files = const_and_utils.list_files(
            config=config_benchmark,
            input_folder=filesystem_folder_benchmark,
            prefix=benchmark_holdings_prefix,
        )
        benchmark_transactions = const_and_utils.read_csv(
            config=config_benchmark,
            file_path=config_benchmark.filesystem_folder
            + "/"
            + benchmark_name
            + "_"
            + const_and_utils.TRANSACTION_FILE_TICKERS,
            start_date=start_date,
            end_date=end_date,
            date_field=fields.TRANSACTION_FIELD_TR_DATE,
        )

        daily_performance = functions_performance_daily.compute_performance_daily(
            config=config,
            prefix=portfolio_holdings_prefix,
            transactions=portfolio_transactions,
            start_date=start_date,
            end_date=end_date,
            holdings_files=portfolio_holdings_files,
            classification=classification,
        )

        daily_performance_benchmark = (
            functions_performance_daily.compute_performance_daily(
                config=config_benchmark,
                prefix=benchmark_holdings_prefix,
                transactions=benchmark_transactions,
                start_date=start_date,
                end_date=end_date,
                holdings_files=benchmark_holdings_files,
                classification=classification,
            )
        )

        if full_time_frame:
            daily_performance, portfolio_r, buckets_r = functions_aggregation.aggregate(
                portfolio_daily_fields=daily_performance,
                classification=classification,
                time_aggregation=True,
            )
            daily_performance_benchmark, benchmark_r, benchmark_buckets_r = (
                functions_aggregation.aggregate(
                    portfolio_daily_fields=daily_performance_benchmark,
                    classification=classification,
                    time_aggregation=True,
                )
            )

        else:
            start_date, end_date = (
                functions_aggregation.resolve_time_portf_vs_bench_time_frame(
                    daily_performance, daily_performance_benchmark, start_date, end_date
                )
            )

            daily_performance, portfolio_r, buckets_r = functions_aggregation.aggregate(
                portfolio_daily_fields=daily_performance,
                start_date=start_date,
                end_date=end_date,
                classification=classification,
                time_aggregation=True,
            )
            daily_performance_benchmark, benchmark_r, benchmark_buckets_r = (
                functions_aggregation.aggregate(
                    portfolio_daily_fields=daily_performance_benchmark,
                    start_date=start_date,
                    end_date=end_date,
                    classification=classification,
                    time_aggregation=True,
                )
            )

        portfolio_r, buckets_r = functions_aggregation.adjust_returns_dicts(
            portfolio_r, buckets_r
        )
        benchmark_r, benchmark_buckets_r = functions_aggregation.adjust_returns_dicts(
            benchmark_r, benchmark_buckets_r
        )

        return portfolio_r, buckets_r, benchmark_r, benchmark_buckets_r

    except Exception as err_msg:
        err_collector.record_error(
            operation_name=ApiNames[PORTFOLIO_TOTALS], error_details=str(err_msg)
        )

    return None, None, None, None


# Encapsulated logic for performance_attribution
def performance_attribution_logic(
    portfolio_name: str,
    start_date: str,
    end_date: str,
    classification: str,
    trend_analysis: bool,
    source: str,
    benchmark_name: str,
    filesystem_folder: Optional[str],
    filesystem_folder_benchmark: Optional[str],
    bucket_name: Optional[str],
    bucket_name_benchmark: Optional[str],
    access_key_id: Optional[str],
    secret_access_key: Optional[str],
    s3_host: Optional[str],
    s3_port: Optional[str],
    ftp_host: Optional[str],
    ftp_user: Optional[str],
    ftp_pass: Optional[str],
) -> pd.DataFrame:
    err_collector = error_collector.get_collector()

    if filesystem_folder is not None:
        filesystem_folder = filesystem_folder.replace("|", "/")
    else:
        filesystem_folder = ""

    if filesystem_folder_benchmark is not None:
        filesystem_folder_benchmark = filesystem_folder_benchmark.replace("|", "/")
    else:
        filesystem_folder_benchmark = ""

    if classification == "":
        classification = None
    else:
        classification = classification.split("|")

    print(
        " portfolio {} VS  benchmark {} - input_folders: portfolio -> {}, benchmark -> {}".format(
            portfolio_name,
            benchmark_name,
            filesystem_folder,
            filesystem_folder_benchmark,
        )
    )

    try:
        import os
        from datetime import datetime
        import pandas as pd

        config = const_and_utils.Config(
            source=source,
            filesystem_folder=filesystem_folder,
            output_file_name=portfolio_name,
            bucket_name=bucket_name,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            s3_host=s3_host,
            s3_port=s3_port,
            ftp_host=ftp_host,
            ftp_user=ftp_user,
            ftp_pass=ftp_pass,
        )
        config.validate()

        config_benchmark = const_and_utils.Config(
            source=source,
            filesystem_folder=filesystem_folder_benchmark,
            output_file_name=benchmark_name,
            bucket_name=bucket_name_benchmark,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            s3_host=s3_host,
            s3_port=s3_port,
            ftp_host=ftp_host,
            ftp_user=ftp_user,
            ftp_pass=ftp_pass,
        )

        config_benchmark.validate()

        if len(start_date) != 0 and len(end_date) != 0:
            try:
                start_date = datetime.strptime(start_date, "%m-%d-%Y").date()
            except ValueError:
                start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
            try:
                end_date = datetime.strptime(end_date, "%m-%d-%Y").date()
            except ValueError:
                end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
            if start_date >= end_date:
                err_msg = "Time frame not valid start-date:{}  end-date:{}".format(
                    start_date, end_date
                )
                raise Exception(err_msg)

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
            date_field=fields.TRANSACTION_FIELD_TR_DATE,
        )

        benchmark_holdings_prefix = (
            benchmark_name + "_" + const_and_utils.DAILY_HOLDINGS_PREFIX
        )
        benchmark_holdings_files = const_and_utils.list_files(
            config=config_benchmark,
            input_folder=filesystem_folder_benchmark,
            prefix=benchmark_holdings_prefix,
        )
        benchmark_transactions = const_and_utils.read_csv(
            config=config_benchmark,
            file_path=config_benchmark.filesystem_folder
            + "/"
            + benchmark_name
            + "_"
            + const_and_utils.TRANSACTION_FILE_TICKERS,
            start_date=start_date,
            end_date=end_date,
            date_field=fields.TRANSACTION_FIELD_TR_DATE,
        )

        performance_daily = functions_performance_daily.compute_performance_daily(
            config=config,
            prefix=portfolio_holdings_prefix,
            transactions=portfolio_transactions,
            start_date=start_date,
            end_date=end_date,
            holdings_files=portfolio_holdings_files,
            classification=classification,
        )
        benchmark_performance_daily = (
            functions_performance_daily.compute_performance_daily(
                config=config_benchmark,
                prefix=benchmark_holdings_prefix,
                transactions=benchmark_transactions,
                start_date=start_date,
                end_date=end_date,
                holdings_files=benchmark_holdings_files,
                classification=classification,
            )
        )

        start_date, end_date = (
            functions_aggregation.resolve_time_portf_vs_bench_time_frame(
                performance_daily, benchmark_performance_daily, start_date, end_date
            )
        )

        performance_daily, portfolio, portfolio_buckets = (
            functions_aggregation.aggregate(
                portfolio_daily_fields=performance_daily,
                start_date=start_date,
                end_date=end_date,
                classification=classification,
                time_aggregation=False,
                metrics=fields.BASIC_HOLDINGS_FIELDS
                + fields.ATTRIBUTION_INPUT_DAILY_FIELDS,
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
                metrics=fields.BASIC_HOLDINGS_FIELDS
                + fields.ATTRIBUTION_INPUT_DAILY_FIELDS,
                performance_attribution=True,
            )
        )

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
            for field in fields.ATTRIBUTION_INPUT_DAILY_FIELDS
            + [fields.FIELD_TREE_POSITION_ID]
        ]
        tree = tree[
            [fields.FIELD_TREE_POSITION_ID]
            + [fields.FIELD_TREE_POSITION_ID_PVSB]
            + fields.BASIC_POSITON_DESCR_FIELDS
            + fields.ATTRIBUTION_INPUT_DAILY_FIELDS
            + ATTRIBUTION_DAILY_FIELDS_BENCH
        ]

        tree = functions_pattribution.compute_performance_attribution(
            tree=tree, start_date=start_date, end_date=end_date, time_aggregation=True
        )

        tree = functions_aggregation.adjust_returns_dataframe(tree)

        if not trend_analysis:
            tree = tree[(tree[fields.FIELD_DATE] == end_date)]
        else:
            if (tree["date"].max() - tree["date"].min()).days > 30:
                dates = pd.date_range(
                    end=tree["date"].max(), start=tree["date"].min(), freq="W"
                )
                tree = tree[tree["date"].isin([d.date() for d in dates])]

        return tree

    except Exception as err_msg:
        err_collector.record_error(
            operation_name=ApiNames[PERFORMANCE_ATTRIBUTION], error_details=str(err_msg)
        )

    return None


def risk_calibration_logic(
    calibration_universe: str,
    start_date: str,
    end_date: str,
    source: str,
    filesystem_folder: Optional[str],
    bucket_name: Optional[str],
    access_key_id: Optional[str],
    secret_access_key: Optional[str],
    s3_host: Optional[str],
    s3_port: Optional[str],
    ftp_host: Optional[str],
    ftp_user: Optional[str],
    ftp_pass: Optional[str],
) -> pd.DataFrame:
    err_collector = error_collector.get_collector()

    if filesystem_folder is not None:
        filesystem_folder = filesystem_folder.replace("|", "/")
    else:
        filesystem_folder = ""

    try:
        from datetime import datetime
        import pandas as pd

        config = const_and_utils.Config(
            source=source,
            filesystem_folder=filesystem_folder,
            output_file_name=calibration_universe,
            bucket_name=bucket_name,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            s3_host=s3_host,
            s3_port=s3_port,
            ftp_host=ftp_host,
            ftp_user=ftp_user,
            ftp_pass=ftp_pass,
        )
        config.validate()

        if len(start_date) != 0 and len(end_date) != 0:
            try:
                dt_start = datetime.strptime(start_date, "%m-%d-%Y").date()
            except ValueError:
                dt_start = datetime.strptime(start_date, "%Y-%m-%d").date()
            try:
                dt_end = datetime.strptime(end_date, "%m-%d-%Y").date()
            except ValueError:
                dt_end = datetime.strptime(end_date, "%Y-%m-%d").date()
        else:
            dt_start = None
            dt_end = None

        # 1. Get tickers from portfolio holdings
        portfolio_holdings_prefix = (
            calibration_universe + "_" + const_and_utils.DAILY_HOLDINGS_PREFIX
        )
        
        # Try to read the full/summary file first using read_csv abstraction
        full_file_path = filesystem_folder + "/" + portfolio_holdings_prefix + ".csv" if filesystem_folder else portfolio_holdings_prefix + ".csv"
        df_h = const_and_utils.read_csv(
            config=config,
            file_path=full_file_path,
            ignore_date=True,
        )

        tickers = []
        if not df_h.empty:
            tickers = [t for t in df_h[fields.FIELD_TICKER].unique().tolist() if isinstance(t, str)]
        else:
            # If empty, try partitioned files
            portfolio_holdings_files = const_and_utils.list_files(
                config=config,
                input_folder=filesystem_folder,
                prefix=portfolio_holdings_prefix,
            )
            for f in portfolio_holdings_files:
                df_h_part = const_and_utils.read_csv(
                    config=config,
                    file_path=filesystem_folder + "/" + f if filesystem_folder else f,
                    ignore_date=True,
                )
                if not df_h_part.empty:
                    tickers.extend([t for t in df_h_part[fields.FIELD_TICKER].unique().tolist() if isinstance(t, str)])
        
        tickers = list(set(tickers))
        print ("calibration universe tickers: ", tickers)
        if not tickers:
            raise ValueError(f"No tickers found for calibration universe {calibration_universe}")

        # 2. Run calibration
        cal, model = functions_risk.initialize_risk_model(
            tickers=tickers, start_date=dt_start, end_date=dt_end
        )

        # 3. Store the calibrated model
        model_path = filesystem_folder + "/" + RISK_MODEL_FILE if filesystem_folder else RISK_MODEL_FILE
        const_and_utils.write_pickle(config, model_path, cal)

        
        print (cal.report())
        print (model.summary())

        # 4. Return summary
        return functions_risk.get_calibration_summary(model)

    except Exception as err_msg:
        err_collector.record_error(
            operation_name=ApiNames[RISK_CALIBRATION], error_details=str(err_msg)
        )

    return None


def get_factor_covariance_logic(
    source: str,
    filesystem_folder: Optional[str],
    bucket_name: Optional[str],
    access_key_id: Optional[str],
    secret_access_key: Optional[str],
    s3_host: Optional[str],
    s3_port: Optional[str],
    ftp_host: Optional[str],
    ftp_user: Optional[str],
    ftp_pass: Optional[str],
) -> pd.DataFrame:
    err_collector = error_collector.get_collector()

    if filesystem_folder is not None:
        filesystem_folder = filesystem_folder.replace("|", "/")
    else:
        filesystem_folder = ""

    try:
        config = const_and_utils.Config(
            source=source,
            filesystem_folder=filesystem_folder,
            output_file_name="",  # Not used for loading common model
            bucket_name=bucket_name,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            s3_host=s3_host,
            s3_port=s3_port,
            ftp_host=ftp_host,
            ftp_user=ftp_user,
            ftp_pass=ftp_pass,
        )
        config.validate()

        model_path = filesystem_folder + "/" + RISK_MODEL_FILE if filesystem_folder else RISK_MODEL_FILE
        cal = const_and_utils.read_pickle(config, model_path)
        if cal is None:
            raise ValueError("No calibrated risk model found. Please run risk_calibration first.")

        return functions_risk.get_factor_covariance(cal.model)

    except Exception as err_msg:
        err_collector.record_error(
            operation_name=ApiNames[FACTOR_COVARIANCE], error_details=str(err_msg)
        )

    return None


def get_portfolio_risk_logic(
    portfolio_name: str,
    source: str,
    filesystem_folder: Optional[str],
    bucket_name: Optional[str],
    access_key_id: Optional[str],
    secret_access_key: Optional[str],
    s3_host: Optional[str],
    s3_port: Optional[str],
    ftp_host: Optional[str],
    ftp_user: Optional[str],
    ftp_pass: Optional[str],
    input_date: Optional[str] = None,
    confidence: float = 0.95,
    horizon_days: int = 1,
    portfolio_mv: Optional[float] = None,
) -> pd.DataFrame:
    err_collector = error_collector.get_collector()

    if filesystem_folder is not None:
        filesystem_folder = filesystem_folder.replace("|", "/")
    else:
        filesystem_folder = ""

    try:
        import pandas as pd
        config = const_and_utils.Config(
            source=source,
            filesystem_folder=filesystem_folder,
            output_file_name=portfolio_name,
            bucket_name=bucket_name,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            s3_host=s3_host,
            s3_port=s3_port,
            ftp_host=ftp_host,
            ftp_user=ftp_user,
            ftp_pass=ftp_pass,
        )
        config.validate()

        # 0. Handle Dates
        from datetime import datetime, timedelta
        if input_date:
            t_date = pd.to_datetime(input_date).date()
        else:
            t_date = datetime.now().date()

        # Business day fallback: if t_date is Monday, prev_date is Friday
        prev_date = t_date - timedelta(days=1)
        # Handle weekends: if yesterday was Sunday or Saturday, go back to Friday
        while prev_date.weekday() >= 5:  # 5=Sat, 6=Sun
            prev_date -= timedelta(days=1)

        # 1. Load the model
        model_path = filesystem_folder + "/" + RISK_MODEL_FILE if filesystem_folder else RISK_MODEL_FILE
        cal = const_and_utils.read_pickle(config, model_path)
        if cal is None:
            raise ValueError("No calibrated risk model found. Please run risk_calibration first.")

        # 2. Get latest weights from portfolio
        portfolio_holdings_prefix = (
            portfolio_name + "_" + const_and_utils.DAILY_HOLDINGS_PREFIX
        )
        
        # Check if portfolio files are in a subdirectory
        current_folder = filesystem_folder
        portfolio_holdings_files = const_and_utils.list_files(
            config=config,
            input_folder=current_folder,
            prefix=portfolio_holdings_prefix,
        )
        
        if not portfolio_holdings_files:
            subfolder = os.path.join(filesystem_folder, portfolio_name)
            if os.path.exists(subfolder) and os.path.isdir(subfolder):
                current_folder = subfolder
                portfolio_holdings_files = const_and_utils.list_files(
                    config=config,
                    input_folder=current_folder,
                    prefix=portfolio_holdings_prefix,
                )
        
        # Create a specific config for portfolio operations with the correct folder
        portfolio_config = const_and_utils.Config(
            source=source,
            filesystem_folder=current_folder,
            output_file_name=portfolio_name,
            bucket_name=bucket_name,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            s3_host=s3_host,
            s3_port=s3_port,
            ftp_host=ftp_host,
            ftp_user=ftp_user,
            ftp_pass=ftp_pass
        )

        # 2. Compute Daily Performance for a 10-day window
        # This ensures we find a valid date with weights even if t_date is a weekend/holiday
        # or if the portfolio has stale data.
        t_date_start = t_date - timedelta(days=10)

        df_h_perf = functions_performance_daily.compute_performance_daily(
            config=portfolio_config,
            prefix=portfolio_holdings_prefix,
            start_date=t_date_start,
            end_date=t_date,
            holdings_files=portfolio_holdings_files,
        )

        if df_h_perf.empty:
            raise ValueError(
                "The portfolio does not have valid weight on the specified date or before."
            )

        # 3. Find the most recent date in performance data that is <= target date
        # and has valid weights (non-null quantity and Close)
        df_valid = df_h_perf.dropna(subset=['quantity', 'Close'])
        available_dates = df_valid[fields.FIELD_DATE].unique()
        
        # available_dates are usually datetime.date objects from performance computation
        valid_dates = [d for d in available_dates if d <= t_date]
        
        if not valid_dates:
            raise ValueError(
                "The portfolio does not have valid weight on the specified date or before."
            )
            
        t_date = max(valid_dates)

        # 4. Aggregate to get weights (FIELD_EVAL_MKTVALUE_WEIGHTS_HLDS)
        df_h_agg, _, _ = functions_aggregation.aggregate(
            portfolio_daily_fields=df_h_perf,
            start_date=t_date,
            end_date=t_date,
        )

        if df_h_agg.empty:
             raise ValueError(f"The portfolio does not have valid weight on the specified date or before.")

        # Extract weights
        weight_col = fields.FIELD_EVAL_MKTVALUE_WEIGHTS_HLDS
        if weight_col not in df_h_agg.columns:
            raise ValueError(
                f"No weight column found after aggregation. Expected {fields.FIELD_EVAL_MKTVALUE_WEIGHTS_HLDS}"
            )

        # Map ticker -> weight
        weights = df_h_agg.groupby(fields.FIELD_TICKER)[weight_col].sum()
        
        # 4. Compute risk
        return functions_risk.get_portfolio_risk(
            cal, weights,
            confidence=confidence,
            horizon_days=horizon_days,
            portfolio_mv=portfolio_mv,
        )

    except Exception as err_msg:
        err_collector.record_error(
            operation_name=ApiNames[PORTFOLIO_RISK], error_details=str(err_msg)
        )

    return None


# Returns a dataframe containing the index constituents of a portfolio.
def get_index_constituents(
    index_ticker: str,
    period: str,
    additional_fields: Optional[str],
    output_file_name: str,
    source: str,
    filesystem_folder: Optional[str],
    bucket_name: Optional[str],
    access_key_id: Optional[str],
    secret_access_key: Optional[str],
    s3_host: Optional[str],
    s3_port: Optional[str],
    ftp_host: Optional[str],
    ftp_user: Optional[str],
    ftp_pass: Optional[str],
) -> pd.DataFrame:
    try:
        err_collector = error_collector.get_collector()

        import os

        # ==============================================================================
        # 1. SECURE CREDENTIAL & COOKIE RETRIEVAL
        # Fetch sensitive values from environment variables instead of hardcoding
        # ==============================================================================
        session_id = os.getenv("TV_SESSION_ID")
        session_id_sign = os.getenv("TV_SESSION_ID_SIGN")
        device_t = os.getenv("TV_DEVICE_T")

        # Optional warning if primary auth tokens are missing
        if not session_id or not session_id_sign:
            print("Warning: TV_SESSION_ID or TV_SESSION_ID_SIGN environment variables are missing.")

        harvested_cookies = {
            'sessionid': session_id or '',
            'sessionid_sign': session_id_sign or '',
            'device_t': device_t or '',
            '_sp_id.cf1a': os.getenv("TV_SP_ID", ""),
            '_sp_ses.cf1a': '*',
            'cookiePrivacyPreferenceBannerProduction': 'ignored',
        }

        # Filter out empty cookie values
        harvested_cookies = {k: v for k, v in harvested_cookies.items() if v}

        # ==============================================================================
        # 2. DYNAMIC & SECURE HEADERS
        # ==============================================================================
        formatted_ticker = index_ticker.replace(":", "-").upper()
        
        request_headers = {
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'accept-encoding': 'gzip, deflate, br, zstd',
            'accept-language': 'en-US,en;q=0.9',
            'cache-control': 'max-age=0',
            'priority': 'u=0, i',
            'referer': f'https://www.tradingview.com/symbols/{formatted_ticker}/',
            'sec-ch-ua': '"Google Chrome";v="143", "Chromium";v="143"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Linux"',
            'sec-fetch-dest': 'document',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-site': 'same-origin',
            'sec-fetch-user': '?1',
            'upgrade-insecure-requests': '1',
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36'
        }

        if filesystem_folder is not None:
            filesystem_folder = filesystem_folder.replace("|", "/")
        else:
            filesystem_folder = ""

        additional_fields = additional_fields.split("|") if additional_fields else []

        # ==============================================================================
        # 4. SESSION INITIALIZATION & REQUEST EXECUTION
        # ==============================================================================
        session = requests.Session()
        session.headers.update(request_headers)
        
        for name, value in harvested_cookies.items():
            session.cookies.set(name, value)
        # ------------------------------------
        # Helper function to check if the ticker exists on TradingView
        def check_ticker_on_tradingview(ticker_symbol):
            urllist = [
                "https://www.tradingview.com/markets/indices/quotes-major/",
                "https://www.tradingview.com/markets/indices/quotes-all/",
                "https://www.tradingview.com/markets/indices/quotes-us/",
                "https://www.tradingview.com/markets/indices/quotes-snp/",
                "https://www.tradingview.com/markets/indices/quotes-currency/",
                "https://www.tradingview.com/markets/indices/quotes-americas/",
                "https://www.tradingview.com/markets/indices/quotes-europe/",
                "https://www.tradingview.com/markets/indices/quotes-asia/",
                "https://www.tradingview.com/markets/indices/quotes-pacific/",
                "https://www.tradingview.com/markets/indices/quotes-middle-east/",
                "https://www.tradingview.com/markets/indices/quotes-africa/",
            ]

            try:
                for url in urllist:
                    response = session.get(url) 
                    response.raise_for_status()  # Raise an exception for bad status codes
                    soup = BeautifulSoup(response.text, "html.parser")

                    # Check table rows with data-rowkey (which uses "INDEXPROVIDER:TICKER" format, e.g. "SPCFD:SPX")
                    target = ticker_symbol.upper().replace(":", "-")
                    
                    rows = soup.find_all("tr", attrs={"data-rowkey": True})
                    for row in rows:
                        rowkey = row.get("data-rowkey")
                        if rowkey and rowkey.upper().replace(":", "-") == target:
                            return True

                    # Fallback to looking at elements with class matching tickerName or tickerName-ixuo49jq
                    ticker_elements = soup.find_all("a", class_=re.compile("tickerName"))
                    for element in ticker_elements:
                        href = element.get("href")
                        if href:
                            match = re.search(r"/symbols/(.*)/", href)
                            if match:
                                extracted_ticker = match.group(1)
                                if extracted_ticker.upper().replace(":", "-") == target:
                                    return True
                    continue
                return False
            except requests.exceptions.RequestException as e:
                err_msg = str(f"Error fetching the page: {e}")
                raise Exception(err_msg)
            except Exception as e:
                err_msg = str(f"Error fetching the page: {e}")
                raise Exception(err_msg)

        #if check_ticker_on_tradingview(index_ticker):
        #    print(f"The ticker {index_ticker} is supported")
        #else:
        #    err_msg = str(f"The ticker {index_ticker} is NOT supported")
        #    url = "https://www.tradingview.com/symbols/" + index_ticker + "/components/"
        #    raise Exception(err_msg + " - url: " + url)

        # Fetch OEX index constituents from TradingView
        url = "https://www.tradingview.com/symbols/" + index_ticker + "/components/"
        session.headers.update({
            'referer': f"https://www.tradingview.com/symbols/" + index_ticker + "/components/"
        })
        response = session.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        print("Status:", response.status_code)
        print("Final URL:", response.url)
        print("Content length:", len(response.text))
        #rows = soup.find_all("tr", {"class": "row-HX5UXsDj listRow"})
        rows = soup.select('tr[class*="row-"]')
        print(f"Found {len(rows)} table rows")


        # Extracting the ticker and market cap
        components = []
        totalMKTcap = 0
        for row in rows:
            cells = row.find_all("td")
            if len(cells) >= 2:
                a_tag = cells[0].find("a")
                comp_ticker = a_tag.text.strip()
                href = a_tag.get("href", "")
                
                exchange = ""
                if href and href.startswith("/symbols/"):
                    parts = href.split("/")[2].split("-")
                    if len(parts) >= 2:
                        exchange = parts[0]
                        
                market_cap_text = cells[1].get_text(strip=True)
                try:
                    market_cap_value = float(
                        market_cap_text.split()[0].replace("\u202f", "")
                    )
                    market_cap_unit = market_cap_text.split()[1]
                    # Converting TUSD to BUSD if needed
                    if market_cap_unit == "TUSD":
                        market_cap_value *= 1000
                except ValueError:
                    market_cap_value = 0
                components.append((comp_ticker, market_cap_value, exchange))
                totalMKTcap += market_cap_value

        # Create a DataFrame to store prices and weights
        df = pd.DataFrame()

        # Helper function to get the most recent 'Basic Avg Shares' before a given date
        def get_basic_avg_shares(quarterly_stmt, date):
            quarterly_dates = sorted(quarterly_stmt.keys())
            for q_date in reversed(quarterly_dates):
                if q_date <= date:
                    return quarterly_stmt[q_date]["Basic Average Shares"]
            return None

        # Fetch data from Yahoo Finance for each ticker over the last 30 days
        cont = 0

        for comp in components:
            raw_ticker = comp[0]
            exchange = comp[2] if len(comp) > 2 else ""
            
            suffixes = [""]
            if exchange == "XETR": suffixes = [".DE"]
            elif exchange == "EURONEXT": suffixes = [".AS", ".PA", ".BR", ".LS"]
            elif exchange == "BME": suffixes = [".MC"]
            elif exchange == "MIL": suffixes = [".MI"]
            elif exchange == "SIX": suffixes = [".SW"]
            elif exchange == "LSE": suffixes = [".L"]
            elif exchange == "OMXSTO": suffixes = [".ST"]
            elif exchange == "OMXC": suffixes = [".CO"]
            elif exchange == "OMXHEL": suffixes = [".HE"]
            elif exchange == "OSL": suffixes = [".OL"]
            elif exchange in ["BSE", "NSE"]: suffixes = [".BO", ".NS"]
            elif exchange == "TSE": suffixes = [".T"]
            elif exchange == "HKEX": suffixes = [".HK"]
            elif exchange == "SZSE": suffixes = [".SZ"]
            elif exchange == "SSE": suffixes = [".SS"]
            elif exchange == "KRX": suffixes = [".KS", ".KQ"]
            elif exchange == "TWSE": suffixes = [".TW"]

            success = False
            for suffix in suffixes:
                full_ticker = raw_ticker + suffix
                try:
                    stock = yf.Ticker(full_ticker)
                    # print(stock.info)
                    history = stock.history(period=period, interval="1d")
                    if history.empty:
                        continue

                    quarterly_stmt = stock.quarterly_income_stmt
                    # Convert history dates to timezone-naive
                    history.index = history.index.tz_localize(None)

                    # Get Basic Avg Shares for each date in the history
                    history["Basic Avg Shares"] = history.index.map(
                        lambda date: get_basic_avg_shares(quarterly_stmt, date)
                    )
                    history["ccy"] = history.index.map(
                        lambda date: stock.fast_info["currency"]
                    )
                    history["securityType"] = history.index.map(lambda date: "Common Stock")

                    for field in additional_fields:
                        history[field] = stock.info.get(field, None)

                    # Calculate market cap
                    history["Market Cap"] = history["Close"] * history["Basic Avg Shares"]
                    history["Ticker"] = full_ticker
                    history["Date"] = history.index
                    # print(history.columns)
                    to_add = history[
                        [
                            "Ticker",
                            "Date",
                            "Market Cap",
                            "Basic Avg Shares",
                            "Close",
                            "ccy",
                            "securityType",
                        ]
                        + additional_fields
                    ]
                    if not to_add.empty:
                        df = pd.concat([df, to_add])
                        success = True
                        break # Successfully fetched, stop trying suffixes
                except Exception as e:
                    print(e)
                    continue
            
            if success:
                cont += 1


        df.rename(
            columns={"Basic Avg Shares": "quantity", "Ticker": "ticker"}, inplace=True
        )
        df["date"] = pd.to_datetime(df.index)
        df.set_index("date", inplace=True)
        total_mkt_cap = (
            df.groupby("date")["Market Cap"]
            .sum()
            .rename("Total Market Cap")
            .reset_index()
        )

        # Merge the total market cap back with the original DataFrame
        df = pd.merge(df, total_mkt_cap, on="date")

        # Calculate weights
        df["Weight"] = df["Market Cap"] / df["Total Market Cap"] * 100
        df["FX Rates"] = df["Market Cap"] / df["Market Cap"]
        df["quantity"] = df["quantity"] / 100000.0
        df["isin"] = df["ticker"]

        # Sort the DataFrame by ticker and date
        df.sort_values(by=["ticker", "date"], inplace=True)

        # Get the full date range across all data
        min_date = df["date"].min()
        max_date = df["date"].max()
        full_date_range = pd.date_range(start=min_date, end=max_date, freq="D")

        # Resample each ticker separately to fill date gaps
        resampled_dfs = []
        for ticker in df["ticker"].unique():
            ticker_df = df[df["ticker"] == ticker].copy()
            ticker_df.set_index("date", inplace=True)

            # Reindex to full date range, then forward fill the data
            ticker_df = ticker_df.reindex(full_date_range)
            ticker_df = ticker_df.ffill()

            ticker_df.reset_index(inplace=True)
            ticker_df.rename(columns={"index": "date"}, inplace=True)
            resampled_dfs.append(ticker_df)

        # Combine all resampled dataframes
        df = pd.concat(resampled_dfs, ignore_index=True)
        df.drop(columns=["Date"], inplace=True)

        # Save the DataFrame to CSV
        config = const_and_utils.Config(
            source=source,
            filesystem_folder=filesystem_folder,
            output_file_name=output_file_name,
            bucket_name=bucket_name,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            s3_host=s3_host,
            s3_port=s3_port,
            ftp_host=ftp_host,
            ftp_user=ftp_user,
            ftp_pass=ftp_pass,
        )
        config.validate()

        write = False
        lock_data = const_and_utils.read_json(
            config=config,
            file_path=config.filesystem_folder + "/" + const_and_utils.constants.WRITE_LOCK
        )
        if lock_data is None or (isinstance(lock_data, dict) and "error" in lock_data):
            write = True
        if not write:
            print("Write lock file found. Skipping write.")

        if write:
            const_and_utils.write_csv(
                config=config,
                df=df,
                file_path=config.filesystem_folder
            + "/"
            + config.output_file_name
            + "_"
            + const_and_utils.DAILY_HOLDINGS_PREFIX
            + ".csv",
        )

        return df

    except Exception as err_msg:
        err_collector.record_error(
            operation_name=ApiNames[CREATE_INDEX_CONSTITUENTS],
            error_details=str(err_msg),
        )

    return None



def regime_calibration_logic(
    source: str,
    filesystem_folder: Optional[str],
    bucket_name: Optional[str],
    access_key_id: Optional[str],
    secret_access_key: Optional[str],
    s3_host: Optional[str],
    s3_port: Optional[str],
    ftp_host: Optional[str],
    ftp_user: Optional[str],
    ftp_pass: Optional[str],
    start_date: str = "",
    end_date: str = "",
    min_regimes: int = 2,
    max_regimes: int = 4,
    estimation_window: Optional[int] = None,
    refit_frequency: int = 21,
    order_selection_holdout: int = 21,
    min_params_multiplier: float = 10.0,
) -> pd.DataFrame:
    """
    Calibrate the Wasserstein-HMM regime model off the factor-return
    time series of an ALREADY-CALIBRATED risk model (run
    risk_calibration_logic first -- this reads the same RISK_MODEL_FILE
    pickle that risk_calibration_logic writes).

    min_params_multiplier guards against fitting an HMM with more free
    parameters than the window can support (see RegimeCalibrator.run);
    lower it only for small illustrative/test fits where you accept the
    resulting fit is not statistically meaningful.

    estimation_window: leave as None (the default) to derive a value that
    actually satisfies min_params_multiplier for the real factor count and
    max_regimes -- a fixed 252 default silently violated that guard for
    any realistic (~9-dimensional) continuous factor set, since the guard
    only depends on estimation_window/max_regimes/min_params_multiplier,
    not on how much history is loaded. Pass an explicit value only if you
    have deliberately sized it yourself (see functions_regime.
    initialize_regime_model).
    """
    err_collector = error_collector.get_collector()

    if filesystem_folder is not None:
        filesystem_folder = filesystem_folder.replace("|", "/")
    else:
        filesystem_folder = ""

    try:
        from datetime import datetime

        config = const_and_utils.Config(
            source=source,
            filesystem_folder=filesystem_folder,
            output_file_name="",  # not used for loading the common model
            bucket_name=bucket_name,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            s3_host=s3_host,
            s3_port=s3_port,
            ftp_host=ftp_host,
            ftp_user=ftp_user,
            ftp_pass=ftp_pass,
        )
        config.validate()

        # 1. Load the already-calibrated risk model
        risk_model_path = (
            filesystem_folder + "/" + RISK_MODEL_FILE if filesystem_folder else RISK_MODEL_FILE
        )
        risk_cal = const_and_utils.read_pickle(config, risk_model_path)
        if risk_cal is None:
            raise ValueError(
                "No calibrated risk model found. Please run risk_calibration first."
            )

        # 2. Pull its factor-return history and optionally window it
        fr = functions_risk.get_factor_returns(risk_cal.model)
        fr.index = pd.to_datetime(fr.index)
        if len(start_date) != 0:
            dt_start = pd.to_datetime(start_date)
            fr = fr[fr.index >= dt_start]
        if len(end_date) != 0:
            dt_end = pd.to_datetime(end_date)
            fr = fr[fr.index <= dt_end]

        # 3. Fit the regime model. estimation_window is only forwarded when
        #    the caller explicitly set it -- otherwise initialize_regime_model
        #    derives a value self-consistent with min_params_multiplier/
        #    max_regimes for the real factor count (see its docstring).
        regime_kwargs = dict(
            min_regimes=min_regimes,
            max_regimes=max_regimes,
            refit_frequency=refit_frequency,
            order_selection_holdout=order_selection_holdout,
            min_params_multiplier=min_params_multiplier,
        )
        if estimation_window is not None:
            regime_kwargs["estimation_window"] = estimation_window
        regime_cal, regime_cfg = functions_regime.initialize_regime_model(
            factor_returns=fr,
            **regime_kwargs,
        )

        # 4. Persist it, same pattern as the risk model pickle
        regime_model_path = (
            filesystem_folder + "/" + REGIME_MODEL_FILE if filesystem_folder else REGIME_MODEL_FILE
        )
        const_and_utils.write_pickle(config, regime_model_path, regime_cal)

        print(regime_cal.report())

        # 5. Return summary
        return functions_regime.get_regime_summary(regime_cal)

    except Exception as err_msg:
        err_collector.record_error(
            operation_name=ApiNames[REGIME_CALIBRATION], error_details=str(err_msg)
        )

    return None


def get_regime_summary_logic(
    source: str,
    filesystem_folder: Optional[str],
    bucket_name: Optional[str],
    access_key_id: Optional[str],
    secret_access_key: Optional[str],
    s3_host: Optional[str],
    s3_port: Optional[str],
    ftp_host: Optional[str],
    ftp_user: Optional[str],
    ftp_pass: Optional[str],
) -> pd.DataFrame:
    """Current regime snapshot + probabilities/transition matrix, all
    stitched into one DataFrame the way get_portfolio_risk does."""
    err_collector = error_collector.get_collector()

    if filesystem_folder is not None:
        filesystem_folder = filesystem_folder.replace("|", "/")
    else:
        filesystem_folder = ""

    try:
        config = const_and_utils.Config(
            source=source,
            filesystem_folder=filesystem_folder,
            output_file_name="",
            bucket_name=bucket_name,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            s3_host=s3_host,
            s3_port=s3_port,
            ftp_host=ftp_host,
            ftp_user=ftp_user,
            ftp_pass=ftp_pass,
        )
        config.validate()

        regime_model_path = (
            filesystem_folder + "/" + REGIME_MODEL_FILE if filesystem_folder else REGIME_MODEL_FILE
        )
        regime_cal = const_and_utils.read_pickle(config, regime_model_path)
        if regime_cal is None:
            raise ValueError(
                "No calibrated regime model found. Please run regime_calibration first."
            )

        current = functions_regime.get_current_regime(regime_cal)
        summary = functions_regime.get_regime_summary(regime_cal)
        return current.join(summary, rsuffix="_summary", how="left")

    except Exception as err_msg:
        err_collector.record_error(
            operation_name=ApiNames[REGIME_SUMMARY], error_details=str(err_msg)
        )

    return None


def construct_portfolio_logic(
    universe_name: str,
    source: str,
    filesystem_folder: Optional[str],
    bucket_name: Optional[str],
    access_key_id: Optional[str],
    secret_access_key: Optional[str],
    s3_host: Optional[str],
    s3_port: Optional[str],
    ftp_host: Optional[str],
    ftp_user: Optional[str],
    ftp_pass: Optional[str],
    input_date: Optional[str] = None,
    target_n: Optional[int] = None,
    max_weight: Optional[float] = None,
    transaction_cost_bps: Optional[float] = None,
    risk_aversion: Optional[float] = None,
) -> pd.DataFrame:
    """
    Regime-aware stock picking over a universe (an index's constituents
    OR an existing portfolio's holdings -- `universe_name` is resolved
    the same way portfolio_name/calibration_universe already are
    elsewhere in this file: via the "<name>_DAILY_HOLDINGS" file(s)).

    Requires both RISK_MODEL_FILE and REGIME_MODEL_FILE to already
    exist (run risk_calibration then regime_calibration first).

    Output columns: weight, prev_weight, trade, expected_return_regime,
    regime_id, regime_confidence, expected_hold_days -- one row per
    selected ticker, indexed by ticker.
    """
    err_collector = error_collector.get_collector()

    if filesystem_folder is not None:
        filesystem_folder = filesystem_folder.replace("|", "/")
    else:
        filesystem_folder = ""

    try:
        from datetime import datetime, timedelta

        config = const_and_utils.Config(
            source=source,
            filesystem_folder=filesystem_folder,
            output_file_name=universe_name,
            bucket_name=bucket_name,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            s3_host=s3_host,
            s3_port=s3_port,
            ftp_host=ftp_host,
            ftp_user=ftp_user,
            ftp_pass=ftp_pass,
        )
        config.validate()

        # 1. Load the calibrated risk + regime models
        risk_model_path = (
            filesystem_folder + "/" + RISK_MODEL_FILE if filesystem_folder else RISK_MODEL_FILE
        )
        risk_cal = const_and_utils.read_pickle(config, risk_model_path)
        if risk_cal is None:
            raise ValueError("No calibrated risk model found. Please run risk_calibration first.")

        regime_model_path = (
            filesystem_folder + "/" + REGIME_MODEL_FILE if filesystem_folder else REGIME_MODEL_FILE
        )
        regime_cal = const_and_utils.read_pickle(config, regime_model_path)
        if regime_cal is None:
            raise ValueError("No calibrated regime model found. Please run regime_calibration first.")

        # 2. Resolve the universe + current weights (same holdings-file
        #    pattern as risk_calibration_logic / get_portfolio_risk_logic).
        #    `universe_name` can itself be an index (e.g. an S&P/Nasdaq
        #    universe carries its own market-cap weights in its holdings
        #    file) -- so "current weights" here doubles as "start the
        #    optimizer from the index" when there is no separate portfolio.
        #    Only when NO holdings snapshot exists at all (anywhere at or
        #    before t_date) does this fall back to a 100%-cash start.
        universe_prefix = universe_name + "_" + const_and_utils.DAILY_HOLDINGS_PREFIX
        holdings_files = const_and_utils.list_files(
            config=config, input_folder=filesystem_folder, prefix=universe_prefix,
        )

        current_weights = None
        universe_tickers = []
        if holdings_files:
            if input_date:
                t_date = pd.to_datetime(input_date).date()
            else:
                t_date = datetime.now().date()
            # A tight +/-10-day window silently produced a cash start
            # whenever the run date didn't line up with a recent holdings
            # snapshot (e.g. running against an index whose file is only
            # refreshed periodically). Look back far enough to find the
            # most recent snapshot ON OR BEFORE t_date instead -- that IS
            # the index's (or portfolio's) current weights as of t_date --
            # and only treat it as missing if there is truly nothing.
            t_date_start = t_date - timedelta(days=400)

            df_perf = functions_performance_daily.compute_performance_daily(
                config=config,
                prefix=universe_prefix,
                start_date=t_date_start,
                end_date=t_date,
                holdings_files=holdings_files,
            )
            df_valid = df_perf.dropna(subset=["quantity", "Close"])
            valid_dates = [d for d in df_valid[fields.FIELD_DATE].unique() if d <= t_date]
            if valid_dates:
                snapshot_date = max(valid_dates)
                staleness_days = (t_date - snapshot_date).days
                if staleness_days > 10:
                    print(
                        "Warning: construct_portfolio_logic: no {} holdings snapshot "
                        "within 10 days of {} -- using the most recent one available, "
                        "from {} ({} days stale).".format(
                            universe_name, t_date, snapshot_date, staleness_days
                        )
                    )
                df_agg, _, _ = functions_aggregation.aggregate(
                    portfolio_daily_fields=df_perf, start_date=snapshot_date, end_date=snapshot_date,
                )
                weight_col = fields.FIELD_EVAL_MKTVALUE_WEIGHTS_HLDS
                if not df_agg.empty and weight_col in df_agg.columns:
                    current_weights = df_agg.groupby(fields.FIELD_TICKER)[weight_col].sum() / 100.0
                    universe_tickers = list(current_weights.index)

        if not universe_tickers:
            # Truly no holdings/index snapshot on file at all -- fall back
            # to every ticker the risk model was calibrated on, starting
            # from cash (full entry cost). This is now the genuine
            # last-resort path, not the common case.
            print(
                "Warning: construct_portfolio_logic: no {} holdings snapshot found "
                "on or before {} -- starting from 100% cash (turnover = 1.0) over "
                "the full risk-model universe instead of the index/portfolio's own "
                "weights.".format(
                    universe_name, t_date if holdings_files else input_date
                )
            )
            exposures = functions_regime.get_latest_exposures(risk_cal)
            universe_tickers = list(exposures.index)

        # 3. Construct the regime-conditioned portfolio
        result = functions_regime.construct_regime_portfolio(
            risk_cal=risk_cal,
            regime_cal=regime_cal,
            universe=universe_tickers,
            current_weights=current_weights,
            target_n=target_n,
            max_weight=max_weight,
            transaction_cost_bps=transaction_cost_bps,
            risk_aversion=risk_aversion,
        )
        return result

    except Exception as err_msg:
        err_collector.record_error(
            operation_name=ApiNames[PORTFOLIO_CONSTRUCTION], error_details=str(err_msg)
        )

    return None


def get_portfolio_tree_risk_logic(
    portfolio_name: str,
    source: str,
    filesystem_folder: Optional[str],
    bucket_name: Optional[str],
    access_key_id: Optional[str],
    secret_access_key: Optional[str],
    s3_host: Optional[str],
    s3_port: Optional[str],
    ftp_host: Optional[str],
    ftp_user: Optional[str],
    ftp_pass: Optional[str],
    input_date: Optional[str] = None,
    confidence: float = 0.95,
    horizon_days: int = 1,
    portfolio_mv: Optional[float] = None,
) -> pd.DataFrame:
    """
    Euler risk-contribution decomposition for a portfolio as of a given date.

    Loads the same calibrated risk model and portfolio weights as
    get_portfolio_risk_logic, then returns one row per holding (standalone
    risk metrics + Euler vol / variance contributions) plus a 'portfolio'
    summary row that matches get_portfolio_risk output.

    Output columns (security rows):
      weight        -- portfolio weight (0-100 scale)
      total_vol     -- annualised standalone vol (%)
      factor_vol    -- factor component of standalone vol (%)
      idio_vol      -- idio component of standalone vol (%)
      factor_share  -- factor share of standalone variance
      RC_vol        -- Euler vol contribution (%), sums to portfolio total_vol
      RC_pct        -- % share of portfolio variance, sums to 100

    Portfolio row additionally carries total_var, factor_var, idio_var,
    confidence, horizon_days, sigma_daily, sigma_horizon, VaR_pct
    (and VaR_value when portfolio_mv is provided).
    """
    err_collector = error_collector.get_collector()

    if filesystem_folder is not None:
        filesystem_folder = filesystem_folder.replace("|", "/")
    else:
        filesystem_folder = ""

    try:
        import pandas as pd
        config = const_and_utils.Config(
            source=source,
            filesystem_folder=filesystem_folder,
            output_file_name=portfolio_name,
            bucket_name=bucket_name,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            s3_host=s3_host,
            s3_port=s3_port,
            ftp_host=ftp_host,
            ftp_user=ftp_user,
            ftp_pass=ftp_pass,
        )
        config.validate()

        from datetime import datetime, timedelta
        if input_date:
            t_date = pd.to_datetime(input_date).date()
        else:
            t_date = datetime.now().date()

        # Load risk model
        model_path = filesystem_folder + "/" + RISK_MODEL_FILE if filesystem_folder else RISK_MODEL_FILE
        cal = const_and_utils.read_pickle(config, model_path)
        if cal is None:
            raise ValueError("No calibrated risk model found. Please run risk_calibration first.")

        # Resolve portfolio holdings folder
        portfolio_holdings_prefix = portfolio_name + "_" + const_and_utils.DAILY_HOLDINGS_PREFIX
        current_folder = filesystem_folder
        portfolio_holdings_files = const_and_utils.list_files(
            config=config,
            input_folder=current_folder,
            prefix=portfolio_holdings_prefix,
        )

        if not portfolio_holdings_files:
            subfolder = os.path.join(filesystem_folder, portfolio_name)
            if os.path.exists(subfolder) and os.path.isdir(subfolder):
                current_folder = subfolder
                portfolio_holdings_files = const_and_utils.list_files(
                    config=config,
                    input_folder=current_folder,
                    prefix=portfolio_holdings_prefix,
                )

        portfolio_config = const_and_utils.Config(
            source=source,
            filesystem_folder=current_folder,
            output_file_name=portfolio_name,
            bucket_name=bucket_name,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            s3_host=s3_host,
            s3_port=s3_port,
            ftp_host=ftp_host,
            ftp_user=ftp_user,
            ftp_pass=ftp_pass,
        )

        # Compute daily performance over a 10-day window to handle weekends/holidays
        t_date_start = t_date - timedelta(days=10)
        df_h_perf = functions_performance_daily.compute_performance_daily(
            config=portfolio_config,
            prefix=portfolio_holdings_prefix,
            start_date=t_date_start,
            end_date=t_date,
            holdings_files=portfolio_holdings_files,
        )

        if df_h_perf.empty:
            raise ValueError("The portfolio does not have valid weight on the specified date or before.")

        # Find the most recent date with valid weights
        df_valid = df_h_perf.dropna(subset=["quantity", "Close"])
        valid_dates = [d for d in df_valid[fields.FIELD_DATE].unique() if d <= t_date]
        if not valid_dates:
            raise ValueError("The portfolio does not have valid weight on the specified date or before.")
        t_date = max(valid_dates)

        df_h_agg, _, _ = functions_aggregation.aggregate(
            portfolio_daily_fields=df_h_perf,
            start_date=t_date,
            end_date=t_date,
        )
        if df_h_agg.empty:
            raise ValueError("The portfolio does not have valid weight on the specified date or before.")

        weight_col = fields.FIELD_EVAL_MKTVALUE_WEIGHTS_HLDS
        if weight_col not in df_h_agg.columns:
            raise ValueError(f"No weight column found after aggregation. Expected {weight_col}")

        weights = df_h_agg.groupby(fields.FIELD_TICKER)[weight_col].sum()

        return functions_risk.get_security_risk_contributions(
            cal,
            weights,
            confidence=confidence,
            horizon_days=horizon_days,
            portfolio_mv=portfolio_mv,
        )

    except Exception as err_msg:
        err_collector.record_error(
            operation_name=ApiNames[PORTFOLIO_TREE_RISK], error_details=str(err_msg)
        )

    return None
