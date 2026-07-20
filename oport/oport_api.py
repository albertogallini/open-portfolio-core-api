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
}

RISK_MODEL_FILE = "risk_model.pkl"


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
    
    # ==============================================================================
    # ⚠️ WARNING: Update these values when the script starts failing (Cookie Expiration)
    # ==============================================================================

    # 1. HARVESTED COOKIES (Updated from fresh working request ~ April 19, 2026)
    HARVESTED_COOKIES = {
        'sessionid': 'REMOVED_SECRET',
        'sessionid_sign': 'REMOVED_SECRET',
        '_sp_id.cf1a': 'REMOVED_SECRET',
        '_sp_ses.cf1a': '*',
        'device_t': 'REMOVED_SECRET',
        'etg': '31fe40b3-4542-4e0a-958a-3ea3949bb1fe',
        'png': '31fe40b3-4542-4e0a-958a-3ea3949bb1fe',
        'cachec': '31fe40b3-4542-4e0a-958a-3ea3949bb1fe',
        'tv_ecuid': '31fe40b3-4542-4e0a-958a-3ea3949bb1fe',
        'cookiePrivacyPreferenceBannerProduction': 'ignored',
        'g_state': '{"i_l":0,"i_ll":1776616386431,"i_e":{"enable_itp_optimization":1},"i_et":1776616386367}',
        'sp': '50b337f0-6623-46fe-989a-2906b4122858',
    }

 
    # 2. HARVESTED REQUEST HEADERS — now matching EXACTLY the working curl/browser request
    REQUEST_HEADERS = {
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'accept-encoding': 'gzip, deflate, br, zstd',
        'accept-language': 'en-US,en;q=0.9,it;q=0.8',
        'cache-control': 'max-age=0',
        'priority': 'u=0, i',
        # ────────────────────────────────────────────────────────────────
        # Critical fixes: exact values from your working request
        'referer': 'https://www.tradingview.com/symbols/DJ-DJI/',  # or make dynamic below
        'sec-ch-ua': '"Google Chrome";v="143", "Chromium";v="143", "Not A(Brand";v="24"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Linux"',
        # ────────────────────────────────────────────────────────────────
        'sec-fetch-dest': 'document',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-site': 'same-origin',
        'sec-fetch-user': '?1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36'
    }
    # ==============================================================================

    try:
        err_collector = error_collector.get_collector()

        if filesystem_folder is not None:
            filesystem_folder = filesystem_folder.replace("|", "/")
        else:
            filesystem_folder = ""

        additional_fields = additional_fields.split("|") if additional_fields else []

        # --- 🛠️ SESSION INITIALIZATION 🛠️ ---
        # 1. Create a session object
        session = requests.Session()
        
        # 2. Add the full set of harvested headers
        session.headers.update(REQUEST_HEADERS)
        
        # 3. Add the harvested cookies
        for name, value in HARVESTED_COOKIES.items():
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
        rows = soup.find_all("tr", {"class": "row-HX5UXsDj listRow"})
        print(f"Found {len(rows)} table rows")


        # Extracting the ticker and market cap
        components = []
        totalMKTcap = 0
        for row in soup.find_all("tr", {"class": "row-HX5UXsDj listRow"}):
            cells = row.find_all("td")
            if len(cells) >= 2:
                ticker = cells[0].find("a").text.strip()
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
                components.append((ticker, market_cap_value))
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

        for ticker in components:
            try:
                stock = yf.Ticker(ticker[0])
                # print(stock.info)
                history = stock.history(period=period, interval="1d")
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
                history["Ticker"] = ticker[0]
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
            except Exception as e:
                print(e)
                continue
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
