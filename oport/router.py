"""open_port router command example with POST and GET methods sharing encapsulated logic."""

from typing import Optional
from openbb_core.app.model.example import PythonEx
from openbb_core.app.model.obbject import OBBject
from openbb_core.app.router import Router
from openbb_core.app.utils import dict_to_basemodel, df_to_basemodel
from fastapi import Query
from typing import Annotated, Literal
from oport.oport_api import (
    impute_positions_logic,
    holdings_logic,
    portfolio_totals_logic,
    performance_attribution_logic,
    get_index_constituents,
    risk_calibration_logic,
    get_factor_covariance_logic,
    get_portfolio_risk_logic,
)
from oport import fields
from oport import error_collector
from oport.oport_api import (
    ApiNames,
    PORTFOLIO_TOTALS,
    HOLDINGS,
    IMPUTE_POSITIONS,
    PERFORMANCE_ATTRIBUTION,
    CREATE_INDEX_CONSTITUENTS,
    RISK_CALIBRATION,
    FACTOR_COVARIANCE,
    PORTFOLIO_RISK,
)

router = Router(prefix="", description="Suite to manage user portfolios.")


def create_error_obbject(api_name: str) -> OBBject:
    err_collector = error_collector.get_collector()
    response = {
        ApiNames[api_name]: ["none"],
        "error": err_collector.pretty_print_errors(),
    }
    error_collector.clear_collector()
    return OBBject(results=list(response.values()))


@router.command(
    methods=["GET"],
    examples=[
        PythonEx(
            description="Impute positions from Transactions.",
            code=[
                'obb.oport.impute_positions(source="filesystem",transaction_file="t1.xls",transactions_field_mapping="transactions_field.json",filesystem_folder=output_folder,output_file_name=config.output_file_name,portfolio_base_ccy="EUR")',  # noqa: E501
            ],
        )
    ],
)
def impute_positions(
    output_file_name: Annotated[
        str,
        Query(
            description="The file name containing the imputed positions.",
            example="positions",
        ),
    ] = "positions",
    portfolio_base_ccy: Annotated[
        str, Query(description="The portfolio base currency.", example="USD")
    ] = "USD",
    transaction_file: Annotated[
        str,
        Query(
            description="The input file containing the portfolio transactions.",
            example="transactions.xls",
        ),
    ] = "transactions.xls",
    transactions_field_mapping: Annotated[
        str,
        Query(
            description="The field mapping for transactions.",
            example="transactions_field.json",
        ),
    ] = "transactions_field.json",
    append: Annotated[
        bool,
        Query(
            description="If true, appends new transactions to existing portfolio.",
            example=False,
        ),
    ] = False,
    select_exchange: Annotated[
        bool,
        Query(
            description="If true, prompts to select exchange code when needed.",
            example=False,
        ),
    ] = False,
    additional_fields: Annotated[
        Optional[str],
        Query(description="Pipes separated field names for bucketing.", example=""),
    ] = "",
    source: Annotated[
        Literal["filesystem", "s3", "ftp"],
        Query(description="The source of data.", example="filesystem"),
    ] = "filesystem",
    filesystem_folder: Annotated[
        Optional[str],
        Query(description="The path where the output file is stored.", example=None),
    ] = None,
    bucket_name: Annotated[
        Optional[str], Query(description="S3 bucket for output file.", example=None)
    ] = None,
    access_key_id: Annotated[
        Optional[str], Query(description="S3 access key.", example=None)
    ] = None,
    secret_access_key: Annotated[
        Optional[str], Query(description="S3 secret key.", example=None)
    ] = None,
    s3_host: Annotated[
        Optional[str], Query(description="S3 host address.", example=None)
    ] = None,
    s3_port: Annotated[
        Optional[str], Query(description="S3 host port.", example=None)
    ] = None,
    ftp_host: Annotated[
        Optional[str], Query(description="FTP host name for output file.", example=None)
    ] = None,
    ftp_user: Annotated[
        Optional[str], Query(description="FTP user name.", example=None)
    ] = None,
    ftp_pass: Annotated[
        Optional[str], Query(description="FTP password.", example=None)
    ] = None,
) -> OBBject:
    """Compute the daily historical imputed positions from a sparse set of transactions

    Parameters
    ----------
    output_file_name: str
        The file name containing the imputed positions. By default "positions"
    portfolio_base_ccy: str
         The portfolio base currency. By default "USD"
    transaction_file: str
         The input file containing the portfolio transactions. By default "transactions.xls"
    transactions_field_mapping: str
         The field mapping. e.g. {"column_mapping": {"tr_sign": "buy_sell", "position amount": "quantity", "transaction cost": "price", "ccy": "currency", "ISIN": "isin", "transaction date": "transaction_date"}, "columns_name_row": 4, "data_row_offset": 5}
    source: Literal['filesystem', 's3', 'ftp']
        The source of data. By default "filesystem"
    filesystem_folder: str, optional
        Filesystem param: the path where the output file with the imputed positions is stored.
    append: bool, optional
        If true, appends new transactions to the existing portfolio.
    select_exchange: bool, optional
        If true, prompts to select the exchange code when needed.
    additional_fields: str, optional
        Pipes separated field names for bucketing in other analyses.
    bucket_name: str, optional
        S3 param: the S3 bucket where the output file is stored.
    access_key_id: str, optional
        S3 param: the access key.
    secret_access_key: str, optional
         S3 param: the secret key.
    s3_host: str, optional
        S3 param: the host address.
    s3_port: str, optional
        S3 param: the host port.
    ftp_host: str, optional
        FTP param: the FTP host name where the output file is stored.
    ftp_user: str, optional
        FTP param: the FTP user name.
    ftp_pass: str, optional
        FTP param: the FTP password.

    Returns
    -------
    OBBject[Dict[str, Any]]
        Historical imputed positions.
    """

    r = impute_positions_logic(
        output_file_name,
        portfolio_base_ccy,
        transaction_file,
        transactions_field_mapping,
        append,
        select_exchange,
        additional_fields,
        source,
        filesystem_folder,
        bucket_name,
        access_key_id,
        secret_access_key,
        s3_host,
        s3_port,
        ftp_host,
        ftp_user,
        ftp_pass,
    )
    if r is None:
        return create_error_obbject(IMPUTE_POSITIONS)

    import pandas as pd

    response = []
    for index, row in r.iterrows():
        # Create an empty dictionary for the current row
        row_dict = {}
        # Iterate through the items in the row (column name, value)
        for column, value in row.items():
            # Check if the value is of NAType (or a similar null type)
            # pd.isna() is the standard and most robust way to check for null values in pandas
            if pd.isna(value):
                row_dict[column] = None  # Replace the null value with None
            else:
                row_dict[column] = value
        response.append(row_dict)
    error_collector.clear_collector()
    return OBBject(results=response, provider="open port")


@router.command(
    methods=["GET"],
    examples=[
        PythonEx(
            description="Read the daily imputed positions and compute the holdings analytics (comprising characteristics and performance) for each day in the timeframe",
            code=[
                'obb.oport.holdings(portfolio_name=config.output_file_name,benchmark_name="oex_idx",classification="ccy|securityType",start_date="10-01-2024",end_date="10-28-2024",characteristics="Diluted_EPS|Basic_Average_Shares|P-E",filesystem_folder=config.filesystem_folder,source="filesystem")',  # noqa: E501
            ],
        )
    ],
)
def holdings(
    portfolio_name: Annotated[
        str, Query(description="The name of the portfolio.", example="test")
    ] = "test",
    start_date: Annotated[
        str,
        Query(description="Timeframe start date. Format: 'MM-DD-YYYY'.", example=""),
    ] = "",
    end_date: Annotated[
        str, Query(description="Timeframe end date. Format: 'MM-DD-YYYY'.", example="")
    ] = "",
    classification: Annotated[
        str, Query(description="Pipes separated field names for bucketing.", example="")
    ] = "",
    characteristics: Annotated[
        str, Query(description="List of characteristics fields to compute.", example="")
    ] = "",
    performance_risk: Annotated[
        str,
        Query(
            description="List of performance and ex-post risk fields to compute.",
            example="",
        ),
    ] = "",
    source: Annotated[
        Literal["filesystem", "s3", "ftp"],
        Query(description="The source of data.", example="filesystem"),
    ] = "filesystem",
    benchmark_name: Annotated[
        Optional[str], Query(description="The name of the benchmark.", example="")
    ] = "",
    filesystem_folder: Annotated[
        Optional[str],
        Query(
            description="Filesystem param: the path where the input file is stored [portfolio].",
            example=None,
        ),
    ] = None,
    filesystem_folder_benchmark: Annotated[
        Optional[str],
        Query(
            description="Filesystem param: the path where the input file is stored [benchmark].",
            example=None,
        ),
    ] = None,
    bucket_name: Annotated[
        Optional[str], Query(description="S3 bucket for output file.", example=None)
    ] = None,
    bucket_name_benchmark: Annotated[
        Optional[str],
        Query(description="S3 bucket for benchmark output file.", example=None),
    ] = None,
    access_key_id: Annotated[
        Optional[str], Query(description="S3 access key.", example=None)
    ] = None,
    secret_access_key: Annotated[
        Optional[str], Query(description="S3 secret key.", example=None)
    ] = None,
    s3_host: Annotated[
        Optional[str], Query(description="S3 host address.", example=None)
    ] = None,
    s3_port: Annotated[
        Optional[str], Query(description="S3 host port.", example=None)
    ] = None,
    ftp_host: Annotated[
        Optional[str], Query(description="FTP host name for output file.", example=None)
    ] = None,
    ftp_user: Annotated[
        Optional[str], Query(description="FTP user name.", example=None)
    ] = None,
    ftp_pass: Annotated[
        Optional[str], Query(description="FTP password.", example=None)
    ] = None,
) -> OBBject:
    """Read the daily imputed positions and compute the holdings analytics (comprising characteristics and performance) for each day in the timeframe

    Parameters
    ----------
    portfolio_name: str
        The name of the portfolio
    start_date: str, optional
        Timeframe start date
    end_date: str, optional
        Timeframe end date
    classification: str, optional
        Pipes separated field names for bucketing
    characteristics: str, optional
        List of characteristics fields to compute
    performance_risk: str, optional
        List of performance and ex-post risk fields to compute
    benchmark_name: str, optional
        The name of the benchmark
    source: Literal['filesystem', 's3', 'ftp']
        The source of data. By default "filesystem"
    filesystem_folder: str, optional
        Filesystem param: the path where the input file with the imputed positions is stored [portfolio].
    filesystem_folder_benchmark: str, optional
        Filesystem param: the path where the input file with the imputed positions is stored [benchmark].
    bucket_name: str, optional
        S3 param: the S3 bucket where the output file is stored
    access_key_id: str, optional
        S3 param: the access key
    secret_access_key: str, optional
        S3 param: the secret key
    s3_host: str, optional
        S3 param: the host address
    s3_port: str, optional
        S3 param: the host port
    ftp_host: str, optional
        FTP param: the FTP host name where the output file is stored
    ftp_user: str, optional
        FTP param: the FTP user name
    ftp_pass: str, optional
        FTP param: the FTP password

    Returns
    -------
    OBBject[Dict[str, Any]]
        Historical imputed positions.
    """
    r = holdings_logic(
        portfolio_name,
        start_date,
        end_date,
        classification,
        characteristics,
        performance_risk,
        source,
        benchmark_name,
        filesystem_folder,
        filesystem_folder_benchmark,
        bucket_name,
        bucket_name_benchmark,
        access_key_id,
        secret_access_key,
        s3_host,
        s3_port,
        ftp_host,
        ftp_user,
        ftp_pass,
    )

    if r is None:
        return create_error_obbject(HOLDINGS)
    import pandas as pd

    response = []
    for index, row in r.iterrows():
        # Create an empty dictionary for the current row
        row_dict = {}
        # Iterate through the items in the row (column name, value)
        for column, value in row.items():
            # Check if the value is of NAType (or a similar null type)
            # pd.isna() is the standard and most robust way to check for null values in pandas
            if pd.isna(value):
                row_dict[column] = None  # Replace the null value with None
            else:
                row_dict[column] = value
        response.append(row_dict)
    error_collector.clear_collector()
    return OBBject(results=response, provider="open port")


@router.command(
    methods=["GET"],
    examples=[
        PythonEx(
            description="Read the daily holdings performance and plot the top/bucket level performance analytics according to the input timeframe.",
            code=[
                'obb.oport.portfolio_totals(portfolio_name=config.output_file_name,filesystem_folder=config.filesystem_folder,source=config.source,start_date="10-01-2024",end_date="10-28-2024")',  # noqa: E501
            ],
        )
    ],
)
def portfolio_totals(
    portfolio_name: Annotated[
        str, Query(description="The name of the portfolio.", example="test_portfolio")
    ] = "test",
    plot_field: Annotated[
        Optional[str],
        Query(
            description="The name of the field to plot in a single chart mode.",
            example="",
        ),
    ] = "",
    start_date: Annotated[
        str,
        Query(description="Timeframe start date. Format: 'MM-DD-YYYY'.", example=""),
    ] = "",
    end_date: Annotated[
        str, Query(description="Timeframe end date. Format: 'MM-DD-YYYY'.", example="")
    ] = "",
    classification: Annotated[
        str, Query(description="Pipes separated field names for bucketing.", example="")
    ] = "",
    source: Annotated[
        Literal["filesystem", "s3", "ftp"],
        Query(description="The source of data.", example="filesystem"),
    ] = "filesystem",
    benchmark_name: Annotated[
        Optional[str], Query(description="The name of the benchmark.", example="")
    ] = "",
    filesystem_folder: Annotated[
        Optional[str],
        Query(
            description="Filesystem param: the path where the input file with daily analytics is stored [portfolio].",
            example=None,
        ),
    ] = None,
    filesystem_folder_benchmark: Annotated[
        Optional[str],
        Query(
            description="Filesystem param: the path where the input file with daily analytics is stored [benchmark].",
            example=None,
        ),
    ] = None,
    bucket_name: Annotated[
        Optional[str], Query(description="S3 bucket for output file.", example=None)
    ] = None,
    access_key_id: Annotated[
        Optional[str], Query(description="S3 access key.", example=None)
    ] = None,
    secret_access_key: Annotated[
        Optional[str], Query(description="S3 secret key.", example=None)
    ] = None,
    s3_host: Annotated[
        Optional[str], Query(description="S3 host address.", example=None)
    ] = None,
    s3_port: Annotated[
        Optional[str], Query(description="S3 host port.", example=None)
    ] = None,
    ftp_host: Annotated[
        Optional[str], Query(description="FTP host name for output file.", example=None)
    ] = None,
    ftp_user: Annotated[
        Optional[str], Query(description="FTP user name.", example=None)
    ] = None,
    ftp_pass: Annotated[
        Optional[str], Query(description="FTP password.", example=None)
    ] = None,
) -> OBBject:
    """Read the daily holdings performance and plot the top/bucket level performance analytics according to the input timeframe.

    Parameters
    ----------
    portfolio_name: str
        The name of the portfolio
    plot_field: str, optional
        The name of the field to plot in a single chart mode
    start_date: str, optional
        Timeframe start date
    end_date: str, optional
        Timeframe end date
    classification: str, optional
        Pipes separated field names for bucketing
    benchmark_name: str, optional
        The name of the benchmark
    source: Literal['filesystem', 's3', 'ftp']
        The source of data. By default "filesystem"
    filesystem_folder: str, optional
        Filesystem param: the path where the input file with daily analytics is stored [portfolio].
    filesystem_folder_benchmark: str, optional
        Filesystem param: the path where the input file with daily analytics is stored [benchmark].
    bucket_name: str, optional
        S3 param: the S3 bucket where the output file is stored
    access_key_id: str, optional
        S3 param: the access key
    secret_access_key: str, optional
        S3 param: the secret key
    s3_host: str, optional
        S3 param: the host address
    s3_port: str, optional
        S3 param: the host port
    ftp_host: str, optional
        FTP param: the FTP host name where the output file is stored
    ftp_user: str, optional
        FTP param: the FTP user name
    ftp_pass: str, optional
        FTP param: the FTP password

    Returns
    -------
    OBBject[Dict[str, Any]]
        Historical imputed positions.
    """

    portfolio_r, buckets_r, benchmark_r, benchmark_buckets_r = portfolio_totals_logic(
        portfolio_name,
        plot_field,
        start_date,
        end_date,
        classification,
        source,
        benchmark_name,
        filesystem_folder,
        filesystem_folder_benchmark,
        bucket_name,
        bucket_name,
        access_key_id,
        secret_access_key,
        s3_host,
        s3_port,
        ftp_host,
        ftp_user,
        ftp_pass,
    )

    if portfolio_r is None or len(portfolio_r) == 0:
        return create_error_obbject(PORTFOLIO_TOTALS)

    results_dict = {}

    def process_data(data_source, prefix):
        if data_source is not None and fields.FIELD_DATE in data_source.keys():
            for d in data_source[fields.FIELD_DATE]:
                date_str = d.isoformat()
                row = {"date": date_str}
                for k, v in data_source.items():
                    if k != fields.FIELD_DATE:
                        row[prefix + k] = v[data_source[fields.FIELD_DATE].index(d)]

                # Merge or add the row to the dictionary
                if date_str in results_dict:
                    results_dict[date_str].update(row)  # Merge with existing row
                else:
                    results_dict[date_str] = row

    process_data(portfolio_r, "portfolio_")
    process_data(benchmark_r, "benchmark_")
    for bucket_name, bucket_data in buckets_r.items():
        process_data(bucket_data, "buckets_" + str(bucket_name[0]).replace(" ", "_") + "_")
    for bucket_name, bucket_data in benchmark_buckets_r.items():
        process_data(
            bucket_data, "benchmark_buckets_" + str(bucket_name[0]).replace(" ", "_") + "_"
        )

    results = list(results_dict.values())

    # results_dict = {"results": data_list}
    # Create the OBBject with the list of tuples as the results
    error_collector.clear_collector()
    return OBBject(results=results, provider="oport")


@router.command(
    methods=["GET"],
    examples=[
        PythonEx(
            description="Read the daily holdings, compute the performance daily of portfolio and benchmark and run the performance attribution on top of it and the input classification strategy.",
            code=[
                'obb.oport.performance_attribution(portfolio_name=config.output_file_name,filesystem_folder=config.filesystem_folder,source=config.source,start_date="10-01-2024",end_date="10-28-2024")',  # noqa: E501
            ],
        )
    ],
)
def performance_attribution(
    portfolio_name: Annotated[
        str, Query(description="The name of the portfolio.", example="test")
    ] = "test",
    start_date: Annotated[
        str,
        Query(description="Timeframe start date. Format: 'MM-DD-YYYY'.", example=""),
    ] = "",
    end_date: Annotated[
        str, Query(description="Timeframe end date. Format: 'MM-DD-YYYY'.", example="")
    ] = "",
    classification: Annotated[
        str, Query(description="Pipes separated field names for bucketing.", example="")
    ] = "",
    trend_analysis: Annotated[
        bool,
        Query(
            description="Whether to run trend analysis on the performance attributin portfolio vs benchmark.",
            example=False,
        ),
    ] = False,
    source: Annotated[
        Literal["filesystem", "s3", "ftp"],
        Query(description="The source of data.", example="filesystem"),
    ] = "filesystem",
    benchmark_name: Annotated[
        str, Query(description="The name of the benchmark.", example="")
    ] = "",
    filesystem_folder: Annotated[
        Optional[str],
        Query(
            description="Filesystem param: the path where the input file with daily analytics is stored [portfolio].",
            example=None,
        ),
    ] = None,
    filesystem_folder_benchmark: Annotated[
        Optional[str],
        Query(
            description="Filesystem param: the path where the input file with daily analytics is stored [benchmark].",
            example=None,
        ),
    ] = None,
    bucket_name: Annotated[
        Optional[str], Query(description="S3 bucket for output file.", example=None)
    ] = None,
    access_key_id: Annotated[
        Optional[str], Query(description="S3 access key.", example=None)
    ] = None,
    secret_access_key: Annotated[
        Optional[str], Query(description="S3 secret key.", example=None)
    ] = None,
    s3_host: Annotated[
        Optional[str], Query(description="S3 host address.", example=None)
    ] = None,
    s3_port: Annotated[
        Optional[str], Query(description="S3 host port.", example=None)
    ] = None,
    ftp_host: Annotated[
        Optional[str], Query(description="FTP host name for output file.", example=None)
    ] = None,
    ftp_user: Annotated[
        Optional[str], Query(description="FTP user name.", example=None)
    ] = None,
    ftp_pass: Annotated[
        Optional[str], Query(description="FTP password.", example=None)
    ] = None,
) -> OBBject:
    """Read the daily holdings, compute the performance daily of portfolio and benchmark and run the performance attribution on top of it and the input classification strategy.

    Parameters
    ----------
    portfolio_name: str
        The name of the portfolio
    start_date: str, optional
        Timeframe start date
    end_date: str, optional
        Timeframe end date
    classification: str, optional
        Pipes separated field names for bucketing
    trend_analysis: bool, optional
        Whether to run trend analysis on the performance attributin on portfolio vs benchmark
    benchmark_name: str, optional
        The name of the benchmark
    source: Literal['filesystem', 's3', 'ftp']
        The source of data. By default "filesystem"
    filesystem_folder: str, optional
        Filesystem param: the path where the input file with daily analytics is stored [portfolio].
    filesystem_folder_benchmark: str, optional
        Filesystem param: the path where the input file with daily analytics is stored [benchmark].
    bucket_name: str, optional
        S3 param: the S3 bucket where the output file is stored
    access_key_id: str, optional
        S3 param: the access key
    secret_access_key: str, optional
        S3 param: the secret key
    s3_host: str, optional
        S3 param: the host address
    s3_port: str, optional
        S3 param: the host port
    ftp_host: str, optional
        FTP param: the FTP host name where the output file is stored
    ftp_user: str, optional
        FTP param: the FTP user name
    ftp_pass: str, optional
        FTP param: the FTP password

    Returns
    -------
    OBBject[Dict[str, Any]]
        Historical imputed positions.
    """
    r = performance_attribution_logic(
        portfolio_name,
        start_date,
        end_date,
        classification,
        trend_analysis,
        source,
        benchmark_name,
        filesystem_folder,
        filesystem_folder_benchmark,
        bucket_name,
        bucket_name,
        access_key_id,
        secret_access_key,
        s3_host,
        s3_port,
        ftp_host,
        ftp_user,
        ftp_pass,
    )
    if r is None:
        return create_error_obbject(PERFORMANCE_ATTRIBUTION)

    import pandas as pd

    response = []
    for index, row in r.iterrows():
        # Create an empty dictionary for the current row
        row_dict = {}
        # Iterate through the items in the row (column name, value)
        for column, value in row.items():
            # Check if the value is of NAType (or a similar null type)
            if pd.isna(value):
                row_dict[column] = None  # Replace the null value with None
            else:
                row_dict[column] = value
        response.append(row_dict)
    error_collector.clear_collector()
    return OBBject(results=response, provider="open port")


@router.command(
    methods=["GET"],
    examples=[
        PythonEx(
            description="Fetch index constituents and their details from Yahoo Finance.",
            code=[
                'obb.oport.create_index(index_ticker="NASDAQ-NDX", period="1y")',  # noqa: E501
            ],
        )
    ],
)
def create_index(
    index_ticker: Annotated[
        str, Query(description="The ticker of the index.", example="NASDAQ-NDX")
    ] = "NASDAQ-NDX",
    period: Annotated[
        Optional[str],
        Query(
            description="the period of the index. Options: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'.",
            example="1y",
        ),
    ] = "",
    additional_fields: Annotated[
        str,
        Query(
            description="Fields to fetch for the index constituents from yahoo Finance. Pipes separated.",
            example="market|sector|industry",
        ),
    ] = "",
    output_file_name: Annotated[
        str,
        Query(description="index file name", example="index"),
    ] = "",
    source: Annotated[
        Literal["filesystem", "s3", "ftp"],
        Query(description="The source of data.", example="filesystem"),
    ] = "filesystem",
    filesystem_folder: Annotated[
        Optional[str],
        Query(
            description="Filesystem param: the path where the input file with daily analytics is stored [portfolio].",
            example=None,
        ),
    ] = None,
    bucket_name: Annotated[
        Optional[str], Query(description="S3 bucket for output file.", example=None)
    ] = None,
    access_key_id: Annotated[
        Optional[str], Query(description="S3 access key.", example=None)
    ] = None,
    secret_access_key: Annotated[
        Optional[str], Query(description="S3 secret key.", example=None)
    ] = None,
    s3_host: Annotated[
        Optional[str], Query(description="S3 host address.", example=None)
    ] = None,
    s3_port: Annotated[
        Optional[str], Query(description="S3 host port.", example=None)
    ] = None,
    ftp_host: Annotated[
        Optional[str], Query(description="FTP host name for output file.", example=None)
    ] = None,
    ftp_user: Annotated[
        Optional[str], Query(description="FTP user name.", example=None)
    ] = None,
    ftp_pass: Annotated[
        Optional[str], Query(description="FTP password.", example=None)
    ] = None,
) -> OBBject:
    """Fetch index constituents and their details from Yahoo Finance..

    Parameters
    ----------
    index_ticker: str
        The ticker of the index
    period: str, optional
        the period of the index. Options: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
    additional_fields: str, optional
        Fields to fetch for the index constituents from yahoo Finance. Pipes separated
    output_file_name: str, optional
        Index file name
    source: Literal['filesystem', 's3', 'ftp']
        The source of data. By default "filesystem"
    filesystem_folder: str, optional
        Filesystem param: the path where the input file with daily analytics is stored [portfolio].
    bucket_name: str, optional
        S3 param: the S3 bucket where the output file is stored
    access_key_id: str, optional
        S3 param: the access key
    secret_access_key: str, optional
        S3 param: the secret key
    s3_host: str, optional
        S3 param: the host address
    s3_port: str, optional
        S3 param: the host port
    ftp_host: str, optional
        FTP param: the FTP host name where the output file is stored
    ftp_user: str, optional
        FTP param: the FTP user name
    ftp_pass: str, optional
        FTP param: the FTP password

    Returns
    -------
    OBBject[Dict[str, Any]]
        Historical imputed positions.
    """

    index_df = get_index_constituents(
        index_ticker,
        period,
        additional_fields,
        output_file_name,
        source,
        filesystem_folder,
        bucket_name,
        access_key_id,
        secret_access_key,
        s3_host,
        s3_port,
        ftp_host,
        ftp_user,
        ftp_pass,
    )

    if index_df is None:
        return create_error_obbject(CREATE_INDEX_CONSTITUENTS)

    import pandas as pd

    response = []
    for index, row in index_df.iterrows():
        # Create an empty dictionary for the current row
        row_dict = {}
        # Iterate through the items in the row (column name, value)
        for column, value in row.items():
            # Check if the value is of NAType (or a similar null type)
            # pd.isna() is the standard and most robust way to check for null values in pandas
            if pd.isna(value):
                row_dict[column] = None  # Replace the null value with None
            else:
                row_dict[column] = value

        response.append(row_dict)
    error_collector.clear_collector()
    return OBBject(results=response, provider="open port")


@router.command(
    methods=["GET"],
    examples=[
        PythonEx(
            description="Calibrate the risk model for a given universe.",
            code=[
                'obb.oport.risk_calibration(calibration_universe="nasdaq_index", start_date="2021-01-01", end_date="2021-11-05", source="filesystem", filesystem_folder="tests/test_attribution_md/test_portfolios/indexes")',
            ],
        )
    ],
)
def risk_calibration(
    calibration_universe: Annotated[
        str, Query(description="The name of the universe for calibration.", example="nasdaq_index")
    ] = "nasdaq_index",
    start_date: Annotated[
        str, Query(description="Start date for calibration. Format: 'YYYY-MM-DD'.", example="2021-01-01")
    ] = "",
    end_date: Annotated[
        str, Query(description="End date for calibration. Format: 'YYYY-MM-DD'.", example="2021-11-05")
    ] = "",
    source: Annotated[
        Literal["filesystem", "s3", "ftp"],
        Query(description="The source of data.", example="filesystem"),
    ] = "filesystem",
    filesystem_folder: Annotated[
        Optional[str],
        Query(description="Filesystem path where universe files are stored.", example=None),
    ] = None,
    bucket_name: Annotated[
        Optional[str], Query(description="S3 bucket for calibration output.", example=None)
    ] = None,
    access_key_id: Annotated[
        Optional[str], Query(description="S3 access key.", example=None)
    ] = None,
    secret_access_key: Annotated[
        Optional[str], Query(description="S3 secret key.", example=None)
    ] = None,
    s3_host: Annotated[
        Optional[str], Query(description="S3 host address.", example=None)
    ] = None,
    s3_port: Annotated[
        Optional[str], Query(description="S3 host port.", example=None)
    ] = None,
    ftp_host: Annotated[
        Optional[str], Query(description="FTP host name.", example=None)
    ] = None,
    ftp_user: Annotated[
        Optional[str], Query(description="FTP user name.", example=None)
    ] = None,
    ftp_pass: Annotated[
        Optional[str], Query(description="FTP password.", example=None)
    ] = None,
) -> OBBject:
    """Calibrate the risk model for a given universe."""
    r = risk_calibration_logic(
        calibration_universe,
        start_date,
        end_date,
        source,
        filesystem_folder,
        bucket_name,
        access_key_id,
        secret_access_key,
        s3_host,
        s3_port,
        ftp_host,
        ftp_user,
        ftp_pass,
    )

    if r is None:
        return create_error_obbject(RISK_CALIBRATION)

    response = []
    for index, row in r.iterrows():
        row_dict = {}
        for column, value in row.items():
            row_dict[column] = None if pd.isna(value) else value
        response.append(row_dict)
    error_collector.clear_collector()
    return OBBject(results=response, provider="open port")


@router.command(
    methods=["GET"],
    examples=[
        PythonEx(
            description="Get the factor covariance matrix from the calibrated model.",
            code=[
                'obb.oport.get_factor_covariance(source="filesystem", filesystem_folder="tests/test_attribution_md/test_portfolios/indexes")',
            ],
        )
    ],
)
def get_factor_covariance(
    source: Annotated[
        Literal["filesystem", "s3", "ftp"],
        Query(description="The source of data.", example="filesystem"),
    ] = "filesystem",
    filesystem_folder: Annotated[
        Optional[str],
        Query(description="Filesystem path where the model is stored.", example=None),
    ] = None,
    bucket_name: Annotated[
        Optional[str], Query(description="S3 bucket for the model.", example=None)
    ] = None,
    access_key_id: Annotated[
        Optional[str], Query(description="S3 access key.", example=None)
    ] = None,
    secret_access_key: Annotated[
        Optional[str], Query(description="S3 secret key.", example=None)
    ] = None,
    s3_host: Annotated[
        Optional[str], Query(description="S3 host address.", example=None)
    ] = None,
    s3_port: Annotated[
        Optional[str], Query(description="S3 host port.", example=None)
    ] = None,
    ftp_host: Annotated[
        Optional[str], Query(description="FTP host name.", example=None)
    ] = None,
    ftp_user: Annotated[
        Optional[str], Query(description="FTP user name.", example=None)
    ] = None,
    ftp_pass: Annotated[
        Optional[str], Query(description="FTP password.", example=None)
    ] = None,
) -> OBBject:
    """Get the factor covariance matrix from the calibrated model."""
    r = get_factor_covariance_logic(
        source,
        filesystem_folder,
        bucket_name,
        access_key_id,
        secret_access_key,
        s3_host,
        s3_port,
        ftp_host,
        ftp_user,
        ftp_pass,
    )

    if r is None:
        return create_error_obbject(FACTOR_COVARIANCE)

    response = []
    for index, row in r.iterrows():
        row_dict = {}
        # index is factor name
        row_dict["factor"] = index
        for column, value in row.items():
            row_dict[column] = None if pd.isna(value) else value
        response.append(row_dict)
    error_collector.clear_collector()
    return OBBject(results=response, provider="open port")


@router.command(
    methods=["GET"],
    examples=[
        PythonEx(
            description="Run risk decomposition for a given portfolio.",
            code=[
                'obb.oport.get_portfolio_risk(portfolio_name="test_e2e_stocksplits", input_date="2021-11-05", source="filesystem", filesystem_folder="tests/test_attribution_md/test_portfolios")',
            ],
        )
    ],
)
def get_portfolio_risk(
    portfolio_name: Annotated[
        str, Query(description="The name of the portfolio.", example="test_e2e_stocksplits")
    ] = "test",
    source: Annotated[
        Literal["filesystem", "s3", "ftp"],
        Query(description="The source of data.", example="filesystem"),
    ] = "filesystem",
    filesystem_folder: Annotated[
        Optional[str],
        Query(description="Filesystem path where portfolio and model are stored.", example=None),
    ] = None,
    bucket_name: Annotated[
        Optional[str], Query(description="S3 bucket for portfolio and model.", example=None)
    ] = None,
    access_key_id: Annotated[
        Optional[str], Query(description="S3 access key.", example=None)
    ] = None,
    secret_access_key: Annotated[
        Optional[str], Query(description="S3 secret key.", example=None)
    ] = None,
    s3_host: Annotated[
        Optional[str], Query(description="S3 host address.", example=None)
    ] = None,
    s3_port: Annotated[
        Optional[str], Query(description="S3 host port.", example=None)
    ] = None,
    ftp_host: Annotated[
        Optional[str], Query(description="FTP host name.", example=None)
    ] = None,
    ftp_user: Annotated[
        Optional[str], Query(description="FTP user name.", example=None)
    ] = None,
    ftp_pass: Annotated[
        Optional[str], Query(description="FTP password.", example=None)
    ] = None,
    input_date: Annotated[
        Optional[str], Query(description="The date for risk decomposition. Format: 'YYYY-MM-DD'.", example="2021-11-05")
    ] = None,
) -> OBBject:
    """Run risk decomposition for a given portfolio."""
    r = get_portfolio_risk_logic(
        portfolio_name,
        source,
        filesystem_folder,
        bucket_name,
        access_key_id,
        secret_access_key,
        s3_host,
        s3_port,
        ftp_host,
        ftp_user,
        ftp_pass,
        input_date,
    )

    if r is None:
        return create_error_obbject(PORTFOLIO_RISK)

    response = []
    for index, row in r.iterrows():
        row_dict = {}
        for column, value in row.items():
            row_dict[column] = None if pd.isna(value) else value
        response.append(row_dict)
    error_collector.clear_collector()
    return OBBject(results=response, provider="open port")
