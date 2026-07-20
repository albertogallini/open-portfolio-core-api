### THIS FILE IS AUTO-GENERATED. DO NOT EDIT. ###

from openbb_core.app.static.container import Container
from openbb_core.app.model.obbject import OBBject
from typing import Optional, Literal, Any
from typing_extensions import Annotated
from openbb_core.app.static.utils.decorators import exception_handler, validate

from openbb_core.app.static.utils.filters import filter_inputs


from openbb_core.app.model.field import OpenBBField


class ROUTER_oport(Container):
    """/oport
    create_index
    holdings
    impute_positions
    performance_attribution
    portfolio_totals
    """

    def __repr__(self) -> str:
        return self.__doc__ or ""

    @exception_handler
    @validate
    def create_index(
        self,
        index_ticker: Annotated[
            str, OpenBBField(description="The ticker of the index.")
        ] = "NASDAQ-NDX",
        period: Annotated[
            Optional[str],
            OpenBBField(
                description="the period of the index. Options: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'."
            ),
        ] = "",
        additional_fields: Annotated[
            str,
            OpenBBField(
                description="Fields to fetch for the index constituents from yahoo Finance. Pipes separated."
            ),
        ] = "",
        output_file_name: Annotated[
            str, OpenBBField(description="index file name")
        ] = "",
        source: Annotated[
            Literal["filesystem", "s3", "ftp"],
            OpenBBField(description="The source of data."),
        ] = "filesystem",
        filesystem_folder: Annotated[
            Optional[str],
            OpenBBField(
                description="Filesystem param: the path where the input file with daily analytics is stored [portfolio]."
            ),
        ] = None,
        bucket_name: Annotated[
            Optional[str], OpenBBField(description="S3 bucket for output file.")
        ] = None,
        access_key_id: Annotated[
            Optional[str], OpenBBField(description="S3 access key.")
        ] = None,
        secret_access_key: Annotated[
            Optional[str], OpenBBField(description="S3 secret key.")
        ] = None,
        s3_host: Annotated[
            Optional[str], OpenBBField(description="S3 host address.")
        ] = None,
        s3_port: Annotated[
            Optional[str], OpenBBField(description="S3 host port.")
        ] = None,
        ftp_host: Annotated[
            Optional[str], OpenBBField(description="FTP host name for output file.")
        ] = None,
        ftp_user: Annotated[
            Optional[str], OpenBBField(description="FTP user name.")
        ] = None,
        ftp_pass: Annotated[
            Optional[str], OpenBBField(description="FTP password.")
        ] = None,
        **kwargs: Any
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

        Examples
        --------
        >>> from openbb import obb
        >>> # Fetch index constituents and their details from Yahoo Finance.
        >>> obb.oport.create_index(index_ticker="NASDAQ-NDX", period="1y")
        """  # noqa: E501

        return self._run(
            "/oport/create_index",
            **filter_inputs(
                index_ticker=index_ticker,
                period=period,
                additional_fields=additional_fields,
                output_file_name=output_file_name,
                source=source,
                filesystem_folder=filesystem_folder,
                bucket_name=bucket_name,
                access_key_id=access_key_id,
                secret_access_key=secret_access_key,
                s3_host=s3_host,
                s3_port=s3_port,
                ftp_host=ftp_host,
                ftp_user=ftp_user,
                ftp_pass=ftp_pass,
                **kwargs,
            ),
        )

    @exception_handler
    @validate
    def holdings(
        self,
        portfolio_name: Annotated[
            str, OpenBBField(description="The name of the portfolio.")
        ] = "test",
        start_date: Annotated[
            str, OpenBBField(description="Timeframe start date. Format: 'MM-DD-YYYY'.")
        ] = "",
        end_date: Annotated[
            str, OpenBBField(description="Timeframe end date. Format: 'MM-DD-YYYY'.")
        ] = "",
        classification: Annotated[
            str, OpenBBField(description="Pipes separated field names for bucketing.")
        ] = "",
        characteristics: Annotated[
            str, OpenBBField(description="list of characteristics fields to compute.")
        ] = "",
        performance_risk: Annotated[
            str,
            OpenBBField(
                description="list of performance and ex-post risk fields to compute."
            ),
        ] = "",
        source: Annotated[
            Literal["filesystem", "s3", "ftp"],
            OpenBBField(description="The source of data."),
        ] = "filesystem",
        benchmark_name: Annotated[
            Optional[str], OpenBBField(description="The name of the benchmark.")
        ] = "",
        filesystem_folder: Annotated[
            Optional[str],
            OpenBBField(
                description="Filesystem param: the path where the input file is stored [portfolio]."
            ),
        ] = None,
        filesystem_folder_benchmark: Annotated[
            Optional[str],
            OpenBBField(
                description="Filesystem param: the path where the input file is stored [benchmark]."
            ),
        ] = None,
        bucket_name: Annotated[
            Optional[str], OpenBBField(description="S3 bucket for output file.")
        ] = None,
        bucket_name_benchmark: Annotated[
            Optional[str],
            OpenBBField(description="S3 bucket for benchmark output file."),
        ] = None,
        access_key_id: Annotated[
            Optional[str], OpenBBField(description="S3 access key.")
        ] = None,
        secret_access_key: Annotated[
            Optional[str], OpenBBField(description="S3 secret key.")
        ] = None,
        s3_host: Annotated[
            Optional[str], OpenBBField(description="S3 host address.")
        ] = None,
        s3_port: Annotated[
            Optional[str], OpenBBField(description="S3 host port.")
        ] = None,
        ftp_host: Annotated[
            Optional[str], OpenBBField(description="FTP host name for output file.")
        ] = None,
        ftp_user: Annotated[
            Optional[str], OpenBBField(description="FTP user name.")
        ] = None,
        ftp_pass: Annotated[
            Optional[str], OpenBBField(description="FTP password.")
        ] = None,
        **kwargs: Any
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
            list of characteristics fields to compute
        performance_risk: str, optional
            list of performance and ex-post risk fields to compute
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

        Examples
        --------
        >>> from openbb import obb
        >>> # Read the daily imputed positions and compute the holdings analytics (comprising characteristics and performance) for each day in the timeframe
        >>> obb.oport.holdings(portfolio_name=config.output_file_name,benchmark_name="oex_idx",classification="ccy|securityType",start_date="10-01-2024",end_date="10-28-2024",characteristics="Diluted_EPS|Basic_Average_Shares|P-E",filesystem_folder=config.filesystem_folder,source="filesystem")
        """  # noqa: E501

        return self._run(
            "/oport/holdings",
            **filter_inputs(
                portfolio_name=portfolio_name,
                start_date=start_date,
                end_date=end_date,
                classification=classification,
                characteristics=characteristics,
                performance_risk=performance_risk,
                source=source,
                benchmark_name=benchmark_name,
                filesystem_folder=filesystem_folder,
                filesystem_folder_benchmark=filesystem_folder_benchmark,
                bucket_name=bucket_name,
                bucket_name_benchmark=bucket_name_benchmark,
                access_key_id=access_key_id,
                secret_access_key=secret_access_key,
                s3_host=s3_host,
                s3_port=s3_port,
                ftp_host=ftp_host,
                ftp_user=ftp_user,
                ftp_pass=ftp_pass,
                **kwargs,
            ),
        )

    @exception_handler
    @validate
    def impute_positions(
        self,
        output_file_name: Annotated[
            str,
            OpenBBField(description="The file name containing the imputed positions."),
        ] = "positions",
        portfolio_base_ccy: Annotated[
            str, OpenBBField(description="The portfolio base currency.")
        ] = "USD",
        transaction_file: Annotated[
            str,
            OpenBBField(
                description="The input file containing the portfolio transactions."
            ),
        ] = "transactions.xls",
        transactions_field_mapping: Annotated[
            str, OpenBBField(description="The field mapping for transactions.")
        ] = "transactions_field.json",
        append: Annotated[
            bool,
            OpenBBField(
                description="If true, appends new transactions to existing portfolio."
            ),
        ] = False,
        select_exchange: Annotated[
            bool,
            OpenBBField(
                description="If true, prompts to select exchange code when needed."
            ),
        ] = False,
        additional_fields: Annotated[
            Optional[str],
            OpenBBField(description="Pipes separated field names for bucketing."),
        ] = "",
        source: Annotated[
            Literal["filesystem", "s3", "ftp"],
            OpenBBField(description="The source of data."),
        ] = "filesystem",
        filesystem_folder: Annotated[
            Optional[str],
            OpenBBField(description="The path where the output file is stored."),
        ] = None,
        bucket_name: Annotated[
            Optional[str], OpenBBField(description="S3 bucket for output file.")
        ] = None,
        access_key_id: Annotated[
            Optional[str], OpenBBField(description="S3 access key.")
        ] = None,
        secret_access_key: Annotated[
            Optional[str], OpenBBField(description="S3 secret key.")
        ] = None,
        s3_host: Annotated[
            Optional[str], OpenBBField(description="S3 host address.")
        ] = None,
        s3_port: Annotated[
            Optional[str], OpenBBField(description="S3 host port.")
        ] = None,
        ftp_host: Annotated[
            Optional[str], OpenBBField(description="FTP host name for output file.")
        ] = None,
        ftp_user: Annotated[
            Optional[str], OpenBBField(description="FTP user name.")
        ] = None,
        ftp_pass: Annotated[
            Optional[str], OpenBBField(description="FTP password.")
        ] = None,
        **kwargs: Any
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

        Examples
        --------
        >>> from openbb import obb
        >>> # Impute positions from Transactions.
        >>> obb.oport.impute_positions(source="filesystem",transaction_file="t1.xls",transactions_field_mapping="transactions_field.json",filesystem_folder=output_folder,output_file_name=config.output_file_name,portfolio_base_ccy="EUR")
        """  # noqa: E501

        return self._run(
            "/oport/impute_positions",
            **filter_inputs(
                output_file_name=output_file_name,
                portfolio_base_ccy=portfolio_base_ccy,
                transaction_file=transaction_file,
                transactions_field_mapping=transactions_field_mapping,
                append=append,
                select_exchange=select_exchange,
                additional_fields=additional_fields,
                source=source,
                filesystem_folder=filesystem_folder,
                bucket_name=bucket_name,
                access_key_id=access_key_id,
                secret_access_key=secret_access_key,
                s3_host=s3_host,
                s3_port=s3_port,
                ftp_host=ftp_host,
                ftp_user=ftp_user,
                ftp_pass=ftp_pass,
                **kwargs,
            ),
        )

    @exception_handler
    @validate
    def performance_attribution(
        self,
        portfolio_name: Annotated[
            str, OpenBBField(description="The name of the portfolio.")
        ] = "test",
        start_date: Annotated[
            str, OpenBBField(description="Timeframe start date. Format: 'MM-DD-YYYY'.")
        ] = "",
        end_date: Annotated[
            str, OpenBBField(description="Timeframe end date. Format: 'MM-DD-YYYY'.")
        ] = "",
        classification: Annotated[
            str, OpenBBField(description="Pipes separated field names for bucketing.")
        ] = "",
        trend_analysis: Annotated[
            bool,
            OpenBBField(
                description="Whether to run trend analysis on the performance attributin portfolio vs benchmark."
            ),
        ] = False,
        source: Annotated[
            Literal["filesystem", "s3", "ftp"],
            OpenBBField(description="The source of data."),
        ] = "filesystem",
        benchmark_name: Annotated[
            str, OpenBBField(description="The name of the benchmark.")
        ] = "",
        filesystem_folder: Annotated[
            Optional[str],
            OpenBBField(
                description="Filesystem param: the path where the input file with daily analytics is stored [portfolio]."
            ),
        ] = None,
        filesystem_folder_benchmark: Annotated[
            Optional[str],
            OpenBBField(
                description="Filesystem param: the path where the input file with daily analytics is stored [benchmark]."
            ),
        ] = None,
        bucket_name: Annotated[
            Optional[str], OpenBBField(description="S3 bucket for output file.")
        ] = None,
        access_key_id: Annotated[
            Optional[str], OpenBBField(description="S3 access key.")
        ] = None,
        secret_access_key: Annotated[
            Optional[str], OpenBBField(description="S3 secret key.")
        ] = None,
        s3_host: Annotated[
            Optional[str], OpenBBField(description="S3 host address.")
        ] = None,
        s3_port: Annotated[
            Optional[str], OpenBBField(description="S3 host port.")
        ] = None,
        ftp_host: Annotated[
            Optional[str], OpenBBField(description="FTP host name for output file.")
        ] = None,
        ftp_user: Annotated[
            Optional[str], OpenBBField(description="FTP user name.")
        ] = None,
        ftp_pass: Annotated[
            Optional[str], OpenBBField(description="FTP password.")
        ] = None,
        **kwargs: Any
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

        Examples
        --------
        >>> from openbb import obb
        >>> # Read the daily holdings, compute the performance daily of portfolio and benchmark and run the performance attribution on top of it and the input classification strategy.
        >>> obb.oport.performance_attribution(portfolio_name=config.output_file_name,filesystem_folder=config.filesystem_folder,source=config.source,start_date="10-01-2024",end_date="10-28-2024")
        """  # noqa: E501

        return self._run(
            "/oport/performance_attribution",
            **filter_inputs(
                portfolio_name=portfolio_name,
                start_date=start_date,
                end_date=end_date,
                classification=classification,
                trend_analysis=trend_analysis,
                source=source,
                benchmark_name=benchmark_name,
                filesystem_folder=filesystem_folder,
                filesystem_folder_benchmark=filesystem_folder_benchmark,
                bucket_name=bucket_name,
                access_key_id=access_key_id,
                secret_access_key=secret_access_key,
                s3_host=s3_host,
                s3_port=s3_port,
                ftp_host=ftp_host,
                ftp_user=ftp_user,
                ftp_pass=ftp_pass,
                **kwargs,
            ),
        )

    @exception_handler
    @validate
    def portfolio_totals(
        self,
        portfolio_name: Annotated[
            str, OpenBBField(description="The name of the portfolio.")
        ] = "test",
        plot_field: Annotated[
            Optional[str],
            OpenBBField(
                description="The name of the field to plot in a single chart mode."
            ),
        ] = "",
        start_date: Annotated[
            str, OpenBBField(description="Timeframe start date. Format: 'MM-DD-YYYY'.")
        ] = "",
        end_date: Annotated[
            str, OpenBBField(description="Timeframe end date. Format: 'MM-DD-YYYY'.")
        ] = "",
        classification: Annotated[
            str, OpenBBField(description="Pipes separated field names for bucketing.")
        ] = "",
        source: Annotated[
            Literal["filesystem", "s3", "ftp"],
            OpenBBField(description="The source of data."),
        ] = "filesystem",
        benchmark_name: Annotated[
            Optional[str], OpenBBField(description="The name of the benchmark.")
        ] = "",
        filesystem_folder: Annotated[
            Optional[str],
            OpenBBField(
                description="Filesystem param: the path where the input file with daily analytics is stored [portfolio]."
            ),
        ] = None,
        filesystem_folder_benchmark: Annotated[
            Optional[str],
            OpenBBField(
                description="Filesystem param: the path where the input file with daily analytics is stored [benchmark]."
            ),
        ] = None,
        bucket_name: Annotated[
            Optional[str], OpenBBField(description="S3 bucket for output file.")
        ] = None,
        access_key_id: Annotated[
            Optional[str], OpenBBField(description="S3 access key.")
        ] = None,
        secret_access_key: Annotated[
            Optional[str], OpenBBField(description="S3 secret key.")
        ] = None,
        s3_host: Annotated[
            Optional[str], OpenBBField(description="S3 host address.")
        ] = None,
        s3_port: Annotated[
            Optional[str], OpenBBField(description="S3 host port.")
        ] = None,
        ftp_host: Annotated[
            Optional[str], OpenBBField(description="FTP host name for output file.")
        ] = None,
        ftp_user: Annotated[
            Optional[str], OpenBBField(description="FTP user name.")
        ] = None,
        ftp_pass: Annotated[
            Optional[str], OpenBBField(description="FTP password.")
        ] = None,
        **kwargs: Any
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

        Examples
        --------
        >>> from openbb import obb
        >>> # Read the daily holdings performance and plot the top/bucket level performance analytics according to the input timeframe.
        >>> obb.oport.portfolio_totals(portfolio_name=config.output_file_name,filesystem_folder=config.filesystem_folder,source=config.source,start_date="10-01-2024",end_date="10-28-2024")
        """  # noqa: E501

        return self._run(
            "/oport/portfolio_totals",
            **filter_inputs(
                portfolio_name=portfolio_name,
                plot_field=plot_field,
                start_date=start_date,
                end_date=end_date,
                classification=classification,
                source=source,
                benchmark_name=benchmark_name,
                filesystem_folder=filesystem_folder,
                filesystem_folder_benchmark=filesystem_folder_benchmark,
                bucket_name=bucket_name,
                access_key_id=access_key_id,
                secret_access_key=secret_access_key,
                s3_host=s3_host,
                s3_port=s3_port,
                ftp_host=ftp_host,
                ftp_user=ftp_user,
                ftp_pass=ftp_pass,
                **kwargs,
            ),
        )
