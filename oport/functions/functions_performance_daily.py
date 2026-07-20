from oport import fields

import pandas as pd
import numpy as np

from datetime import datetime
from oport.ticker_resolver import *
from oport.metadata.asset_types import *
from oport import error_collector


def compute_transaction_pnl(
    p: pd.DataFrame,
    daycount: int,
    transactions: pd.DataFrame = None,
    verbose: bool = False,
) -> None:
    """
    compute transactions pnl from holdings snapshot an transactions on the snapshot date
    """

    if transactions is not None:

        p[fields.FIELD_EVAL_PNL] = p[fields.FIELD_EVAL_PNL].astype(float)
        p[fields.FIELD_EVAL_TRANSACTION_PNL] = p[
            fields.FIELD_EVAL_TRANSACTION_PNL
        ].astype(float)
        p[fields.FIELD_EVAL_TRANSACTION_PNL_LCL_CCY] = p[
            fields.FIELD_EVAL_TRANSACTION_PNL_LCL_CCY
        ].astype(float)

        holdings_date = p.at[daycount, fields.FIELD_DATE]
        ticker_transactions = transactions[
            transactions[fields.TRANSACTION_FIELD_TR_DATE] == holdings_date
        ]
        transaction_value_open = 0
        transaction_value_open_acc = 0
        holds_pnl = p.at[daycount, fields.FIELD_EVAL_PNL]

        for _, t in ticker_transactions.iterrows():

            if t[fields.TRANSACTION_FIELD_PRICE] == 0:
                continue

            transaction_value = (
                t[fields.TRANSACTION_FIELD_QUANTITIY]
                * t[fields.TRANSACTION_FIELD_PRICE]
            )
            transaction_value_open_acc += abs(transaction_value)
            eod_mktvalue = (
                t[fields.TRANSACTION_FIELD_QUANTITIY]
                * p.at[daycount, fields.FIELD_CLOSE]
            )
            transaction_pnl = 0

            if t[fields.TRANSACTION_FIELD_BUY_SELL] == fields.VALUE_TRANSACTION_BUY:
                transaction_value_open += transaction_value
                transaction_pnl = eod_mktvalue - transaction_value
                p.at[daycount, fields.FIELD_EVAL_PNL_LCL_CCY] += transaction_pnl
                p.at[daycount, fields.FIELD_EVAL_PNL] += (
                    transaction_pnl * p.at[daycount, fields.FIELD_FX_RATES]
                )
                p.at[
                    daycount, fields.FIELD_EVAL_TRANSACTION_PNL_LCL_CCY
                ] += transaction_pnl
                p.at[daycount, fields.FIELD_EVAL_TRANSACTION_PNL] += (
                    transaction_pnl * p.at[daycount, fields.FIELD_FX_RATES]
                )
                transaction_type = ("Buy", transaction_value)

            if (
                t[fields.TRANSACTION_FIELD_BUY_SELL] == fields.VALUE_TRANSACTION_SELL
            ):  # TODO: generalize 'V'
                transaction_value_open -= transaction_value
                transaction_pnl = (transaction_value - eod_mktvalue) * p.at[
                    daycount, fields.FIELD_FX_RATES
                ]
                p.at[daycount, fields.FIELD_EVAL_PNL_LCL_CCY] += transaction_pnl
                p.at[daycount, fields.FIELD_EVAL_PNL] += (
                    transaction_pnl * p.at[daycount, fields.FIELD_FX_RATES]
                )
                p.at[
                    daycount, fields.FIELD_EVAL_TRANSACTION_PNL_LCL_CCY
                ] += transaction_pnl
                p.at[daycount, fields.FIELD_EVAL_TRANSACTION_PNL] += (
                    transaction_pnl * p.at[daycount, fields.FIELD_FX_RATES]
                )
                transaction_type = ("Sell", transaction_value)

            ticker = ticker_transactions[fields.FIELD_TICKER].values[0]
            if verbose:
                print(
                    "{} {} {} -> transcation pnl: {}  hold pnl {}  mvbod {}  trbasis {}, quantity {}, close price {}, trans. price {}".format(
                        ticker,
                        holdings_date,
                        transaction_type,
                        transaction_pnl,
                        holds_pnl,
                        p.at[daycount, fields.FIELD_EVAL_MKTVALUE_BOD],
                        transaction_value_open,
                        t[fields.TRANSACTION_FIELD_QUANTITIY],
                        p.at[daycount, fields.FIELD_CLOSE],
                        t[fields.TRANSACTION_FIELD_PRICE],
                    )
                )

        if (
            p.at[daycount, fields.FIELD_EVAL_TOTAL_RETURN_BASIS] == 0
            and not ticker_transactions.empty
        ):
            p.at[daycount, fields.FIELD_EVAL_TOTAL_RETURN_BASIS_LCL_CCY] = abs(
                transaction_value_open_acc
            )
            p.at[daycount, fields.FIELD_EVAL_TOTAL_RETURN_BASIS] = (
                abs(transaction_value_open_acc) * p.at[daycount, fields.FIELD_FX_RATES]
            )


def compute_return(p: pd.DataFrame, daycount: int) -> None:
    """
    compute return and return component as  pnl / total return basis.
    """

    if (p.loc[daycount, fields.FIELD_EVAL_TOTAL_RETURN_BASIS]) != 0:
        p.at[daycount, fields.FIELD_EVAL_RETURN] = (
            p.loc[daycount, fields.FIELD_EVAL_PNL]
            / p.loc[daycount, fields.FIELD_EVAL_TOTAL_RETURN_BASIS]
        ) + 1
        p.at[daycount, fields.FIELD_EVAL_RETURN_LCL_CCY] = (
            p.loc[daycount, fields.FIELD_EVAL_PNL_LCL_CCY]
            / p.loc[daycount, fields.FIELD_EVAL_TOTAL_RETURN_BASIS_LCL_CCY]
        ) + 1
        p.at[daycount, fields.FIELD_EVAL_CCY_RETURN] = (
            p.loc[daycount, fields.FIELD_EVAL_RETURN]
            - p.loc[daycount, fields.FIELD_EVAL_RETURN_LCL_CCY]
            + 1
        )  # both are 1 based.

        p.at[daycount, fields.FIELD_EVAL_TRANSACTION_RETURN] = (
            p.loc[daycount, fields.FIELD_EVAL_TRANSACTION_PNL]
            / p.loc[daycount, fields.FIELD_EVAL_TOTAL_RETURN_BASIS]
        ) + 1
        p.at[daycount, fields.FIELD_EVAL_TRANSACTION_RETURN_LCL_CCY] = (
            p.loc[daycount, fields.FIELD_EVAL_TRANSACTION_PNL_LCL_CCY]
            / p.loc[daycount, fields.FIELD_EVAL_TOTAL_RETURN_BASIS_LCL_CCY]
        ) + 1

        if p.at[daycount, fields.FIELD_EVAL_POSITION_BASIS] != 0:
            p.at[daycount, fields.FIELD_EVAL_RETURN_HLDS] = (
                (
                    p.loc[daycount, fields.FIELD_EVAL_PNL]
                    - p.loc[daycount, fields.FIELD_EVAL_TRANSACTION_PNL]
                )
                / p.loc[daycount, fields.FIELD_EVAL_POSITION_BASIS]
            ) + 1
            p.at[daycount, fields.FIELD_EVAL_RETURN_HLDS_LCL_CCY] = (
                (
                    p.loc[daycount, fields.FIELD_EVAL_PNL_LCL_CCY]
                    - p.loc[daycount, fields.FIELD_EVAL_TRANSACTION_PNL_LCL_CCY]
                )
                / p.loc[daycount, fields.FIELD_EVAL_POSITION_BASIS_LCL_CCY]
            ) + 1
        else:
            p.at[daycount, fields.FIELD_EVAL_RETURN_HLDS] = 1.0
            p.at[daycount, fields.FIELD_EVAL_RETURN_HLDS_LCL_CCY] = 1.0
    else:
        p.at[daycount, fields.FIELD_EVAL_RETURN] = 1.0
        p.at[daycount, fields.FIELD_EVAL_RETURN_LCL_CCY] = 1.0
        p.at[daycount, fields.FIELD_EVAL_CCY_RETURN] = 1.0
        p.at[daycount, fields.FIELD_EVAL_RETURN_HLDS] = 1.0
        p.at[daycount, fields.FIELD_EVAL_RETURN_HLDS_LCL_CCY] = 1.0
        p.at[daycount, fields.FIELD_EVAL_TRANSACTION_RETURN_LCL_CCY] = 1.0
        p.at[daycount, fields.FIELD_EVAL_TRANSACTION_RETURN] = 1.0


def set_mark2market(p: pd.DataFrame, daycount: int) -> None:
    """
    init market value fields
    """
    futures_types = list(
        [
            ASSET_TYPE_PHYSICAL_INDEX_FUTURE,
            ASSET_TYPE_CURRENCY_FUTURE,
            ASSET_TYPE_FINANCIAL_COMMODITY_FUTURE,
            ASSET_TYPE_GENERIC_INDEX_FUTURE,
            ASSET_TYPE_FINANCIAL_INDEX_FUTURE,
            ASSET_TYPE_GENERIC_CURRENCY_FUTURE,
        ]
    )

    if p.loc[daycount, fields.FIELD_SECURITY_TYPE] in futures_types:
        p[fields.FIELD_EVAL_MKTVALUE_LCL_CCY] = 0.0
        p[fields.FIELD_EVAL_MKTVALUE] = 0.0
        p[fields.FIELD_EVAL_M2M_BOD] = 0.0
        p[fields.FIELD_EVAL_M2M_BOD_LCL_CCY] = 0.0


def performance_daily_single_position(
    p: pd.DataFrame,
    transactions: pd.DataFrame = None,
    classification: list = None,
) -> pd.DataFrame:

    if p.empty:
        return pd.DataFrame()

    daycount = 0

    # Data adjustment
    p[fields.FIELD_CLOSE] = p[fields.FIELD_CLOSE].ffill()
    p[fields.FIELD_FX_RATES] = p[fields.FIELD_FX_RATES].ffill()
    p[fields.FIELD_FX_RATES] = p[fields.FIELD_FX_RATES].fillna(1)
    p[fields.FIELD_EVAL_QUANTITY] = p[fields.FIELD_EVAL_QUANTITY].ffill()

    if fields.FIELD_DIVIDENDS in p.columns:
        p.replace({fields.FIELD_DIVIDENDS: np.nan}, value=0.0, inplace=True)
    else:
        p[fields.FIELD_DIVIDENDS] = np.zeros(len(p))

    has_split = False
    if fields.FIELD_STOCK_SPLIT in p.columns:
        has_split = True
        p[fields.FIELD_STOCK_SPLIT] = p[fields.FIELD_STOCK_SPLIT].ffill()

    for r in p.iterrows():

        # position inception:
        if daycount == 0:
            p[fields.FIELD_EVAL_MKTVALUE_LCL_CCY] = (
                float(p.at[daycount, fields.FIELD_EVAL_QUANTITY])
                * p.at[daycount, fields.FIELD_CLOSE]
            )
            p[fields.FIELD_EVAL_MKTVALUE_BOD_LCL_CCY] = 0.0
            p[fields.FIELD_EVAL_MKTVALUE_EOD_LCL_CCY] = 0.0
            p[fields.FIELD_EVAL_TOTAL_RETURN_BASIS] = 0.0
            p[fields.FIELD_EVAL_TOTAL_RETURN_BASIS_LCL_CCY] = 0.0
            p[fields.FIELD_EVAL_POSITION_BASIS] = 0.0
            p[fields.FIELD_EVAL_POSITION_BASIS_LCL_CCY] = 0.0
            p[fields.FIELD_EVAL_MKTVALUE] = (
                p[fields.FIELD_EVAL_MKTVALUE_LCL_CCY]
                * p.at[daycount, fields.FIELD_FX_RATES]
            )
            p[fields.FIELD_EVAL_MKTVALUE_BOD] = 0.0
            p[fields.FIELD_EVAL_MKTVALUE_EOD] = 0.0
            p[fields.FIELD_EVAL_M2M_BOD_LCL_CCY] = 0.0
            p[fields.FIELD_EVAL_M2M_BOD] = 0.0

            p[fields.FIELD_EVAL_PNL] = 0.0
            p[fields.FIELD_EVAL_INCOME_PNL] = 0.0
            p[fields.FIELD_EVAL_TRANSACTION_PNL] = 0.0
            p[fields.FIELD_EVAL_PNL_LCL_CCY] = 0.0
            p[fields.FIELD_EVAL_INCOME_PNL_LCL_CCY] = 0.0
            p[fields.FIELD_EVAL_TRANSACTION_PNL_LCL_CCY] = 0.0

            p[fields.FIELD_EVAL_RETURN_LCL_CCY] = 1.0
            p[fields.FIELD_EVAL_RETURN] = 1.0
            p[fields.FIELD_EVAL_CCY_RETURN] = 1.0

            set_mark2market(p, daycount)
            compute_transaction_pnl(p, daycount, transactions)
            compute_return(p, daycount)

            daycount += 1
            continue

        split_adj_factor_eod = 1
        if daycount < len(p) - 1 and has_split:
            split_adj_factor_eod = (
                p.at[daycount, fields.FIELD_STOCK_SPLIT]
                / p.at[daycount + 1, fields.FIELD_STOCK_SPLIT]
            )

        p.at[daycount, fields.FIELD_EVAL_MKTVALUE_LCL_CCY] = (
            float(p.at[daycount, fields.FIELD_EVAL_QUANTITY])
            * p.at[daycount, fields.FIELD_CLOSE]
        )

        p.at[daycount, fields.FIELD_EVAL_MKTVALUE_BOD_LCL_CCY] = (
            float(p.at[daycount - 1, fields.FIELD_EVAL_QUANTITY])
            * p.at[daycount - 1, fields.FIELD_CLOSE]
        )

        p[fields.FIELD_EVAL_M2M_BOD_LCL_CCY] = p[fields.FIELD_EVAL_MKTVALUE_BOD_LCL_CCY]

        p.at[daycount, fields.FIELD_EVAL_MKTVALUE_EOD_LCL_CCY] = (
            float(p.at[daycount - 1, fields.FIELD_EVAL_QUANTITY])
            * split_adj_factor_eod
            * (
                p.at[daycount, fields.FIELD_CLOSE]
                + p.at[daycount, fields.FIELD_DIVIDENDS]
            )
        )

        p.at[daycount, fields.FIELD_EVAL_POSITION_BASIS_LCL_CCY] = (
            float(p.at[daycount - 1, fields.FIELD_EVAL_QUANTITY])
            * p.at[daycount - 1, fields.FIELD_CLOSE]
        )

        p.at[daycount, fields.FIELD_EVAL_TOTAL_RETURN_BASIS_LCL_CCY] = p.at[
            daycount, fields.FIELD_EVAL_POSITION_BASIS_LCL_CCY
        ]

        p.at[daycount, fields.FIELD_EVAL_PNL_LCL_CCY] = (
            p.loc[daycount, fields.FIELD_EVAL_MKTVALUE_EOD_LCL_CCY]
            - p.loc[daycount, fields.FIELD_EVAL_MKTVALUE_BOD_LCL_CCY]
        )
        p.at[daycount, fields.FIELD_EVAL_INCOME_PNL_LCL_CCY] = (
            float(p.at[daycount - 1, fields.FIELD_EVAL_QUANTITY])
            * split_adj_factor_eod
            * p.at[daycount, fields.FIELD_DIVIDENDS]
        )

        p.at[daycount, fields.FIELD_EVAL_MKTVALUE] = (
            p.at[daycount, fields.FIELD_EVAL_MKTVALUE_LCL_CCY]
            * p.at[daycount, fields.FIELD_FX_RATES]
        )

        p.at[daycount, fields.FIELD_EVAL_MKTVALUE_BOD] = (
            p.at[daycount, fields.FIELD_EVAL_MKTVALUE_BOD_LCL_CCY]
            * p.at[daycount - 1, fields.FIELD_FX_RATES]
        )

        p[fields.FIELD_EVAL_M2M_BOD] = p[fields.FIELD_EVAL_MKTVALUE_BOD]

        p.at[daycount, fields.FIELD_EVAL_MKTVALUE_EOD] = (
            p.at[daycount, fields.FIELD_EVAL_MKTVALUE_EOD_LCL_CCY]
            * p.at[daycount, fields.FIELD_FX_RATES]
        )

        p.at[daycount, fields.FIELD_EVAL_POSITION_BASIS] = (
            float(p.at[daycount, fields.FIELD_EVAL_POSITION_BASIS_LCL_CCY])
            * p.at[daycount - 1, fields.FIELD_FX_RATES]
        )

        p.at[daycount, fields.FIELD_EVAL_TOTAL_RETURN_BASIS] = p.at[
            daycount, fields.FIELD_EVAL_POSITION_BASIS
        ]

        p.at[daycount, fields.FIELD_EVAL_PNL] = (
            p.loc[daycount, fields.FIELD_EVAL_MKTVALUE_EOD]
            - p.loc[daycount, fields.FIELD_EVAL_MKTVALUE_BOD]
        )
        p.at[daycount, fields.FIELD_EVAL_INCOME_PNL] = (
            float(p.at[daycount - 1, fields.FIELD_EVAL_QUANTITY])
            * split_adj_factor_eod
            * p.at[daycount, fields.FIELD_DIVIDENDS]
            * p.at[daycount, fields.FIELD_FX_RATES]
        )

        set_mark2market(p, daycount)
        compute_transaction_pnl(p, daycount, transactions)
        compute_return(p, daycount)

        daycount += 1

    if classification is None:
        return p[
            fields.BASIC_POSITON_DESCR_FIELDS + fields.BASIC_PERFORMANCE_DAILY_FIELDS
        ]
    else:
        for c in classification:
            if c not in p.columns:
                p[c] = fields.TREE_NOT_CLASSIFIED_VALUE

        combined_list = (
            fields.BASIC_POSITON_DESCR_FIELDS
            + fields.BASIC_PERFORMANCE_DAILY_FIELDS
            + classification
        )
        unique_values = list(set(combined_list))
        return p[unique_values]


def performance_daily_mfiles(
    config: const_and_utils.Config,
    transactions: pd.DataFrame,
    prefix: str,
    files: list,
    start_date: datetime.date = None,
    end_date: datetime.date = None,
    classification: list = None,
) -> pd.DataFrame:
    """
    Loads holdings data based on the provided configuration.

    Args:
        config (Config): The configuration object containing necessary parameters to load hlds files.
        transactions(pd.DataFrame): the transaction file with the valid holdings
        prefix(str): the holdings prefix files
        file(list): the holding file names


    Returns:
        dict: A pd.DataFrame containing the loaded holdings data.
    """
    hlds = pd.DataFrame()

    transactions_df = None
    if transactions is not None and not transactions.empty:
        transactions_df = transactions[
            [
                fields.TRANSACTION_FIELD_TR_DATE,
                fields.TRANSACTION_FIELD_BUY_SELL,
                fields.TRANSACTION_FIELD_QUANTITIY,
                fields.TRANSACTION_FIELD_PRICE,
                fields.FIELD_TICKER,
            ]
        ]

    # Load position files and compute P&L
    current_file = ""
    err_collector = error_collector.get_collector()
    for f in files:
        try:

            f_ticker = f.replace(prefix, "").replace(".csv", "")
            print(
                "processing {}                                                ".format(
                    f_ticker
                ),
                end="\r",
            )
            ticker_transactions = None
            if transactions_df is not None:
                ticker_transactions = transactions_df[
                    transactions_df.ticker == f_ticker
                ]

            current_file = config.filesystem_folder + "/" + f
            hlds_t = performance_daily_single_position(
                p=const_and_utils.read_csv(
                    config=config,
                    file_path=current_file,
                    start_date=start_date,
                    end_date=end_date,
                ),
                transactions=ticker_transactions,
                classification=classification,
            )
            if not hlds_t.empty:
                hlds = pd.concat([hlds, hlds_t], ignore_index=True)

        except Exception as e:
            err_collector.record_error(
                operation_name="Loading Holdings Files", error_details=str(e)
            )
            print("Error in loading holdgins files {} : {} ".format(current_file, e))
            continue

    # Write holdings data to CSV
    if not hlds.empty:
        hlds = hlds.reset_index(drop=True)
        const_and_utils.write_csv(
            config=config,
            df=hlds,
            file_path=config.filesystem_folder
            + "/"
            + config.output_file_name
            + "_"
            + const_and_utils.DAILY_PERFORMANCE,
        )

    return hlds


def performance_daily_single_file(
    config: const_and_utils.Config,
    prefix: str,
    transactions: pd.DataFrame = None,
    start_date: datetime.date = None,
    end_date: datetime.date = None,
    classification: list = None,
) -> pd.DataFrame:
    """
     Loads holdings data based on the provided configuration.

    Args:
        config (Config): The configuration object containing necessary parameters to load hlds files.
        transactions(pd.DataFrame): the transaction file with the valid holdings
        prefix(str): the holdings prefix files
        prefix(str): the holding file prefix name.

    Returns:
        dict: A pd.DataFrame containing the loaded holdings data.
    """
    hlds = pd.DataFrame()
    err_collector = error_collector.get_collector()
    try:
        transactions_df = None
        if transactions is not None and not transactions.empty:
            transactions_df = transactions[
                [
                    fields.TRANSACTION_FIELD_TR_DATE,
                    fields.TRANSACTION_FIELD_BUY_SELL,
                    fields.TRANSACTION_FIELD_QUANTITIY,
                    fields.TRANSACTION_FIELD_PRICE,
                    fields.FIELD_TICKER,
                ]
            ]

        holdings = const_and_utils.read_csv(
            config=config,
            file_path=config.filesystem_folder + "/" + prefix + ".csv",
            start_date=start_date,
            end_date=end_date,
        )

        if holdings.empty:
            return hlds
        # Load position files and compute P&L
        for ticker, h in holdings.groupby([fields.FIELD_TICKER]):
            ticker_transactions = None
            if transactions_df is not None:
                ticker_transactions = transactions_df[transactions_df.ticker == ticker]
            print(
                "processing {}                                                ".format(
                    ticker
                ),
                end="\r",
            )
            h = performance_daily_single_position(
                p=h.sort_values(fields.FIELD_DATE).reset_index(drop=True),
                transactions=ticker_transactions,
                classification=classification,
            )
            hlds = pd.concat([hlds, h], ignore_index=True)

        # Write holdings data to CSV
        if not hlds.empty:
            hlds = hlds.reset_index(drop=True)
            const_and_utils.write_csv(
                config=config,
                df=hlds,
                file_path=config.filesystem_folder
                + "/"
                + prefix.replace(const_and_utils.DAILY_HOLDINGS_PREFIX, "")
                + const_and_utils.DAILY_PERFORMANCE,
            )
        return hlds
    except Exception as e:
        err_collector.record_error(
            operation_name="Loading Holdings File", error_details=str(e)
        )
        print(
            "Error in loading holdgins file {} : {} ".format(
                config.filesystem_folder + "/" + prefix + ".csv", e
            )
        )
        return hlds


import pandas as pd


def compute_performance_daily(
    config: const_and_utils.Config,
    prefix: str,
    transactions: pd.DataFrame = None,
    start_date: datetime.date = None,
    end_date: datetime.date = None,
    holdings_files: list = None,
    classification: list = None,
) -> pd.DataFrame:

    print(
        "Performance daily for {} from {} to {}.".format(prefix, start_date, end_date)
    )
    hlds = pd.DataFrame()
    if len(holdings_files) > 0:
        # Load holdings files and compute PnLs
        hlds = performance_daily_mfiles(
            config=config,
            transactions=transactions,
            start_date=start_date,
            end_date=end_date,
            prefix=prefix,
            files=holdings_files,
            classification=classification,
        )
    else:
        # Load holdings from a single file and compute PnLs.
        hlds = performance_daily_single_file(
            config=config,
            transactions=transactions,
            start_date=start_date,
            end_date=end_date,
            prefix=prefix,
            classification=classification,
        )

    return hlds
