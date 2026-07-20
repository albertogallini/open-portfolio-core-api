from oport.const_and_utils import io_config
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import os
import warnings

from oport.ticker_resolver import *
from oport import const_and_utils
from oport import fields
from oport import error_collector


def generate_transaction_file(
    config: const_and_utils.Config,
    transaction_file_path: str = "./transactions.xls",
    fields_config: dict = {},
    select_exchange: bool = False,
    mode: str = "append",
):
    try:
        error_collector.get_collector()
        if mode != "append":
            print(
                "Full dump mode: previous data for {} will be lost.".format(
                    config.output_file_name
                )
            )

        transactions_input_df = const_and_utils.read_xls(
            config=config, file_path=transaction_file_path
        )

        transactions_fields = [
            fields.TRANSACTION_FIELD_ISIN,
            fields.FIELD_FIGI,
            fields.TRANSACTION_FIELD_TR_DATE,
            fields.TRANSACTION_FIELD_BUY_SELL,
            fields.TRANSACTION_FIELD_QUANTITIY,
            fields.TRANSACTION_FIELD_PRICE,
            fields.TRANSACTION_FIELD_CCY,
        ]

        all_in_dict = all(
            item not in transactions_fields
            for item in set(fields_config["column_mapping"].keys())
        )

        if len(fields_config) == 0 or not all_in_dict:
            print(fields_config)
            raise Exception("Transaction field mapping is not valid")

        # Extract column mapping, columns name row, and data row offset from the JSON
        column_mapping = fields_config["column_mapping"]
        columns_name_row = fields_config["columns_name_row"]
        data_row_offset = fields_config["data_row_offset"]
        # Set the DataFrame columns based on the specified row
        transactions_input_df.columns = transactions_input_df.iloc[columns_name_row]
        # Select the data starting from the specified row offset
        transactions = transactions_input_df.iloc[data_row_offset:]
        # Rename the columns in the DataFrame based on the mapping

        with pd.option_context(
            "mode.chained_assignment", None
        ):  # Suppress the warning for this specific operation
            # method 'rename' simply ignore it and proceed with renaming the columns that do exist. It won’t raise an error.
            transactions.rename(columns=column_mapping, inplace=True)

        # Suppress the specific warning
        warnings.simplefilter(
            action="ignore", category=pd.errors.SettingWithCopyWarning
        )
        transactions[fields.TRANSACTION_FIELD_TR_DATE] = pd.to_datetime(
            transactions[fields.TRANSACTION_FIELD_TR_DATE],
            format=fields_config["date_format"],
            dayfirst=True,
        ).dt.date

        # Replace the source values of buy sell with B and V.
        buys_sell_mapping = fields_config["buy_sell_mapping"]
        for k in buys_sell_mapping.keys():
            transactions.loc[:, fields.TRANSACTION_FIELD_BUY_SELL] = transactions.loc[
                :, fields.TRANSACTION_FIELD_BUY_SELL
            ].replace(k, buys_sell_mapping[k])

        if fields.FIELD_FIGI not in transactions.columns:
            asset_cols = [fields.TRANSACTION_FIELD_ISIN, fields.TRANSACTION_FIELD_CCY]
            assets = pd.DataFrame(columns=asset_cols)
            assets = transactions[asset_cols].drop_duplicates()
            tickers = get_tickers_and_suffix(assets[asset_cols], select_exchange)
            assets = assets.merge(tickers, on=asset_cols, how="left")
            resolve_ccy(assets=assets, id_field=fields.TRANSACTION_FIELD_ISIN)

        else:
            print("Figi id detetect. This will be used to resolve the ticker")
            asset_cols = [
                fields.TRANSACTION_FIELD_ISIN,
                fields.TRANSACTION_FIELD_CCY,
                fields.FIELD_FIGI,
            ]
            assets = pd.DataFrame(columns=asset_cols)
            assets = transactions[asset_cols].drop_duplicates()
            tickers = get_tickers_and_suffix(assets[asset_cols])
            assets = assets.merge(tickers, on=asset_cols, how="left")

            resolve_ccy(assets=assets, id_field=fields.FIELD_FIGI)

        # write the resolved asset file
        asset_file_path = (
            config.filesystem_folder
            + "/"
            + config.output_file_name
            + "_"
            + const_and_utils.PORTFOLIO_ASSETS_FILE
        )
        if mode == "append":
            prev_assets = const_and_utils.read_csv(
                config=config, file_path=asset_file_path, ignore_date=True
            )
            assets = pd.concat([assets, prev_assets]).drop_duplicates(
                subset=[fields.FIELD_ISIN, fields.FIELD_FIGI]
            )
            print("Previous transaction set detected and merged.")
        else:
            assets = assets.drop_duplicates()

        const_and_utils.write_csv(
            config=config,
            file_path=config.filesystem_folder
            + "/"
            + config.output_file_name
            + "_"
            + const_and_utils.PORTFOLIO_ASSETS_FILE,
            df=assets,
        )

        # discarded asset - no ticker resolved:
        assets_no_ticker = assets[pd.isna(assets[fields.FIELD_TICKER])]
        if not assets_no_ticker.empty:
            assets_no_ticker_file_name = (
                config.filesystem_folder
                + "/"
                + config.output_file_name
                + "_discarded_"
                + const_and_utils.PORTFOLIO_ASSETS_FILE
            )
            print(
                "Assets not recognized saved in {}.".format(assets_no_ticker_file_name)
            )
            const_and_utils.write_csv(
                config=config, file_path=assets_no_ticker_file_name, df=assets_no_ticker
            )

        # write the resolved transaction file:
        transactions = transactions.merge(
            assets,
            on=[fields.TRANSACTION_FIELD_ISIN, fields.TRANSACTION_FIELD_CCY],
            how="left",
        )
        transaction_file_path = (
            config.filesystem_folder
            + "/"
            + config.output_file_name
            + "_"
            + const_and_utils.TRANSACTION_FILE_TICKERS
        )
        if mode == "append":
            prev_transactions = const_and_utils.read_csv(
                config=config,
                file_path=transaction_file_path,
                date_field=fields.TRANSACTION_FIELD_TR_DATE,
            )
            transactions = pd.concat(
                [transactions, prev_transactions]
            )  # .drop_duplicates()
        else:
            transactions = transactions  # .drop_duplicates()

        # remove not resolved transactions:
        transactions = transactions[~pd.isna(transactions[fields.FIELD_TICKER])]

        const_and_utils.write_csv(
            config=config,
            file_path=config.filesystem_folder
            + "/"
            + config.output_file_name
            + "_"
            + const_and_utils.TRANSACTION_FILE_TICKERS,
            df=transactions,
        )

        return assets.to_dict(orient="records")

    except Exception as e:
        error_collector.get_collector().record_error(
            operation_name="generate_transaction_file", error_details=str(e)
        )
        return dict()


def generate_holdings(config: const_and_utils.Config):
    try:
        transactions = const_and_utils.read_csv(
            config=config,
            file_path=config.filesystem_folder
            + "/"
            + config.output_file_name
            + "_"
            + const_and_utils.TRANSACTION_FILE_TICKERS,
            date_field=fields.TRANSACTION_FIELD_TR_DATE,
        )

        transactions[fields.FIELD_DATE] = transactions[fields.TRANSACTION_FIELD_TR_DATE]

        # adjust the transactions: ##########################################################
        # let's check the transactions for splits are in synch with yahoo fianace split dates
        # Based on the assumption there is not other split  on +/- 1 business days - which is dangerous
        splits = transactions[
            transactions[fields.TRANSACTION_FIELD_PRICE] == 0
        ]  # transaction
        for index, row in splits.iterrows():
            ticker = row[fields.FIELD_TICKER]
            tr_date = row[fields.FIELD_DATE]

            # Fetch stock splits from Yahoo Finance
            yhsplits = yf.Ticker(ticker).splits
            yhsplits.index = yhsplits.index.date

            if tr_date.weekday() != 0:  # If tr_date is not Monday
                start_date = tr_date - timedelta(days=1)
            else:
                start_date = tr_date - timedelta(days=3)

            if tr_date.weekday() != 4:  # If tr_date is not Friday
                end_date = tr_date + timedelta(days=1)
            else:
                end_date = tr_date + timedelta(days=3)

            splits_data = yhsplits[
                (yhsplits.index >= start_date) & (yhsplits.index <= end_date)
            ]

            if not splits_data.empty:
                # Replace the transaction date with the date of the stock split
                new_date = splits_data.index[0]
                transactions.at[index, fields.FIELD_DATE] = new_date
        ############################################################################################

        const_and_utils.write_csv(
            config=config,
            df=transactions,
            file_path=config.filesystem_folder
            + "/"
            + config.output_file_name
            + "_yhf_adj_transactions.csv",
        )

        transactions.sort_values(by=[fields.FIELD_DATE])
        holdings = pd.DataFrame(
            columns=[
                fields.FIELD_TICKER,
                fields.FIELD_ISIN,
                fields.FIELD_SECURITY_TYPE,
                fields.FIELD_MARKET_SECTOR,
                fields.FIELD_DATE,
                fields.FIELD_EVAL_QUANTITY,
                fields.FIELD_CCY,
            ]
        )
        dates = [d for d in sorted(transactions[fields.FIELD_DATE].unique())]

        for d in dates:
            t_snapshot = transactions[
                transactions[fields.FIELD_DATE].apply(lambda td: td == d)
            ]

            for tindex, t in t_snapshot.iterrows():
                last_q = 0
                last_d = datetime(1999, 1, 1)

                if holdings.shape[0] > 0:
                    # display(holdings)
                    if len(holdings[fields.FIELD_TICKER]) == 0:
                        continue
                    asset_holds = holdings[
                        holdings[fields.FIELD_TICKER] == t[fields.FIELD_TICKER]
                    ]
                    if len(asset_holds) > 0:
                        last_q = float(asset_holds[fields.FIELD_EVAL_QUANTITY].iloc[-1])
                        last_d = asset_holds[fields.FIELD_DATE].iloc[-1]

                if t[fields.FIELD_DATE] != last_d:
                    if (
                        t[fields.TRANSACTION_FIELD_BUY_SELL]
                        == fields.VALUE_TRANSACTION_BUY
                    ):
                        new_hold = {
                            fields.FIELD_TICKER: t[fields.FIELD_TICKER],
                            fields.FIELD_ISIN: t[fields.TRANSACTION_FIELD_ISIN],
                            fields.FIELD_SECURITY_TYPE: t[fields.FIELD_SECURITY_TYPE],
                            fields.FIELD_MARKET_SECTOR: t[fields.FIELD_MARKET_SECTOR],
                            fields.FIELD_DATE: t[fields.FIELD_DATE],
                            fields.FIELD_EVAL_QUANTITY: last_q
                            + float(t[fields.TRANSACTION_FIELD_QUANTITIY]),
                            fields.FIELD_CCY: t[fields.TRANSACTION_FIELD_CCY],
                        }
                        holdings.loc[len(holdings)] = new_hold

                    if (
                        t[fields.TRANSACTION_FIELD_BUY_SELL]
                        == fields.VALUE_TRANSACTION_SELL
                    ):
                        new_hold = {
                            fields.FIELD_TICKER: t[fields.FIELD_TICKER],
                            fields.FIELD_ISIN: t[fields.TRANSACTION_FIELD_ISIN],
                            fields.FIELD_SECURITY_TYPE: t[fields.FIELD_SECURITY_TYPE],
                            fields.FIELD_MARKET_SECTOR: t[fields.FIELD_MARKET_SECTOR],
                            fields.FIELD_DATE: t[fields.FIELD_DATE],
                            fields.FIELD_EVAL_QUANTITY: last_q
                            - float(t[fields.TRANSACTION_FIELD_QUANTITIY]),
                            fields.FIELD_CCY: t[fields.TRANSACTION_FIELD_CCY],
                        }
                        holdings.loc[len(holdings)] = new_hold

                else:

                    if (
                        t[fields.TRANSACTION_FIELD_BUY_SELL]
                        == fields.VALUE_TRANSACTION_BUY
                    ):
                        new_amt = last_q + float(t[fields.TRANSACTION_FIELD_QUANTITIY])
                        # DEBUG - print("buy {} : {} + {} = {}".format(t[fields.FIELD_TICKER],last_q, t[fields.TRANSACTION_FIELD_QUANTITIY], new_amt ))
                        filtered_rows = holdings[
                            holdings[fields.FIELD_TICKER] == t[fields.FIELD_TICKER]
                        ]
                        last_row_index = filtered_rows.index[-1]
                        holdings.at[last_row_index, fields.FIELD_EVAL_QUANTITY] = (
                            new_amt
                        )

                    if (
                        t[fields.TRANSACTION_FIELD_BUY_SELL]
                        == fields.VALUE_TRANSACTION_SELL
                    ):
                        new_amt = last_q - float(t[fields.TRANSACTION_FIELD_QUANTITIY])
                        # DEBUG - print("sell {} : {} - {} = {}".format(t[fields.FIELD_TICKER],last_q, t[fields.TRANSACTION_FIELD_QUANTITIY], new_amt ))
                        filtered_rows = holdings[
                            holdings[fields.FIELD_TICKER] == t[fields.FIELD_TICKER]
                        ]
                        last_row_index = filtered_rows.index[-1]
                        holdings.at[last_row_index, fields.FIELD_EVAL_QUANTITY] = (
                            new_amt
                        )

        const_and_utils.write_csv(
            config=config,
            df=holdings,
            file_path=config.filesystem_folder
            + "/"
            + config.output_file_name
            + "_"
            + const_and_utils.PORTFOLIO_HOLDINGS,
        )

        holdings[fields.FIELD_DATE] = holdings[fields.FIELD_DATE].astype(str)
        return holdings.to_dict(orient="records")

    except Exception as e:
        error_collector.get_collector().record_error(
            operation_name="generate_holdings", error_details=str(e)
        )
        return dict()


def unroll_daily_holdings(config: const_and_utils.Config):

    holdings = const_and_utils.read_csv(
        config=config,
        file_path=config.filesystem_folder
        + "/"
        + config.output_file_name
        + "_"
        + const_and_utils.PORTFOLIO_HOLDINGS,
    ).sort_values(by=[fields.FIELD_DATE])[
        [
            fields.FIELD_TICKER,
            fields.FIELD_ISIN,
            fields.FIELD_SECURITY_TYPE,
            fields.FIELD_MARKET_SECTOR,
            fields.FIELD_DATE,
            fields.FIELD_EVAL_QUANTITY,
            fields.FIELD_CCY,
        ]
    ]

    daily_snapshots = pd.DataFrame(columns=holdings.columns)

    start_date = holdings.iloc[0][fields.FIELD_DATE]
    print("Unrolling daily holdings - Portfolio inception is {}".format(start_date))
    prev_date = pd.to_datetime(start_date).date()
    prev_snapshot = pd.DataFrame(columns=holdings.columns)

    snap_cont = 0
    for s_date, snapshot in holdings.groupby(fields.FIELD_DATE):

        # print("*** {} -> {}".format(prev_date,s_date))
        # print(snapshot)

        daily_snapshots = pd.concat(
            [df for df in [daily_snapshots, snapshot] if not df.empty],
            ignore_index=True,
        )

        if s_date == prev_date:
            prev_date = s_date
            prev_snapshot = snapshot
            continue

        date_range = pd.date_range(
            start=prev_date + timedelta(days=1), end=s_date, freq="D"
        )

        for ticker in prev_snapshot[fields.FIELD_TICKER].unique():
            if pd.isna(ticker):
                continue

            print(
                "{}: {} -> {}                                                                  ".format(
                    ticker, prev_date, s_date
                ),
                end="\r",
            )

            prev_ticker_snapshot = prev_snapshot[
                prev_snapshot[fields.FIELD_TICKER] == ticker
            ]

            if prev_ticker_snapshot[fields.FIELD_EVAL_QUANTITY].values[0] != 0:
                for d in date_range:
                    # do not add to the daily snapshot the position if it is already in the next snapshot
                    if ticker in snapshot.ticker.values and d.date() == s_date:
                        # DEBUG -- print("removed {}".format(ticker))
                        continue

                    ffill_hold = {
                        fields.FIELD_TICKER: ticker,
                        fields.FIELD_ISIN: prev_ticker_snapshot[
                            fields.FIELD_ISIN
                        ].values[0],
                        fields.FIELD_SECURITY_TYPE: prev_ticker_snapshot[
                            fields.FIELD_SECURITY_TYPE
                        ].values[0],
                        fields.FIELD_MARKET_SECTOR: prev_ticker_snapshot[
                            fields.FIELD_MARKET_SECTOR
                        ].values[0],
                        fields.FIELD_DATE: pd.to_datetime(d).date(),
                        fields.FIELD_EVAL_QUANTITY: prev_ticker_snapshot[
                            fields.FIELD_EVAL_QUANTITY
                        ].values[0],
                        fields.FIELD_CCY: prev_ticker_snapshot[fields.FIELD_CCY].values[
                            0
                        ],
                    }

                    daily_snapshots.loc[daily_snapshots.index[-1] + 1] = ffill_hold

        lps = daily_snapshots[daily_snapshots[fields.FIELD_DATE] == s_date]
        close_positions = list(
            prev_ticker_snapshot[
                prev_ticker_snapshot[fields.FIELD_EVAL_QUANTITY] == 0
            ].ticker.values
        )

        if (
            not lps.empty
        ):  #  copy to the next snapshot the postions that are not already redefined there (if any)
            lps = lps[lps.ticker.apply(lambda t: t not in close_positions)]
            lps = lps[lps.ticker.apply(lambda t: t not in snapshot.ticker.values)]
            prev_snapshot = pd.concat([snapshot, lps], ignore_index=True)
        else:
            prev_snapshot = snapshot

        prev_date = s_date
        snap_cont += 1

    const_and_utils.write_csv(
        config=config,
        df=daily_snapshots,
        file_path=config.filesystem_folder
        + "/"
        + config.output_file_name
        + "_"
        + const_and_utils.PORTFOLIO_HOLDINGS_DAILY,
    )
    return daily_snapshots


def fetch_portfolio_prices(
    config: const_and_utils.Config,
    mode: str = "offline",
    base_currency: str = "USD",
    additional_fields="",
) -> pd.DataFrame:

    try:
        print("Fetching Prices, FX and CAs. Base ccy is {}".format(base_currency))
        portfolio_and_market_data = dict()
        portfolio = const_and_utils.read_csv(
            config=config,
            file_path=config.filesystem_folder
            + "/"
            + config.output_file_name
            + "_"
            + const_and_utils.PORTFOLIO_HOLDINGS_DAILY,
        )
        unique_assets = portfolio[
            [fields.FIELD_TICKER, fields.FIELD_SECURITY_TYPE, fields.FIELD_CCY]
        ].drop_duplicates()

        for index, row in unique_assets.iterrows():
            try:
                ticker = row[fields.FIELD_TICKER]
                security_type = row[fields.FIELD_SECURITY_TYPE]
                ticker_ccy = row[fields.FIELD_CCY]
                start_date = portfolio[portfolio[fields.FIELD_TICKER] == ticker][
                    fields.FIELD_DATE
                ].values[0]
                positions = portfolio[
                    portfolio[fields.FIELD_TICKER] == ticker
                ].sort_values(by=fields.FIELD_DATE)
                ticker_data = fetch_equity_ticker_data(
                    config=config,
                    ticker=ticker,
                    ticker_currency=ticker_ccy,
                    security_type=security_type,
                    base_currency=base_currency,
                    start_date=start_date,
                    positions=positions,
                    mode=mode,
                    additional_fields=additional_fields,
                )
                if ticker_data is not None and not ticker_data.empty:
                    portfolio_and_market_data[ticker] = ticker_data
            except Exception as e:
                print(
                    "Ticker [{}] --> ! Error in fetching price {}                                     ".format(
                        ticker, e
                    )
                )
                continue
        print(
            "Done!                                                                     "
        )

        write = False
        lock_data = const_and_utils.read_json(
            config=config,
            file_path=config.filesystem_folder + "/" + const_and_utils.constants.WRITE_LOCK
        )
        if lock_data is None or (isinstance(lock_data, dict) and "error" in lock_data):
            write = True
        if not write:
            print("Write lock file found. Skipping write.")
        
        for pos in portfolio_and_market_data.items():
            if write:
                const_and_utils.write_csv(
                    config=config,
                    df=pos[1],
                    file_path=config.filesystem_folder
                    + "/"
                    + config.output_file_name
                    + "_"
                    + const_and_utils.DAILY_HOLDINGS_PREFIX
                    + pos[0]
                    + ".csv",
                )
            pos[1][fields.FIELD_DATE] = pos[1][fields.FIELD_DATE].astype(str)

        non_empty_dfs = [
            df for df in portfolio_and_market_data.values() if not df.empty
        ]
        response = None
        if non_empty_dfs:
            response = pd.concat(non_empty_dfs, ignore_index=True)
        return response

    except Exception as e:
        error_collector.get_collector().record_error(
            operation_name="fetch_portfolio_prices", error_details=str(e)
        )

    return pd.DataFrame()


def fetch_equity_ticker_data(
    config: const_and_utils.Config,
    ticker: str,
    ticker_currency: str,
    security_type: str,
    base_currency: str,
    start_date: datetime.date,
    positions: pd.DataFrame,
    mode: str,
    additional_fields: str,
) -> pd.DataFrame:
    """
    Fetches historical market data for a given equity ticker. Here errors are not collected as non blocking for the analysis.

    Parameters:
        config (Config): The configuration object containing necessary parameters to load hlds files.
        ticker (str): The equity ticker symbol.
        ticker_currency (str): The currency of the equity ticker.
        security_type (str): The type of the security, e.g. stock, currency, etc.
        base_currency (str): The base currency of the portfolio.
        start_date (datetime.date): The start date of the data to be fetched.
        positions (pd.DataFrame): The current positions of the portfolio.
        mode (str): The mode of data fetching, either "online" or "offline".
        additional_fields (str): A string of additional fields to be fetched and merged with the positions data.

    Returns:
        pd.DataFrame: The fetched market data merged with the positions data.
    """

    if pd.isna(ticker):
        return

    print(
        "Ticker {} [type: {} - fx: {}{}=X] -> Fetching ...                            ".format(
            ticker, security_type, ticker_currency, base_currency
        ),
        end="\r",
    )
    ticker_prices = pd.DataFrame()
    if additional_fields:
        additional_fields = additional_fields.split("|")
    else:
        additional_fields = []

    if mode == "offline":
        ticker_prices = const_and_utils.read_csv(
            config=config,
            file_path=config.filesystem_folder
            + "/data/"
            + const_and_utils.PREFIX_FILE_PRICE
            + ticker
            + ".csv",
        )
        additional_fields = [
            f
            for f in ticker_prices.columns
            if f
            not in fields.BASIC_POSITION_FIELDS + fields.BASIC_MARKET_DATA_INPUT_FIELDS
        ]

    else:  # online data fetching:

        end_date = (datetime.today() + timedelta(days=1)).date()

        try:  # let's see first if data are availabe offline - this is alwyas preferred.
            path = "/data/" + const_and_utils.PREFIX_FILE_PRICE + ticker + ".csv"
            if os.path.isfile(config.filesystem_folder + path):
                print(
                    "Looking for offline fetched data {}                                                              ".format(
                        path
                    ),
                    end="\r",
                )
                ticker_prices = const_and_utils.read_csv(
                    config=config,
                    file_path=config.filesystem_folder + path,
                    date_field="Date",
                    date_format="%Y-%m-%d",
                )
                ticker_prices = ticker_prices[ticker_prices["Date"] >= start_date]
        except Exception as e:
            print(
                "No offline data for {}: {}                                                                   ".format(
                    ticker, e
                ),
                end="\r",
            )

        # if no data is avalaiable let's fetch form yfinanace
        if ticker_prices.empty:
            try:
                tickerData = yf.Ticker(ticker)

                if security_type == "Currency":  # if the ticker is a ccy position
                    tickerData = yf.Ticker(ticker + ticker + "=X")
                    ticker_prices = tickerData.history(
                        start=start_date, end=end_date, auto_adjust=False
                    )
                    ticker_prices.reset_index(inplace=True)
                    ticker_prices.drop(columns=["Close"], inplace=True)
                    if fields.FIELD_STOCK_SPLIT in ticker_prices.columns:
                        ticker_prices.drop(
                            columns=[fields.FIELD_STOCK_SPLIT], inplace=True
                        )
                    date_range = pd.date_range(
                        start=start_date, end=end_date, freq="D"
                    )  # 'B' for business days
                    ones_series = pd.Series(1, index=date_range)
                    date_df = pd.DataFrame({"Date": date_range, "Close": ones_series})
                    date_df["Date"] = date_df["Date"].apply(
                        lambda d: pd.to_datetime(d).date()
                    )
                    ticker_prices["Date"] = ticker_prices["Date"].apply(
                        lambda d: pd.to_datetime(d).date()
                    )
                    ticker_prices = ticker_prices.merge(date_df, on="Date", how="right")
                    last_date = ticker_prices["Date"].iloc[-1]
                    # Drop the last row if it's a Saturday or Sunday
                    if last_date.weekday() == 0:
                        ticker_prices = ticker_prices.iloc[:-3]
                    elif last_date.weekday() == 6:
                        ticker_prices = ticker_prices.iloc[:-2]
                    elif last_date.weekday() == 5:
                        ticker_prices = ticker_prices.iloc[:-1]

                else:  # if is a regualr equity ticker:
                    ticker_prices = tickerData.history(
                        start=start_date, end=end_date, auto_adjust=False
                    )
                    ticker_prices.reset_index(inplace=True)

                if additional_fields:
                    if len(additional_fields) > 0:
                        ticker_prices_add = {
                            key: ["" + str(tickerData.info[key])] * len(ticker_prices)
                            for key in additional_fields
                            if key in tickerData.info
                        }
                        if len(ticker_prices_add) > 0:
                            ticker_prices = pd.concat(
                                [ticker_prices, pd.DataFrame(ticker_prices_add)], axis=1
                            )
                        else:
                            print(
                                "No additional fields found for ticker {}                            ".format(
                                    ticker,
                                    security_type,
                                    ticker_currency,
                                    base_currency,
                                ),
                                end="\r",
                            )
                            additional_fields = []
                        additional_fields = [
                            f for f in additional_fields if f in ticker_prices.columns
                        ]

            except Exception as e:
                print("Error fetching data for {}: {}".format(ticker, e))
                return

    # Fetch FX rate
    try:
        if not ticker_prices.empty and not pd.isna(ticker_currency):
            fx_data = yf.Ticker(f"{ticker_currency}{base_currency}=X")
            fx_rates = fx_data.history(
                start=start_date, end=end_date, auto_adjust=False
            )
            fx_rates.reset_index(inplace=True)
            fx_rates = fx_rates[["Date", "Close"]].rename(
                columns={"Close": fields.FIELD_FX_RATES}
            )
            fx_rates["Date"] = fx_rates["Date"].apply(
                lambda d: pd.to_datetime(d).date()
            )
            ticker_prices["Date"] = ticker_prices["Date"].apply(
                lambda d: pd.to_datetime(d).date()
            )
            ticker_prices = ticker_prices.merge(fx_rates, on="Date", how="left")
    except Exception as e:
        print(
            "Error fetching data for {}: {}".format(
                f"{ticker_currency}{base_currency}=X", e
            )
        )
        return

    ticker_prices = ticker_prices[ticker_prices["Date"] < datetime.today().date()]

    if ticker_prices.empty:
        print(
            "No data for {}                                                              ".format(
                ticker
            )
        )
        return

    try:
        if fields.FIELD_STOCK_SPLIT in ticker_prices.columns:
            # split adjustment : make yahoo price raw!
            ticker_prices[fields.FIELD_STOCK_SPLIT] = ticker_prices[
                fields.FIELD_STOCK_SPLIT
            ].replace({0: 1})
            ticker_prices[fields.FIELD_STOCK_SPLIT] = ticker_prices[
                fields.FIELD_STOCK_SPLIT
            ][::-1].cumprod()[::-1]
            last_value = ticker_prices[fields.FIELD_CLOSE].iloc[-1].copy()
            ticker_prices[fields.FIELD_CLOSE] = ticker_prices[
                fields.FIELD_CLOSE
            ] * ticker_prices[fields.FIELD_STOCK_SPLIT].shift(-1)
            # ticker_prices[fields.FIELD_CLOSE].iloc[-1] = last_value
            ticker_prices.loc[ticker_prices.index[-1], fields.FIELD_CLOSE] = last_value
            t_p = (
                ticker_prices[
                    [
                        "Date",
                        fields.FIELD_CLOSE,
                        fields.FIELD_FX_RATES,
                        fields.FIELD_DIVIDENDS,
                        fields.FIELD_STOCK_SPLIT,
                    ]
                    + additional_fields
                ]
                .rename(columns={"Date": fields.FIELD_DATE})
                .sort_values(by=fields.FIELD_DATE)
            )
        else:
            t_p = (
                ticker_prices[
                    ["Date", fields.FIELD_CLOSE, fields.FIELD_FX_RATES]
                    + additional_fields
                ]
                .rename(columns={"Date": fields.FIELD_DATE})
                .sort_values(by=fields.FIELD_DATE)
            )

        # fill to today if last quantity is not 0
        last_snapshot = positions.iloc[-1]
        last_price = t_p.iloc[-1]

        if (
            last_snapshot[fields.FIELD_DATE] < last_price[fields.FIELD_DATE]
            and last_snapshot[fields.FIELD_EVAL_QUANTITY] > 0
        ):
            date_range = pd.date_range(
                last_snapshot[fields.FIELD_DATE] + timedelta(days=1),
                last_price[fields.FIELD_DATE],
            )
            datetime_range = [ts.to_pydatetime().date() for ts in date_range]

            fillfwd = pd.DataFrame(columns=fields.BASIC_POSITION_FIELDS)

            for d in datetime_range:
                t = {k: last_snapshot[k] for k in fields.BASIC_POSITION_FIELDS}
                t[fields.FIELD_DATE] = d
                # print(t)
                fillfwd.loc[len(fillfwd)] = t
            positions = pd.concat([positions, fillfwd], ignore_index=True)

    except Exception as e:
        print("Error de-adjusting prices for {}: {}".format(ticker, e))
        return

    return positions[fields.BASIC_POSITION_FIELDS].merge(
        t_p, on=fields.FIELD_DATE, how="left"
    )
