from oport import const_and_utils
from oport import fields

import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

from openfigipy import OpenFigiClient
import yfinance as yf


#  --- Deprecated
def get_ticker(isin, sleep_time=1):
    try:
        time.sleep(sleep_time)
        ofc = OpenFigiClient()
        ofc.connect()  # Establish a requests session

        # Create a dataframe with ISINs (you can modify this with your own data)
        df = pd.DataFrame(
            {
                "idType": ["ID_ISIN"],
                "idValue": [isin],  # Replace with your ISIN
            }
        )

        # Map ISINs to tickers
        result = ofc.map(df)
        print(result[fields.FIELD_TICKER][0])
        return result[fields.FIELD_TICKER][0]
    except Exception as e:
        print(" {} No ticker found : {} ".format(isin, e))
        return ""


def get_tickers(isins):
    try:
        ofc = OpenFigiClient()
        ofc.connect()  # Establish a requests session

        # Create a dataframe with ISINs
        df = pd.DataFrame(
            {
                "idType": ["ID_ISIN"] * len(isins.to_list()),
                "idValue": isins.to_list(),
            }
        )

        # Map ISINs to tickers
        result = ofc.map(df)
        result.rename(
            columns={"q_idValue": fields.TRANSACTION_FIELD_ISIN}, inplace=True
        )
        return result[
            [
                fields.TRANSACTION_FIELD_ISIN,
                fields.FIELD_TICKER,
                fields.FIELD_SECURITY_TYPE,
                fields.FIELD_MARKET_SECTOR,
            ]
        ].drop_duplicates(subset=[fields.TRANSACTION_FIELD_ISIN])

    except Exception as e:
        print("Error in retrieving tickers: {}".format(e))
        print([""] * len(isins))
        return ""


# Function to prompt user for selection
def get_user_choice(isin: str, exchanges: np.array, suffixes: pd.DataFrame):
    print(f"ISIN: {isin}")
    while True:
        for i, exchange in enumerate(exchanges, 1):
            exchange_descr = suffixes[suffixes[fields.FIELD_EXCHANGE] == exchange][
                fields.FIELD_EXCHANGE_DESCR
            ].values[0]
            print(f"{i}: {exchange} - {exchange_descr}")
        try:
            choice = int(
                input("Select the exchange by entering the corresponding number: ")
            )
            if 1 <= choice <= len(exchanges):
                return exchanges[choice - 1]
            else:
                print(
                    "Invalid choice. Please enter a number corresponding to the listed exchanges."
                )
        except ValueError:
            print("Invalid input. Please enter a valid number.")


def get_tickers_and_suffix(data: pd.DataFrame, select_exchange: bool = False):
    try:
        suffixes = pd.read_csv(const_and_utils.METADATA_EXCHANCE_CODES)
        ofc = OpenFigiClient()
        ofc.connect()  # Establish a requests session

        if fields.FIELD_FIGI in data.columns:
            df = pd.DataFrame(
                {
                    "idType": ["ID_BB_GLOBAL"] * len(data[fields.FIELD_FIGI].to_list()),
                    "idValue": data[fields.FIELD_FIGI].to_list(),
                }
            )
            result = ofc.map(df)

            # result.rename(columns={'q_idValue':FIELD_FIGI}, inplace=True)
            result = result.merge(data, on=fields.FIELD_FIGI, how="left")
        else:
            # Create a dataframe with ISINs
            df = pd.DataFrame(
                {
                    "idType": ["ID_ISIN"] * len(data["isin"].to_list()),
                    "idValue": data["isin"].to_list(),
                }
            )
            # Map ISINs to tickers
            result = ofc.map(df)
            result.rename(
                columns={"q_idValue": fields.TRANSACTION_FIELD_ISIN}, inplace=True
            )
            result = result.merge(data, on="isin", how="left")
            result2 = pd.DataFrame()
            for ccy in suffixes.currency.unique():
                suffixes_ccys = (
                    suffixes[suffixes[fields.TRANSACTION_FIELD_CCY] == ccy]
                    .sort_values([fields.FIELD_EXCHANGE])
                    .exchCode.unique()
                )
                isins = result[fields.TRANSACTION_FIELD_ISIN].unique()
                for isin in isins:
                    # print("{}-{}--{}".format(isin,ccy,suffixes_ccys))
                    r = result[
                        (result[fields.TRANSACTION_FIELD_ISIN] == isin)
                        & (result[fields.TRANSACTION_FIELD_CCY] == ccy)
                        & (result[fields.FIELD_EXCHANGE].isin(suffixes_ccys))
                    ]
                    result2 = pd.concat([result2, r])
            result = result2

        if not select_exchange:
            result = result[
                [
                    fields.TRANSACTION_FIELD_ISIN,
                    fields.FIELD_TICKER,
                    fields.FIELD_SECURITY_TYPE,
                    fields.FIELD_SECURITY_TYPE2,
                    fields.FIELD_MARKET_SECTOR,
                    fields.FIELD_EXCHANGE,
                    fields.FIELD_FIGI,
                    fields.TRANSACTION_FIELD_CCY,
                ]
            ].drop_duplicates(subset=[fields.TRANSACTION_FIELD_ISIN])
        else:
            # Group by ISIN and collect possible FIELD_TICKER values
            grouped = result.groupby(fields.TRANSACTION_FIELD_ISIN)
            # Dictionary to store user selections
            user_selections = {}
            # Prompt user for each ISIN
            for isin, exchange in grouped:
                exchange = exchange.drop_duplicates(subset=[fields.FIELD_EXCHANGE])
                if len(exchange[fields.FIELD_EXCHANGE].values) > 1:
                    user_selections[isin] = get_user_choice(
                        isin, exchange[fields.FIELD_EXCHANGE].values, suffixes
                    )
            # Filter the DataFrame based on user selections
            for isin, exchange in user_selections.items():

                result = result[
                    ~(
                        (result[fields.TRANSACTION_FIELD_ISIN] == isin)
                        & (result[fields.FIELD_EXCHANGE] != exchange)
                    )
                ]

        for idx, row in result.iterrows():
            ss = suffixes[
                suffixes[fields.FIELD_EXCHANGE] == row[fields.FIELD_EXCHANGE]
            ]["Suffix"].sort_values()
            if len(ss) > 0:
                s = ss.values[0]
                if not pd.isna(s):
                    result.at[idx, fields.FIELD_TICKER] = row[
                        fields.FIELD_TICKER
                    ] + "{}".format(s)

        return result

    except Exception as e:
        print("Error in retrieving tickers: {}".format(e))
        print([""] * len(data["isin"]))
        return ""


def resolve_ccy(assets: pd.DataFrame, id_field: str):
    ccy_df = pd.read_csv(const_and_utils.METADATA_CCY_TICKERS)
    for idx, row in assets.iterrows():
        if row[id_field] in ccy_df["ticker"].values:
            assets.loc[idx, fields.FIELD_TICKER] = ccy_df[
                ccy_df["ticker"] == row[id_field]
            ]["ticker"].values[0]
            assets.loc[idx, fields.FIELD_SECURITY_TYPE] = "Currency"
            assets.loc[idx, fields.FIELD_SECURITY_TYPE2] = "SPOT"
