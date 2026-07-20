# %%
import yfinance as yf
import requests


ticker = yf.Ticker("NVDA")
tickerData = ticker.splits
print(tickerData)

# %%
import yfinance as yf
import requests
import random
import time

ticker = "GSBD"
print(ticker)

# List of public proxies (HTTP/HTTPS, high anonymity, sourced from public lists)
proxies_list = [
    {"http": "http://47.251.43.115:3128", "https": "http://47.251.43.115:3128"},
    {"http": "http://72.10.164.178:25353", "https": "http://72.10.164.178:25353"},
    {"http": "http://67.43.236.20:33667", "https": "http://67.43.236.20:33667"},
    {"http": "http://159.223.176.158:8080", "https": "http://159.223.176.158:8080"},
    {"http": "http://185.199.110.137:8443", "https": "http://185.199.110.137:8443"},
    {"http": "http://154.202.121.180:3128", "https": "http://154.202.121.180:3128"},
]


# Function to test if a proxy is working
def test_proxy(proxy):
    try:
        response = requests.get("https://www.google.com", proxies=proxy, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


# Filter working proxies
working_proxies = []
for proxy in proxies_list:
    if test_proxy(proxy):
        working_proxies.append(proxy)
        print(f"Proxy {proxy['http']} is working")
    else:
        print(f"Proxy {proxy['http']} is not working")

if not working_proxies:
    print(
        "No working proxies found. Consider using premium proxies or checking proxy sources."
    )
    exit()

# Number of requests to attempt
max_attempts = 5

for attempt in range(max_attempts):
    # Select a random proxy
    selected_proxy = random.choice(working_proxies)
    print(f"Attempt {attempt + 1}/{max_attempts} using proxy: {selected_proxy['http']}")

    try:
        # Create a session with the selected proxy
        session = requests.Session()
        session.proxies = selected_proxy

        # Fetch stock splits using yfinance
        yhsplits = yf.Ticker(ticker, session=session).splits

        # Print results
        print("Stock splits retrieved successfully:")
        print(yhsplits)
        break  # Exit loop on success

    except requests.exceptions.ProxyError as e:
        print(f"Proxy error with {selected_proxy['http']}: {e}")
        working_proxies.remove(selected_proxy)  # Remove failed proxy
        if not working_proxies:
            print("No more working proxies available.")
            break

    except yf.utils.exceptions.YFRateLimitError as e:
        print(f"Yahoo Finance rate limit error: {e}")
        time.sleep(5)  # Wait before retrying
        continue

    except Exception as e:
        print(f"An unexpected error occurred with {selected_proxy['http']}: {e}")
        working_proxies.remove(selected_proxy)  # Remove failed proxy
        if not working_proxies:
            print("No more working proxies available.")
            break

    # Delay to avoid rate-limiting
    time.sleep(2)

else:
    print("Failed to retrieve data after all attempts.")


# %%
import pandas as pd
import json


def csv_to_json(csv_file):
    df = pd.read_csv(csv_file, header=0, names=["index", "ISIN", "ticker"])
    print(df.head())
    df.drop(columns=["index"], inplace=True)  # Drop the unnecessary 'index' column
    data = df.to_dict(orient="records")
    return data


csv_file = "/home/agallini/portfolios/portfolio_assets.csv"
json_data = csv_to_json(csv_file)
print(json.dumps(json_data, indent=2))

# %%
import pandas as pd
from oport.functions.functions_characteristics import get_char_indicator

# Define the stock ticker and target date
ticker_symbol = "GSBD"  # Example: Apple Inc.
target_date = pd.to_datetime("2024-10-02").date()

d = get_char_indicator(ticker_symbol, target_date)
d

# %%


# %%
import pandas as pd
from oport.const_and_utils import read_csv
from oport.const_and_utils import Config

from oport.functions import functions_aggregation
from oport.functions.functions_aggregation import aggregate
from oport.fields import *
from oport.functions.functions_performance import compute_performance_indicators


import os

print("Current working directory: {}".format(os.getcwd()))

of = "/Users/albertogallini/pythonprj/oport/tests/test_portfolios/test_e2e_stocksplits/"
config = Config(
    source="filesystem",
    filesystem_folder=of,
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

classification = "ccy|securityType"
daily_performance = read_csv(
    config=config, file_path=of + "test_e2e_stocksplits_performance.csv"
)
daily_performance, portfolio_r, buckets_r = functions_aggregation.aggregate(
    portfolio_daily_fields=daily_performance,
    classification=classification.split("|"),
    time_aggregation=True,
)

d, portfolio_r, buckets_r = compute_performance_indicators(
    daily_performance, buckets_r, portfolio_r, PERFORMANCE_FIELDS
)

# Assuming `d` is your DataFrame
result = d[
    [FIELD_TICKER, FIELD_DATE] + [FIELD_EVAL_VAR_95, FIELD_EVAL_VAR_99, FIELD_EVAL_MEAN]
].groupby(FIELD_DATE)
n = 5  # Number of first dates to print
count = 0
# Loop through the groups
for date, group in result:
    if count < n:
        print(f"Date: {date}")
        display(group)
        count += 1
    else:
        break

print("Portfolio")
display(
    pd.DataFrame(portfolio_r)[
        [FIELD_DATE] + [FIELD_EVAL_VAR_95, FIELD_EVAL_VAR_99, FIELD_EVAL_MEAN]
    ]
)
for bucket_name, bucket_data in buckets_r.items():
    print(bucket_name)
    bucket_df = display(
        pd.DataFrame(bucket_data)[
            [FIELD_DATE] + [FIELD_EVAL_VAR_95, FIELD_EVAL_VAR_99, FIELD_EVAL_MEAN]
        ]
    )

# %%

import numpy as np
import pandas as pd
from oport.functions import (
    functions_aggregation,
    functions_performance_daily,
    functions_tree,
)
from oport.functions import functions_pattribution
from .const_and_utils import Config
from oport.functions.functions_pattribution import is_root
from oport.fields import (
    ATTRIBUTION_INPUT_DAILY_FIELDS,
    BASIC_HOLDINGS_FIELDS,
    BASIC_POSITON_DESCR_FIELDS,
    FIELD_EVAL_CCY_RETURN,
    FIELD_EVAL_CTR_TO_RETURN,
    FIELD_EVAL_TRANSACTION_RETURN_LCL_CCY,
    FIELD_TREE_POSITION_ID,
    FIELD_TREE_POSITION_ID_PVSB,
    FIELD_EVAL_ALLOCATION,
    FIELD_EVAL_SELECTION,
    FIELD_EVAL_TOTAL_RETURN_BASIS_WEIGHTS,
    FIELD_EVAL_LEVERAGE,
    FIELD_EVAL_RETURN_HLDS_LCL_CCY,
    TRANSACTION_FIELD_TR_DATE,
    TREE_ACTIVE_SUFFIX,
    TREE_BENCHMARK_SUFFIX,
    FIELD_DATE,
)
import warnings
from datetime import date

portfolio_name = "dec2024"
benchmark_name = "oex_index"
classification = "ccy|securityType"

filesystem_folder = ".|..|..|tests|test_attribution|".replace("|", "/")
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
end_date = date(2024, 10, 10)


########################    Load portfolio
portfolio_holdings_prefix = portfolio_name + "_" + const_and_utils.DAILY_HOLDINGS_PREFIX
portfolio_holdings_files = const_and_utils.list_files(
    config=config, input_folder=filesystem_folder, prefix=portfolio_holdings_prefix
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
benchmark_holdings_prefix = benchmark_name + "_" + const_and_utils.DAILY_HOLDINGS_PREFIX
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
benchmark_performance_daily = functions_performance_daily.compute_performance_daily(
    config=config_benchmark,
    prefix=benchmark_holdings_prefix,
    transactions=benchmark_transactions,
    start_date=start_date,
    end_date=end_date,
    holdings_files=benchmark_holdings_files,
)

start_date, end_date = functions_aggregation.resolve_time_portf_vs_bench_time_frame(
    performance_daily, benchmark_performance_daily, start_date, end_date
)

########################   aggregate at bucket/top level
portfolio = None
portfolio_buckets = None
benchmark = None
benchmark_buckets = None
print("Bucketing by {} ... ".format(classification))
classification = classification.split("|")
performance_daily, portfolio, portfolio_buckets = functions_aggregation.aggregate(
    portfolio_daily_fields=performance_daily,
    start_date=start_date,
    end_date=end_date,
    classification=classification,
    time_aggregation=False,
    metrics=BASIC_HOLDINGS_FIELDS + ATTRIBUTION_INPUT_DAILY_FIELDS,
    performance_attribution=True,
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


row = "[USD\\]"
from IPython.display import display

pd.set_option("display.max_rows", 100)
tree_1003 = tree[tree[FIELD_DATE] == date(2024, 10, 3)]
root_1003 = tree_1003[tree_1003[FIELD_TREE_POSITION_ID_PVSB].apply(lambda x: row in x)]
display(
    root_1003[
        [
            FIELD_TREE_POSITION_ID_PVSB,
            FIELD_EVAL_TOTAL_RETURN_BASIS_WEIGHTS + TREE_BENCHMARK_SUFFIX,
            FIELD_EVAL_RETURN_HLDS_LCL_CCY + TREE_BENCHMARK_SUFFIX,
        ]
    ]
)
tree_1003 = tree[tree[FIELD_DATE] == date(2024, 10, 4)]
root_1003 = tree_1003[tree_1003[FIELD_TREE_POSITION_ID_PVSB].apply(lambda x: row in x)]
display(
    root_1003[
        [
            FIELD_TREE_POSITION_ID_PVSB,
            FIELD_EVAL_TOTAL_RETURN_BASIS_WEIGHTS + TREE_BENCHMARK_SUFFIX,
            FIELD_EVAL_RETURN_HLDS_LCL_CCY + TREE_BENCHMARK_SUFFIX,
        ]
    ]
)


root_1003 = tree[tree[FIELD_TREE_POSITION_ID_PVSB].apply(lambda x: row in x)]
display(
    root_1003[
        [
            FIELD_DATE,
            FIELD_TREE_POSITION_ID_PVSB,
            FIELD_EVAL_TOTAL_RETURN_BASIS_WEIGHTS + TREE_BENCHMARK_SUFFIX,
            FIELD_EVAL_RETURN_HLDS_LCL_CCY + TREE_BENCHMARK_SUFFIX,
        ]
    ]
)
display(
    root_1003[
        [
            FIELD_DATE,
            FIELD_TREE_POSITION_ID_PVSB,
            FIELD_EVAL_TOTAL_RETURN_BASIS_WEIGHTS,
            FIELD_EVAL_RETURN_HLDS_LCL_CCY,
        ]
    ]
)


display(benchmark_buckets[("USD",)])
display(portfolio_buckets[("USD",)])
# %%
