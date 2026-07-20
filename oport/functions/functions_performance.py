import pandas as pd
import datetime
import numpy as np
from scipy.stats import skew


from . import functions_aggregation
from oport.fields import *
from .linking import carino_linking_return_split

import datetime
import pandas as pd

import datetime
import pandas as pd


def get_perf_indicator(
    ticker_symbol: str,
    target_date: datetime.date,
    hlds: pd.DataFrame,
    perf_fields: list,
    benchmark_r: pd.DataFrame = None,
) -> dict:
    """
    This function computes the performance indicator in the input list
    for a given ticker as of the target date.
    """
    result = dict()

    # Filter by ticker symbol
    ticker_data = hlds[hlds[FIELD_TICKER] == ticker_symbol]

    ticker_data.loc[:, FIELD_EVAL_RETURN] = ticker_data[FIELD_EVAL_RETURN].fillna(1.0)
    ticker_data.loc[:, FIELD_EVAL_RETURN_LCL_CCY] = ticker_data[
        FIELD_EVAL_RETURN_LCL_CCY
    ].fillna(1.0)

    if target_date not in ticker_data[FIELD_DATE].values:
        return {perf_field: None for perf_field in perf_fields}

    target_index = ticker_data.index[ticker_data[FIELD_DATE] == target_date][0]

    for perf_field in perf_fields:

        if "std" in perf_field:
            if "std_lcl" in perf_field:  # Standard deviation in local currency
                result[perf_field] = np.std(
                    100
                    * (ticker_data.loc[:target_index, FIELD_EVAL_RETURN_LCL_CCY] - 1)
                )
            else:  # Standard deviation in base currency
                result[perf_field] = np.std(
                    100 * (ticker_data.loc[:target_index, FIELD_EVAL_RETURN]) - 1
                )
        elif "lcl" in perf_field:  # linking
            if "1d" in perf_field:
                result[perf_field] = 100 * (
                    ticker_data.loc[target_index, FIELD_EVAL_RETURN_LCL_CCY] - 1
                )
            elif "1w" in perf_field:
                if target_index - 7 >= 0:
                    acc_total_return_i = (
                        ticker_data.loc[
                            target_index - 7 : target_index, FIELD_EVAL_RETURN
                        ]
                    ).prod()
                    lcl_ccy_return_slice = np.array(
                        ticker_data.loc[
                            target_index - 7 : target_index, FIELD_EVAL_RETURN_LCL_CCY
                        ]
                    )
                    return_slice = np.array(
                        ticker_data.loc[
                            target_index - 7 : target_index, FIELD_EVAL_RETURN
                        ]
                    )
                    result[perf_field] = 100 * (
                        carino_linking_return_split(
                            lcl_ccy_return_slice, return_slice, acc_total_return_i
                        )
                        - 1
                    )
                else:
                    result[perf_field] = None
            elif "1M" in perf_field:
                if target_index - 30 >= 0:
                    acc_total_return_i = (
                        ticker_data.loc[
                            target_index - 30 : target_index, FIELD_EVAL_RETURN
                        ]
                    ).prod()
                    lcl_ccy_return_slice = np.array(
                        ticker_data.loc[
                            target_index - 30 : target_index, FIELD_EVAL_RETURN_LCL_CCY
                        ]
                    )
                    return_slice = np.array(
                        ticker_data.loc[
                            target_index - 30 : target_index, FIELD_EVAL_RETURN
                        ]
                    )
                    result[perf_field] = 100 * (
                        carino_linking_return_split(
                            lcl_ccy_return_slice, return_slice, acc_total_return_i
                        )
                        - 1
                    )
                else:
                    result[perf_field] = None
            elif "1Q" in perf_field:
                if target_index - 90 >= 0:
                    acc_total_return_i = (
                        ticker_data.loc[
                            target_index - 90 : target_index, FIELD_EVAL_RETURN
                        ]
                    ).prod()
                    lcl_ccy_return_slice = np.array(
                        ticker_data.loc[
                            target_index - 90 : target_index, FIELD_EVAL_RETURN_LCL_CCY
                        ]
                    )
                    return_slice = np.array(
                        ticker_data.loc[
                            target_index - 90 : target_index, FIELD_EVAL_RETURN
                        ]
                    )
                    result[perf_field] = 100 * (
                        carino_linking_return_split(
                            lcl_ccy_return_slice, return_slice, acc_total_return_i
                        )
                        - 1
                    )
                else:
                    result[perf_field] = None
            elif "1Y" in perf_field:
                if target_index - 365 >= 0:
                    acc_total_return_i = (
                        ticker_data.loc[
                            target_index - 365 : target_index, FIELD_EVAL_RETURN
                        ]
                    ).prod()
                    lcl_ccy_return_slice = np.array(
                        ticker_data.loc[
                            target_index - 365 : target_index, FIELD_EVAL_RETURN_LCL_CCY
                        ]
                    )
                    return_slice = np.array(
                        ticker_data.loc[
                            target_index - 365 : target_index, FIELD_EVAL_RETURN
                        ]
                    )
                    result[perf_field] = 100 * (
                        carino_linking_return_split(
                            lcl_ccy_return_slice, return_slice, acc_total_return_i
                        )
                        - 1
                    )
                else:
                    result[perf_field] = None
            else:
                continue
        else:  # compounding
            if "1d" in perf_field:
                result[perf_field] = 100 * (
                    ticker_data.loc[target_index, FIELD_EVAL_RETURN] - 1
                )
            elif "1w" in perf_field:
                if target_index - 7 >= 0:
                    result[perf_field] = (
                        (
                            ticker_data.loc[
                                target_index - 7 : target_index, FIELD_EVAL_RETURN
                            ]
                        ).prod()
                        - 1
                    ) * 100
                else:
                    result[perf_field] = None
            elif "1M" in perf_field:
                if target_index - 30 >= 0:
                    result[perf_field] = (
                        (
                            ticker_data.loc[
                                target_index - 30 : target_index, FIELD_EVAL_RETURN
                            ]
                        ).prod()
                        - 1
                    ) * 100
                else:
                    result[perf_field] = None
            elif "1Q" in perf_field:
                if target_index - 90 >= 0:
                    result[perf_field] = (
                        (
                            ticker_data.loc[
                                target_index - 90 : target_index, FIELD_EVAL_RETURN
                            ]
                        ).prod()
                        - 1
                    ) * 100
                else:
                    result[perf_field] = None
            elif "1Y" in perf_field:
                if target_index - 365 >= 0:
                    result[perf_field] = (
                        (
                            ticker_data.loc[
                                target_index - 365 : target_index, FIELD_EVAL_RETURN
                            ]
                        ).prod()
                        - 1
                    ) * 100
                else:
                    result[perf_field] = None
            elif FIELD_EVAL_DOWNSIDE_RISK in perf_field:
                downside_returns = ticker_data.loc[:target_index, FIELD_EVAL_RETURN][
                    ticker_data[FIELD_EVAL_RETURN] < 0
                ]
                result[perf_field] = np.std(downside_returns) * np.sqrt(
                    252
                )  # Annualized
            elif FIELD_EVAL_SKEWNESS in perf_field:
                result[perf_field] = skew(
                    ticker_data.loc[:target_index, FIELD_EVAL_RETURN]
                )
            elif FIELD_EVAL_MEAN in perf_field:
                mean_return = np.mean(
                    ticker_data.loc[:target_index, FIELD_EVAL_RETURN] - 1
                )
                result[perf_field] = mean_return * 100
            elif FIELD_EVAL_VAR_95 in perf_field:
                """
                Value at Risk (VaR) Explanation
                VaR at the 95% confidence level means that there is a 95% chance that the loss on an investment will not exceed a certain amount over a specified period. Conversely, there is a 5% chance that the loss will exceed this amount.

                Steps to Calculate VaR Using the Variance-Covariance Method
                Collect Historical Data: Gather historical return data for the asset.

                Calculate Mean and Standard Deviation: Compute the mean (average) return and the standard deviation of returns.

                Determine the Z-Score: For a 95% confidence level, the Z-score is approximately 1.64485.

                Compute VaR: Multiply the Z-score by the standard deviation and subtract the mean return (considering negative returns):
                """
                returns = ticker_data.loc[:target_index, FIELD_EVAL_RETURN]
                log_returns = np.log(returns.where(returns != 0, 1))
                mean_return = np.mean(log_returns)
                std_dev = np.std(log_returns)
                z_score = -1.64485  # 95% confidence level
                var_95 = mean_return + z_score * std_dev
                result[perf_field] = (1 - np.exp(var_95)) * 100
            elif FIELD_EVAL_VAR_99 in perf_field:
                # Compute log returns
                returns = ticker_data.loc[:target_index, FIELD_EVAL_RETURN]
                log_returns = np.log(returns.where(returns != 0, 1))
                mean_return = np.mean(log_returns)
                std_dev = np.std(log_returns)
                z_score = -2.33  # 99% confidence level
                var_99 = mean_return + z_score * std_dev
                result[perf_field] = (1 - np.exp(var_99)) * 100
            elif FIELD_EVAL_TRACKING_ERROR in perf_field and not benchmark_r.empty:
                result[perf_field] = np.std(
                    ticker_data[FIELD_EVAL_RETURN] - benchmark_r[FIELD_EVAL_RETURN]
                ) * np.sqrt(
                    252
                )  # Annualized
            elif FIELD_EVAL_SHARPE_RATIO in perf_field:
                risk_free_rate = 0.01  # Assumed risk-free rate #TODO: acquire the risk-free rates (1M Tsy bond yield is a good proxy)
                result[perf_field] = (
                    (
                        np.mean(ticker_data.loc[:target_index, FIELD_EVAL_RETURN])
                        - risk_free_rate
                    )
                    / np.std(ticker_data.loc[:target_index, FIELD_EVAL_RETURN])
                    * np.sqrt(252)
                )  # Annualized
            elif FIELD_EVAL_BETA in perf_field and not benchmark_r.empty:
                covariance_matrix = np.cov(
                    ticker_data[FIELD_EVAL_RETURN], benchmark_r[FIELD_EVAL_RETURN]
                )
                beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]
                result[perf_field] = beta
            else:
                continue

    return result


def compute_performance_indicators(
    hlds: pd.DataFrame,
    buckets_r: dict,
    portfolio_r: dict,
    benchmark_r: dict,
    perf_fields: list = PERFORMANCE_FIELDS,
) -> pd.DataFrame:

    perf_fields = [cf for cf in perf_fields if cf in PERFORMANCE_FIELDS]
    if len(perf_fields) == 0:
        print("No valid peformance field provided.")
        return hlds
    if hlds.empty:
        return hlds

    # Initialize the new columns with NaN
    for indicator in perf_fields:
        hlds[indicator] = pd.NA

    # Iterate over each row and update the hlds DataFrame
    for idx, row in hlds.iterrows():
        indicators = get_perf_indicator(
            row["ticker"], row["date"], hlds, perf_fields, pd.DataFrame(benchmark_r)
        )
        for indicator, value in indicators.items():
            if (
                indicator in perf_fields
            ):  # TODO: this condition should be moved inside 'get_char_indicator' and used to do less calcs.
                hlds.at[idx, indicator] = value

    # Compute the performance indicators for the buckets
    for bucket_name, bucket_data in buckets_r.items():
        bucket_df = pd.DataFrame(bucket_data)
        bucket_df[FIELD_TICKER] = [str(bucket_name)] * len(bucket_df)

        for date in bucket_df[FIELD_DATE]:
            indicators = get_perf_indicator(
                bucket_name, date, bucket_df, perf_fields, pd.DataFrame(benchmark_r)
            )
            for field in indicators:
                if field not in buckets_r[bucket_name]:
                    buckets_r[bucket_name][field] = []
                buckets_r[bucket_name][field].append(indicators[field])

    # Compute the performance indicators for the entire portfolio
    portfolio_df = pd.DataFrame(portfolio_r)
    portfolio_df[FIELD_TICKER] = "Portfolio"
    portfolio_df[FIELD_TICKER] = ["Portfolio"] * len(portfolio_df)

    for date in portfolio_df[FIELD_DATE]:
        indicators = get_perf_indicator(
            "Portfolio", date, portfolio_df, perf_fields, pd.DataFrame(benchmark_r)
        )
        for field in indicators:
            if field not in portfolio_r:
                portfolio_r[field] = []
            portfolio_r[field].append(indicators[field])

    return hlds, portfolio_r, buckets_r
