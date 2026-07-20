from typing import Tuple
from oport import fields
from oport.ticker_resolver import *


import pandas as pd
import numpy as np

from datetime import datetime

import numba

import warnings


DENOM_TOTAL_RETURN_BASIS = "DENOM_TRB"
DENOM_MARKET_VALUE_WEIGHTS_HLDS = "DENOM_MVW"
NOT_TO_AGGREGATE_FIELDS = list(
    set(
        fields.BASIC_POSITION_FIELDS
        + fields.BASIC_POSITON_DESCR_FIELDS
        + fields.BASIC_MARKET_DATA_INPUT_FIELDS
    )
)


def get_denom_for_date(field: str, date: datetime.date, data: dict):
    try:
        index = data[fields.FIELD_DATE].index(date)
        return data[field][index]
    except:
        return 0


def trim_trailing_nones(dict_of_lists):
    for key, lst in dict_of_lists.items():
        last_non_none = next(
            (i for i in reversed(range(len(lst))) if lst[i] is not None), -1
        )
        dict_of_lists[key] = lst[: last_non_none + 1]
    return dict_of_lists


# numba.jit

from typing import Tuple, List
import warnings


def aggregate(
    portfolio_daily_fields: pd.DataFrame,
    start_date: datetime.date = None,
    end_date: datetime.date = None,
    classification: List = None,
    time_aggregation: bool = False,
    metrics=fields.BASIC_HOLDINGS_FIELDS + fields.CHARACTERISTICS_FIELDS,
    performance_attribution: bool = False,
) -> Tuple[pd.DataFrame, dict, dict]:

    if portfolio_daily_fields.empty:
        return portfolio_daily_fields, {}, {}

    # Ensure we have a value for classification even if it's empty
    if classification is not None:
        for field in classification:
            if field in portfolio_daily_fields.columns:
                portfolio_daily_fields[field] = portfolio_daily_fields[field].ffill()
                portfolio_daily_fields[field] = portfolio_daily_fields[field].fillna(
                    fields.TREE_NOT_CLASSIFIED_VALUE
                )

    warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)

    # Filter metrics and conditionally add performance metrics
    metrics = [m for m in metrics if m in portfolio_daily_fields.columns]
    # Weights has to be addedd to the metrics
    metrics.extend(
        [
            fields.FIELD_EVAL_MKTVALUE_WEIGHTS_HLDS,
            fields.FIELD_EVAL_TOTAL_RETURN_BASIS_WEIGHTS,
            fields.FIELD_EVAL_POSITION_BASIS_WEIGHTS,
            fields.FIELD_EVAL_CTR_TO_RETURN,
            fields.FIELD_EVAL_CTR_TO_RETURN_LCL_CCY,
        ]
    )
    metrics = list(set(metrics))

    # Filter by date
    DATE_D = fields.FIELD_DATE + "date"
    portfolio_daily_fields[DATE_D] = portfolio_daily_fields[fields.FIELD_DATE]
    if start_date and end_date:
        portfolio_daily_fields = portfolio_daily_fields[
            (portfolio_daily_fields[DATE_D] >= start_date)
            & (portfolio_daily_fields[DATE_D] <= end_date)
        ]
        print(f"Aggregation from {start_date} to {end_date}.")
        if portfolio_daily_fields.empty:
            return portfolio_daily_fields, {}, {}
    else:
        print("Aggregation since inception")

    # Initialize portfolio_series with the correct shape using numpy for speed
    portfolio_time_frame_len = len(portfolio_daily_fields[DATE_D])
    portfolio_series = {
        k: [None] * portfolio_time_frame_len
        for k in metrics
        + [fields.FIELD_DATE, DENOM_TOTAL_RETURN_BASIS, DENOM_MARKET_VALUE_WEIGHTS_HLDS]
    }

    # Vectorized operations for date and weights
    last_date = max(portfolio_daily_fields[DATE_D])
    daily_sanpshots = portfolio_daily_fields.groupby([DATE_D])
    portfolio_series[fields.FIELD_DATE] = daily_sanpshots.first().index.tolist()
    portfolio_series[DENOM_TOTAL_RETURN_BASIS] = (
        daily_sanpshots[fields.FIELD_EVAL_M2M_BOD].sum().tolist()
    )
    portfolio_series[DENOM_MARKET_VALUE_WEIGHTS_HLDS] = (
        daily_sanpshots[fields.FIELD_EVAL_MKTVALUE].sum().tolist()
    )
    portfolio_series_denominator = pd.DataFrame(
        {
            fields.FIELD_DATE: portfolio_series[fields.FIELD_DATE],
            DENOM_TOTAL_RETURN_BASIS: portfolio_series[DENOM_TOTAL_RETURN_BASIS],
            DENOM_MARKET_VALUE_WEIGHTS_HLDS: portfolio_series[
                DENOM_MARKET_VALUE_WEIGHTS_HLDS
            ],
        }
    )

    # assets per day : portfolio
    for _, snapshot in daily_sanpshots:
        compute_weights_and_ctr(
            snapshot,
            portfolio_series_denominator,
            performance_attribution=performance_attribution,
        )
        # set internally computed fields into the source dataframe:
        if fields.FIELD_EVAL_MKTVALUE_WEIGHTS_HLDS in snapshot.columns:
            portfolio_daily_fields.loc[
                snapshot.index, fields.FIELD_EVAL_MKTVALUE_WEIGHTS_HLDS
            ] = snapshot[fields.FIELD_EVAL_MKTVALUE_WEIGHTS_HLDS]
        if fields.FIELD_EVAL_POSITION_BASIS_WEIGHTS in snapshot.columns:
            portfolio_daily_fields.loc[
                snapshot.index, fields.FIELD_EVAL_POSITION_BASIS_WEIGHTS
            ] = snapshot[fields.FIELD_EVAL_POSITION_BASIS_WEIGHTS]
        if fields.FIELD_EVAL_TOTAL_RETURN_BASIS_WEIGHTS in snapshot.columns:
            portfolio_daily_fields.loc[
                snapshot.index, fields.FIELD_EVAL_TOTAL_RETURN_BASIS_WEIGHTS
            ] = snapshot[fields.FIELD_EVAL_TOTAL_RETURN_BASIS_WEIGHTS]
            portfolio_daily_fields.loc[
                snapshot.index, fields.FIELD_EVAL_CTR_TO_RETURN
            ] = snapshot[fields.FIELD_EVAL_TOTAL_RETURN_BASIS_WEIGHTS] * (
                snapshot[fields.FIELD_EVAL_RETURN] - 1
            )
            portfolio_daily_fields.loc[
                snapshot.index, fields.FIELD_EVAL_CTR_TO_RETURN_LCL_CCY
            ] = snapshot[fields.FIELD_EVAL_TOTAL_RETURN_BASIS_WEIGHTS] * (
                snapshot[fields.FIELD_EVAL_RETURN_LCL_CCY] - 1
            )
        # set the position id for positions row if there is no classification (i.e. no bucketing)
        if classification == None:
            position_id = ""
            portfolio_daily_fields.loc[
                snapshot.index, fields.FIELD_TREE_POSITION_ID
            ] = [position_id] * len(snapshot[fields.FIELD_DATE])
            portfolio_daily_fields.loc[
                snapshot.index, fields.FIELD_TREE_POSITION_ID
            ] = ("\\" + portfolio_daily_fields.loc[snapshot.index, fields.FIELD_TICKER])

    for metric in metrics:

        warnings.filterwarnings("ignore")
        # Here we ensure we're working with a DataFrame for the grouped data
        result = daily_sanpshots.apply(
            lambda x: aggregate_metrics_on_single_period_snapshot(
                x.reset_index(drop=True),
                metric,
                portfolio_series_denominator,
                bucket=False,
                classification=classification,
            ),
            include_groups=True,  # This will be deprecated in future pandas versions!!!!
        )
        # Since apply might return a Series, we convert it to a list here
        portfolio_series[metric] = (
            result.tolist()
            if isinstance(result, pd.Series)
            else [result] * portfolio_time_frame_len
        )

    # Handle classification if provided
    buckets_series = {}
    if classification:
        for level in range(1, len(classification) + 1):
            for name, group in portfolio_daily_fields.groupby(classification[:level]):
                bucket = tuple(name) if isinstance(name, tuple) else (name,)
                bucket_series = {
                    k: [None] * portfolio_time_frame_len
                    for k in metrics
                    + [
                        fields.FIELD_DATE,
                        fields.FIELD_EVAL_MKTVALUE_WEIGHTS_HLDS,
                        fields.FIELD_EVAL_TOTAL_RETURN_BASIS_WEIGHTS,
                        fields.FIELD_EVAL_POSITION_BASIS_WEIGHTS,
                    ]
                }
                bucket_grouped = group.groupby(DATE_D)
                bucket_series[fields.FIELD_DATE] = bucket_grouped.first().index.tolist()
                for metric in metrics:
                    result = bucket_grouped.apply(
                        lambda x: aggregate_metrics_on_single_period_snapshot(
                            x.reset_index(drop=True),
                            metric,
                            portfolio_series_denominator,
                            bucket=True,
                            classification=classification,
                        ),
                        include_groups=False,
                    )
                    bucket_series[metric] = (
                        result.tolist()
                        if isinstance(result, pd.Series)
                        else [result] * portfolio_time_frame_len
                    )
                buckets_series[bucket] = bucket_series
                # init bucket dictionary:
                position_id = (
                    str(bucket)
                    .replace("(", "[")
                    .replace(")", "]")
                    .replace(" ", "")
                    .replace(",", "\\")
                    .replace("'", "")
                )
                # set position ids if the current level is the last one
                if level == len(classification):
                    portfolio_daily_fields.loc[
                        group.index, fields.FIELD_TREE_POSITION_ID
                    ] = [position_id] * len(group[fields.FIELD_DATE])
                    portfolio_daily_fields.loc[
                        group.index, fields.FIELD_TREE_POSITION_ID
                    ] = (
                        portfolio_daily_fields.loc[
                            group.index, fields.FIELD_TREE_POSITION_ID
                        ]
                        + "\\"
                        + portfolio_daily_fields.loc[group.index, fields.FIELD_TICKER]
                    )

    # Post-processing
    for series in [portfolio_series] + list(buckets_series.values()):
        for key in series:
            if key == fields.FIELD_DATE:
                series[key] = [date for date in series[key]]
            else:
                trim_trailing_nones(series)

    if time_aggregation:
        portfolio_series = accumulte_over_time(portfolio_series)
        for bucket in buckets_series:
            buckets_series[bucket] = accumulte_over_time(buckets_series[bucket])

    portfolio_daily_fields.drop(DATE_D, axis=1, inplace=True)
    return portfolio_daily_fields, portfolio_series, buckets_series


def compute_weights_and_ctr(
    snapshot: pd.DataFrame,
    portfolio_series_denominator: pd.DataFrame,
    bucket: bool = False,
    performance_attribution: bool = False,
) -> float:

    snapshot_date = snapshot[fields.FIELD_DATE].values[0]
    denom = portfolio_series_denominator.loc[
        portfolio_series_denominator[fields.FIELD_DATE] == snapshot_date,
        DENOM_TOTAL_RETURN_BASIS,
    ].values[0]
    denom_mktvalue = portfolio_series_denominator.loc[
        portfolio_series_denominator[fields.FIELD_DATE] == snapshot_date,
        DENOM_MARKET_VALUE_WEIGHTS_HLDS,
    ].values[0]

    # Compute weights and contributions
    if denom != 0:
        snapshot[fields.FIELD_EVAL_TOTAL_RETURN_BASIS_WEIGHTS] = (
            snapshot[fields.FIELD_EVAL_TOTAL_RETURN_BASIS] / denom
        )
        snapshot[fields.FIELD_EVAL_POSITION_BASIS_WEIGHTS] = (
            snapshot[fields.FIELD_EVAL_POSITION_BASIS] / denom
        )
        if performance_attribution:

            snapshot[fields.FIELD_EVAL_CTR_TO_RETURN] = snapshot[
                fields.FIELD_EVAL_TOTAL_RETURN_BASIS_WEIGHTS
            ] * (snapshot[fields.FIELD_EVAL_TRANSACTION_RETURN] - 1) + snapshot[
                fields.FIELD_EVAL_POSITION_BASIS_WEIGHTS
            ] * (
                snapshot[fields.FIELD_EVAL_RETURN_HLDS] - 1
            )

            snapshot[fields.FIELD_EVAL_CTR_TO_RETURN_LCL_CCY] = snapshot[
                fields.FIELD_EVAL_TOTAL_RETURN_BASIS_WEIGHTS
            ] * (snapshot[fields.FIELD_EVAL_TRANSACTION_RETURN_LCL_CCY] - 1) + snapshot[
                fields.FIELD_EVAL_POSITION_BASIS_WEIGHTS
            ] * (
                snapshot[fields.FIELD_EVAL_RETURN_HLDS_LCL_CCY] - 1
            )

    if denom_mktvalue != 0:
        snapshot[fields.FIELD_EVAL_MKTVALUE_WEIGHTS_HLDS] = (
            snapshot[fields.FIELD_EVAL_MKTVALUE] / denom_mktvalue * 100
        )


def aggregate_metrics_on_single_period_snapshot(
    snapshot: pd.DataFrame,
    m: str,
    portfolio_series_denominator: pd.DataFrame,
    bucket: bool = False,
    classification: List = None,
) -> float:
    # Check if 'm' is in the DataFrame's columns
    """
    Aggregate metrics on a single period snapshot.

    This function processes a DataFrame snapshot to aggregate various financial metrics.
    It checks if the specified metric 'm' exists in the DataFrame's columns, and calculates
    the aggregate value based on the type of metric and whether it is part of a bucket.

    Args:
        snapshot (pd.DataFrame): DataFrame containing financial data for a specific period.
        m (str): The metric to be aggregated from the snapshot.
        portfolio_series_denominator (pd.DataFrame): DataFrame providing denominators for
            weight calculations.
        bucket (bool, optional): Flag indicating if the aggregation is for a bucket. Defaults
            to False.

    Returns:
        float: The aggregated value for the specified metric 'm'.
    """

    if m not in snapshot.columns:
        return None

    snapshot_date = snapshot[fields.FIELD_DATE].values[0]
    denom = portfolio_series_denominator.loc[
        portfolio_series_denominator[fields.FIELD_DATE] == snapshot_date,
        DENOM_TOTAL_RETURN_BASIS,
    ].values[0]

    if denom != 0:
        denom_bucket = (
            1.0
            if not bucket
            else snapshot[fields.FIELD_EVAL_TOTAL_RETURN_BASIS_WEIGHTS].sum(skipna=True)
        )
        denom_bucket_hlds = (
            1.0
            if not bucket
            else snapshot[fields.FIELD_EVAL_POSITION_BASIS_WEIGHTS].sum(skipna=True)
        )

    snapshot = snapshot.dropna(subset=[m])
    if snapshot.empty:
        return 1 if "return" in m and "ctr" not in m else 0

    if classification is not None:
        if m in classification:
            return "nan"

    if m in fields.CHARACTERISTICS_FIELDS:
        return (
            snapshot[m] * snapshot[fields.FIELD_EVAL_MKTVALUE_WEIGHTS_HLDS] / 100
        ).sum(skipna=True)

    elif m in [
        fields.FIELD_EVAL_TRANSACTION_RETURN,
        fields.FIELD_EVAL_TRANSACTION_RETURN_LCL_CCY,
    ]:
        if denom != 0:
            if bucket and not np.isnan(denom_bucket):
                if denom_bucket == 0:
                    return float("nan")
                else:
                    return (
                        (snapshot[m] - 1)
                        * snapshot[fields.FIELD_EVAL_TOTAL_RETURN_BASIS_WEIGHTS]
                    ).sum(skipna=True) / denom_bucket + 1
            else:
                return (
                    (snapshot[m] - 1)
                    * snapshot[fields.FIELD_EVAL_TOTAL_RETURN_BASIS_WEIGHTS]
                ).sum(skipna=True) + 1
        else:
            return float("nan")

    elif m in [fields.FIELD_EVAL_RETURN_HLDS, fields.FIELD_EVAL_RETURN_HLDS_LCL_CCY]:
        if denom != 0:
            if bucket and not np.isnan(denom_bucket_hlds):
                if denom_bucket_hlds == 0:
                    return float("nan")
                else:
                    return (
                        (snapshot[m] - 1)
                        * snapshot[fields.FIELD_EVAL_POSITION_BASIS_WEIGHTS]
                    ).sum(skipna=True) / denom_bucket_hlds + 1
            else:
                return (
                    (snapshot[m] - 1)
                    * snapshot[fields.FIELD_EVAL_POSITION_BASIS_WEIGHTS]
                ).sum(skipna=True) + 1
        else:
            return float("nan")

    elif m == fields.FIELD_EVAL_RETURN:
        holdings_return = 0
        trading_return = 0
        if denom != 0:
            if bucket:
                if denom_bucket_hlds == 0 or np.isnan(denom_bucket_hlds):
                    holdings_return = 1
                else:
                    holdings_return = (
                        (snapshot[fields.FIELD_EVAL_RETURN_HLDS] - 1)
                        * snapshot[fields.FIELD_EVAL_POSITION_BASIS_WEIGHTS]
                    ).sum(skipna=True) / denom_bucket_hlds + 1
                if denom_bucket == 0 or np.isnan(denom_bucket):
                    trading_return = 1
                else:
                    trading_return = (
                        (snapshot[fields.FIELD_EVAL_TRANSACTION_RETURN] - 1)
                        * snapshot[fields.FIELD_EVAL_TOTAL_RETURN_BASIS_WEIGHTS]
                    ).sum(skipna=True) / denom_bucket + 1
            else:
                holdings_return = (
                    (snapshot[fields.FIELD_EVAL_RETURN_HLDS] - 1)
                    * snapshot[fields.FIELD_EVAL_POSITION_BASIS_WEIGHTS]
                ).sum(skipna=True) + 1
                trading_return = (
                    (snapshot[fields.FIELD_EVAL_TRANSACTION_RETURN] - 1)
                    * snapshot[fields.FIELD_EVAL_TOTAL_RETURN_BASIS_WEIGHTS]
                ).sum(skipna=True) + 1

            total_return = holdings_return + trading_return - 1
            return total_return
        else:
            return float("nan")

    elif m == fields.FIELD_EVAL_RETURN_LCL_CCY:
        holdings_return = 0
        trading_return = 0
        if denom != 0:
            if bucket:
                if denom_bucket_hlds == 0 or np.isnan(denom_bucket_hlds):
                    holdings_return = 1
                else:
                    holdings_return = (
                        (snapshot[fields.FIELD_EVAL_RETURN_HLDS_LCL_CCY] - 1)
                        * snapshot[fields.FIELD_EVAL_POSITION_BASIS_WEIGHTS]
                    ).sum(skipna=True) / denom_bucket_hlds + 1
                if denom_bucket == 0 or np.isnan(denom_bucket):
                    trading_return = 1
                else:
                    trading_return = (
                        (snapshot[fields.FIELD_EVAL_TRANSACTION_RETURN_LCL_CCY] - 1)
                        * snapshot[fields.FIELD_EVAL_TOTAL_RETURN_BASIS_WEIGHTS]
                    ).sum(skipna=True) / denom_bucket + 1
            else:
                holdings_return = (
                    (snapshot[fields.FIELD_EVAL_RETURN_HLDS_LCL_CCY] - 1)
                    * snapshot[fields.FIELD_EVAL_POSITION_BASIS_WEIGHTS]
                ).sum(skipna=True) + 1
                trading_return = (
                    (snapshot[fields.FIELD_EVAL_TRANSACTION_RETURN_LCL_CCY] - 1)
                    * snapshot[fields.FIELD_EVAL_TOTAL_RETURN_BASIS_WEIGHTS]
                ).sum(skipna=True) + 1

            total_return = holdings_return + trading_return - 1
            return total_return
        else:
            return float("nan")

    elif m == fields.FIELD_EVAL_CCY_RETURN:
        holdings_return = 0
        trading_return = 0
        if denom != 0:
            if bucket:
                if denom_bucket_hlds == 0 or np.isnan(denom_bucket_hlds):
                    holdings_return = 1
                else:
                    holdings_return = (
                        (
                            (snapshot[fields.FIELD_EVAL_RETURN_HLDS] - 1)
                            - (snapshot[fields.FIELD_EVAL_RETURN_HLDS_LCL_CCY] - 1)
                        )
                        * snapshot[fields.FIELD_EVAL_POSITION_BASIS_WEIGHTS]
                    ).sum(skipna=True) / denom_bucket_hlds + 1
                if denom_bucket == 0 or np.isnan(denom_bucket):
                    trading_return = 1
                else:
                    trading_return = (
                        (
                            (snapshot[fields.FIELD_EVAL_TRANSACTION_RETURN] - 1)
                            - (
                                snapshot[fields.FIELD_EVAL_TRANSACTION_RETURN_LCL_CCY]
                                - 1
                            )
                        )
                        * snapshot[fields.FIELD_EVAL_TOTAL_RETURN_BASIS_WEIGHTS]
                    ).sum(skipna=True) / denom_bucket + 1
            else:
                holdings_return = (
                    (
                        (snapshot[fields.FIELD_EVAL_RETURN_HLDS] - 1)
                        - (snapshot[fields.FIELD_EVAL_RETURN_HLDS_LCL_CCY] - 1)
                    )
                    * snapshot[fields.FIELD_EVAL_POSITION_BASIS_WEIGHTS]
                ).sum(skipna=True) + 1
                trading_return = (
                    (
                        (snapshot[fields.FIELD_EVAL_TRANSACTION_RETURN] - 1)
                        - (snapshot[fields.FIELD_EVAL_TRANSACTION_RETURN_LCL_CCY] - 1)
                    )
                    * snapshot[fields.FIELD_EVAL_TOTAL_RETURN_BASIS_WEIGHTS]
                ).sum(skipna=True) + 1

            total_return = holdings_return + trading_return - 1
            return total_return
        else:
            return float("nan")

    elif m in [
        fields.FIELD_EVAL_CTR_TO_RETURN,
        fields.FIELD_EVAL_CTR_TO_RETURN_LCL_CCY,
    ] or ("pnl" in m and "lcl_ccy" not in m):
        if denom != 0:
            return snapshot[m].sum(skipna=True)
        else:
            return 0
    else:
        if m in NOT_TO_AGGREGATE_FIELDS:
            if m == fields.FIELD_DATE:
                return snapshot_date
            else:
                return float("nan")
        else:
            return snapshot[m].sum(skipna=True)


def accumulte_over_time(snapshots: dict) -> dict:
    """
    Aggregate metrics over time into the snapshots dictionary.
    Applies Carino linking for return fields and cumulative summation for PNL fields.
    Modifies the 'snapshots' dictionary in place by adding accumulated metrics.

    Args:
        snapshots (dict): Dictionary containing time-series data for metrics.

    Returns:
        dict: The modified snapshots dictionary with new accumulated metric fields.
    """
    from .linking import carino_linking_return_split
    import numpy as np
    import pandas as pd

    # Select daily metrics present in snapshots and defined in ACC_OVER_TIME_DAILY_FIELDS
    daily_metrics = [
        field for field in fields.ACC_OVER_TIME_DAILY_FIELDS if field in snapshots
    ]

    if not daily_metrics:
        return snapshots  # Return unchanged if no valid metrics found

    # Map daily metrics to their accumulated counterparts
    acc_metrics = [
        fields.accumulate_over_time_field_realtion_dict[field]
        for field in daily_metrics
        if field in fields.accumulate_over_time_field_realtion_dict
    ]

    # Initialize accumulators for all metrics
    accumulated_fields = {}
    for daily_field, acc_field in zip(daily_metrics, acc_metrics):
        # Initialize with 1.0 for returns, 0.0 for PNLs
        initial_value = 1.0 if "return" in daily_field.lower() else 0.0
        accumulated_fields[acc_field] = [initial_value]
        # Set initial snapshot value for daily return fields
        if "return" in daily_field.lower():
            snapshots[daily_field][0] = 1.0

    # Process each time step
    for i in range(len(snapshots[daily_metrics[0]])):
        # Handle total return first (needed for Carino linking)
        if fields.FIELD_EVAL_RETURN in daily_metrics:
            # Replace 0.0 with 1.0 to prevent errors in return calculations
            if snapshots[fields.FIELD_EVAL_RETURN][i] == 0:
                snapshots[fields.FIELD_EVAL_RETURN][i] = 1.0
            acc_total_return_i = (
                accumulated_fields[fields.FIELD_EVAL_ACC_T_RETURN][-1]
                * snapshots[fields.FIELD_EVAL_RETURN][i]
            )
            accumulated_fields[fields.FIELD_EVAL_ACC_T_RETURN].append(
                acc_total_return_i
            )
        else:
            acc_total_return_i = 1.0  # Default if total return not provided

        # Process other metrics
        for daily_field, acc_field in zip(daily_metrics, acc_metrics):
            # Skip total return as it's already handled
            if daily_field == fields.FIELD_EVAL_RETURN:
                continue

            # Handle return metrics with Carino linking
            if "return" in daily_field.lower():
                # Replace 0.0 with 1.0 to prevent errors
                if snapshots[daily_field][i] == 0:
                    snapshots[daily_field][i] = 1.0
                return_slice = (
                    np.array(snapshots[fields.FIELD_EVAL_RETURN][: i + 1])
                    if fields.FIELD_EVAL_RETURN in daily_metrics
                    else np.ones(i + 1)
                )
                return_component_slice = np.array(snapshots[daily_field][: i + 1])
                acc_return_component = carino_linking_return_split(
                    return_component_slice, return_slice, acc_total_return_i
                )
                accumulated_fields[acc_field].append(acc_return_component)

            # Handle PNL metrics with summation
            elif "pnl" in daily_field.lower():
                acc_pnl_i = (
                    accumulated_fields[acc_field][-1] + snapshots[daily_field][i]
                )
                accumulated_fields[acc_field].append(acc_pnl_i)

    # Assign accumulated values back to snapshots
    for acc_field in acc_metrics:
        snapshots[acc_field] = pd.Series(data=accumulated_fields[acc_field][1:])

    return snapshots


def accumulte_over_time_positions(portfolio_daily_fields: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate metrics over time for each asset in the portfolio_daily_fields DataFrame.
    Applies Carino linking for return fields and cumulative summation for PNL.
    Modifies the input DataFrame in place by adding accumulated metrics.

    Args:
        portfolio_daily_fields (pd.DataFrame): DataFrame containing daily portfolio data with asset IDs and metrics.

    Returns:
        pd.DataFrame: The modified DataFrame with new columns for accumulated metrics.
    """
    from .linking import carino_linking_return_split

    if portfolio_daily_fields.empty:
        return portfolio_daily_fields

    # Ensure date is in datetime.date format
    portfolio_daily_fields[fields.FIELD_DATE] = pd.to_datetime(
        portfolio_daily_fields[fields.FIELD_DATE]
    ).dt.date

    # Initialize new columns for accumulated metrics
    daily_metrics = [
        daily_field
        for daily_field in fields.ACC_OVER_TIME_DAILY_FIELDS
        if daily_field in portfolio_daily_fields.columns
    ]
    acc_metrics = [
        fields.accumulate_over_time_field_realtion_dict[daily_field]
        for daily_field in daily_metrics
    ]

    print(daily_metrics)
    print(acc_metrics)

    for metric in acc_metrics:
        if metric not in portfolio_daily_fields.columns:
            portfolio_daily_fields[metric] = np.nan

    # Group by asset ID
    grouped = portfolio_daily_fields.groupby(fields.FIELD_ASSET_ID)

    for asset_id, group in grouped:
        # Sort by date to ensure chronological order
        group = group.sort_values(fields.FIELD_DATE)
        n_rows = len(group)

        # Initialize accumulators for all fields in ACC_OVER_TIME_FIELDS
        accumulated_fields = {
            field: [0.0] if "pnl" in field else [1.0] for field in acc_metrics
        }
        daily_fields = dict()
        for field in daily_metrics:
            if "pnl" in field:
                daily_fields[field] = group[field].fillna(0.0).values
            elif "return" in field:
                daily_fields[field] = group[field].fillna(1.0).values
            else:
                continue

        for i in range(n_rows):
            # Total return
            if fields.FIELD_EVAL_RETURN in daily_fields:
                acc_total_return_i = (
                    accumulated_fields[fields.FIELD_EVAL_ACC_T_RETURN][-1]
                    * daily_fields[fields.FIELD_EVAL_RETURN][i]
                )
                accumulated_fields[fields.FIELD_EVAL_ACC_T_RETURN].append(
                    acc_total_return_i
                )

            # Slice up to current point for linking
            return_slice = accumulated_fields[fields.FIELD_EVAL_ACC_T_RETURN][: i + 1]

            for field in daily_metrics:
                if (
                    "return" in field
                    and field != fields.FIELD_EVAL_RETURN
                    and fields.FIELD_EVAL_RETURN in daily_fields
                ):
                    return_component_slice = daily_fields[field][: i + 1]
                    acc_return_component = carino_linking_return_split(
                        return_component_slice, return_slice, acc_total_return_i
                    )
                    accumulated_fields[
                        fields.accumulate_over_time_field_realtion_dict[field]
                    ].append(acc_return_component)
                elif "pnl" in field:
                    acc_pnl_i = (
                        accumulated_fields[
                            fields.accumulate_over_time_field_realtion_dict[field]
                        ][-1]
                        + daily_fields[field][i]
                    )
                    accumulated_fields[
                        fields.accumulate_over_time_field_realtion_dict[field]
                    ].append(acc_pnl_i)
                else:
                    continue

        # Assign accumulated values back to the group
        indices = group.index
        for acc_metric in acc_metrics:
            if len(accumulated_fields[acc_metric][1:]) != n_rows:
                print(
                    f"Warning: Length mismatch for asset {asset_id}, metric {acc_metric}. Expected {n_rows}, got {len(accumulated_fields[acc_metric][1:])}"
                )
            else:
                portfolio_daily_fields.loc[indices, acc_metric] = accumulated_fields[
                    acc_metric
                ][1:]

    return portfolio_daily_fields


def adjust_returns_dataframe(hlds: pd.DataFrame) -> pd.DataFrame:
    """
    Adjust the returns in the portfolio_hlds DataFrame from the format X -> (X - 1) * 100.
    """
    if hlds.empty:
        return hlds

    # Adjust return fields in the portfolio_hlds dataframe
    for field in fields.RETURNS_1_BASE:
        if field in hlds.columns:
            hlds[field] = (
                (hlds[field] - 1) * 100
                if field
                not in [
                    fields.FIELD_EVAL_CTR_TO_RETURN + fields.TREE_BENCHMARK_SUFFIX,
                    fields.FIELD_EVAL_CTR_TO_RETURN_LCL_CCY
                    + fields.TREE_BENCHMARK_SUFFIX,
                    fields.FIELD_EVAL_CTR_TO_RETURN,
                    fields.FIELD_EVAL_CTR_TO_RETURN_LCL_CCY,
                ]
                else hlds[field] * 100
            )

    return hlds


def adjust_returns_dicts(portfolio_r: dict, buckets_r: dict):
    """
    Converts the returns in the given dictionaries from the format X -> (X - 1) * 100.
    """
    if portfolio_r:
        for key, value in portfolio_r.items():
            if key in [
                fields.FIELD_EVAL_RETURN,
                fields.FIELD_EVAL_RETURN_LCL_CCY,
                fields.FIELD_EVAL_CCY_RETURN,
                fields.FIELD_EVAL_RETURN_HLDS,
                fields.FIELD_EVAL_RETURN_HLDS_LCL_CCY,
                fields.FIELD_EVAL_ACC_T_RETURN,
                fields.FIELD_EVAL_ACC_T_RETURN_LCL_CCY,
                fields.FIELD_EVAL_ACC_T_CCY_RETURN,
            ]:
                portfolio_r[key] = [
                    (v - 1) * 100 if v is not None else None for v in value
                ]
            else:
                portfolio_r[key] = [v if v is not None else None for v in value]

    if buckets_r:
        for bucket_name, bucket_data in buckets_r.items():
            for key, value in bucket_data.items():
                if key in [
                    fields.FIELD_EVAL_RETURN,
                    fields.FIELD_EVAL_RETURN_LCL_CCY,
                    fields.FIELD_EVAL_CCY_RETURN,
                    fields.FIELD_EVAL_RETURN_HLDS,
                    fields.FIELD_EVAL_RETURN_HLDS_LCL_CCY,
                    fields.FIELD_EVAL_ACC_T_RETURN,
                    fields.FIELD_EVAL_ACC_T_RETURN_LCL_CCY,
                    fields.FIELD_EVAL_ACC_T_CCY_RETURN,
                ]:
                    buckets_r[bucket_name][key] = [
                        (v - 1) * 100 if v is not None else None for v in value
                    ]
                else:
                    buckets_r[bucket_name][key] = [
                        v if v is not None else None for v in value
                    ]

    return portfolio_r, buckets_r


def resolve_time_portf_vs_bench_time_frame(
    daily_performance, daily_performance_benchmark, start_date, end_date
):
    """
    Adjust the start and end date to the maximum of the start dates and the minimum of the end dates
    of the two dataframes.
    """
    if not daily_performance_benchmark.empty:
        portfolio_start_date = min(daily_performance[fields.FIELD_DATE])
        benchmark_start_date = min(daily_performance_benchmark[fields.FIELD_DATE])
        portfolio_end_date = max(daily_performance[fields.FIELD_DATE])
        benchmark_end_date = max(daily_performance_benchmark[fields.FIELD_DATE])
        effective_start_date = max(portfolio_start_date, benchmark_start_date)
        effective_end_date = min(portfolio_end_date, benchmark_end_date)
        if start_date < effective_start_date:
            start_date = effective_start_date
        if end_date > effective_end_date:
            end_date = effective_end_date
    return start_date, end_date
