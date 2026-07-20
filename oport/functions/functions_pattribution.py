import warnings
import datetime

import pandas as pd
import numpy as np

from datetime import datetime

from oport import fields
from .functions_tree import build_snapshot_maps, is_bucket, is_root
from oport import error_collector


def subtract_with_nan_and_suffix(
    snapshot,
    field_base,
):
    """
    Subtracts values from two columns in a DataFrame, handling NaN values.

    This function computes the difference between values in two columns
    derived from a base field with different suffixes, and stores the result
    in a new column. If both values are NaN, the result is NaN. If one value
    is NaN, it is treated as 0 for the subtraction.

    Args:
        snapshot (pd.DataFrame): The DataFrame containing the data.
        field_base (str): The base name of the field to process.

    Modifies:
        snapshot: A new column is added to the DataFrame with the
        differences between the active and benchmark fields.
    """

    active_field = field_base + fields.TREE_ACTIVE_SUFFIX
    benchmark_field = field_base + fields.TREE_BENCHMARK_SUFFIX
    snapshot[active_field] = np.where(
        (np.isnan(snapshot[field_base]) & np.isnan(snapshot[benchmark_field])),
        np.nan,
        np.where(np.isnan(snapshot[field_base]), 0, snapshot[field_base])
        - np.where(np.isnan(snapshot[benchmark_field]), 0, snapshot[benchmark_field]),
    )


def subtract_with_nan_and_suffix_1based(
    snapshot,
    field_base,
):
    """
    Subtracts values from two columns in a DataFrame, handling NaN values.

    This function computes the difference between values in two columns
    derived from a base field with different suffixes, and stores the result
    in a new column. If both values are NaN, the result is NaN. If one value
    is NaN, it is treated as 1 for the subtraction.

    Args:
        snapshot (pd.DataFrame): The DataFrame containing the data.
        field_base (str): The base name of the field to process.

    Modifies:
        snapshot: A new column is added to the DataFrame with the
        differences between the active and benchmark fields.
    """

    active_field = field_base + fields.TREE_ACTIVE_SUFFIX
    benchmark_field = field_base + fields.TREE_BENCHMARK_SUFFIX
    snapshot[active_field] = np.where(
        (np.isnan(snapshot[field_base]) & np.isnan(snapshot[benchmark_field])),
        np.nan,
        np.where(np.isnan(snapshot[field_base]), 1, snapshot[field_base])
        - np.where(np.isnan(snapshot[benchmark_field]), 1, snapshot[benchmark_field]),
    )


def get_portf_vs_bench_indicator(snapshot, field_base, default_value=0):
    """
    Returns the portfolio and benchmark values for a given field_base.
    If either value is NaN, it is replaced with the default_value.
    """
    benchmark_field = field_base + fields.TREE_BENCHMARK_SUFFIX
    portfolio_value = snapshot[field_base].fillna(default_value)
    benchmark_value = snapshot[benchmark_field].fillna(default_value)
    return portfolio_value, benchmark_value


def compute_performance_attribution(
    tree: pd.DataFrame,
    start_date: datetime.date = None,
    end_date: datetime.date = None,
    time_aggregation: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:

    err_collector = error_collector.get_collector()

    if tree.empty:
        err_collector.record_error(
            operation_name="compute_performance_attribution",
            error_details="Attribution tree is empty. No valid constituents",
        )
        raise ValueError("Attribution tree is empty. No valid constituents")

    tree = tree[tree[".tree"] != "nan"]

    if tree.empty:
        err_collector.record_error(
            operation_name="compute_performance_attribution",
            error_details="Attribution tree is empty. No valid constituents",
        )
        raise ValueError("Attribution tree is empty. No valid constituents")

    # Disable copy warnings
    warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)

    # Process each day's snapshot
    for day, snapshot in tree.groupby([fields.FIELD_DATE]):

        if day[0] < start_date:
            continue
        if day[0] > end_date:
            break

        print(
            f"Computing performance attribution for date: {day[0].strftime('%m/%d/%Y')} ..."
        )
        position_id_to_index, parent_map, children_map, parent_index_map = (
            build_snapshot_maps(snapshot)
        )
        # note: inefficient.
        hurdle_rate = snapshot[
            snapshot[fields.FIELD_TREE_POSITION_ID_PVSB].apply(lambda x: is_root(x))
        ][fields.FIELD_EVAL_RETURN_HLDS_LCL_CCY + fields.TREE_BENCHMARK_SUFFIX].values[
            0
        ]
        portfolio_weight = snapshot[
            snapshot[fields.FIELD_TREE_POSITION_ID_PVSB].apply(lambda x: is_root(x))
        ][fields.FIELD_EVAL_POSITION_BASIS_WEIGHTS].values[0]
        benchmark_weight = snapshot[
            snapshot[fields.FIELD_TREE_POSITION_ID_PVSB].apply(lambda x: is_root(x))
        ][
            fields.FIELD_EVAL_POSITION_BASIS_WEIGHTS + fields.TREE_BENCHMARK_SUFFIX
        ].values[
            0
        ]

        # Common active indicators. No special logic at bucket or root level:
        # active total ctr
        subtract_with_nan_and_suffix_1based(snapshot, fields.FIELD_EVAL_RETURN)
        subtract_with_nan_and_suffix_1based(snapshot, fields.FIELD_EVAL_RETURN_LCL_CCY)
        subtract_with_nan_and_suffix(snapshot, fields.FIELD_EVAL_CTR_TO_RETURN)
        subtract_with_nan_and_suffix(snapshot, fields.FIELD_EVAL_CTR_TO_RETURN_LCL_CCY)
        # active total weight
        subtract_with_nan_and_suffix(snapshot, fields.FIELD_EVAL_POSITION_BASIS_WEIGHTS)
        # active trading return in local currency
        subtract_with_nan_and_suffix_1based(
            snapshot, fields.FIELD_EVAL_TRANSACTION_RETURN_LCL_CCY
        )
        # active trading return in local currency
        subtract_with_nan_and_suffix_1based(
            snapshot, fields.FIELD_EVAL_TRANSACTION_RETURN
        )
        # active local returns
        subtract_with_nan_and_suffix_1based(
            snapshot, fields.FIELD_EVAL_RETURN_HLDS_LCL_CCY
        )
        subtract_with_nan_and_suffix_1based(snapshot, fields.FIELD_EVAL_RETURN_HLDS)
        # active currency return
        subtract_with_nan_and_suffix_1based(snapshot, fields.FIELD_EVAL_CCY_RETURN)

        p_weight_snapshot, b_weight_snapshot = get_portf_vs_bench_indicator(
            snapshot, fields.FIELD_EVAL_POSITION_BASIS_WEIGHTS
        )
        p_return_lcl_ccy_snapshot, b_return_lcl_ccy_snapshot = (
            get_portf_vs_bench_indicator(
                snapshot, fields.FIELD_EVAL_RETURN_HLDS_LCL_CCY, default_value=1.0
            )
        )

        if portfolio_weight == 0 or benchmark_weight == 0:
            snapshot.loc[fields.FIELD_EVAL_ALLOCATION] = np.nan
            snapshot.loc[fields.FIELD_EVAL_SELECTION] = np.nan
            snapshot.loc[fields.FIELD_EVAL_LEVERAGE] = np.nan
            continue

        root_index = np.nan

        for p in snapshot[fields.FIELD_TREE_POSITION_ID_PVSB]:
            row_index = position_id_to_index[p]

            p_weight = p_weight_snapshot[row_index]
            b_weight = b_weight_snapshot[row_index]
            p_return_lcl_ccy = p_return_lcl_ccy_snapshot[row_index]
            b_return_lcl_ccy = b_return_lcl_ccy_snapshot[row_index]

            if is_root(p):
                # Access root row if needed: snapshot.loc[row_index]
                if verbose:
                    print(
                        f"Root: {day} : {p} | Active Total Ctr : {snapshot.loc[row_index, fields.FIELD_EVAL_CTR_TO_RETURN + fields.TREE_ACTIVE_SUFFIX]}"
                    )
                    print(
                        f"Root: {day} : {p} | Active Total r : {snapshot.loc[row_index, fields.FIELD_EVAL_RETURN + fields.TREE_ACTIVE_SUFFIX]}"
                    )
                    print(
                        f"Root: {day} : {p} | Acrive trading lcl return: {snapshot.loc[row_index, fields.FIELD_EVAL_TRANSACTION_RETURN_LCL_CCY + fields.TREE_ACTIVE_SUFFIX]}"
                    )
                    print(
                        f"Root: {day} : {p} | Acrive trading return: {snapshot.loc[row_index, fields.FIELD_EVAL_TRANSACTION_RETURN + fields.TREE_ACTIVE_SUFFIX]}"
                    )
                    print(
                        f"Root: {day} : {p} | Acrive ccy return: {snapshot.loc[row_index, fields.FIELD_EVAL_CCY_RETURN + fields.TREE_ACTIVE_SUFFIX]}"
                    )
                    print(
                        f"Root: {day} : {p} | Active Return hlds: {snapshot.loc[row_index, fields.FIELD_EVAL_RETURN_HLDS_LCL_CCY + fields.TREE_ACTIVE_SUFFIX]}, Active Weight: {snapshot.loc[row_index, fields.FIELD_EVAL_POSITION_BASIS_WEIGHTS + fields.TREE_ACTIVE_SUFFIX]}"
                    )
                    print(
                        f"Root: {day} : {p} | r[p]: {snapshot.loc[row_index, fields.FIELD_EVAL_RETURN_HLDS_LCL_CCY]}, r[b]: {snapshot.loc[row_index, fields.FIELD_EVAL_RETURN_HLDS_LCL_CCY + fields.TREE_BENCHMARK_SUFFIX]}"
                    )
                    print(
                        f"Root: {day} : {p} | tr[p]: {snapshot.loc[row_index, fields.FIELD_EVAL_TRANSACTION_RETURN_LCL_CCY]}, tr[b]: {snapshot.loc[row_index, fields.FIELD_EVAL_TRANSACTION_RETURN_LCL_CCY + fields.TREE_BENCHMARK_SUFFIX]}"
                    )
                    print(
                        f"Root: {day} : {p} | ccyr[p]: {snapshot.loc[row_index, fields.FIELD_EVAL_CCY_RETURN]}, ccyr[b]: {snapshot.loc[row_index, fields.FIELD_EVAL_CCY_RETURN + fields.TREE_BENCHMARK_SUFFIX]}"
                    )
                    print(
                        f"Root: {day} : {p} | total r[p]: {snapshot.loc[row_index, fields.FIELD_EVAL_RETURN]}, total r[b]: {snapshot.loc[row_index, fields.FIELD_EVAL_RETURN + fields.TREE_BENCHMARK_SUFFIX]}"
                    )
                    print(
                        f"Root: {day} : {p} | total ctr[p]: {snapshot.loc[row_index, fields.FIELD_EVAL_CTR_TO_RETURN]+1}, total r[b]: {snapshot.loc[row_index, fields.FIELD_EVAL_CTR_TO_RETURN + fields.TREE_BENCHMARK_SUFFIX]+1}"
                    )
                    print(
                        f"Root: {day} : {p} | local r[p]: {snapshot.loc[row_index, fields.FIELD_EVAL_RETURN_LCL_CCY]}, local r[b]: {snapshot.loc[row_index, fields.FIELD_EVAL_RETURN_LCL_CCY + fields.TREE_BENCHMARK_SUFFIX]}"
                    )
                    print(
                        f"Root: {day} : {p} | local hlds r[p]: {snapshot.loc[row_index, fields.FIELD_EVAL_RETURN_HLDS_LCL_CCY]}, local hlds r[b]: {snapshot.loc[row_index, fields.FIELD_EVAL_RETURN_HLDS_LCL_CCY + fields.TREE_BENCHMARK_SUFFIX]}"
                    )
                    print(
                        f"Root: {day} : {p} | Weight[p]: {p_weight}, Weight[b]: {b_weight}"
                    )

                root_p = p
                # Leverage:
                leverage = (p_weight - b_weight) * (b_return_lcl_ccy - 1)
                root_index = row_index
                snapshot.loc[root_index, fields.FIELD_EVAL_LEVERAGE] = leverage

            elif is_bucket(p):
                parent_bucket_idx = parent_index_map.get(p)
                sp_weight = snapshot.loc[
                    parent_bucket_idx, fields.FIELD_EVAL_POSITION_BASIS_WEIGHTS
                ]
                sb_weight = snapshot.loc[
                    parent_bucket_idx,
                    fields.FIELD_EVAL_POSITION_BASIS_WEIGHTS
                    + fields.TREE_BENCHMARK_SUFFIX,
                ]

                # Access bucket row if needed: snapshot.loc[row_index]
                # Access parent index: parent_index_map.get(p)
                # Access offspring indices: children_map[p]

                if is_root(
                    snapshot.loc[parent_bucket_idx][fields.FIELD_TREE_POSITION_ID_PVSB]
                ):

                    # Allocation:
                    allocation = (p_weight - b_weight) * (
                        b_return_lcl_ccy - hurdle_rate
                    )

                    # Selection:
                    selection = np.nan
                    # if p_weight != 0:
                    selection = p_weight * (p_return_lcl_ccy - b_return_lcl_ccy)
                    # else:
                    #    selection = b_weight * (p_return_lcl_ccy - b_return_lcl_ccy)

                    if verbose:
                        print(
                            f"1. level Bucket: {day} : {p} | Allocation: {allocation}, Selection: {selection}"
                        )
                        print(
                            f"1. level Bucket: {day} : {p} | r[p]: {p_return_lcl_ccy}, r[b]: {b_return_lcl_ccy}"
                        )
                        print(
                            f"1. level Bucket: {day} : {p} | Weight[p]: {p_weight}, Weight[b]: {b_weight}"
                        )

                else:
                    # Allocation:
                    allocation = np.nan
                    if sb_weight != 0 and ~np.isnan(sb_weight):
                        allocation = (p_weight - sp_weight * b_weight / sb_weight) * (
                            p_return_lcl_ccy - hurdle_rate
                        )
                    else:
                        allocation = (p_weight) * (p_return_lcl_ccy - hurdle_rate)

                    # Selection:
                    selection = np.nan

                    # if p_weight != 0:
                    selection = p_weight * (p_return_lcl_ccy - b_return_lcl_ccy)
                    # else:
                    #    selection = b_weight * (p_return_lcl_ccy - b_return_lcl_ccy)

                #    pass
            else:  # Leaf
                # Access leaf row if needed: snapshot.loc[row_index]
                # Access parent index: parent_index_map.get(p)
                # Leaf has no children, so children_map[p] is empty
                parent_bucket_idx = parent_index_map.get(p)
                sp_weight = snapshot.loc[
                    parent_bucket_idx, fields.FIELD_EVAL_POSITION_BASIS_WEIGHTS
                ]
                sb_weight = snapshot.loc[
                    parent_bucket_idx,
                    fields.FIELD_EVAL_POSITION_BASIS_WEIGHTS
                    + fields.TREE_BENCHMARK_SUFFIX,
                ]

                allocation = np.nan
                selection = np.nan
                if sb_weight != 0 and ~np.isnan(sb_weight):
                    selection = (p_weight - sp_weight * b_weight / sb_weight) * (
                        b_return_lcl_ccy - hurdle_rate
                    )
                else:
                    selection = (p_weight) * (p_return_lcl_ccy - hurdle_rate)

                # print(f"Leaf: {day} : {p} | Allocation: {allocation}, Selection: {selection}")

            snapshot.loc[row_index, fields.FIELD_EVAL_ALLOCATION] = allocation
            snapshot.loc[row_index, fields.FIELD_EVAL_SELECTION] = selection

            # Example: Access parent and children
            parent_idx = parent_index_map.get(p)
            if parent_idx is not None:
                # parent_row = snapshot.loc[parent_idx]  # Uncomment if needed
                pass
            children_indices = children_map[p]
            if children_indices:
                # children_rows = snapshot.loc[children_indices]  # Uncomment if needed
                pass

        if verbose:
            for pp in children_map[root_p]:
                print(snapshot.loc[pp, fields.FIELD_TREE_POSITION_ID_PVSB])

        allocation = snapshot.loc[
            children_map[root_p], fields.FIELD_EVAL_ALLOCATION
        ].sum()
        selection = snapshot.loc[
            children_map[root_p], fields.FIELD_EVAL_SELECTION
        ].sum()
        snapshot.loc[root_index, fields.FIELD_EVAL_ALLOCATION] = allocation
        snapshot.loc[root_index, fields.FIELD_EVAL_SELECTION] = selection
        if verbose:
            print(
                f"Root: {day} : {root_p} | Allocation: {allocation}, Selection: {selection}, Leverage: {leverage}, Total Active: {allocation + selection + leverage}"
            )
            print("--------------------------")

        # Populate the tree and format in % :
        tree.loc[snapshot.index, fields.FIELD_EVAL_ALLOCATION] = (
            snapshot[fields.FIELD_EVAL_ALLOCATION] * 100
        )
        tree.loc[snapshot.index, fields.FIELD_EVAL_SELECTION] = (
            snapshot[fields.FIELD_EVAL_SELECTION] * 100
        )
        tree.loc[snapshot.index, fields.FIELD_EVAL_LEVERAGE] = (
            snapshot[fields.FIELD_EVAL_LEVERAGE] * 100
        )
        tree.loc[
            snapshot.index, fields.FIELD_EVAL_CTR_TO_RETURN + fields.TREE_ACTIVE_SUFFIX
        ] = (
            snapshot[fields.FIELD_EVAL_CTR_TO_RETURN + fields.TREE_ACTIVE_SUFFIX] * 100
        )
        tree.loc[
            snapshot.index,
            fields.FIELD_EVAL_CTR_TO_RETURN_LCL_CCY + fields.TREE_ACTIVE_SUFFIX,
        ] = (
            snapshot[
                fields.FIELD_EVAL_CTR_TO_RETURN_LCL_CCY + fields.TREE_ACTIVE_SUFFIX
            ]
            * 100
        )
        tree.loc[
            snapshot.index,
            fields.FIELD_EVAL_POSITION_BASIS_WEIGHTS + fields.TREE_ACTIVE_SUFFIX,
        ] = (
            snapshot[
                fields.FIELD_EVAL_POSITION_BASIS_WEIGHTS + fields.TREE_ACTIVE_SUFFIX
            ]
            * 100
        )
        tree.loc[
            snapshot.index,
            fields.FIELD_EVAL_TRANSACTION_RETURN_LCL_CCY + fields.TREE_ACTIVE_SUFFIX,
        ] = (
            snapshot[
                fields.FIELD_EVAL_TRANSACTION_RETURN_LCL_CCY + fields.TREE_ACTIVE_SUFFIX
            ]
            * 100
        )
        tree.loc[
            snapshot.index,
            fields.FIELD_EVAL_TRANSACTION_RETURN + fields.TREE_ACTIVE_SUFFIX,
        ] = (
            snapshot[fields.FIELD_EVAL_TRANSACTION_RETURN + fields.TREE_ACTIVE_SUFFIX]
            * 100
        )
        tree.loc[
            snapshot.index,
            fields.FIELD_EVAL_RETURN_HLDS_LCL_CCY + fields.TREE_ACTIVE_SUFFIX,
        ] = (
            snapshot[fields.FIELD_EVAL_RETURN_HLDS_LCL_CCY + fields.TREE_ACTIVE_SUFFIX]
            * 100
        )
        tree.loc[
            snapshot.index, fields.FIELD_EVAL_RETURN_HLDS + fields.TREE_ACTIVE_SUFFIX
        ] = (snapshot[fields.FIELD_EVAL_RETURN_HLDS + fields.TREE_ACTIVE_SUFFIX] * 100)
        tree.loc[
            snapshot.index, fields.FIELD_EVAL_CCY_RETURN + fields.TREE_ACTIVE_SUFFIX
        ] = (snapshot[fields.FIELD_EVAL_CCY_RETURN + fields.TREE_ACTIVE_SUFFIX] * 100)
        tree.loc[
            snapshot.index, fields.FIELD_EVAL_RETURN + fields.TREE_ACTIVE_SUFFIX
        ] = (snapshot[fields.FIELD_EVAL_RETURN + fields.TREE_ACTIVE_SUFFIX] * 100)
        tree.loc[
            snapshot.index, fields.FIELD_EVAL_RETURN_LCL_CCY + fields.TREE_ACTIVE_SUFFIX
        ] = (
            snapshot[fields.FIELD_EVAL_RETURN_LCL_CCY + fields.TREE_ACTIVE_SUFFIX] * 100
        )

    if time_aggregation:
        accumulte_over_time(tree)

    return tree


def accumulte_over_time(tree: pd.DataFrame):
    """
    aggregate over time the metrics into the snapshots collector.
    'snapshots' is modified in place adding the aggregated over time metrics.
    """
    from .linking import carino_linking_active

    linked_effects = [
        fields.FIELD_EVAL_ACC_T_RETURN_LCL_CCY,
        fields.FIELD_EVAL_ACC_T_RETURN_HLDS_LCL_CCY,
        fields.FIELD_EVAL_ACC_T_CTR_TO_RETURN_LCL_CCY,
        fields.FIELD_EVAL_ACC_T_CCY_RETURN,
        fields.FIELD_EVAL_ACC_T_CTR_TO_RETURN,
        fields.FIELD_EVAL_ACC_T_RETURN_HLDS,
        fields.FIELD_EVAL_ACC_T_ALLOCATION,
        fields.FIELD_EVAL_ACC_T_SELECTION,
        fields.FIELD_EVAL_ACC_T_LEVERAGE,
        fields.FIELD_EVAL_ACC_T_TRANSACTION_RETURN_LCL_CCY,
        fields.FIELD_EVAL_ACC_T_TRANSACTION_RETURN,
    ]

    effects_to_link = [
        fields.FIELD_EVAL_RETURN_LCL_CCY,
        fields.FIELD_EVAL_RETURN_HLDS_LCL_CCY,
        fields.FIELD_EVAL_CTR_TO_RETURN_LCL_CCY,
        fields.FIELD_EVAL_CCY_RETURN,
        fields.FIELD_EVAL_CTR_TO_RETURN,
        fields.FIELD_EVAL_RETURN_HLDS,
        fields.FIELD_EVAL_ALLOCATION,
        fields.FIELD_EVAL_SELECTION,
        fields.FIELD_EVAL_LEVERAGE,
        fields.FIELD_EVAL_TRANSACTION_RETURN_LCL_CCY,
        fields.FIELD_EVAL_TRANSACTION_RETURN,
    ]

    effects_dict = dict(zip(effects_to_link, linked_effects))

    tree[fields.FIELD_EVAL_ACC_T_RETURN + fields.TREE_ACTIVE_SUFFIX] = np.nan

    for lf in linked_effects:
        if lf not in fields.ATTRIBUTION_OUTPUT_DAILY_FIELDS and lf not in [
            fields.FIELD_EVAL_ACC_T_ALLOCATION,
            fields.FIELD_EVAL_ACC_T_SELECTION,
            fields.FIELD_EVAL_ACC_T_LEVERAGE,
        ]:
            tree[lf + fields.TREE_ACTIVE_SUFFIX] = np.nan

    # root series
    root_fields = tree[
        tree[fields.FIELD_TREE_POSITION_ID_PVSB].apply(lambda x: is_root(x))
    ]

    for row_id, tree_row_fields in tree.groupby([fields.FIELD_TREE_POSITION_ID_PVSB]):

        row_id = row_id[0]
        acc_portfolio_linking_r = [1.0]
        acc_benchmark_linking_r = [1.0]
        acc_portfolio_r = [1.0]
        acc_benchmark_r = [1.0]

        linked_effect_to_store = {}
        for lf in linked_effects:
            linked_effect_to_store.update({lf: [0.0]})

        for i in range(1, len(tree_row_fields[fields.FIELD_EVAL_RETURN] - 1)):

            # total return: must be the first as it is needed for all the effects
            if is_root(row_id) or is_bucket(
                row_id
            ):  # total return of the portfolio is needed for the linking coefficients
                acc_total_return_portfolio_i = acc_portfolio_linking_r[i - 1] * (
                    root_fields[fields.FIELD_EVAL_RETURN].values[i]
                )
                acc_portfolio_linking_r.append(acc_total_return_portfolio_i)
                acc_portfolio_r.append(
                    acc_portfolio_r[i - 1]
                    * (tree_row_fields[fields.FIELD_EVAL_RETURN].values[i])
                )

                acc_total_return_benchmark_i = acc_benchmark_linking_r[i - 1] * (
                    root_fields[
                        fields.FIELD_EVAL_RETURN + fields.TREE_BENCHMARK_SUFFIX
                    ].values[i]
                )
                acc_benchmark_linking_r.append(acc_total_return_benchmark_i)
                acc_benchmark_r.append(
                    acc_benchmark_r[i - 1]
                    * (
                        tree_row_fields[
                            fields.FIELD_EVAL_RETURN + fields.TREE_BENCHMARK_SUFFIX
                        ].values[i]
                    )
                )

                return_portfolio_slice = np.array(
                    root_fields[fields.FIELD_EVAL_RETURN].values[: i + 1]
                )
                return_benchmark_slice = np.array(
                    root_fields[
                        fields.FIELD_EVAL_RETURN + fields.TREE_BENCHMARK_SUFFIX
                    ].values[: i + 1]
                )
            else:  # total return of the tree row is needed for the linking coefficients
                row_rp = tree_row_fields[fields.FIELD_EVAL_RETURN].values[i]
                if pd.isna(row_rp):
                    acc_total_return_portfolio_i = acc_portfolio_linking_r[i - 1] * 1
                else:
                    acc_total_return_portfolio_i = (
                        acc_portfolio_linking_r[i - 1] * row_rp
                    )
                acc_portfolio_linking_r.append(acc_total_return_portfolio_i)
                acc_portfolio_r = acc_portfolio_linking_r

                row_rp = tree_row_fields[
                    fields.FIELD_EVAL_RETURN + fields.TREE_BENCHMARK_SUFFIX
                ].values[i]
                if pd.isna(row_rp):
                    acc_total_return_benchmark_i = acc_benchmark_linking_r[i - 1] * 1
                else:
                    acc_total_return_benchmark_i = (
                        acc_benchmark_linking_r[i - 1] * row_rp
                    )
                acc_benchmark_linking_r.append(acc_total_return_benchmark_i)
                acc_benchmark_r = acc_benchmark_linking_r

                return_portfolio_slice = np.array(
                    tree_row_fields[fields.FIELD_EVAL_RETURN].values[: i + 1]
                )
                return_benchmark_slice = np.array(
                    tree_row_fields[
                        fields.FIELD_EVAL_RETURN + fields.TREE_BENCHMARK_SUFFIX
                    ].values[: i + 1]
                )

            # link effects:
            for lf in effects_to_link:
                if lf in fields.ATTRIBUTION_OUTPUT_DAILY_FIELDS:
                    acc_active_i = carino_linking_active(
                        np.array(tree_row_fields[lf].values[: i + 1]),
                        return_portfolio_slice,
                        return_benchmark_slice,
                        acc_total_return_portfolio_i,
                        acc_total_return_benchmark_i,
                    )
                else:
                    acc_active_i = carino_linking_active(
                        np.array(
                            tree_row_fields[lf + fields.TREE_ACTIVE_SUFFIX].values[
                                : i + 1
                            ]
                        ),
                        return_portfolio_slice,
                        return_benchmark_slice,
                        acc_total_return_portfolio_i,
                        acc_total_return_benchmark_i,
                    )

                linked_effect_to_store[effects_dict[lf]].append(acc_active_i)

        # Store effects into the tree:
        tree.loc[
            tree_row_fields.index,
            fields.FIELD_EVAL_ACC_T_RETURN + fields.TREE_ACTIVE_SUFFIX,
        ] = pd.Series(
            data=[float(x - y) * 100 for x, y in zip(acc_portfolio_r, acc_benchmark_r)],
            index=tree_row_fields.index,
        )

        for lf in effects_to_link:
            if lf in fields.ATTRIBUTION_OUTPUT_DAILY_FIELDS:
                tree.loc[tree_row_fields.index, effects_dict[lf]] = pd.Series(
                    data=linked_effect_to_store[effects_dict[lf]],
                    index=tree_row_fields.index,
                )
            else:
                tree.loc[
                    tree_row_fields.index, effects_dict[lf] + fields.TREE_ACTIVE_SUFFIX
                ] = pd.Series(
                    data=linked_effect_to_store[effects_dict[lf]],
                    index=tree_row_fields.index,
                )

    # /// DEBUG ONLY --- tree.to_csv("attribution.csv")
    return tree
