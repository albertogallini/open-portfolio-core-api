import warnings
from oport import fields

import pandas as pd


def merge_buckets(
    portfolio: dict,
    portfolio_buckets: dict,
    portfolio_hlds: pd.DataFrame,
    benchmark: dict,
    benchmark_buckets: dict,
    benchmark_hlds: pd.DataFrame,
    indexing: bool = False,
) -> pd.DataFrame:
    """
    Returns a dataframe containing both the holdings, buckets, and top level (portfolio).
    If portfolio and buckets are None, return holdings.
    """
    print("Merging buckets and holdings in a tree view ...")

    response = None

    if portfolio is not None and bool(portfolio):
        portfolio[fields.FIELD_TREE_POSITION_ID] = [fields.TREE_TOP_ROW_ID] * len(
            portfolio[fields.FIELD_DATE]
        )
        portfolio = pd.DataFrame.from_dict(portfolio)
        common_columns = portfolio.columns.intersection(
            portfolio_hlds.columns
        ).to_list()
        response = portfolio_hlds
        for _, row in portfolio[common_columns].iterrows():
            response = pd.concat([response, pd.DataFrame([row])], ignore_index=True)
    else:
        response = portfolio_hlds

    if portfolio_buckets is not None and bool(portfolio_buckets):
        for bucket in portfolio_buckets:
            bucket_position_id = (
                str(bucket)
                .replace("(", "[")
                .replace(")", "]")
                .replace(" ", "")
                .replace(",", "\\")
                .replace("'", "")
            )
            portfolio_buckets[bucket][fields.FIELD_TREE_POSITION_ID] = [
                bucket_position_id
            ] * len(portfolio_buckets[bucket][fields.FIELD_DATE])
            portfolio_buckets[bucket] = pd.DataFrame.from_dict(
                portfolio_buckets[bucket]
            )
            common_columns = (
                portfolio_buckets[bucket]
                .columns.intersection(response.columns)
                .to_list()
            )
            for _, row in portfolio_buckets[bucket][common_columns].iterrows():
                response = pd.concat([response, pd.DataFrame([row])], ignore_index=True)

    response = fillTreeValues(response)

    if benchmark is not None and bool(benchmark):
        benchmark[fields.FIELD_TREE_POSITION_ID] = [fields.TREE_TOP_ROW_ID] * len(
            benchmark[fields.FIELD_DATE]
        )
        benchmark = pd.DataFrame.from_dict(benchmark)
        common_columns_bench = benchmark.columns.intersection(
            response.columns
        ).to_list()
        common_columns_bench = [
            item
            for item in common_columns_bench
            if item not in fields.BASIC_POSITON_DESCR_FIELDS
        ]
        # Prepare benchmark for updating existing rows (only common columns, renamed to _bench)
        update_bench = benchmark[[fields.FIELD_DATE] + common_columns_bench]
        update_bench = update_bench.rename(
            columns={col: f"{col}_bench" for col in common_columns_bench}
        )

        update_bench = fillTreeValues(update_bench)
        # Merge to update response with _bench columns where keys match (vectorized, no loop)
        response = pd.merge(
            response,
            update_bench,
            left_on=[fields.FIELD_TREE_POSITION_ID, fields.FIELD_DATE],
            right_on=[f"{fields.FIELD_TREE_POSITION_ID}_bench", fields.FIELD_DATE],
            how="left",
        )

        # Find dates in benchmark not present in response for TREE_TOP_ROW_ID
        bucket_response_dates = response.loc[
            response[fields.FIELD_TREE_POSITION_ID] == fields.TREE_TOP_ROW_ID,
            fields.FIELD_DATE,
        ].unique()

        new_dates_mask = ~benchmark[fields.FIELD_DATE].isin(bucket_response_dates)
        new_bench_bucket = benchmark[new_dates_mask].copy()

        if not new_bench_bucket.empty:
            # Rename only common columns to _bench for new rows (keep basics and others as-is)
            new_bench_bucket = new_bench_bucket.rename(
                columns={col: f"{col}_bench" for col in common_columns_bench}
            )

            # Concat new rows (aligns columns, fills NaNs where needed)
            response = pd.concat([response, new_bench_bucket], ignore_index=True)

    if benchmark_buckets is not None and bool(benchmark_buckets):
        for bucket in benchmark_buckets:
            bucket_position_id = (
                str(bucket)
                .replace("(", "[")
                .replace(")", "]")
                .replace(" ", "")
                .replace(",", "\\")
                .replace("'", "")
            )
            benchmark_buckets[bucket][fields.FIELD_TREE_POSITION_ID] = [
                bucket_position_id
            ] * len(benchmark_buckets[bucket][fields.FIELD_DATE])
            benchmark_buckets[bucket] = pd.DataFrame.from_dict(
                benchmark_buckets[bucket]
            )
            common_columns_bench = (
                benchmark_buckets[bucket]
                .columns.intersection(response.columns)
                .to_list()
            )
            common_columns_bench = [
                item
                for item in common_columns_bench
                if item not in fields.BASIC_POSITON_DESCR_FIELDS + [fields.FIELD_DATE]
            ]

            # Prepare benchmark for updating existing rows (only common columns, renamed to _bench)
            update_bench_bucket = benchmark_buckets[bucket][
                [fields.FIELD_DATE] + common_columns_bench
            ]
            update_bench_bucket = update_bench_bucket.rename(
                columns={col: f"{col}_bench" for col in common_columns_bench}
            )

            update_bench_bucket = fillTreeValues(update_bench_bucket)

            # Merge to update response with _bench columns where keys match (vectorized, no loop)
            response = pd.merge(
                response,
                update_bench_bucket,
                left_on=[fields.FIELD_TREE_POSITION_ID, fields.FIELD_DATE],
                right_on=[f"{fields.FIELD_TREE_POSITION_ID}_bench", fields.FIELD_DATE],
                how="left",
                suffixes=(
                    None,
                    "_new",
                ),  # Use None to keep existing _bench columns, _new for conflicts
            )

            for c in response.columns:
                if c.endswith("_new"):
                    c_original = c.replace("_new", "")
                    # Fill NaN in existing _bench columns with values from temp columns
                    response[c_original] = response[c_original].combine_first(
                        response[c]
                    )
            response.drop(
                columns=[c for c in response.columns if c.endswith("_new")],
                inplace=True,
            )

            # Find dates in benchmark not present in response for bucket_position_id
            bucket_response_dates = response.loc[
                response[fields.FIELD_TREE_POSITION_ID] == bucket_position_id,
                fields.FIELD_DATE,
            ].unique()

            new_dates_mask = ~benchmark_buckets[bucket][fields.FIELD_DATE].isin(
                bucket_response_dates
            )
            new_bench_bucket = benchmark_buckets[bucket][new_dates_mask].copy()

            if not new_bench_bucket.empty:
                # Rename only common columns to _bench for new rows (keep basics and others as-is)
                new_bench_bucket = new_bench_bucket.rename(
                    columns={col: f"{col}_bench" for col in common_columns_bench}
                )

                # Concat new rows (aligns columns, fills NaNs where needed)
                response = pd.concat([response, new_bench_bucket], ignore_index=True)

    if benchmark_hlds is not None and not benchmark_hlds.empty:
        existing_position_ids = response[fields.FIELD_TREE_POSITION_ID].unique()
        common_columns_bench = benchmark_hlds.columns.intersection(
            response.columns
        ).to_list()
        common_columns_bench = [
            item
            for item in common_columns_bench
            if item not in fields.BASIC_POSITON_DESCR_FIELDS
        ]

        benchmark_hlds = fillTreeValues(benchmark_hlds)
        
        for _, row in benchmark_hlds.iterrows():
            condition = (
                response[fields.FIELD_TREE_POSITION_ID]
                == row[fields.FIELD_TREE_POSITION_ID]
            ) & (response[fields.FIELD_DATE] == row[fields.FIELD_DATE])
            if condition.any():
                for col in common_columns_bench:
                    response.loc[condition, f"{col}_bench"] = row[col]
            else:
                new_row = row.rename(
                    {col: f"{col}_bench" for col in common_columns_bench}
                )
                response = pd.concat(
                    [response, pd.DataFrame([new_row])], ignore_index=True
                )

    response[fields.FIELD_TREE_POSITION_ID_PVSB] = response.apply(
        lambda row: (
            row[fields.FIELD_TREE_POSITION_ID]
            if pd.notna(row[fields.FIELD_TREE_POSITION_ID])
            else row[fields.FIELD_TREE_POSITION_ID + fields.TREE_BENCHMARK_SUFFIX]
        ),
        axis=1,
    )

    if indexing:
        # Function to find the parent bucket for a given position ID
        def find_parent(position_id):
            if position_id == fields.TREE_TOP_ROW_ID:
                return fields.TREE_TOP_ROW_ID
            tokens = position_id.split("]\\")
            if len(tokens) == 1:  # If it's already at the top level, return itself
                return fields.TREE_TOP_ROW_ID
            parent_id = "]\\".join(tokens[:-1]) + "]"
            return parent_id

        # Find the index of the parent for each row
        def find_parent_index(row, df):
            parent_id = find_parent(row[fields.FIELD_TREE_POSITION_ID_PVSB])
            parents = df[df[fields.FIELD_TREE_POSITION_ID_PVSB] == parent_id]
            if not parents.empty:
                matching_parent = parents[
                    parents[fields.FIELD_DATE] == row[fields.FIELD_DATE]
                ]
                if not matching_parent.empty:
                    return matching_parent.index[0]
            return None

        # Add 'parent' column with the index of the parent at the same date
        response["parent"] = response.apply(
            lambda row: find_parent_index(row, response), axis=1
        )

    return response


# Helper functions to identify row types
is_bucket = lambda p: p.endswith("]") and not p.startswith(".")
is_root = lambda p: p.startswith(".")


# Helper function to determine the parent position_id
def get_parent_id(position_id: str) -> str | None:
    """
    Determine the parent position ID for a given position ID.

    Args:
        position_id (str): The position ID to analyze. It can represent a root,
                           bucket, or leaf node in a tree structure.

    Returns:
        str | None: The parent position ID if applicable, or None if the
                    position ID is a root node.

    Raises:
        ValueError: If the position ID is a leaf but not in a valid format.

    The function distinguishes between root, bucket, and leaf nodes:
    - Root nodes start with a dot (.) and have no parent.
    - Bucket nodes end with a bracket (]) and have a parent determined by
      the path within the brackets.
    - Leaf nodes are expected to have at least one backslash (\), and their
      parent is determined by the preceding path.
    """

    if position_id.startswith("."):
        return None  # Root has no parent
    elif position_id.endswith("]"):
        # Bucket: e.g., [EUR\ETP\]
        path = position_id[1:-2]  # Remove [ and \]
        levels = path.split("\\")
        if len(levels) == 1:
            return fields.TREE_TOP_ROW_ID  # Top-level bucket's parent is root
        else:
            parent_path = "\\".join(levels[:-1])
            return f"[{parent_path}\\]"
    else:
        # Leaf: e.g., EUR\ETP\leaf_id
        levels = position_id.split("\\")
        if len(levels) < 2:
            raise ValueError(f"Invalid leaf position_id: {position_id}")
        bucket_path = "\\".join(levels[:-1])
        return f"{bucket_path}"


def build_snapshot_maps(snapshot):
    from collections import defaultdict

    # Map position_id to row index
    """
        Build maps from a tree snapshot.

        Maps are used to traverse the tree efficiently:
        - position_id_to_index: maps position_id to row index.
        - parent_map: maps position_id to parent position_id.
        - children_map: maps position_id to list of children indices.
        - parent_index_map: maps position_id to parent row index.

        Args:
            snapshot (pd.DataFrame): Tree snapshot.

        Returns:
            Tuple of 4 dictionaries:
                position_id_to_index (dict): position_id -> row index.
                parent_map (dict): position_id -> parent position_id.
                children_map (dict): position_id -> list of children indices.
                parent_index_map (dict): position_id -> parent row index.
        """
    position_id_to_index = {
        row[fields.FIELD_TREE_POSITION_ID_PVSB]: index
        for index, row in snapshot.iterrows()
    }

    # Map position_id to parent position_id
    parent_map = {
        position_id: get_parent_id(position_id) for position_id in position_id_to_index
    }

    # Map position_id to list of children indices
    children_map = defaultdict(list)
    for position_id, index in position_id_to_index.items():
        parent_id = parent_map[position_id]
        if parent_id is not None:
            children_map[parent_id].append(index)

    # Map position_id to parent row index
    parent_index_map = {
        position_id: position_id_to_index[parent_id]
        for position_id, parent_id in parent_map.items()
        if parent_id is not None
    }

    return position_id_to_index, parent_map, children_map, parent_index_map


def fillTreeValues(tree: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in the attribution tree DataFrame.

    This function fills missing values in specific columns of the DataFrame
    using forward-fill and backward-fill methods. It ensures that key fields
    have no missing values by propagating the last valid observation forward
    and then backward.

    Args:
        tree (pd.DataFrame): The DataFrame containing the attribution tree data.

    Returns:
        pd.DataFrame: The DataFrame with filled missing values.
    """
    # Disable copy warnings
    warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)

    if tree.empty:
        return tree

    fields_to_fill = [
        fields.FIELD_TREE_POSITION_ID_PVSB,
        fields.FIELD_TREE_POSITION_ID,
    ]
    for field in fields_to_fill:
        if field in tree.columns:
            tree[field] = tree[field].ffill()

    for col in fields.BASIC_PERFORMANCE_DAILY_FIELDS:
        if (
            col not in tree.columns
            or f"{col}{fields.TREE_BENCHMARK_SUFFIX}" not in tree.columns
        ):
            continue
        if "return" in col.lower() and "basis" not in col.lower():
            tree[col] = tree[col].fillna(1)
            tree[f"{col}{fields.TREE_BENCHMARK_SUFFIX}"] = tree[
                f"{col}{fields.TREE_BENCHMARK_SUFFIX}"
            ].fillna(1)
        elif "pnl" in col.lower():
            tree[col] = tree[col].fillna(0)
            tree[f"{col}{fields.TREE_BENCHMARK_SUFFIX}"] = tree[
                f"{col}{fields.TREE_BENCHMARK_SUFFIX}"
            ].fillna(0)
        else:
            tree[col] = tree[col].fillna(method="ffill")
            tree[f"{col}{fields.TREE_BENCHMARK_SUFFIX}"] = tree[
                f"{col}{fields.TREE_BENCHMARK_SUFFIX}"
            ].fillna(method="ffill")

    return tree
