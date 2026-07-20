# df_utils.py
# DataFrame and JSON validation utilities.

import pandas as pd
from typing import Dict, Any, List

from oport import error_collector
from .schemas import PortfolioDescriptor
from pydantic import BaseModel


def calculate_ratio(val, prev_val):
    return val / prev_val


def fill_value(column_name: str, df: pd.DataFrame) -> pd.DataFrame:
    df[column_name] = df[column_name].replace(0.0, pd.NaT)
    df[column_name] = df[column_name].fillna(method="ffill")
    return df


def backfill_value(column_name: str, df: pd.DataFrame) -> pd.DataFrame:
    df[column_name] = df[column_name].fillna(method="bfill")
    return df


def validate_json_data(data: Dict[str, Any], schema_class: BaseModel = PortfolioDescriptor) -> Dict[str, Any]:
    """Validate JSON data against a Pydantic schema."""
    try:
        validated = schema_class(**data)
        return validated.dict()
    except Exception as e:
        err_collector = error_collector.get_collector()
        err_collector.record_error(
            operation_name="validate_json_data",
            error_details=f"JSON validation failed: {e}"
        )
        return data  # Return original data if validation fails


def validate_dataframe_schema(df: pd.DataFrame, expected_columns: List[str] = None) -> pd.DataFrame:
    """Basic DataFrame column validation."""
    if expected_columns:
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            err_collector = error_collector.get_collector()
            err_collector.record_error(
                operation_name="validate_dataframe_schema",
                error_details=f"Missing columns: {missing_cols}"
            )
    return df
