"""
K-EDA — Column Type Detection Engine
==========================================
Automatically infers semantic column types: numeric, categorical, boolean, datetime, text.
"""

import numpy as np
import pandas as pd

from .config import (
    BOOLEAN_VALUES,
    CATEGORICAL_RATIO_THRESHOLD,
    CATEGORICAL_THRESHOLD,
)


class ColumnType:
    """Enum-like class for column types."""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    CONSTANT = "constant"
    UNIQUE_ID = "unique_id"


def detect_types(df):
    """
    Detect the semantic type of each column in the DataFrame.

    Returns
    -------
    dict : {column_name: ColumnType}
    """
    type_map = {}
    n_rows = len(df)

    for col in df.columns:
        series = df[col]
        type_map[col] = _detect_single(series, n_rows)

    return type_map


def _detect_single(series, n_rows):
    """Detect the type of a single column."""
    non_null = series.dropna()
    n_non_null = len(non_null)

    # Empty column
    if n_non_null == 0:
        return ColumnType.CONSTANT

    n_unique = non_null.nunique()

    # Constant column (single unique value)
    if n_unique <= 1:
        return ColumnType.CONSTANT

    # Check boolean
    if _is_boolean(non_null, n_unique):
        return ColumnType.BOOLEAN

    # Check datetime
    if _is_datetime(series):
        return ColumnType.DATETIME

    # Check numeric
    if pd.api.types.is_numeric_dtype(series):
        # Could be categorical if very few unique values
        if n_unique <= CATEGORICAL_THRESHOLD and n_rows > 0:
            ratio = n_unique / n_rows
            if ratio < CATEGORICAL_RATIO_THRESHOLD:
                return ColumnType.CATEGORICAL
        # Unique ID check
        if n_unique == n_rows and n_rows > 10:
            return ColumnType.UNIQUE_ID
        return ColumnType.NUMERIC

    # String/object types
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        # Try numeric conversion
        numeric_converted = pd.to_numeric(non_null, errors="coerce")
        non_null_numeric = numeric_converted.dropna()
        if len(non_null_numeric) > 0.8 * n_non_null:
            return ColumnType.NUMERIC

        # Unique ID check
        if n_unique == n_rows and n_rows > 10:
            return ColumnType.UNIQUE_ID

        # Text vs categorical
        if n_unique <= CATEGORICAL_THRESHOLD:
            return ColumnType.CATEGORICAL
        if n_rows > 0 and (n_unique / n_rows) < CATEGORICAL_RATIO_THRESHOLD:
            return ColumnType.CATEGORICAL

        # Check average string length — long strings are text
        avg_len = non_null.astype(str).str.len().mean()
        if avg_len > 50:
            return ColumnType.TEXT

        return ColumnType.CATEGORICAL

    # Fallback for category dtype
    if pd.api.types.is_categorical_dtype(series):
        return ColumnType.CATEGORICAL

    return ColumnType.TEXT


def _is_boolean(series, n_unique):
    """Check if a column is boolean."""
    if n_unique > 2:
        return False

    # Check pandas boolean dtype
    if pd.api.types.is_bool_dtype(series):
        return True

    # Check value set
    unique_vals = set(series.unique())
    # Check if values are a subset of known boolean values
    str_vals = {str(v).strip() for v in unique_vals}
    return str_vals.issubset({str(b) for b in BOOLEAN_VALUES})


def _is_datetime(series):
    """Check if a column is datetime."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    # Try parsing string columns as dates
    if pd.api.types.is_object_dtype(series):
        non_null = series.dropna()
        if len(non_null) == 0:
            return False
        sample = non_null.head(100)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*Could not infer format.*")
            try:
                parsed = pd.to_datetime(sample, errors="coerce")
                success_rate = parsed.notna().sum() / len(sample)
                return success_rate > 0.8
            except Exception:
                return False

    return False


def get_type_summary(type_map):
    """
    Get a summary count of each column type.

    Returns
    -------
    dict : {type_name: count}
    """
    summary = {}
    for col_type in type_map.values():
        summary[col_type] = summary.get(col_type, 0) + 1
    return summary


def get_columns_by_type(type_map, col_type):
    """Get list of column names matching the given type."""
    return [col for col, t in type_map.items() if t == col_type]
