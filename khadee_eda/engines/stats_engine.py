"""
Khadee EDA — Statistics Engine
===============================
Descriptive and inferential statistics for numeric and categorical columns.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def descriptive_stats(series):
    """
    Compute comprehensive descriptive statistics for a numeric series.

    Returns
    -------
    dict : All descriptive statistics.
    """
    clean = series.dropna()
    n = len(clean)

    if n == 0:
        return _empty_numeric_stats(series)

    values = clean.values.astype(float)

    result = {
        "count": len(series),
        "n_valid": n,
        "n_missing": int(series.isna().sum()),
        "missing_pct": series.isna().sum() / len(series) if len(series) > 0 else 0,
        "n_distinct": int(clean.nunique()),
        "distinct_pct": clean.nunique() / n if n > 0 else 0,
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if n > 1 else 0.0,
        "variance": float(np.var(values, ddof=1)) if n > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "range": float(np.ptp(values)),
        "median": float(np.median(values)),
        "sum": float(np.sum(values)),
        "mad": float(np.median(np.abs(values - np.median(values)))),
        "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
        "cv": float(np.std(values, ddof=1) / np.mean(values)) if np.mean(values) != 0 and n > 1 else 0.0,
        "skewness": float(scipy_stats.skew(values)) if n > 2 else 0.0,
        "kurtosis": float(scipy_stats.kurtosis(values)) if n > 3 else 0.0,
        "p1": float(np.percentile(values, 1)),
        "p5": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "n_zeros": int(np.sum(values == 0)),
        "n_negatives": int(np.sum(values < 0)),
        "n_inf": int(np.sum(np.isinf(values))),
        "zeros_pct": int(np.sum(values == 0)) / len(series) if len(series) > 0 else 0,
        "negatives_pct": int(np.sum(values < 0)) / len(series) if len(series) > 0 else 0,
        "inf_pct": int(np.sum(np.isinf(values))) / len(series) if len(series) > 0 else 0,
        "memory_size": int(series.memory_usage(deep=True)),
        "monotonic_increasing": bool(clean.is_monotonic_increasing),
        "monotonic_decreasing": bool(clean.is_monotonic_decreasing),
    }

    # Mode (may be multiple)
    mode_result = clean.mode()
    result["mode"] = float(mode_result.iloc[0]) if len(mode_result) > 0 else None
    result["n_modes"] = len(mode_result)

    return result


def categorical_stats(series):
    """
    Compute statistics for a categorical series.

    Returns
    -------
    dict : Categorical statistics.
    """
    clean = series.dropna()
    n = len(clean)
    n_total = len(series)

    value_counts = clean.value_counts()

    result = {
        "count": n_total,
        "n_valid": n,
        "n_missing": int(series.isna().sum()),
        "missing_pct": series.isna().sum() / n_total if n_total > 0 else 0,
        "n_distinct": int(clean.nunique()),
        "distinct_pct": clean.nunique() / n if n > 0 else 0,
        "most_common": str(value_counts.index[0]) if len(value_counts) > 0 else None,
        "most_common_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
        "most_common_pct": float(value_counts.iloc[0] / n) if n > 0 and len(value_counts) > 0 else 0,
        "least_common": str(value_counts.index[-1]) if len(value_counts) > 0 else None,
        "least_common_count": int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0,
        "value_counts": {str(k): int(v) for k, v in value_counts.head(20).items()},
        "memory_size": int(series.memory_usage(deep=True)),
    }

    # Entropy
    if n > 0:
        probs = value_counts.values / n
        result["entropy"] = float(-np.sum(probs * np.log2(probs + 1e-10)))
    else:
        result["entropy"] = 0.0

    # Imbalance ratio
    if len(value_counts) > 1:
        result["imbalance_ratio"] = float(value_counts.iloc[0] / value_counts.iloc[-1])
    else:
        result["imbalance_ratio"] = 1.0

    return result


def boolean_stats(series):
    """Compute statistics for a boolean series."""
    clean = series.dropna()
    n = len(clean)
    n_total = len(series)

    # Normalize to True/False
    bool_map = {
        True: True, False: False,
        1: True, 0: False,
        "true": True, "false": False,
        "True": True, "False": False,
        "TRUE": True, "FALSE": False,
        "yes": True, "no": False,
        "Yes": True, "No": False,
        "YES": True, "NO": False,
        "1": True, "0": False,
    }

    mapped = clean.map(lambda x: bool_map.get(x, x))
    true_count = int((mapped == True).sum())
    false_count = int((mapped == False).sum())

    return {
        "count": n_total,
        "n_valid": n,
        "n_missing": int(series.isna().sum()),
        "missing_pct": series.isna().sum() / n_total if n_total > 0 else 0,
        "true_count": true_count,
        "false_count": false_count,
        "true_pct": true_count / n if n > 0 else 0,
        "false_pct": false_count / n if n > 0 else 0,
    }


def datetime_stats(series):
    """Compute statistics for a datetime series."""
    clean = series.dropna()

    # Try converting to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(clean):
        try:
            clean = pd.to_datetime(clean, errors="coerce").dropna()
        except Exception:
            return {"count": len(series), "n_valid": 0, "error": "Could not parse dates"}

    n = len(clean)
    n_total = len(series)

    if n == 0:
        return {"count": n_total, "n_valid": 0, "n_missing": n_total}

    return {
        "count": n_total,
        "n_valid": n,
        "n_missing": int(series.isna().sum()),
        "missing_pct": series.isna().sum() / n_total if n_total > 0 else 0,
        "min": str(clean.min()),
        "max": str(clean.max()),
        "range": str(clean.max() - clean.min()),
        "n_distinct": int(clean.nunique()),
    }


def _empty_numeric_stats(series):
    """Return empty stats for all-null numeric column."""
    return {
        "count": len(series),
        "n_valid": 0,
        "n_missing": len(series),
        "missing_pct": 1.0,
        "n_distinct": 0,
        "mean": None, "std": None, "variance": None,
        "min": None, "max": None, "range": None,
        "median": None, "sum": None, "mad": None,
        "iqr": None, "cv": None, "skewness": None,
        "kurtosis": None, "mode": None,
        "p1": None, "p5": None, "p25": None, "p50": None,
        "p75": None, "p95": None, "p99": None,
        "n_zeros": 0, "n_negatives": 0, "n_inf": 0,
    }
