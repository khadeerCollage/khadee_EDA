"""
K-EDA — Statistics Engine
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
    
    # Exact values from full data (fast operations)
    val_min = float(np.min(values))
    val_max = float(np.max(values))
    val_sum = float(np.sum(values))
    
    # Sample for heavier/slower computations (percentiles, skewness, kurtosis, etc.)
    if len(values) > 100_000:
        rng = np.random.RandomState(42)
        values_sampled = rng.choice(values, 100_000, replace=False)
    else:
        values_sampled = values

    n_sampled = len(values_sampled)
    mean_val = float(np.mean(values_sampled))
    std_val = float(np.std(values_sampled, ddof=1)) if n_sampled > 1 else 0.0
    var_val = float(np.var(values_sampled, ddof=1)) if n_sampled > 1 else 0.0
    median_val = float(np.median(values_sampled))
    mad_val = float(np.median(np.abs(values_sampled - median_val)))
    iqr_val = float(np.percentile(values_sampled, 75) - np.percentile(values_sampled, 25))
    cv_val = float(std_val / mean_val) if mean_val != 0 and n_sampled > 1 else 0.0
    skew_val = float(scipy_stats.skew(values_sampled)) if n_sampled > 2 else 0.0
    kurt_val = float(scipy_stats.kurtosis(values_sampled)) if n_sampled > 3 else 0.0

    result = {
        "count": len(series),
        "n_valid": n,
        "n_missing": int(series.isna().sum()),
        "missing_pct": series.isna().sum() / len(series) if len(series) > 0 else 0,
        "n_distinct": int(clean.nunique()),
        "distinct_pct": clean.nunique() / n if n > 0 else 0,
        "mean": mean_val,
        "std": std_val,
        "variance": var_val,
        "min": val_min,
        "max": val_max,
        "range": val_max - val_min,
        "median": median_val,
        "sum": val_sum,
        "mad": mad_val,
        "iqr": iqr_val,
        "cv": cv_val,
        "skewness": skew_val,
        "kurtosis": kurt_val,
        "p1": float(np.percentile(values_sampled, 1)),
        "p5": float(np.percentile(values_sampled, 5)),
        "p25": float(np.percentile(values_sampled, 25)),
        "p50": median_val,
        "p75": float(np.percentile(values_sampled, 75)),
        "p95": float(np.percentile(values_sampled, 95)),
        "p99": float(np.percentile(values_sampled, 99)),
        "n_zeros": int(np.sum(values_sampled == 0)),
        "n_negatives": int(np.sum(values_sampled < 0)),
        "n_inf": int(np.sum(np.isinf(values_sampled))),
        "zeros_pct": int(np.sum(values_sampled == 0)) / n_sampled if n_sampled > 0 else 0,
        "negatives_pct": int(np.sum(values_sampled < 0)) / n_sampled if n_sampled > 0 else 0,
        "inf_pct": int(np.sum(np.isinf(values_sampled))) / n_sampled if n_sampled > 0 else 0,
        "memory_size": int(series.memory_usage(deep=(len(series) <= 100_000))),
        "monotonic_increasing": bool(clean.is_monotonic_increasing) if len(clean) < 100_000 else False,
        "monotonic_decreasing": bool(clean.is_monotonic_decreasing) if len(clean) < 100_000 else False,
    }

    # Mode (may be multiple)
    # Check size for mode calculation
    if len(clean) > 100_000:
        mode_sample = clean.sample(100_000, random_state=42)
        mode_result = mode_sample.mode()
    else:
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

    # Downsample value_counts if extremely large
    if n > 500_000:
        clean_sampled = clean.sample(500_000, random_state=42)
        value_counts = clean_sampled.value_counts()
        # Scale back counts to match actual size roughly
        scale = n / 500_000
        value_counts = (value_counts * scale).astype(int)
    else:
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
        "memory_size": int(series.memory_usage(deep=(len(series) <= 100_000))),
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
