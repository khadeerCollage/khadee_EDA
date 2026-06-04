"""
Khadee EDA — Outlier Detection Engine
=======================================
IQR, Z-score, Modified Z-score (MAD), and Isolation Forest methods.
"""

import numpy as np
import pandas as pd

from ..config import IQR_MULTIPLIER, MODIFIED_ZSCORE_THRESHOLD, ZSCORE_THRESHOLD


def detect_outliers_iqr(series, multiplier=None):
    """
    Detect outliers using the IQR method.

    Returns
    -------
    dict : {mask, count, lower_bound, upper_bound, percentage}
    """
    if multiplier is None:
        multiplier = IQR_MULTIPLIER

    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) == 0:
        return {"mask": pd.Series(dtype=bool), "count": 0, "percentage": 0.0}

    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr

    mask = (clean < lower) | (clean > upper)

    return {
        "mask": mask,
        "count": int(mask.sum()),
        "percentage": float(mask.sum() / len(clean)) if len(clean) > 0 else 0.0,
        "lower_bound": lower,
        "upper_bound": upper,
    }


def detect_outliers_zscore(series, threshold=None):
    """
    Detect outliers using the Z-score method.

    Returns
    -------
    dict : {mask, count, percentage, threshold}
    """
    if threshold is None:
        threshold = ZSCORE_THRESHOLD

    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < 3:
        return {"mask": pd.Series(dtype=bool), "count": 0, "percentage": 0.0}

    mean = float(clean.mean())
    std = float(clean.std())

    if std == 0:
        return {"mask": pd.Series(False, index=clean.index), "count": 0, "percentage": 0.0}

    z_scores = np.abs((clean - mean) / std)
    mask = z_scores > threshold

    return {
        "mask": mask,
        "count": int(mask.sum()),
        "percentage": float(mask.sum() / len(clean)) if len(clean) > 0 else 0.0,
        "threshold": threshold,
    }


def detect_outliers_mad(series, threshold=None):
    """
    Detect outliers using Modified Z-score (MAD-based).
    More robust than Z-score for non-normal distributions.

    Returns
    -------
    dict : {mask, count, percentage}
    """
    if threshold is None:
        threshold = MODIFIED_ZSCORE_THRESHOLD

    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < 3:
        return {"mask": pd.Series(dtype=bool), "count": 0, "percentage": 0.0}

    median = float(np.median(clean))
    mad = float(np.median(np.abs(clean - median)))

    if mad == 0:
        return {"mask": pd.Series(False, index=clean.index), "count": 0, "percentage": 0.0}

    modified_z = 0.6745 * (clean - median) / mad
    mask = np.abs(modified_z) > threshold

    return {
        "mask": mask,
        "count": int(mask.sum()),
        "percentage": float(mask.sum() / len(clean)) if len(clean) > 0 else 0.0,
        "threshold": threshold,
    }


def detect_outliers_isolation_forest(df, numeric_cols, contamination=0.05):
    """
    Detect outliers using Isolation Forest.

    Returns
    -------
    dict : {labels, scores, n_outliers, percentage}
    """
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        return {"error": "scikit-learn not installed"}

    if len(numeric_cols) == 0 or len(df) < 10:
        return {"labels": np.array([]), "scores": np.array([]), "n_outliers": 0}

    # Prepare data
    data = df[numeric_cols].copy()
    data = data.fillna(data.median())

    # Limit columns for performance
    if len(numeric_cols) > 20:
        # Use top 20 by variance
        variances = data.var().sort_values(ascending=False)
        selected = variances.head(20).index.tolist()
        data = data[selected]

    # Sample for large datasets
    if len(data) > 10000:
        sample_idx = data.sample(10000, random_state=42).index
        data_sample = data.loc[sample_idx]
    else:
        data_sample = data

    try:
        iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        labels = iso.fit_predict(data_sample)
        scores = iso.decision_function(data_sample)

        n_outliers = int((labels == -1).sum())

        return {
            "labels": labels,
            "scores": scores,
            "n_outliers": n_outliers,
            "percentage": n_outliers / len(labels) if len(labels) > 0 else 0.0,
            "n_samples": len(labels),
        }
    except Exception as e:
        return {"error": str(e), "n_outliers": 0}


def outlier_summary(df, numeric_cols):
    """
    Generate outlier summary for all numeric columns using multiple methods.

    Returns
    -------
    list of dict : Per-column outlier summary.
    """
    results = []

    for col in numeric_cols:
        series = df[col]
        iqr_result = detect_outliers_iqr(series)
        zscore_result = detect_outliers_zscore(series)
        mad_result = detect_outliers_mad(series)

        results.append({
            "column": col,
            "iqr_count": iqr_result["count"],
            "iqr_pct": iqr_result["percentage"],
            "zscore_count": zscore_result["count"],
            "zscore_pct": zscore_result["percentage"],
            "mad_count": mad_result["count"],
            "mad_pct": mad_result["percentage"],
            "iqr_lower": iqr_result.get("lower_bound"),
            "iqr_upper": iqr_result.get("upper_bound"),
        })

    return results
