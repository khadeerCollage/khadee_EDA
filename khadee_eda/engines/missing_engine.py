"""
Khadee EDA — Missing Values Engine
====================================
Analyze missing value patterns, co-occurrence, and missingness type estimation.
"""

import numpy as np
import pandas as pd


def missing_summary(df):
    """
    Compute per-column missing value summary.

    Returns
    -------
    pd.DataFrame : Columns [column, count, percentage, type_suggestion]
    """
    n_rows = len(df)
    records = []

    for col in df.columns:
        n_miss = int(df[col].isna().sum())
        pct = n_miss / n_rows if n_rows > 0 else 0.0
        suggestion = _estimate_missingness_type(df, col, n_miss, n_rows)
        records.append({
            "column": col,
            "count": n_miss,
            "percentage": pct,
            "type_suggestion": suggestion,
        })

    result = pd.DataFrame(records)
    result = result.sort_values("count", ascending=False).reset_index(drop=True)
    return result


def missing_matrix(df, max_cols=50, max_rows=200):
    """
    Generate a nullity matrix (boolean: True=present, False=missing).
    Sampled for visualization if too large.

    Returns
    -------
    pd.DataFrame : Boolean matrix (True=present)
    """
    sub = df.iloc[:max_rows, :max_cols] if len(df) > max_rows else df.iloc[:, :max_cols]
    return sub.notna().astype(int)


def missing_correlation(df):
    """
    Compute correlation between missingness patterns of columns.
    High correlation indicates columns that tend to be missing together.

    Returns
    -------
    pd.DataFrame : Correlation matrix of missingness indicators.
    """
    # Only include columns that have at least some missing values
    missing_cols = [col for col in df.columns if df[col].isna().any()]
    if len(missing_cols) < 2:
        return pd.DataFrame()

    miss_indicators = df[missing_cols].isna().astype(int)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return miss_indicators.corr()


def imputation_recommendations(df, type_map):
    """
    Suggest imputation strategy for each column with missing values.

    Returns
    -------
    list of dict : [{column, missing_pct, strategy, reason}, ...]
    """
    from .stats_engine import descriptive_stats

    recommendations = []
    n_rows = len(df)

    for col in df.columns:
        n_miss = int(df[col].isna().sum())
        if n_miss == 0:
            continue

        pct = n_miss / n_rows if n_rows > 0 else 0

        col_type = type_map.get(col, "unknown")

        if pct > 0.50:
            strategy = "Drop column"
            reason = f"Too many missing values ({pct:.0%}). Consider removing this feature."
        elif col_type == "numeric":
            # Check skewness
            clean = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(clean) > 0:
                skew = abs(float(clean.skew()))
                if skew > 1.0:
                    strategy = "Median imputation"
                    reason = f"Skewed distribution (skew={skew:.2f}), median is more robust."
                else:
                    strategy = "Mean imputation"
                    reason = f"Roughly symmetric distribution (skew={skew:.2f})."
            else:
                strategy = "Median imputation"
                reason = "Default for numeric columns."
        elif col_type in ("categorical", "boolean"):
            strategy = "Mode imputation"
            reason = "Most frequent value for categorical data."
        elif col_type == "datetime":
            strategy = "Forward fill / Interpolation"
            reason = "Time-based data benefits from temporal interpolation."
        else:
            strategy = "Mode or custom"
            reason = "Inspect column manually."

        recommendations.append({
            "column": col,
            "missing_pct": pct,
            "strategy": strategy,
            "reason": reason,
        })

    return recommendations


def _estimate_missingness_type(df, col, n_miss, n_rows):
    """
    Estimate whether missingness is MCAR, MAR, or MNAR.
    This is a rough heuristic, not a formal test.
    """
    if n_miss == 0:
        return "None"
    if n_miss == n_rows:
        return "Completely missing"

    pct = n_miss / n_rows

    # Simple heuristic:
    # - If missingness in this column is correlated with values in other columns → MAR
    # - If missingness seems random → MCAR
    # - Otherwise → potentially MNAR

    if pct < 0.05:
        return "Likely MCAR (low rate)"
    elif pct > 0.50:
        return "Investigate — high rate"

    # Sample for performance if large
    if len(df) > 20000:
        df_sample = df.sample(20000, random_state=42)
    else:
        df_sample = df

    # Check if missing pattern correlates with other columns
    miss_indicator = df_sample[col].isna().astype(int)
    numeric_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != col]

    if len(numeric_cols) > 0:
        correlations = []
        for other_col in numeric_cols[:10]:  # Check up to 10 columns
            try:
                valid_mask = df_sample[other_col].notna()
                if valid_mask.sum() > 10:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        corr = abs(miss_indicator[valid_mask].corr(df_sample[other_col][valid_mask]))
                    if not np.isnan(corr):
                        correlations.append(corr)
            except Exception:
                pass

        if correlations and max(correlations) > 0.3:
            return "Likely MAR"

    return "Likely MCAR"
