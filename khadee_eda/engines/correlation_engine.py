"""
Khadee EDA — Correlation Engine
================================
Compute Pearson, Spearman, Kendall, Cramér's V, and Phik correlations.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def pearson_correlation(df, numeric_cols):
    """Compute Pearson correlation matrix for numeric columns."""
    # Filter out zero-variance columns to prevent numpy divide-by-zero warnings
    valid_cols = [c for c in numeric_cols if df[c].nunique() > 1]
    if len(valid_cols) < 2:
        return pd.DataFrame()
    return df[valid_cols].corr(method="pearson")


def spearman_correlation(df, numeric_cols):
    """Compute Spearman rank correlation matrix."""
    valid_cols = [c for c in numeric_cols if df[c].nunique() > 1]
    if len(valid_cols) < 2:
        return pd.DataFrame()
    return df[valid_cols].corr(method="spearman")


def kendall_correlation(df, numeric_cols):
    """Compute Kendall Tau correlation matrix."""
    valid_cols = [c for c in numeric_cols if df[c].nunique() > 1]
    if len(valid_cols) < 2:
        return pd.DataFrame()
    # Kendall can be slow, sample if needed
    sample = df[valid_cols]
    if len(sample) > 5000:
        sample = sample.sample(5000, random_state=42)
    return sample.corr(method="kendall")


def cramers_v_matrix(df, categorical_cols):
    """
    Compute Cramér's V association matrix for categorical columns.
    Cramér's V measures association between two categorical variables.
    """
    if len(categorical_cols) < 2:
        return pd.DataFrame()

    n = len(categorical_cols)
    matrix = pd.DataFrame(np.ones((n, n)), index=categorical_cols, columns=categorical_cols)

    for i in range(n):
        for j in range(i + 1, n):
            v = _cramers_v(df[categorical_cols[i]], df[categorical_cols[j]])
            matrix.iloc[i, j] = v
            matrix.iloc[j, i] = v

    return matrix


def _cramers_v(x, y):
    """Compute Cramér's V between two categorical series."""
    try:
        confusion = pd.crosstab(x, y)
        chi2 = scipy_stats.chi2_contingency(confusion)[0]
        n = confusion.sum().sum()
        phi2 = chi2 / n if n > 0 else 0
        r, k = confusion.shape
        phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1)) if n > 1 else 0
        r_corr = r - ((r - 1) ** 2) / (n - 1) if n > 1 else r
        k_corr = k - ((k - 1) ** 2) / (n - 1) if n > 1 else k
        denom = min(k_corr - 1, r_corr - 1)
        if denom <= 0:
            return 0.0
        return float(np.sqrt(phi2_corr / denom))
    except Exception:
        return 0.0


def top_correlations(corr_matrix, n=10):
    """
    Get the top N strongest correlations from a correlation matrix.

    Returns
    -------
    list of tuples : [(col1, col2, correlation_value), ...]
    """
    if corr_matrix.empty:
        return []

    # Get upper triangle
    mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    pairs = []

    for i in range(corr_matrix.shape[0]):
        for j in range(i + 1, corr_matrix.shape[1]):
            val = corr_matrix.iloc[i, j]
            if not np.isnan(val):
                pairs.append((
                    corr_matrix.index[i],
                    corr_matrix.columns[j],
                    float(val),
                ))

    # Sort by absolute value
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs[:n]


def multicollinearity_warnings(corr_matrix, threshold=0.80):
    """
    Find pairs with correlation above threshold.

    Returns
    -------
    list of tuples : [(col1, col2, correlation_value), ...]
    """
    pairs = top_correlations(corr_matrix, n=100)
    return [(c1, c2, v) for c1, c2, v in pairs if abs(v) > threshold]
