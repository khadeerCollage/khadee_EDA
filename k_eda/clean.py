"""
K-EDA — Data Cleaning & Standardizing Module (Dataprep Clean equivalent)
=============================================================================
Unified, high-performance API to clean, standardizing, and prepare DataFrames.
"""

import re
import numpy as np
import pandas as pd
import warnings


def clean_headers(df, case="snake"):
    """
    Standardize and clean DataFrame column headers.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    case : str, optional
        Target case format: 'snake', 'camel', 'pascal', 'upper', or 'lower'.
        Default: 'snake'.

    Returns
    -------
    pd.DataFrame : Cleaned copy of the DataFrame.
    """
    df_clean = df.copy()
    new_cols = []

    for col in df_clean.columns:
        col_str = str(col).strip()
        # Remove accents
        import unicodedata
        col_str = unicodedata.normalize('NFKD', col_str).encode('ASCII', 'ignore').decode('utf-8')
        
        # Replace non-alphanumeric chars with spaces
        col_str = re.sub(r'[^a-zA-Z0-9]', ' ', col_str)
        # Collapse multiple spaces
        col_str = re.sub(r'\s+', ' ', col_str).strip()

        words = col_str.split(' ')

        if case == "snake":
            new_col = '_'.join(w.lower() for w in words)
        elif case == "camel":
            if not words:
                new_col = ""
            else:
                new_col = words[0].lower() + ''.join(w.capitalize() for w in words[1:])
        elif case == "pascal":
            new_col = ''.join(w.capitalize() for w in words)
        elif case == "upper":
            new_col = '_'.join(w.upper() for w in words)
        elif case == "lower":
            new_col = '_'.join(w.lower() for w in words)
        else:
            new_col = '_'.join(words)

        # Handle empty names or duplicate columns
        if not new_col:
            new_col = f"col_{len(new_cols)}"
        
        new_cols.append(new_col)

    # De-duplicate column names
    seen = {}
    final_cols = []
    for col in new_cols:
        if col in seen:
            seen[col] += 1
            final_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            final_cols.append(col)

    df_clean.columns = final_cols
    return df_clean


def clean_missing(df, columns=None, strategy="median", fill_value=None):
    """
    Handle missing values in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list or str, optional
        Target columns to clean. If None, cleans all columns.
    strategy : str, optional
        Imputation strategy: 'mean', 'median', 'mode', 'constant', or 'drop'.
        Default: 'median'.
    fill_value : any, optional
        Value to use if strategy='constant'.

    Returns
    -------
    pd.DataFrame : Cleaned DataFrame.
    """
    df_clean = df.copy()
    if columns is None:
        columns = df_clean.columns
    elif isinstance(columns, str):
        columns = [columns]

    for col in columns:
        if col not in df_clean.columns:
            continue
        
        if strategy == "drop":
            df_clean = df_clean.dropna(subset=[col])
            continue

        if df_clean[col].isna().sum() == 0:
            continue

        if strategy == "constant":
            val = fill_value if fill_value is not None else ("Missing" if df_clean[col].dtype == "object" else 0)
            df_clean[col] = df_clean[col].fillna(val)
        elif strategy == "mean":
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            else:
                # Fallback to mode for categoricals
                mode_val = df_clean[col].mode()
                if not mode_val.empty:
                    df_clean[col] = df_clean[col].fillna(mode_val.iloc[0])
        elif strategy == "median":
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            else:
                mode_val = df_clean[col].mode()
                if not mode_val.empty:
                    df_clean[col] = df_clean[col].fillna(mode_val.iloc[0])
        elif strategy == "mode":
            mode_val = df_clean[col].mode()
            if not mode_val.empty:
                df_clean[col] = df_clean[col].fillna(mode_val.iloc[0])
                
    return df_clean


def clean_duplicates(df, columns=None, keep="first"):
    """
    Remove duplicate rows from DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list, optional
        Columns to check for duplicates. If None, checks all columns.
    keep : str, optional
        Which duplicates to keep: 'first', 'last', or False (drop all).
        Default: 'first'.

    Returns
    -------
    pd.DataFrame : Cleaned DataFrame.
    """
    return df.drop_duplicates(subset=columns, keep=keep)


def clean_outliers(df, columns=None, method="iqr", threshold=1.5, strategy="clip"):
    """
    Identify and handle outliers in numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list or str, optional
        Target numeric columns. If None, processes all numeric columns.
    method : str, optional
        Outlier detection method: 'iqr' (Interquartile Range) or 'zscore'.
        Default: 'iqr'.
    threshold : float, optional
        IQR multiplier (default 1.5) or Z-score limit (default 3.0).
    strategy : str, optional
        Outlier handling strategy: 'clip' (winsorize to bounds) or 'drop' (remove rows).
        Default: 'clip'.

    Returns
    -------
    pd.DataFrame : Cleaned DataFrame.
    """
    df_clean = df.copy()
    if columns is None:
        columns = [c for c in df_clean.columns if pd.api.types.is_numeric_dtype(df_clean[c])]
    elif isinstance(columns, str):
        columns = [columns]

    for col in columns:
        if col not in df_clean.columns or not pd.api.types.is_numeric_dtype(df_clean[col]):
            continue

        series = pd.to_numeric(df_clean[col], errors="coerce")
        clean = series.dropna()
        if len(clean) == 0:
            continue

        if method == "iqr":
            q1, q3 = clean.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
        elif method == "zscore":
            mean, std = clean.mean(), clean.std()
            if std == 0:
                continue
            lower = mean - threshold * std
            upper = mean + threshold * std
        else:
            continue

        if strategy == "clip":
            df_clean[col] = df_clean[col].clip(lower, upper)
        elif strategy == "drop":
            df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper) | (df_clean[col].isna())]

    return df_clean


def clean_text(df, columns, lowercase=True, strip=True, remove_special=True):
    """
    Clean and normalize string/text columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list or str
        Target column name(s).
    lowercase : bool, optional
        Convert text to lower case. Default: True.
    strip : bool, optional
        Strip leading/trailing spaces. Default: True.
    remove_special : bool, optional
        Remove punctuation and special symbols. Default: True.

    Returns
    -------
    pd.DataFrame : Cleaned DataFrame.
    """
    df_clean = df.copy()
    if isinstance(columns, str):
        columns = [columns]

    for col in columns:
        if col not in df_clean.columns:
            continue
        
        # Cast to string
        s = df_clean[col].astype(str)
        
        if lowercase:
            s = s.str.lower()
        if strip:
            s = s.str.strip()
        if remove_special:
            s = s.apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x) if x != 'nan' else x)
            
        # Re-introduce NaNs if they were cast to 'nan'
        df_clean[col] = s.replace('nan', np.nan)
        
    return df_clean


def clean_df(df):
    """
    Comprehensive dataframe clean-up. Standardizes headers, drops full duplicates,
    and returns clean data.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame : Cleaned DataFrame.
    """
    df_clean = clean_headers(df)
    df_clean = clean_duplicates(df_clean)
    return df_clean
