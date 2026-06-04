"""
Khadee EDA — Multi-Format Dataset Loader
=========================================
Auto-detects file format from extension and loads into pandas DataFrame.
Supports: CSV, TSV, Excel, JSON, Parquet, Feather, HDF5, SQLite, XML, Pickle, Stata, SAS, SPSS
"""

import os
import pandas as pd


# ── Format-to-reader mapping ──────────────────────────────────────────────────
EXTENSION_MAP = {
    # CSV / delimited
    ".csv": "csv",
    ".tsv": "tsv",
    ".txt": "csv",
    # Excel
    ".xlsx": "excel",
    ".xls": "excel",
    ".xlsm": "excel",
    ".xlsb": "excel",
    # JSON
    ".json": "json",
    # Parquet / Feather
    ".parquet": "parquet",
    ".feather": "feather",
    # HDF5
    ".h5": "hdf5",
    ".hdf5": "hdf5",
    ".hdf": "hdf5",
    # SQLite
    ".db": "sqlite",
    ".sqlite": "sqlite",
    ".sqlite3": "sqlite",
    # XML
    ".xml": "xml",
    # Pickle
    ".pkl": "pickle",
    ".pickle": "pickle",
    # Statistical formats
    ".dta": "stata",
    ".sas7bdat": "sas",
    ".sav": "spss",
}


def load_dataset(source, **kwargs):
    """
    Load a dataset from a file path or DataFrame.

    Parameters
    ----------
    source : str or pd.DataFrame
        File path (auto-detects format) or an existing DataFrame.
    **kwargs : dict
        Extra keyword arguments passed to the pandas reader function.

    Returns
    -------
    tuple : (pd.DataFrame, dict)
        The loaded DataFrame and metadata about the loading process.
    """
    metadata = {
        "source_type": None,
        "file_name": None,
        "file_size": None,
        "file_format": None,
    }

    # Already a DataFrame
    if isinstance(source, pd.DataFrame):
        metadata["source_type"] = "dataframe"
        metadata["file_name"] = "DataFrame (in-memory)"
        metadata["file_format"] = "DataFrame"
        return source.copy(), metadata

    # File path
    if not isinstance(source, str):
        raise TypeError(
            f"Expected a file path (str) or pandas DataFrame, got {type(source).__name__}"
        )

    if not os.path.exists(source):
        raise FileNotFoundError(f"File not found: {source}")

    file_name = os.path.basename(source)
    file_size = os.path.getsize(source)
    ext = os.path.splitext(source)[1].lower()

    if ext not in EXTENSION_MAP:
        raise ValueError(
            f"Unsupported file format: '{ext}'. "
            f"Supported formats: {', '.join(sorted(EXTENSION_MAP.keys()))}"
        )

    fmt = EXTENSION_MAP[ext]
    metadata["source_type"] = "file"
    metadata["file_name"] = file_name
    metadata["file_size"] = file_size
    metadata["file_format"] = fmt

    df = _read_file(source, fmt, **kwargs)
    return df, metadata


def _read_file(path, fmt, **kwargs):
    """Dispatch to the appropriate pandas reader."""
    readers = {
        "csv": _read_csv,
        "tsv": _read_tsv,
        "excel": _read_excel,
        "json": _read_json,
        "parquet": _read_parquet,
        "feather": _read_feather,
        "hdf5": _read_hdf5,
        "sqlite": _read_sqlite,
        "xml": _read_xml,
        "pickle": _read_pickle,
        "stata": _read_stata,
        "sas": _read_sas,
        "spss": _read_spss,
    }
    reader = readers.get(fmt)
    if reader is None:
        raise ValueError(f"No reader available for format: {fmt}")
    return reader(path, **kwargs)


def _read_csv(path, **kwargs):
    kwargs.setdefault("encoding", "utf-8")
    try:
        return pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        kwargs["encoding"] = "latin-1"
        return pd.read_csv(path, **kwargs)


def _read_tsv(path, **kwargs):
    kwargs.setdefault("sep", "\t")
    return _read_csv(path, **kwargs)


def _read_excel(path, **kwargs):
    return pd.read_excel(path, **kwargs)


def _read_json(path, **kwargs):
    try:
        return pd.read_json(path, **kwargs)
    except ValueError:
        # Try lines=True for JSON-lines format
        return pd.read_json(path, lines=True, **kwargs)


def _read_parquet(path, **kwargs):
    return pd.read_parquet(path, **kwargs)


def _read_feather(path, **kwargs):
    return pd.read_feather(path, **kwargs)


def _read_hdf5(path, **kwargs):
    # Try to auto-detect the key
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        store = pd.HDFStore(path, mode="r")
        keys = store.keys()
        store.close()
    if not keys:
        raise ValueError(f"No datasets found in HDF5 file: {path}")
    kwargs.setdefault("key", keys[0])
    return pd.read_hdf(path, **kwargs)


def _read_sqlite(path, **kwargs):
    import sqlite3

    conn = sqlite3.connect(path)
    # Get all table names
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
    if tables.empty:
        conn.close()
        raise ValueError(f"No tables found in SQLite database: {path}")
    table_name = kwargs.pop("table", tables["name"].iloc[0])
    df = pd.read_sql(f"SELECT * FROM [{table_name}]", conn, **kwargs)
    conn.close()
    return df


def _read_xml(path, **kwargs):
    return pd.read_xml(path, **kwargs)


def _read_pickle(path, **kwargs):
    return pd.read_pickle(path, **kwargs)


def _read_stata(path, **kwargs):
    return pd.read_stata(path, **kwargs)


def _read_sas(path, **kwargs):
    return pd.read_sas(path, **kwargs)


def _read_spss(path, **kwargs):
    return pd.read_spss(path, **kwargs)


def format_file_size(size_bytes):
    """Convert bytes to human-readable format."""
    if size_bytes is None:
        return "N/A"
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"
