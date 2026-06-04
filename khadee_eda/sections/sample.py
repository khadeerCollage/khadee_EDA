"""
Khadee EDA — Section 10: Data Sample & Duplicates
==================================================
Head, tail, random sample, duplicates, and auto-generated data dictionary.
"""

import pandas as pd

from ..type_detector import ColumnType
from ..utils import format_number, format_percentage, section_header, safe_str


def generate(df, type_map):
    """Generate the Sample & Duplicates section HTML."""
    html = section_header("section-sample", " Sample & Duplicates",
                          "Data preview, duplicate analysis, and auto-generated data dictionary")

    # ── Head (First 10 rows) ──
    html += '<div class="card">'
    html += '<h3 class="card-title"> First 10 Rows</h3>'
    html += _df_to_html(df.head(10))
    html += '</div>'

    # ── Tail (Last 10 rows) ──
    html += '<div class="card">'
    html += '<h3 class="card-title"> Last 10 Rows</h3>'
    html += _df_to_html(df.tail(10))
    html += '</div>'

    # ── Random Sample ──
    html += '<div class="card">'
    html += '<h3 class="card-title"> Random Sample (10 Rows)</h3>'
    sample_size = min(10, len(df))
    if sample_size > 0:
        html += _df_to_html(df.sample(sample_size, random_state=42))
    else:
        html += '<p class="empty-message">No data to sample.</p>'
    html += '</div>'

    # ── Duplicates ──
    n_dup = int(df.duplicated().sum())
    html += '<div class="card">'
    html += '<h3 class="card-title"> Duplicate Rows</h3>'
    if n_dup > 0:
        html += f'<div class="alert-box warning"><strong>{format_number(n_dup)}</strong> duplicate rows found ({format_percentage(n_dup / len(df) if len(df) > 0 else 0)})</div>'
        # Show first few duplicates
        dup_df = df[df.duplicated(keep="first")].head(10)
        html += '<h4 class="sub-heading">First 10 Duplicates:</h4>'
        html += _df_to_html(dup_df)
    else:
        html += '<div class="success-banner"> No duplicate rows found!</div>'
    html += '</div>'

    # ── Data Dictionary ──
    html += '<div class="card">'
    html += '<h3 class="card-title"> Data Dictionary (Auto-Generated)</h3>'
    html += '<div class="table-responsive"><table class="styled-table">'
    html += '<thead><tr><th>#</th><th>Column</th><th>Type</th><th>Dtype</th><th>Non-Null</th><th>Unique</th><th>Sample Values</th></tr></thead>'
    html += '<tbody>'

    type_name_map = {
        ColumnType.NUMERIC: "Numeric",
        ColumnType.CATEGORICAL: "Categorical",
        ColumnType.BOOLEAN: "Boolean",
        ColumnType.DATETIME: "DateTime",
        ColumnType.TEXT: "Text",
        ColumnType.CONSTANT: "Constant",
        ColumnType.UNIQUE_ID: "Unique ID",
    }

    for i, col in enumerate(df.columns):
        col_type = type_name_map.get(type_map.get(col, ""), "Unknown")
        dtype = str(df[col].dtype)
        n_valid = int(df[col].notna().sum())
        n_unique = int(df[col].nunique())
        # Sample values
        sample_vals = df[col].dropna().head(3).tolist()
        sample_str = ", ".join(str(v)[:20] for v in sample_vals)

        html += f'<tr><td>{i+1}</td><td class="col-name">{safe_str(col)}</td>'
        html += f'<td><span class="type-badge">{col_type}</span></td>'
        html += f'<td class="dtype">{dtype}</td>'
        html += f'<td>{format_number(n_valid)}</td>'
        html += f'<td>{format_number(n_unique)}</td>'
        html += f'<td class="sample-vals">{safe_str(sample_str)}</td></tr>'

    html += '</tbody></table></div></div>'

    return html


def _df_to_html(df):
    """Convert a DataFrame to styled HTML table."""
    html = '<div class="table-responsive"><table class="styled-table compact">'
    html += '<thead><tr>'
    html += '<th>#</th>'
    for col in df.columns:
        html += f'<th>{safe_str(str(col)[:25])}</th>'
    html += '</tr></thead><tbody>'

    for idx, row in df.iterrows():
        html += f'<tr><td class="idx">{idx}</td>'
        for val in row:
            cell_val = safe_str(str(val)[:30]) if pd.notna(val) else '<span class="null-val">NaN</span>'
            html += f'<td>{cell_val}</td>'
        html += '</tr>'

    html += '</tbody></table></div>'
    return html
