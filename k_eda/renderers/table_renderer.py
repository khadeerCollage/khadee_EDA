"""
K-EDA — Table Renderer
=============================
Styled HTML table generation utilities.
"""

import pandas as pd
from ..utils import safe_str, format_number


def df_to_styled_table(df, max_rows=100, compact=False):
    """Convert a DataFrame to a styled HTML table."""
    cls = "styled-table compact" if compact else "styled-table"

    html = f'<div class="table-responsive"><table class="{cls}">'
    html += '<thead><tr>'
    for col in df.columns:
        html += f'<th>{safe_str(str(col))}</th>'
    html += '</tr></thead><tbody>'

    display_df = df.head(max_rows)
    for _, row in display_df.iterrows():
        html += '<tr>'
        for val in row:
            if pd.isna(val):
                html += '<td><span class="null-val">NaN</span></td>'
            elif isinstance(val, float):
                html += f'<td>{format_number(val)}</td>'
            else:
                html += f'<td>{safe_str(str(val)[:50])}</td>'
        html += '</tr>'

    html += '</tbody></table></div>'

    if len(df) > max_rows:
        html += f'<p class="table-truncated">Showing {max_rows} of {len(df)} rows</p>'

    return html
