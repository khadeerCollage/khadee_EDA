"""
K-EDA — Section 5: Missing Values
========================================
Bar charts, matrix plot, co-occurrence heatmap, imputation recommendations.
"""

import numpy as np
import pandas as pd

from ..engines.missing_engine import (
    missing_summary, missing_matrix, missing_correlation,
    imputation_recommendations,
)
from ..utils import format_number, format_percentage, section_header, safe_str
from ..renderers.chart_renderer import bar_chart, heatmap_chart, missing_matrix_chart


def generate(df, type_map):
    """Generate the Missing Values section HTML."""
    html = section_header("section-missing", " Missing Values",
                          "Comprehensive missing data analysis and imputation recommendations")

    total_missing = int(df.isna().sum().sum())
    total_cells = df.shape[0] * df.shape[1]

    if total_missing == 0:
        html += '<div class="card">'
        html += '<div class="success-banner"> No missing values found in this dataset!</div>'
        html += '</div>'
        return html

    # Overall summary
    html += '<div class="stat-grid" style="margin-bottom:1.5rem">'
    from ..utils import stat_card
    html += stat_card("Total Missing", format_number(total_missing), "❓", "#ef4444")
    html += stat_card("Missing %", format_percentage(total_missing / total_cells if total_cells > 0 else 0), "📊", "#f59e0b")
    cols_with_missing = int(df.isna().any().sum())
    html += stat_card("Columns Affected", f"{cols_with_missing} / {df.shape[1]}", "📋", "#8b5cf6")
    complete_rows = int((~df.isna().any(axis=1)).sum())
    html += stat_card("Complete Rows", f"{format_number(complete_rows)} ({format_percentage(complete_rows / len(df) if len(df) > 0 else 0)})", "✅", "#10b981")
    html += '</div>'

    # ── Missing Count Bar Chart ──
    summary = missing_summary(df)
    missing_cols = summary[summary["count"] > 0]

    if len(missing_cols) > 0:
        html += '<div class="card">'
        html += '<h3 class="card-title"> Missing Values per Column</h3>'
        labels = missing_cols["column"].tolist()[:30]
        values = missing_cols["count"].tolist()[:30]
        html += bar_chart(labels, values, "missing_bar", title="Missing Count by Column",
                         color="#ef4444")
        html += '</div>'

    # ── Missing Summary Table ──
    html += '<div class="card">'
    html += '<h3 class="card-title">📋 Missing Values Summary</h3>'
    html += '<div class="table-responsive"><table class="styled-table">'
    html += '<thead><tr><th>Column</th><th>Missing Count</th><th>Missing %</th><th>Missingness Type</th></tr></thead>'
    html += '<tbody>'
    for _, row in missing_cols.iterrows():
        pct_class = "danger" if row["percentage"] > 0.5 else "warning" if row["percentage"] > 0.1 else ""
        html += f'<tr><td class="col-name">{safe_str(row["column"])}</td>'
        html += f'<td>{format_number(row["count"])}</td>'
        html += f'<td class="{pct_class}">{format_percentage(row["percentage"])}</td>'
        html += f'<td>{safe_str(row["type_suggestion"])}</td></tr>'
    html += '</tbody></table></div></div>'

    # ── Missing Matrix ──
    html += '<div class="card">'
    html += '<h3 class="card-title"> Nullity Matrix</h3>'
    html += '<p class="card-desc">White = present, dark = missing. Patterns reveal systematic missingness.</p>'
    matrix = missing_matrix(df)
    html += missing_matrix_chart(matrix, "missing_matrix")
    html += '</div>'

    # ── Missing Co-occurrence ──
    miss_corr = missing_correlation(df)
    if not miss_corr.empty:
        html += '<div class="card">'
        html += '<h3 class="card-title"> Missing Value Co-occurrence</h3>'
        html += '<p class="card-desc">High correlation indicates columns that tend to be missing together.</p>'
        html += heatmap_chart(miss_corr, "missing_corr_heatmap", title="Missingness Correlation")
        html += '</div>'

    # ── Imputation Recommendations ──
    recs = imputation_recommendations(df, type_map)
    if recs:
        html += '<div class="card">'
        html += '<h3 class="card-title"> Imputation Recommendations</h3>'
        html += '<div class="table-responsive"><table class="styled-table">'
        html += '<thead><tr><th>Column</th><th>Missing %</th><th>Strategy</th><th>Reason</th></tr></thead>'
        html += '<tbody>'
        for rec in recs:
            html += f'<tr><td class="col-name">{safe_str(rec["column"])}</td>'
            html += f'<td>{format_percentage(rec["missing_pct"])}</td>'
            html += f'<td><strong>{safe_str(rec["strategy"])}</strong></td>'
            html += f'<td>{safe_str(rec["reason"])}</td></tr>'
        html += '</tbody></table></div></div>'

    return html
