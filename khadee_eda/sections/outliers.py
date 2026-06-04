"""
Khadee EDA — Section 6: Outlier Detection
==========================================
Multi-method outlier analysis: IQR, Z-score, MAD, Isolation Forest.
"""

import numpy as np
import pandas as pd

from ..type_detector import ColumnType
from ..engines.outlier_engine import (
    outlier_summary, detect_outliers_isolation_forest,
)
from ..utils import format_number, format_percentage, section_header, safe_str, stat_card
from ..renderers.chart_renderer import box_plot_multi, bar_chart


def generate(df, type_map):
    """Generate the Outlier Detection section HTML."""
    numeric_cols = [c for c, t in type_map.items()
                    if t == ColumnType.NUMERIC and df[c].notna().sum() > 3]

    html = section_header("section-outliers", " Outlier Detection",
                          "Multi-method outlier analysis using IQR, Z-score, MAD, and Isolation Forest")

    if not numeric_cols:
        html += '<div class="card"><p class="empty-message">No numeric columns available for outlier detection.</p></div>'
        return html

    # Compute outlier summary
    summary = outlier_summary(df, numeric_cols)

    # ── Overview Cards ──
    total_iqr = sum(r["iqr_count"] for r in summary)
    total_zscore = sum(r["zscore_count"] for r in summary)
    total_mad = sum(r["mad_count"] for r in summary)
    affected_cols = sum(1 for r in summary if r["iqr_count"] > 0)

    html += '<div class="stat-grid" style="margin-bottom:1.5rem">'
    html += stat_card("IQR Outliers", format_number(total_iqr), "📦", "#3b82f6")
    html += stat_card("Z-Score Outliers", format_number(total_zscore), "📊", "#8b5cf6")
    html += stat_card("MAD Outliers", format_number(total_mad), "📐", "#06b6d4")
    html += stat_card("Affected Columns", f"{affected_cols} / {len(numeric_cols)}", "📋", "#f59e0b")
    html += '</div>'

    # ── Outlier Summary Table ──
    html += '<div class="card">'
    html += '<h3 class="card-title"> Outlier Summary by Column</h3>'
    html += '<div class="table-responsive"><table class="styled-table">'
    html += '<thead><tr><th>Column</th><th>IQR Count</th><th>IQR %</th><th>Z-Score Count</th><th>Z-Score %</th><th>MAD Count</th><th>MAD %</th><th>IQR Bounds</th></tr></thead>'
    html += '<tbody>'
    for r in summary:
        html += f'<tr><td class="col-name">{safe_str(r["column"])}</td>'
        html += f'<td>{format_number(r["iqr_count"])}</td>'
        html += f'<td>{format_percentage(r["iqr_pct"])}</td>'
        html += f'<td>{format_number(r["zscore_count"])}</td>'
        html += f'<td>{format_percentage(r["zscore_pct"])}</td>'
        html += f'<td>{format_number(r["mad_count"])}</td>'
        html += f'<td>{format_percentage(r["mad_pct"])}</td>'
        lower = format_number(r.get("iqr_lower")) if r.get("iqr_lower") is not None else "N/A"
        upper = format_number(r.get("iqr_upper")) if r.get("iqr_upper") is not None else "N/A"
        html += f'<td>[{lower}, {upper}]</td></tr>'
    html += '</tbody></table></div></div>'

    # ── Box Plots ──
    display_cols = numeric_cols[:12]
    html += '<div class="card">'
    html += '<h3 class="card-title"> Box Plots (Outlier Visualization)</h3>'
    series_list = [pd.to_numeric(df[c], errors="coerce").dropna() for c in display_cols]
    html += box_plot_multi(series_list, display_cols, "outlier_boxplots")
    html += '</div>'

    # ── Outlier Percentage Bar Chart ──
    cols_with_outliers = [r for r in summary if r["iqr_count"] > 0]
    if cols_with_outliers:
        html += '<div class="card">'
        html += '<h3 class="card-title"> Outlier Percentage by Column (IQR Method)</h3>'
        labels = [r["column"] for r in cols_with_outliers[:20]]
        values = [r["iqr_pct"] * 100 for r in cols_with_outliers[:20]]
        html += bar_chart(labels, values, "outlier_pct_bar",
                         title="Outlier % (IQR)", color="#ef4444")
        html += '</div>'

    # ── Isolation Forest ──
    if len(numeric_cols) >= 2:
        iso_result = detect_outliers_isolation_forest(df, numeric_cols[:20])
        if "error" not in iso_result and iso_result.get("n_outliers", 0) > 0:
            html += '<div class="card">'
            html += '<h3 class="card-title"> Isolation Forest Analysis</h3>'
            html += f'<div class="alert-box info">Detected <strong>{format_number(iso_result["n_outliers"])}</strong> anomalous rows '
            html += f'({format_percentage(iso_result["percentage"])}) using Isolation Forest on {iso_result.get("n_samples", "N/A")} samples.</div>'
            html += '</div>'

    return html
