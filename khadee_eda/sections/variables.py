"""
Khadee EDA — Section 2: Variables (Per-Column Deep Dive)
========================================================
Interactive select dropdown and detailed cards for every column.
"""

import numpy as np
import pandas as pd

from ..type_detector import ColumnType
from ..engines.stats_engine import (
    descriptive_stats, categorical_stats, boolean_stats, datetime_stats,
)
from ..utils import (
    format_number, format_percentage, section_header, safe_str, collapsible,
)
from ..renderers.chart_renderer import (
    histogram_chart, bar_chart, box_plot, pie_chart, kde_plot, qq_plot,
)


def _format_bytes(size_bytes):
    """Convert bytes to human-readable format."""
    if size_bytes is None:
        return "N/A"
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TiB"


def generate(df, type_map):
    """Generate the Variables section HTML."""
    html = section_header("section-variables", " Variables",
                          "Detailed analysis for each column in your dataset")

    # Column dropdown selector
    html += '<div class="card variables-selector-card" style="margin-bottom:1.5rem;">'
    html += '<div style="display:flex; align-items:center; gap:1rem; flex-wrap:wrap;">'
    html += '<span style="font-weight:600; font-size:0.95rem; color:var(--text-primary);"> Explore Column:</span>'
    html += '<select id="variables-select" class="variables-select-dropdown" onchange="showVariableCard(this.value)" style="'
    html += 'background:rgba(10,10,30,0.8); color:var(--text-primary); border:1px solid var(--border); '
    html += 'border-radius:var(--radius-sm); padding:0.5rem 1rem; font-family:var(--font); font-size:0.9rem; '
    html += 'cursor:pointer; min-width:250px; outline:none; transition:var(--transition);">'
    for col in df.columns:
        col_id = col.replace(" ", "_").replace(".", "_").replace("/", "_").replace("-", "_")
        html += f'<option value="{safe_str(col_id)}">{safe_str(col)} ({type_map.get(col, "Unknown").upper()})</option>'
    html += '</select>'
    html += '</div>'
    html += '</div>'

    for i, col in enumerate(df.columns):
        col_type = type_map.get(col, ColumnType.TEXT)
        series = df[col]

        # First card visible, rest hidden
        style = 'style="display: block;"' if i == 0 else 'style="display: none;"'

        if col_type == ColumnType.NUMERIC or col_type == ColumnType.UNIQUE_ID:
            card_html = _numeric_card(col, series, style)
        elif col_type == ColumnType.CATEGORICAL:
            card_html = _categorical_card(col, series, style)
        elif col_type == ColumnType.BOOLEAN:
            card_html = _boolean_card(col, series, style)
        elif col_type == ColumnType.DATETIME:
            card_html = _datetime_card(col, series, style)
        elif col_type == ColumnType.CONSTANT:
            card_html = _constant_card(col, series, style)
        else:
            card_html = _text_card(col, series, style)

        html += card_html

    return html


def _numeric_card(col, series, style=""):
    """Generate a variable card for a numeric column."""
    stats = descriptive_stats(series)
    col_id = col.replace(" ", "_").replace(".", "_").replace("/", "_").replace("-", "_")

    # Quick stats bar
    html = f'<div class="variable-card" id="var-{safe_str(col_id)}" {style}>'
    html += f'<div class="var-header">'
    html += f'<h4 class="var-name">{safe_str(col)}</h4>'
    html += f'<span class="type-badge type-numeric">Numeric</span>'
    if stats["n_missing"] > 0:
        html += f'<span class="missing-badge">{format_percentage(stats["missing_pct"])} missing</span>'
    html += '</div>'

    # Quick stats row
    html += '<div class="var-quick-stats">'
    html += f'<div class="quick-stat"><span class="qs-label">Distinct</span><span class="qs-value">{format_number(stats["n_distinct"])}</span></div>'
    html += f'<div class="quick-stat"><span class="qs-label">Mean</span><span class="qs-value">{format_number(stats["mean"])}</span></div>'
    html += f'<div class="quick-stat"><span class="qs-label">Std</span><span class="qs-value">{format_number(stats["std"])}</span></div>'
    html += f'<div class="quick-stat"><span class="qs-label">Min</span><span class="qs-value">{format_number(stats["min"])}</span></div>'
    html += f'<div class="quick-stat"><span class="qs-label">Max</span><span class="qs-value">{format_number(stats["max"])}</span></div>'
    html += f'<div class="quick-stat"><span class="qs-label">Memory size</span><span class="qs-value">{_format_bytes(stats["memory_size"])}</span></div>'
    html += '</div>'

    # Tabs container
    html += '<div class="tab-container variable-tabs" style="margin-top: 1.5rem;">'
    html += '<div class="tab-buttons">'
    html += f'<button class="tab-btn active" onclick="switchTab(this, \'tab-stats-{col_id}\')">Stats</button>'
    html += f'<button class="tab-btn" onclick="switchTab(this, \'tab-hist-{col_id}\')">Histogram</button>'
    html += f'<button class="tab-btn" onclick="switchTab(this, \'tab-kde-{col_id}\')">KDE Plot</button>'
    html += f'<button class="tab-btn" onclick="switchTab(this, \'tab-qq-{col_id}\')">Normal Q-Q Plot</button>'
    html += f'<button class="tab-btn" onclick="switchTab(this, \'tab-box-{col_id}\')">Box Plot</button>'
    html += '</div>'

    # Stats tab content (Overview, Descriptive, Quantile Statistics)
    html += f'<div class="tab-content" id="tab-stats-{col_id}" style="display:block;">'
    html += '<div class="stats-tab-layout">'

    # Overview table
    html += '<div class="stats-column">'
    html += '<h4 class="sub-heading">Overview</h4>'
    html += '<div class="stats-table-wrapper"><table class="stats-table">'
    html += f'<tr><td class="label">Distinct Count</td><td class="value">{format_number(stats["n_distinct"])}</td></tr>'
    html += f'<tr><td class="label">Unique (%)</td><td class="value">{format_percentage(stats["distinct_pct"])}</td></tr>'
    html += f'<tr><td class="label">Missing</td><td class="value">{format_number(stats["n_missing"])}</td></tr>'
    html += f'<tr><td class="label">Missing (%)</td><td class="value">{format_percentage(stats["missing_pct"])}</td></tr>'
    html += f'<tr><td class="label">Infinite</td><td class="value">{format_number(stats["n_inf"])}</td></tr>'
    html += f'<tr><td class="label">Infinite (%)</td><td class="value">{format_percentage(stats["inf_pct"])}</td></tr>'
    html += f'<tr><td class="label">Memory Size</td><td class="value">{_format_bytes(stats["memory_size"])}</td></tr>'
    html += f'<tr><td class="label">Mean</td><td class="value">{format_number(stats["mean"])}</td></tr>'
    html += f'<tr><td class="label">Minimum</td><td class="value">{format_number(stats["min"])}</td></tr>'
    html += f'<tr><td class="label">Maximum</td><td class="value">{format_number(stats["max"])}</td></tr>'
    html += f'<tr><td class="label">Zeros</td><td class="value">{format_number(stats["n_zeros"])}</td></tr>'
    html += f'<tr><td class="label">Zeros (%)</td><td class="value">{format_percentage(stats["zeros_pct"])}</td></tr>'
    html += f'<tr><td class="label">Negatives</td><td class="value">{format_number(stats["n_negatives"])}</td></tr>'
    html += f'<tr><td class="label">Negatives (%)</td><td class="value">{format_percentage(stats["negatives_pct"])}</td></tr>'
    html += '</table></div></div>'

    # Descriptive stats table
    html += '<div class="stats-column">'
    html += '<h4 class="sub-heading">Descriptive Statistics</h4>'
    html += '<div class="stats-table-wrapper"><table class="stats-table">'
    html += f'<tr><td class="label">Mean</td><td class="value">{format_number(stats["mean"])}</td></tr>'
    html += f'<tr><td class="label">Standard Deviation</td><td class="value">{format_number(stats["std"])}</td></tr>'
    html += f'<tr><td class="label">Variance</td><td class="value">{format_number(stats["variance"])}</td></tr>'
    html += f'<tr><td class="label">Sum</td><td class="value">{format_number(stats["sum"])}</td></tr>'
    html += f'<tr><td class="label">Skewness</td><td class="value">{format_number(stats["skewness"])}</td></tr>'
    html += f'<tr><td class="label">Kurtosis</td><td class="value">{format_number(stats["kurtosis"])}</td></tr>'
    html += f'<tr><td class="label">Coefficient of Variation</td><td class="value">{format_number(stats["cv"])}</td></tr>'
    html += '</table></div></div>'

    html += '</div>' # End stats-tab-layout

    # Quantile statistics row
    html += '<div class="stats-bottom-row" style="margin-top: 1rem;">'
    html += '<h4 class="sub-heading">Quantile Statistics</h4>'
    html += '<div class="stats-table-wrapper"><table class="stats-table">'
    html += f'<tr><td class="label">Minimum</td><td class="value">{format_number(stats["min"])}</td></tr>'
    html += f'<tr><td class="label">5-th Percentile</td><td class="value">{format_number(stats["p5"])}</td></tr>'
    html += f'<tr><td class="label">Q1 (25%)</td><td class="value">{format_number(stats["p25"])}</td></tr>'
    html += f'<tr><td class="label">Median (50%)</td><td class="value">{format_number(stats["p50"])}</td></tr>'
    html += f'<tr><td class="label">Q3 (75%)</td><td class="value">{format_number(stats["p75"])}</td></tr>'
    html += f'<tr><td class="label">95-th Percentile</td><td class="value">{format_number(stats["p95"])}</td></tr>'
    html += f'<tr><td class="label">99-th Percentile</td><td class="value">{format_number(stats["p99"])}</td></tr>'
    html += f'<tr><td class="label">Maximum</td><td class="value">{format_number(stats["max"])}</td></tr>'
    html += f'<tr><td class="label">Range</td><td class="value">{format_number(stats["range"])}</td></tr>'
    html += f'<tr><td class="label">IQR</td><td class="value">{format_number(stats["iqr"])}</td></tr>'
    html += '</table></div></div>'
    html += '</div>' # End stats-bottom-row

    html += '</div>' # End stats tab content

    numeric_series = pd.to_numeric(series, errors="coerce").dropna()

    # Histogram tab
    html += f'<div class="tab-content" id="tab-hist-{col_id}" style="display:none;">'
    if len(numeric_series) > 0:
        html += histogram_chart(numeric_series, f"hist_{col_id}", title="Distribution Histogram")
    else:
        html += '<p class="empty-message">No numeric data available.</p>'
    html += '</div>'

    # KDE Plot tab
    html += f'<div class="tab-content" id="tab-kde-{col_id}" style="display:none;">'
    if len(numeric_series) > 0:
        html += kde_plot(numeric_series, f"kde_{col_id}", title="Kernel Density Estimation")
    else:
        html += '<p class="empty-message">No numeric data available.</p>'
    html += '</div>'

    # Normal Q-Q Plot tab
    html += f'<div class="tab-content" id="tab-qq-{col_id}" style="display:none;">'
    if len(numeric_series) > 0:
        html += qq_plot(numeric_series, f"qq_{col_id}", title="Normal Q-Q Plot")
    else:
        html += '<p class="empty-message">No numeric data available.</p>'
    html += '</div>'

    # Box Plot tab
    html += f'<div class="tab-content" id="tab-box-{col_id}" style="display:none;">'
    if len(numeric_series) > 0:
        html += box_plot(numeric_series, f"box_{col_id}", title="Box Plot")
    else:
        html += '<p class="empty-message">No numeric data available.</p>'
    html += '</div>'

    html += '</div>' # End tab-container
    html += '</div>' # End variable-card
    return html


def _categorical_card(col, series, style=""):
    """Generate a variable card for a categorical column."""
    stats = categorical_stats(series)
    col_id = col.replace(" ", "_").replace(".", "_").replace("/", "_").replace("-", "_")

    html = f'<div class="variable-card" id="var-{safe_str(col_id)}" {style}>'
    html += f'<div class="var-header">'
    html += f'<h4 class="var-name">{safe_str(col)}</h4>'
    html += f'<span class="type-badge type-categorical">Categorical</span>'
    if stats["n_missing"] > 0:
        html += f'<span class="missing-badge">{format_percentage(stats["missing_pct"])} missing</span>'
    html += '</div>'

    # Categorical layout (side-by-side stats and chart)
    html += '<div class="categorical-layout">'

    # Left stats table
    html += '<div class="categorical-left">'
    html += '<table class="stats-table">'
    html += f'<tr><td class="label">Distinct</td><td class="value">{format_number(stats["n_distinct"])}</td></tr>'
    html += f'<tr><td class="label">Distinct (%)</td><td class="value">{format_percentage(stats["distinct_pct"])}</td></tr>'
    html += f'<tr><td class="label">Missing</td><td class="value">{format_number(stats["n_missing"])}</td></tr>'
    html += f'<tr><td class="label">Missing (%)</td><td class="value">{format_percentage(stats["missing_pct"])}</td></tr>'
    html += f'<tr><td class="label">Memory size</td><td class="value">{_format_bytes(stats["memory_size"])}</td></tr>'
    html += '</table>'
    html += '</div>'

    # Right horizontal bar chart
    html += '<div class="categorical-right">'
    if stats["value_counts"]:
        labels = list(stats["value_counts"].keys())[:10]
        values = [stats["value_counts"][k] for k in labels]
        # Reverse for horizontal display (largest at top)
        labels.reverse()
        values.reverse()
        html += bar_chart(labels, values, f"bar_{col_id}", title="", horizontal=True)
    else:
        html += '<p class="empty-message">No categorical values available.</p>'
    html += '</div>'

    html += '</div>' # End categorical-layout

    # Collapsible details (Frequency table)
    detail_html = '<div class="table-responsive"><table class="styled-table compact">'
    detail_html += '<thead><tr><th>Value</th><th>Count</th><th>Frequency</th></tr></thead>'
    detail_html += '<tbody>'
    n_valid = stats["n_valid"]
    for val, count in stats["value_counts"].items():
        pct = count / n_valid if n_valid > 0 else 0
        detail_html += f'<tr><td>{safe_str(str(val)[:40])}</td><td>{format_number(count)}</td><td>{format_percentage(pct)}</td></tr>'
    detail_html += '</tbody></table></div>'

    html += collapsible("📋 More Details", detail_html)
    html += '</div>' # End variable-card
    return html


def _boolean_card(col, series, style=""):
    """Generate a variable card for a boolean column."""
    stats = boolean_stats(series)
    col_id = col.replace(" ", "_").replace(".", "_").replace("/", "_").replace("-", "_")

    html = f'<div class="variable-card" id="var-{safe_str(col_id)}" {style}>'
    html += f'<div class="var-header">'
    html += f'<h4 class="var-name">{safe_str(col)}</h4>'
    html += f'<span class="type-badge type-boolean">Boolean</span>'
    html += '</div>'

    html += '<div class="var-quick-stats">'
    html += f'<div class="quick-stat"><span class="qs-label">True</span><span class="qs-value">{format_number(stats["true_count"])} ({format_percentage(stats["true_pct"])})</span></div>'
    html += f'<div class="quick-stat"><span class="qs-label">False</span><span class="qs-value">{format_number(stats["false_count"])} ({format_percentage(stats["false_pct"])})</span></div>'
    html += f'<div class="quick-stat"><span class="qs-label">Missing</span><span class="qs-value">{format_number(stats["n_missing"])}</span></div>'
    html += '</div>'

    # Pie chart
    html += '<div class="var-chart-half" style="margin: 0 auto;">'
    html += pie_chart(["True", "False"], [stats["true_count"], stats["false_count"]],
                      ["#10b981", "#ef4444"], f"pie_{col_id}")
    html += '</div>'

    html += '</div>'
    return html


def _datetime_card(col, series, style=""):
    """Generate a variable card for a datetime column."""
    stats = datetime_stats(series)
    col_id = col.replace(" ", "_").replace(".", "_").replace("/", "_").replace("-", "_")

    html = f'<div class="variable-card" id="var-{safe_str(col_id)}" {style}>'
    html += f'<div class="var-header">'
    html += f'<h4 class="var-name">{safe_str(col)}</h4>'
    html += f'<span class="type-badge type-datetime">DateTime</span>'
    html += '</div>'

    html += '<div class="var-quick-stats">'
    html += f'<div class="quick-stat"><span class="qs-label">Min</span><span class="qs-value">{safe_str(str(stats.get("min", "N/A"))[:20])}</span></div>'
    html += f'<div class="quick-stat"><span class="qs-label">Max</span><span class="qs-value">{safe_str(str(stats.get("max", "N/A"))[:20])}</span></div>'
    html += f'<div class="quick-stat"><span class="qs-label">Range</span><span class="qs-value">{safe_str(str(stats.get("range", "N/A"))[:20])}</span></div>'
    html += f'<div class="quick-stat"><span class="qs-label">Distinct</span><span class="qs-value">{format_number(stats.get("n_distinct", 0))}</span></div>'
    html += '</div>'

    html += '</div>'
    return html


def _constant_card(col, series, style=""):
    """Generate a variable card for a constant column."""
    col_id = col.replace(" ", "_").replace(".", "_").replace("/", "_").replace("-", "_")
    val = series.dropna().iloc[0] if series.notna().any() else "N/A"

    html = f'<div class="variable-card" id="var-{safe_str(col_id)}" {style}>'
    html += f'<div class="var-header">'
    html += f'<h4 class="var-name">{safe_str(col)}</h4>'
    html += f'<span class="type-badge type-constant">Constant</span>'
    html += f'<span class="missing-badge" style="background:#f59e0b20;color:#f59e0b;"> Zero Variance</span>'
    html += '</div>'
    html += f'<div class="var-quick-stats"><div class="quick-stat"><span class="qs-label">Value</span><span class="qs-value">{safe_str(str(val)[:50])}</span></div></div>'
    html += '</div>'
    return html


def _text_card(col, series, style=""):
    """Generate a variable card for a text column."""
    col_id = col.replace(" ", "_").replace(".", "_").replace("/", "_").replace("-", "_")
    clean = series.dropna()

    html = f'<div class="variable-card" id="var-{safe_str(col_id)}" {style}>'
    html += f'<div class="var-header">'
    html += f'<h4 class="var-name">{safe_str(col)}</h4>'
    html += f'<span class="type-badge type-text">Text</span>'
    html += '</div>'

    avg_len = clean.astype(str).str.len().mean() if len(clean) > 0 else 0
    html += '<div class="var-quick-stats">'
    html += f'<div class="quick-stat"><span class="qs-label">Distinct</span><span class="qs-value">{format_number(clean.nunique())}</span></div>'
    html += f'<div class="quick-stat"><span class="qs-label">Avg Length</span><span class="qs-value">{format_number(avg_len)}</span></div>'
    html += f'<div class="quick-stat"><span class="qs-label">Missing</span><span class="qs-value">{format_number(series.isna().sum())}</span></div>'
    html += '</div></div>'
    return html
