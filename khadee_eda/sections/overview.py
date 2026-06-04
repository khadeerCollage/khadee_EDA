"""
Khadee EDA — Section 1: Dataset Overview
=========================================
High-level snapshot: summary cards, alerts, reproduction info, type breakdown.
"""

import time
import numpy as np
import pandas as pd

from ..type_detector import ColumnType, get_type_summary, get_columns_by_type
from ..utils import (
    format_number, format_percentage, format_memory,
    stat_card, section_header, alert_badge, safe_str,
)
from ..renderers.chart_renderer import donut_chart


def generate(df, type_map, metadata, start_time):
    """Generate the Overview section HTML."""
    n_rows, n_cols = df.shape
    n_cells = n_rows * n_cols
    n_missing = int(df.isna().sum().sum())
    n_duplicates = int(df.duplicated().sum())
    memory = df.memory_usage(deep=True).sum()

    type_summary = get_type_summary(type_map)
    alerts = _detect_alerts(df, type_map, type_summary)

    elapsed = time.time() - start_time

    html = section_header("section-overview", " Dataset Overview",
                          "High-level snapshot of your dataset")

    # ── Summary Cards ──
    html += '<div class="stat-grid">'
    html += stat_card("Rows", format_number(n_rows), "📊", "#3b82f6")
    html += stat_card("Columns", format_number(n_cols), "📋", "#8b5cf6")
    html += stat_card("Total Cells", format_number(n_cells), "🔢", "#06b6d4")
    html += stat_card("Missing Cells", f"{format_number(n_missing)} ({format_percentage(n_missing / n_cells if n_cells > 0 else 0)})", "❓", "#ef4444")
    html += stat_card("Duplicate Rows", f"{format_number(n_duplicates)} ({format_percentage(n_duplicates / n_rows if n_rows > 0 else 0)})", "📑", "#f59e0b")
    html += stat_card("Memory", format_memory(memory), "💾", "#10b981")
    html += stat_card("Numeric Features", format_number(type_summary.get(ColumnType.NUMERIC, 0) + type_summary.get(ColumnType.UNIQUE_ID, 0)), "🔢", "#3b82f6")
    html += stat_card("Categorical Features", format_number(type_summary.get(ColumnType.CATEGORICAL, 0)), "🏷️", "#8b5cf6")
    html += stat_card("DateTime Features", format_number(type_summary.get(ColumnType.DATETIME, 0)), "📅", "#06b6d4")
    html += stat_card("Boolean Features", format_number(type_summary.get(ColumnType.BOOLEAN, 0)), "✅", "#10b981")
    html += '</div>'

    # ── Alerts ──
    if alerts:
        html += '<div class="card">'
        html += '<h3 class="card-title">⚠️ Alerts</h3>'
        html += '<div class="alerts-list">'
        for alert in alerts:
            html += f'<div class="alert-item">{alert_badge(alert["level"], alert["type"])} {safe_str(alert["message"])}</div>'
        html += '</div></div>'

    # ── Data Types Breakdown ──
    type_labels = []
    type_values = []
    type_colors = ["#3b82f6", "#8b5cf6", "#06b6d4", "#10b981", "#f59e0b", "#ef4444", "#6366f1"]
    type_name_map = {
        ColumnType.NUMERIC: "Numeric",
        ColumnType.CATEGORICAL: "Categorical",
        ColumnType.BOOLEAN: "Boolean",
        ColumnType.DATETIME: "DateTime",
        ColumnType.TEXT: "Text",
        ColumnType.CONSTANT: "Constant",
        ColumnType.UNIQUE_ID: "Unique ID",
    }
    for t, label in type_name_map.items():
        count = type_summary.get(t, 0)
        if count > 0:
            type_labels.append(label)
            type_values.append(count)

    if type_labels:
        html += '<div class="card">'
        html += '<h3 class="card-title"> Data Types Breakdown</h3>'
        html += donut_chart(type_labels, type_values, type_colors[:len(type_labels)], "types_donut")
        html += '</div>'

    # ── Column Overview Table ──
    html += '<div class="card">'
    html += '<h3 class="card-title"> Column Overview</h3>'
    html += '<div class="table-responsive"><table class="styled-table">'
    html += '<thead><tr><th>#</th><th>Column</th><th>Type</th><th>Non-Null</th><th>Missing %</th><th>Unique</th><th>Dtype</th></tr></thead>'
    html += '<tbody>'
    for i, col in enumerate(df.columns):
        n_valid = int(df[col].notna().sum())
        miss_pct = df[col].isna().sum() / n_rows if n_rows > 0 else 0
        n_unique = int(df[col].nunique())
        col_type = type_name_map.get(type_map.get(col, ""), type_map.get(col, "unknown"))
        dtype = str(df[col].dtype)
        miss_class = "danger" if miss_pct > 0.5 else "warning" if miss_pct > 0.1 else ""
        html += f'<tr><td>{i+1}</td><td class="col-name">{safe_str(col)}</td>'
        html += f'<td><span class="type-badge">{col_type}</span></td>'
        html += f'<td>{format_number(n_valid)}</td>'
        html += f'<td class="{miss_class}">{format_percentage(miss_pct)}</td>'
        html += f'<td>{format_number(n_unique)}</td>'
        html += f'<td class="dtype">{dtype}</td></tr>'
    html += '</tbody></table></div></div>'

    # ── Reproduction Info ──
    html += '<div class="card">'
    html += '<h3 class="card-title"> Reproduction</h3>'
    html += '<div class="repro-grid">'
    html += f'<div class="repro-item"><span class="repro-label">File</span><span class="repro-value">{safe_str(metadata.get("file_name", "N/A"))}</span></div>'
    html += f'<div class="repro-item"><span class="repro-label">Format</span><span class="repro-value">{safe_str(metadata.get("file_format", "N/A"))}</span></div>'
    if metadata.get("file_size"):
        html += f'<div class="repro-item"><span class="repro-label">File Size</span><span class="repro-value">{format_memory(metadata["file_size"])}</span></div>'
    html += f'<div class="repro-item"><span class="repro-label">Analysis Time</span><span class="repro-value">{elapsed:.2f}s</span></div>'
    html += f'<div class="repro-item"><span class="repro-label">Package</span><span class="repro-value">Khadee EDA v1.0.0</span></div>'
    html += f'<div class="repro-item"><span class="repro-label">Timestamp</span><span class="repro-value">{time.strftime("%Y-%m-%d %H:%M:%S")}</span></div>'
    html += '</div></div>'

    return html


def _detect_alerts(df, type_map, type_summary):
    """Detect data quality issues and generate alerts."""
    alerts = []
    n_rows = len(df)

    for col in df.columns:
        col_type = type_map.get(col)
        miss_pct = df[col].isna().sum() / n_rows if n_rows > 0 else 0

        # High missing
        if miss_pct > 0.50:
            alerts.append({"level": "danger", "type": "HIGH MISSING",
                           "message": f"'{col}' has {format_percentage(miss_pct)} missing values"})
        elif miss_pct > 0.10:
            alerts.append({"level": "warning", "type": "MISSING",
                           "message": f"'{col}' has {format_percentage(miss_pct)} missing values"})

        # Constant column
        if col_type == ColumnType.CONSTANT:
            alerts.append({"level": "warning", "type": "CONSTANT",
                           "message": f"'{col}' has constant or single value"})

        # Unique ID
        if col_type == ColumnType.UNIQUE_ID:
            alerts.append({"level": "info", "type": "UNIQUE ID",
                           "message": f"'{col}' appears to be a unique identifier"})

        # High cardinality categorical
        if col_type == ColumnType.CATEGORICAL:
            n_unique = df[col].nunique()
            if n_unique > 50:
                alerts.append({"level": "warning", "type": "HIGH CARDINALITY",
                               "message": f"'{col}' has {n_unique} unique categories"})

        # Skewness for numeric
        if col_type == ColumnType.NUMERIC:
            try:
                clean = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(clean) > 3:
                    skew = abs(float(clean.skew()))
                    if skew > 2.0:
                        alerts.append({"level": "warning", "type": "SKEWED",
                                       "message": f"'{col}' is highly skewed (skew={skew:.2f})"})
            except Exception:
                pass

    # Duplicate rows
    n_dup = df.duplicated().sum()
    if n_dup > 0:
        pct = n_dup / n_rows if n_rows > 0 else 0
        level = "danger" if pct > 0.10 else "warning"
        alerts.append({"level": level, "type": "DUPLICATES",
                       "message": f"{format_number(n_dup)} duplicate rows ({format_percentage(pct)})"})

    return alerts
