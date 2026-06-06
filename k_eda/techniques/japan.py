"""
K-EDA — 🇯🇵 Japan Technique: Quality Control & Process Analytics (Kaizen)
=================================================================================
Shewhart control charts, Cp/Cpk indices, Pareto analysis, stability metrics.
"""

import numpy as np
import pandas as pd

from ..type_detector import ColumnType
from ..utils import format_number, format_percentage, safe_str
from ..renderers.chart_renderer import control_chart, bar_chart


def generate(df, type_map):
    """Generate Japan technique analysis HTML."""
    numeric_cols = [c for c, t in type_map.items()
                    if t == ColumnType.NUMERIC and df[c].notna().sum() > 10]
    categorical_cols = [c for c, t in type_map.items()
                        if t == ColumnType.CATEGORICAL]

    html = ""

    # ── Control Charts (Shewhart) ──
    html += '<div class="card">'
    html += '<h4 class="card-title">📉 Shewhart Control Charts (UCL/LCL)</h4>'
    html += '<p class="card-desc">Statistical process control — detect out-of-control data points.</p>'

    if numeric_cols:
        for col in numeric_cols[:4]:
            clean = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(clean) < 10:
                continue

            mean = float(clean.mean())
            std = float(clean.std())
            ucl = mean + 3 * std
            lcl = mean - 3 * std

            # Count out-of-control points
            ooc = int(((clean > ucl) | (clean < lcl)).sum())
            ooc_pct = ooc / len(clean) if len(clean) > 0 else 0

            chart_id = f"control_{col}".replace(" ", "_").replace(".", "_")
            html += f'<div class="control-chart-info">'
            html += f'<strong>{safe_str(col)}</strong> — Mean: {format_number(mean)}, UCL: {format_number(ucl)}, LCL: {format_number(lcl)}, '
            html += f'Out-of-Control: {format_number(ooc)} ({format_percentage(ooc_pct)})'
            html += '</div>'
            html += control_chart(clean, mean, ucl, lcl, chart_id, title=f"{col} — Control Chart")
    else:
        html += '<p class="empty-message">No numeric columns for control chart analysis.</p>'
    html += '</div>'

    # ── Process Capability (Cp/Cpk) ──
    html += '<div class="card">'
    html += '<h4 class="card-title">📐 Process Capability Indices (Cp/Cpk)</h4>'
    html += '<p class="card-desc">Cp measures spread vs tolerance, Cpk measures centering. Using ±3σ as spec limits.</p>'

    if numeric_cols:
        html += '<div class="table-responsive"><table class="styled-table compact">'
        html += '<thead><tr><th>Feature</th><th>Mean</th><th>Std Dev</th><th>USL (μ+3σ)</th><th>LSL (μ-3σ)</th><th>Cp</th><th>Cpk</th><th>Assessment</th></tr></thead>'
        html += '<tbody>'

        for col in numeric_cols[:15]:
            clean = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(clean) < 10 or clean.std() == 0:
                continue

            mean = float(clean.mean())
            std = float(clean.std())
            usl = mean + 3 * std
            lsl = mean - 3 * std

            cp = (usl - lsl) / (6 * std) if std > 0 else 0
            cpu = (usl - mean) / (3 * std) if std > 0 else 0
            cpl = (mean - lsl) / (3 * std) if std > 0 else 0
            cpk = min(cpu, cpl)

            if cpk >= 1.33:
                assessment = '<span class="badge-pass">Excellent</span>'
            elif cpk >= 1.0:
                assessment = '<span class="badge-info">Capable</span>'
            else:
                assessment = '<span class="badge-warning">Needs Improvement</span>'

            html += f'<tr><td class="col-name">{safe_str(col)}</td>'
            html += f'<td>{format_number(mean)}</td>'
            html += f'<td>{format_number(std)}</td>'
            html += f'<td>{format_number(usl)}</td>'
            html += f'<td>{format_number(lsl)}</td>'
            html += f'<td>{format_number(cp, 3)}</td>'
            html += f'<td>{format_number(cpk, 3)}</td>'
            html += f'<td>{assessment}</td></tr>'

        html += '</tbody></table></div>'
    html += '</div>'

    # ── Pareto Analysis ──
    if categorical_cols:
        html += '<div class="card">'
        html += '<h4 class="card-title">📊 Pareto Analysis (80/20 Rule)</h4>'
        html += '<p class="card-desc">Identifying the vital few categories that account for the majority of data.</p>'

        for col in categorical_cols[:3]:
            vc = df[col].value_counts()
            total = vc.sum()
            cumulative = vc.cumsum()
            cumulative_pct = cumulative / total

            # Find the 80% cutoff
            n_for_80 = int((cumulative_pct <= 0.80).sum()) + 1
            pct_categories = n_for_80 / len(vc) * 100 if len(vc) > 0 else 0

            html += f'<div class="pareto-result">'
            html += f'<strong>{safe_str(col)}</strong>: '
            html += f'Top <strong>{n_for_80}</strong> categories ({pct_categories:.0f}% of {len(vc)} total) account for 80% of data.'
            html += '</div>'

        html += '</div>'

    # ── Stability Metrics (Rolling Stats) ──
    html += '<div class="card">'
    html += '<h4 class="card-title">📈 Data Stability Analysis</h4>'
    html += '<p class="card-desc">Rolling mean/std deviation to detect drift in data sequence.</p>'

    if numeric_cols:
        html += '<div class="table-responsive"><table class="styled-table compact">'
        html += '<thead><tr><th>Feature</th><th>Overall Mean</th><th>First Half Mean</th><th>Second Half Mean</th><th>Drift</th><th>Stable?</th></tr></thead>'
        html += '<tbody>'

        for col in numeric_cols[:15]:
            clean = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(clean) < 20:
                continue

            overall_mean = float(clean.mean())
            half = len(clean) // 2
            first_half = float(clean.iloc[:half].mean())
            second_half = float(clean.iloc[half:].mean())

            drift = abs(second_half - first_half) / (abs(overall_mean) + 1e-10) * 100
            stable = drift < 10  # Less than 10% drift
            stable_badge = '<span class="badge-pass">Stable</span>' if stable else '<span class="badge-warning">Drift Detected</span>'

            html += f'<tr><td class="col-name">{safe_str(col)}</td>'
            html += f'<td>{format_number(overall_mean)}</td>'
            html += f'<td>{format_number(first_half)}</td>'
            html += f'<td>{format_number(second_half)}</td>'
            html += f'<td>{format_number(drift, 1)}%</td>'
            html += f'<td>{stable_badge}</td></tr>'

        html += '</tbody></table></div>'
    html += '</div>'

    return html


def analyze(df):
    """Standalone Japan technique analysis."""
    from ..type_detector import detect_types
    type_map = detect_types(df)
    numeric_cols = [c for c, t in type_map.items() if t == ColumnType.NUMERIC]

    results = {"control_charts": {}, "stability": {}}

    for col in numeric_cols:
        clean = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(clean) < 10:
            continue
        mean = float(clean.mean())
        std = float(clean.std())
        results["control_charts"][col] = {
            "mean": mean, "ucl": mean + 3 * std, "lcl": mean - 3 * std,
            "out_of_control": int(((clean > mean + 3 * std) | (clean < mean - 3 * std)).sum()),
        }

    return results
