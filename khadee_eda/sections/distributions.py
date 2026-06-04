"""
Khadee EDA — Section 3: Distributions
======================================
Normality tests, histogram grid, QQ plots, violin plots, skewness/kurtosis table.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from ..type_detector import ColumnType
from ..utils import format_number, section_header, safe_str
from ..renderers.chart_renderer import histogram_grid, qq_plot, violin_plot


def generate(df, type_map):
    """Generate the Distributions section HTML."""
    numeric_cols = [c for c, t in type_map.items()
                    if t in (ColumnType.NUMERIC,) and df[c].notna().sum() > 2]

    html = section_header("section-distributions", " Distributions",
                          "Distribution analysis and normality testing for numeric features")

    if not numeric_cols:
        html += '<div class="card"><p class="empty-message">No numeric columns available for distribution analysis.</p></div>'
        return html

    # ── Normality Tests Table ──
    html += '<div class="card">'
    html += '<h3 class="card-title"> Normality Tests</h3>'
    html += '<p class="card-desc">Shapiro-Wilk & Jarque-Bera tests. p < 0.05 suggests non-normal distribution.</p>'
    html += '<div class="table-responsive"><table class="styled-table">'
    html += '<thead><tr><th>Column</th><th>Skewness</th><th>Kurtosis</th><th>Shapiro-Wilk p</th><th>Jarque-Bera p</th><th>Normal?</th></tr></thead>'
    html += '<tbody>'

    normality_results = []
    for col in numeric_cols:
        clean = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(clean) < 8:
            continue

        skew = float(clean.skew())
        kurt = float(clean.kurtosis())

        # Shapiro-Wilk (sample if too large)
        sample = clean.sample(min(5000, len(clean)), random_state=42) if len(clean) > 5000 else clean
        try:
            sw_stat, sw_p = scipy_stats.shapiro(sample)
        except Exception:
            sw_p = None

        # Jarque-Bera
        try:
            jb_stat, jb_p = scipy_stats.jarque_bera(clean)
        except Exception:
            jb_p = None

        is_normal = (sw_p is not None and sw_p > 0.05) and (jb_p is not None and jb_p > 0.05)
        normal_badge = '<span class="badge-pass">Yes</span>' if is_normal else '<span class="badge-fail">No</span>'

        html += f'<tr><td class="col-name">{safe_str(col)}</td>'
        html += f'<td>{format_number(skew)}</td>'
        html += f'<td>{format_number(kurt)}</td>'
        html += f'<td>{format_number(sw_p, 4) if sw_p is not None else "N/A"}</td>'
        html += f'<td>{format_number(jb_p, 4) if jb_p is not None else "N/A"}</td>'
        html += f'<td>{normal_badge}</td></tr>'

        normality_results.append({
            "col": col, "skew": skew, "kurt": kurt,
            "sw_p": sw_p, "jb_p": jb_p, "is_normal": is_normal,
        })

    html += '</tbody></table></div></div>'

    # ── Histogram Grid ──
    html += '<div class="card">'
    html += '<h3 class="card-title">📊 Distribution Grid</h3>'
    display_cols = numeric_cols[:20]  # Limit for performance
    series_list = [pd.to_numeric(df[c], errors="coerce").dropna() for c in display_cols]
    html += histogram_grid(series_list, display_cols, "dist_grid")
    html += '</div>'

    # ── Skewness / Kurtosis Summary ──
    if normality_results:
        html += '<div class="card">'
        html += '<h3 class="card-title">📐 Skewness & Kurtosis Summary</h3>'

        skewed = [r for r in normality_results if abs(r["skew"]) > 1.0]
        if skewed:
            html += '<div class="alert-box warning">'
            html += f'<strong>⚠️ {len(skewed)} features</strong> with high skewness (|skew| > 1.0). Consider log or power transforms.'
            html += '</div>'

        heavy_tail = [r for r in normality_results if abs(r["kurt"]) > 3.0]
        if heavy_tail:
            html += '<div class="alert-box info">'
            html += f'<strong>ℹ️ {len(heavy_tail)} features</strong> with excess kurtosis (|kurt| > 3.0). Heavy-tailed distributions.'
            html += '</div>'

        html += '</div>'

    return html
