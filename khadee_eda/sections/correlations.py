"""
Khadee EDA — Section 4: Correlations
=====================================
Pearson, Spearman, Kendall, Cramér's V heatmaps with tab toggles.
"""

import numpy as np
import pandas as pd

from ..type_detector import ColumnType
from ..engines.correlation_engine import (
    pearson_correlation, spearman_correlation, kendall_correlation,
    cramers_v_matrix, top_correlations, multicollinearity_warnings,
)
from ..utils import format_number, section_header, safe_str
from ..renderers.chart_renderer import heatmap_chart


def generate(df, type_map):
    """Generate the Correlations section HTML."""
    numeric_cols = [c for c, t in type_map.items()
                    if t in (ColumnType.NUMERIC,) and df[c].notna().sum() > 1]
    categorical_cols = [c for c, t in type_map.items()
                        if t == ColumnType.CATEGORICAL]

    html = section_header("section-correlations", " Correlations",
                          "Multi-method correlation analysis between variables")

    if len(numeric_cols) < 2 and len(categorical_cols) < 2:
        html += '<div class="card"><p class="empty-message">Not enough columns for correlation analysis.</p></div>'
        return html

    # ── Correlation Heatmaps (Tabbed) ──
    html += '<div class="card">'
    html += '<h3 class="card-title"> Correlation Matrices</h3>'

    # Tab buttons
    tabs = []
    if len(numeric_cols) >= 2:
        tabs.extend(["Pearson", "Spearman", "Kendall"])
    if len(categorical_cols) >= 2:
        tabs.append("Cramér's V")

    html += '<div class="tab-container">'
    html += '<div class="tab-buttons">'
    for i, tab in enumerate(tabs):
        active = "active" if i == 0 else ""
        html += f'<button class="tab-btn {active}" onclick="switchTab(this, \'corr-tab-{i}\')" data-tab="corr-tab-{i}">{tab}</button>'
    html += '</div>'

    tab_idx = 0
    if len(numeric_cols) >= 2:
        display_cols = numeric_cols[:30]  # Limit for readability

        # Pearson
        pearson = pearson_correlation(df, display_cols)
        style = "display:block" if tab_idx == 0 else "display:none"
        html += f'<div class="tab-content" id="corr-tab-{tab_idx}" style="{style}">'
        html += heatmap_chart(pearson, f"pearson_heatmap", title="Pearson Correlation")
        html += '</div>'
        tab_idx += 1

        # Spearman
        spearman = spearman_correlation(df, display_cols)
        html += f'<div class="tab-content" id="corr-tab-{tab_idx}" style="display:none">'
        html += heatmap_chart(spearman, f"spearman_heatmap", title="Spearman Rank Correlation")
        html += '</div>'
        tab_idx += 1

        # Kendall
        kendall = kendall_correlation(df, display_cols)
        html += f'<div class="tab-content" id="corr-tab-{tab_idx}" style="display:none">'
        html += heatmap_chart(kendall, f"kendall_heatmap", title="Kendall Tau Correlation")
        html += '</div>'
        tab_idx += 1

    if len(categorical_cols) >= 2:
        display_cats = categorical_cols[:20]
        cramers = cramers_v_matrix(df, display_cats)
        style = "display:block" if tab_idx == 0 else "display:none"
        html += f'<div class="tab-content" id="corr-tab-{tab_idx}" style="{style}">'
        html += heatmap_chart(cramers, f"cramers_heatmap", title="Cramér's V (Categorical Association)")
        html += '</div>'

    html += '</div></div>'

    # ── Top Correlations Table ──
    if len(numeric_cols) >= 2:
        pearson_matrix = pearson_correlation(df, numeric_cols[:30])
        top_pairs = top_correlations(pearson_matrix, n=15)

        if top_pairs:
            html += '<div class="card">'
            html += '<h3 class="card-title"> Top Correlations (Pearson)</h3>'
            html += '<div class="table-responsive"><table class="styled-table">'
            html += '<thead><tr><th>#</th><th>Variable 1</th><th>Variable 2</th><th>Correlation</th><th>Strength</th></tr></thead>'
            html += '<tbody>'
            for i, (c1, c2, val) in enumerate(top_pairs):
                abs_val = abs(val)
                if abs_val > 0.8:
                    strength = '<span class="badge-danger">Very Strong</span>'
                elif abs_val > 0.6:
                    strength = '<span class="badge-warning">Strong</span>'
                elif abs_val > 0.4:
                    strength = '<span class="badge-info">Moderate</span>'
                else:
                    strength = '<span class="badge-muted">Weak</span>'
                html += f'<tr><td>{i+1}</td><td>{safe_str(c1)}</td><td>{safe_str(c2)}</td>'
                html += f'<td>{format_number(val, 4)}</td><td>{strength}</td></tr>'
            html += '</tbody></table></div></div>'

        # Multicollinearity Warnings
        warnings = multicollinearity_warnings(pearson_matrix)
        if warnings:
            html += '<div class="card">'
            html += '<h3 class="card-title"> Multicollinearity Warnings</h3>'
            html += f'<div class="alert-box danger"><strong>{len(warnings)} pairs</strong> with |correlation| > 0.80. Consider removing redundant features.</div>'
            html += '<div class="table-responsive"><table class="styled-table compact">'
            html += '<thead><tr><th>Variable 1</th><th>Variable 2</th><th>Correlation</th></tr></thead>'
            html += '<tbody>'
            for c1, c2, val in warnings:
                html += f'<tr><td>{safe_str(c1)}</td><td>{safe_str(c2)}</td><td>{format_number(val, 4)}</td></tr>'
            html += '</tbody></table></div></div>'

    return html
