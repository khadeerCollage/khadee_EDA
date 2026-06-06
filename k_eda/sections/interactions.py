"""
K-EDA — Section 7: Interactions
=====================================
Pairwise variable relationships: scatter plots, pair grid, cross-tabulation.
"""

import numpy as np
import pandas as pd

from ..type_detector import ColumnType
from ..engines.correlation_engine import pearson_correlation, top_correlations
from ..utils import format_number, section_header, safe_str
from ..renderers.chart_renderer import scatter_plot, grouped_box_plot


def generate(df, type_map):
    """Generate the Interactions section HTML."""
    numeric_cols = [c for c, t in type_map.items()
                    if t == ColumnType.NUMERIC and df[c].notna().sum() > 1]
    categorical_cols = [c for c, t in type_map.items()
                        if t == ColumnType.CATEGORICAL]

    html = section_header("section-interactions", " Interactions",
                          "Pairwise variable relationships and interaction patterns")

    if len(numeric_cols) < 2 and not categorical_cols:
        html += '<div class="card"><p class="empty-message">Not enough variables for interaction analysis.</p></div>'
        return html

    # ── Top Correlated Scatter Plots ──
    if len(numeric_cols) >= 2:
        corr_matrix = pearson_correlation(df, numeric_cols[:20])
        top_pairs = top_correlations(corr_matrix, n=6)

        if top_pairs:
            html += '<div class="card">'
            html += '<h3 class="card-title"> Top Correlated Pairs (Scatter Plots)</h3>'
            html += '<div class="scatter-grid">'
            for c1, c2, val in top_pairs:
                x = pd.to_numeric(df[c1], errors="coerce").dropna()
                y = pd.to_numeric(df[c2], errors="coerce").dropna()
                # Align indices
                common = x.index.intersection(y.index)
                if len(common) > 0:
                    x_aligned = x.loc[common]
                    y_aligned = y.loc[common]
                    # Sample for performance
                    if len(x_aligned) > 2000:
                        idx = x_aligned.sample(2000, random_state=42).index
                        x_aligned = x_aligned.loc[idx]
                        y_aligned = y_aligned.loc[idx]
                    title = f"{c1} vs {c2} (r={val:.3f})"
                    chart_id = f"scatter_{c1}_{c2}".replace(" ", "_").replace(".", "_")
                    html += '<div class="scatter-item">'
                    html += scatter_plot(x_aligned, y_aligned, c1, c2, chart_id, title=title)
                    html += '</div>'
            html += '</div></div>'

    # ── Numeric vs Categorical (Grouped Box Plots) ──
    if numeric_cols and categorical_cols:
        html += '<div class="card">'
        html += '<h3 class="card-title"> Numeric vs Categorical</h3>'
        html += '<p class="card-desc">Box plots showing distribution of numeric variables across categorical groups.</p>'

        shown = 0
        for cat_col in categorical_cols[:3]:
            n_unique = df[cat_col].nunique()
            if 2 <= n_unique <= 10:
                for num_col in numeric_cols[:2]:
                    chart_id = f"grouped_{cat_col}_{num_col}".replace(" ", "_").replace(".", "_")
                    html += grouped_box_plot(df, num_col, cat_col, chart_id,
                                            title=f"{num_col} by {cat_col}")
                    shown += 1
                    if shown >= 4:
                        break
            if shown >= 4:
                break

        if shown == 0:
            html += '<p class="empty-message">No suitable categorical variables (2-10 groups) found for grouped analysis.</p>'

        html += '</div>'

    return html
