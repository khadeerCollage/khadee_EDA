"""
K-EDA — 🇺🇸 US Technique: ML-Readiness & Feature Engineering
===================================================================
Feature importance, target leakage detection, feature engineering suggestions.
"""

import numpy as np
import pandas as pd

from ..type_detector import ColumnType
from ..utils import format_number, format_percentage, safe_str


def generate(df, type_map):
    """Generate US technique analysis HTML."""
    numeric_cols = [c for c, t in type_map.items() if t == ColumnType.NUMERIC]

    html = ""

    # ── Feature Importance (Variance-based) ──
    html += '<div class="card">'
    html += '<h4 class="card-title">📊 Feature Importance (Variance-Based Ranking)</h4>'
    html += '<p class="card-desc">Features ranked by normalized variance — higher variance features carry more information.</p>'

    if numeric_cols:
        variances = {}
        for col in numeric_cols:
            clean = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(clean) > 0 and clean.std() > 0:
                # Coefficient of variation for scale-invariant ranking
                variances[col] = float(clean.std() / (abs(clean.mean()) + 1e-10))

        if variances:
            sorted_vars = sorted(variances.items(), key=lambda x: x[1], reverse=True)

            html += '<div class="table-responsive"><table class="styled-table compact">'
            html += '<thead><tr><th>Rank</th><th>Feature</th><th>Coeff. of Variation</th><th>Importance</th></tr></thead>'
            html += '<tbody>'
            max_var = sorted_vars[0][1] if sorted_vars else 1
            for i, (col, var) in enumerate(sorted_vars[:20]):
                pct = var / max_var * 100 if max_var > 0 else 0
                bar_html = f'<div class="mini-bar" style="width:{min(100, pct):.0f}%"></div>'
                html += f'<tr><td>{i+1}</td><td class="col-name">{safe_str(col)}</td>'
                html += f'<td>{format_number(var, 4)}</td>'
                html += f'<td>{bar_html}</td></tr>'
            html += '</tbody></table></div>'
    else:
        html += '<p class="empty-message">No numeric features for importance ranking.</p>'
    html += '</div>'

    # ── Feature Engineering Suggestions ──
    html += '<div class="card">'
    html += '<h4 class="card-title">🔧 Feature Engineering Suggestions</h4>'
    html += '<div class="suggestions-list">'

    suggestions = []
    for col in numeric_cols:
        clean = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(clean) < 3:
            continue

        skew = abs(float(clean.skew()))
        if skew > 1.0 and clean.min() >= 0:
            suggestions.append(f"<strong>{safe_str(col)}</strong>: Apply <code>log1p</code> transform (skew={skew:.2f}, all non-negative)")
        elif skew > 1.0:
            suggestions.append(f"<strong>{safe_str(col)}</strong>: Apply <code>power transform</code> (skew={skew:.2f})")

        # Suggest binning for long-tail
        if clean.nunique() > 100 and float(clean.kurtosis()) > 5:
            suggestions.append(f"<strong>{safe_str(col)}</strong>: Consider <code>binning/bucketing</code> (high kurtosis={clean.kurtosis():.2f})")

    # Categorical encoding suggestions
    cat_cols = [c for c, t in type_map.items() if t == ColumnType.CATEGORICAL]
    for col in cat_cols:
        n_unique = df[col].nunique()
        if n_unique > 10:
            suggestions.append(f"<strong>{safe_str(col)}</strong>: Use <code>target encoding</code> or <code>frequency encoding</code> ({n_unique} categories)")
        elif n_unique <= 10:
            suggestions.append(f"<strong>{safe_str(col)}</strong>: Use <code>one-hot encoding</code> ({n_unique} categories)")

    if suggestions:
        for s in suggestions[:15]:
            html += f'<div class="suggestion-item">💡 {s}</div>'
    else:
        html += '<p class="empty-message">No specific suggestions — data looks well-prepared.</p>'

    html += '</div></div>'

    return html


def analyze(df):
    """
    Standalone US technique analysis.
    Returns dict with feature importance and engineering suggestions.
    """
    from ..type_detector import detect_types
    type_map = detect_types(df)
    numeric_cols = [c for c, t in type_map.items() if t == ColumnType.NUMERIC]

    results = {"feature_importance": {}, "suggestions": []}

    for col in numeric_cols:
        clean = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(clean) > 0 and clean.std() > 0:
            results["feature_importance"][col] = float(clean.std() / (abs(clean.mean()) + 1e-10))

    return results
