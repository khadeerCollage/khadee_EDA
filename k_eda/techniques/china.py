"""
K-EDA — 🇨🇳 China Technique: Large-Scale Pattern Recognition
===================================================================
PCA, Hopkins statistic, K-Means preview, data density analysis.
"""

import numpy as np
import pandas as pd

from ..type_detector import ColumnType
from ..engines.dim_reduction import pca_analysis, kmeans_preview, hopkins_statistic
from ..utils import format_number, format_percentage, safe_str
from ..renderers.chart_renderer import pca_variance_chart, cluster_scatter


def generate(df, type_map):
    """Generate China technique analysis HTML."""
    numeric_cols = [c for c, t in type_map.items()
                    if t == ColumnType.NUMERIC and df[c].notna().sum() > 3]

    html = ""

    if len(numeric_cols) < 2:
        html += '<div class="card"><p class="empty-message">Need at least 2 numeric features for pattern recognition analysis.</p></div>'
        return html

    # ── PCA Variance Explained ──
    pca_result = pca_analysis(df, numeric_cols)
    if "error" not in pca_result:
        html += '<div class="card">'
        html += '<h4 class="card-title">📊 PCA — Variance Explained</h4>'
        html += f'<p class="card-desc">Analyzed {pca_result["n_features"]} features. '
        html += f'<strong>{pca_result["n_components_95"]} components</strong> explain 95% of variance.</p>'

        html += pca_variance_chart(
            pca_result["explained_variance_ratio"],
            pca_result["cumulative_variance"],
            "pca_variance"
        )

        # Dimensionality reduction suggestion
        ratio = pca_result["n_components_95"] / pca_result["n_features"] if pca_result["n_features"] > 0 else 1
        if ratio < 0.5:
            html += f'<div class="alert-box success">💡 <strong>High redundancy detected!</strong> You can reduce from {pca_result["n_features"]} to {pca_result["n_components_95"]} features ({ratio:.0%}) while retaining 95% information.</div>'
        elif ratio < 0.8:
            html += f'<div class="alert-box info">ℹ️ Moderate redundancy. Can reduce to {pca_result["n_components_95"]} features while retaining 95% information.</div>'
        else:
            html += f'<div class="alert-box warning">⚠️ Low redundancy — most features carry unique information. Dimensionality reduction may lose important signal.</div>'
        html += '</div>'

    # ── Hopkins Statistic (Cluster Tendency) ──
    hopkins = hopkins_statistic(df, numeric_cols)
    html += '<div class="card">'
    html += '<h4 class="card-title">🔍 Cluster Tendency (Hopkins Statistic)</h4>'
    html += f'<div class="hopkins-result">'
    html += f'<div class="hopkins-value" style="font-size:2rem;font-weight:700;color:#3b82f6">{format_number(hopkins, 4)}</div>'

    if hopkins > 0.75:
        html += '<div class="hopkins-interpretation badge-pass">Strong clustering tendency — data has natural groups!</div>'
    elif hopkins > 0.5:
        html += '<div class="hopkins-interpretation badge-info">Moderate clustering tendency — some structure exists.</div>'
    else:
        html += '<div class="hopkins-interpretation badge-warning">Low clustering tendency — data appears random/uniform.</div>'

    html += '<p class="card-desc" style="margin-top:0.5rem">Hopkins statistic: 0.5 = random, →1.0 = highly clustered, →0.0 = uniform.</p>'
    html += '</div></div>'

    # ── K-Means Preview ──
    kmeans_result = kmeans_preview(df, numeric_cols)
    if "error" not in kmeans_result:
        html += '<div class="card">'
        html += '<h4 class="card-title">🎯 K-Means Clustering Preview (k=3)</h4>'
        html += '<p class="card-desc">Auto-clustered on top-2 PCA components for visualization.</p>'
        html += cluster_scatter(kmeans_result, "kmeans_preview")
        html += '</div>'

    # ── Data Density Analysis ──
    html += '<div class="card">'
    html += '<h4 class="card-title">🔬 Data Density Analysis</h4>'
    html += '<p class="card-desc">Analyzing sparse vs dense regions in feature space.</p>'

    html += '<div class="table-responsive"><table class="styled-table compact">'
    html += '<thead><tr><th>Feature</th><th>Range</th><th>IQR</th><th>IQR/Range</th><th>Density</th></tr></thead>'
    html += '<tbody>'

    for col in numeric_cols[:15]:
        clean = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(clean) < 3:
            continue

        data_range = float(clean.max() - clean.min())
        iqr = float(clean.quantile(0.75) - clean.quantile(0.25))
        density_ratio = iqr / data_range if data_range > 0 else 0

        if density_ratio > 0.5:
            density = '<span class="badge-pass">Dense</span>'
        elif density_ratio > 0.25:
            density = '<span class="badge-info">Moderate</span>'
        else:
            density = '<span class="badge-warning">Sparse</span>'

        html += f'<tr><td class="col-name">{safe_str(col)}</td>'
        html += f'<td>{format_number(data_range)}</td>'
        html += f'<td>{format_number(iqr)}</td>'
        html += f'<td>{format_number(density_ratio, 3)}</td>'
        html += f'<td>{density}</td></tr>'

    html += '</tbody></table></div></div>'

    return html


def analyze(df):
    """Standalone China technique analysis."""
    from ..type_detector import detect_types
    type_map = detect_types(df)
    numeric_cols = [c for c, t in type_map.items() if t == ColumnType.NUMERIC]

    results = {}
    if len(numeric_cols) >= 2:
        results["pca"] = pca_analysis(df, numeric_cols)
        results["hopkins"] = hopkins_statistic(df, numeric_cols)
        results["kmeans"] = kmeans_preview(df, numeric_cols)

    return results
