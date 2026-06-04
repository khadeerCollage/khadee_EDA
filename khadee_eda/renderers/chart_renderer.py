"""
Khadee EDA — Chart Renderer
=============================
Plotly-based chart generation. Each function returns an HTML string with an embedded Plotly chart.
"""

import json
import numpy as np
import pandas as pd

from ..config import COLORS, PLOTLY_COLORSCALE, CHART_HEIGHT, HISTOGRAM_BINS


def _plotly_div(fig_json, chart_id, height=None):
    """Wrap a Plotly figure JSON into an HTML div with inline rendering."""
    h = height or CHART_HEIGHT
    return f"""
    <div id="{chart_id}" class="plotly-chart" style="width:100%;height:{h}px;"></div>
    <script>
    (function() {{
        var fig = {fig_json};
        fig.layout = fig.layout || {{}};
        fig.layout.paper_bgcolor = 'rgba(0,0,0,0)';
        fig.layout.plot_bgcolor = 'rgba(0,0,0,0)';
        fig.layout.font = {{color: '#e2e8f0', family: 'Inter, sans-serif', size: 12}};
        fig.layout.margin = fig.layout.margin || {{l:50,r:30,t:40,b:40}};
        fig.layout.xaxis = Object.assign(fig.layout.xaxis || {{}}, {{gridcolor:'rgba(99,102,241,0.1)', zerolinecolor:'rgba(99,102,241,0.2)'}});
        fig.layout.yaxis = Object.assign(fig.layout.yaxis || {{}}, {{gridcolor:'rgba(99,102,241,0.1)', zerolinecolor:'rgba(99,102,241,0.2)'}});
        fig.layout.autosize = true;
        fig.layout.height = {h};
        Plotly.newPlot('{chart_id}', fig.data, fig.layout, {{responsive:true, displayModeBar:false}});
    }})();
    </script>
    """


def histogram_chart(series, chart_id, title="", bins=None):
    """Generate a histogram with KDE overlay."""
    if bins is None:
        bins = min(HISTOGRAM_BINS, max(10, len(series) // 20))

    values = series.values.tolist()

    fig = {
        "data": [{
            "type": "histogram",
            "x": values,
            "nbinsx": bins,
            "marker": {"color": COLORS["primary"], "opacity": 0.7,
                       "line": {"color": COLORS["primary"], "width": 0.5}},
            "name": "Distribution",
        }],
        "layout": {
            "title": {"text": title, "font": {"size": 14}},
            "xaxis": {"title": ""},
            "yaxis": {"title": "Count"},
            "bargap": 0.05,
        }
    }
    return _plotly_div(json.dumps(fig, default=str), chart_id)


def histogram_grid(series_list, col_names, chart_id):
    """Generate a grid of small histograms."""
    n = len(series_list)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig = {"data": [], "layout": {
        "title": {"text": "", "font": {"size": 14}},
        "height": max(300, rows * 200),
        "showlegend": False,
    }}

    # Use subplots via domain
    for i, (series, name) in enumerate(zip(series_list, col_names)):
        row = i // cols
        col = i % cols
        x_domain = [col / cols + 0.02, (col + 1) / cols - 0.02]
        y_domain = [1 - (row + 1) / rows + 0.05, 1 - row / rows - 0.05]

        axis_num = "" if i == 0 else str(i + 1)

        fig["data"].append({
            "type": "histogram",
            "x": series.values.tolist()[:5000],
            "nbinsx": 20,
            "marker": {"color": [COLORS["primary"], COLORS["secondary"], COLORS["accent"],
                                 COLORS["success"], COLORS["info"]][i % 5], "opacity": 0.7},
            "xaxis": f"x{axis_num}",
            "yaxis": f"y{axis_num}",
            "name": name[:15],
        })

        fig["layout"][f"xaxis{axis_num}"] = {
            "domain": x_domain, "title": name[:15], "titlefont": {"size": 10},
            "gridcolor": "rgba(99,102,241,0.1)",
        }
        fig["layout"][f"yaxis{axis_num}"] = {
            "domain": y_domain, "title": "",
            "gridcolor": "rgba(99,102,241,0.1)",
        }

    return _plotly_div(json.dumps(fig, default=str), chart_id, height=max(350, rows * 200))


def box_plot(series, chart_id, title=""):
    """Generate a box plot."""
    fig = {
        "data": [{
            "type": "box",
            "y": series.values.tolist()[:10000],
            "marker": {"color": COLORS["accent"]},
            "line": {"color": COLORS["accent"]},
            "fillcolor": f"{COLORS['accent']}30",
            "boxpoints": "outliers",
            "name": "",
        }],
        "layout": {
            "title": {"text": title, "font": {"size": 14}},
            "yaxis": {"title": ""},
        }
    }
    return _plotly_div(json.dumps(fig, default=str), chart_id)


def box_plot_multi(series_list, col_names, chart_id):
    """Generate multiple box plots side by side."""
    colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"],
              COLORS["success"], COLORS["warning"], COLORS["info"],
              "#ec4899", "#14b8a6", "#a855f7", "#f97316", "#06b6d4", "#84cc16"]

    fig = {"data": [], "layout": {
        "title": {"text": "", "font": {"size": 14}},
        "showlegend": False,
        "height": 400,
    }}

    for i, (series, name) in enumerate(zip(series_list, col_names)):
        fig["data"].append({
            "type": "box",
            "y": series.values.tolist()[:5000],
            "name": name[:15],
            "marker": {"color": colors[i % len(colors)]},
            "boxpoints": "outliers",
        })

    return _plotly_div(json.dumps(fig, default=str), chart_id, height=400)


def bar_chart(labels, values, chart_id, title="", color=None, horizontal=False):
    """Generate a bar chart."""
    if color is None:
        color = COLORS["primary"]

    orientation = "h" if horizontal else "v"
    x_data = values if horizontal else labels
    y_data = labels if horizontal else values

    fig = {
        "data": [{
            "type": "bar",
            "x": x_data,
            "y": y_data,
            "orientation": orientation,
            "marker": {"color": color, "opacity": 0.8,
                       "line": {"color": color, "width": 0.5}},
        }],
        "layout": {
            "title": {"text": title, "font": {"size": 14}},
            "xaxis": {"tickangle": -45 if not horizontal and len(labels) > 5 else 0},
        }
    }
    return _plotly_div(json.dumps(fig, default=str), chart_id)


def pie_chart(labels, values, colors, chart_id, title=""):
    """Generate a pie/donut chart."""
    fig = {
        "data": [{
            "type": "pie",
            "labels": labels,
            "values": values,
            "marker": {"colors": colors},
            "hole": 0.0,
            "textinfo": "label+percent",
            "textfont": {"color": "#e2e8f0"},
        }],
        "layout": {
            "title": {"text": title, "font": {"size": 14}},
            "showlegend": True,
            "legend": {"font": {"color": "#e2e8f0"}},
        }
    }
    return _plotly_div(json.dumps(fig, default=str), chart_id, height=300)


def donut_chart(labels, values, colors, chart_id, title=""):
    """Generate a donut chart."""
    fig = {
        "data": [{
            "type": "pie",
            "labels": labels,
            "values": values,
            "marker": {"colors": colors},
            "hole": 0.45,
            "textinfo": "label+percent",
            "textfont": {"color": "#e2e8f0"},
        }],
        "layout": {
            "title": {"text": title, "font": {"size": 14}},
            "showlegend": True,
            "legend": {"font": {"color": "#e2e8f0"}},
            "height": 350,
        }
    }
    return _plotly_div(json.dumps(fig, default=str), chart_id, height=350)


def heatmap_chart(corr_df, chart_id, title=""):
    """Generate a correlation heatmap."""
    if corr_df.empty:
        return '<p class="empty-message">No data for heatmap.</p>'

    z = corr_df.values.tolist()
    x_labels = [str(c)[:15] for c in corr_df.columns]
    y_labels = [str(c)[:15] for c in corr_df.index]

    # Create annotation text
    annotations = []
    for i, row in enumerate(z):
        for j, val in enumerate(row):
            if not np.isnan(val):
                annotations.append({
                    "x": x_labels[j], "y": y_labels[i],
                    "text": f"{val:.2f}", "showarrow": False,
                    "font": {"color": "#fff" if abs(val) > 0.5 else "#94a3b8", "size": 9},
                })

    fig = {
        "data": [{
            "type": "heatmap",
            "z": z,
            "x": x_labels,
            "y": y_labels,
            "colorscale": [
                [0, "#3b82f6"], [0.25, "#06b6d4"],
                [0.5, "#1e1e3e"], [0.75, "#f59e0b"], [1, "#ef4444"]
            ],
            "zmin": -1, "zmax": 1,
            "colorbar": {"tickfont": {"color": "#e2e8f0"}},
        }],
        "layout": {
            "title": {"text": title, "font": {"size": 14}},
            "annotations": annotations if len(x_labels) <= 15 else [],
            "xaxis": {"tickangle": -45},
            "height": max(400, len(y_labels) * 25 + 100),
        }
    }
    return _plotly_div(json.dumps(fig, default=_json_default), chart_id,
                       height=max(400, len(y_labels) * 25 + 100))


def scatter_plot(x_series, y_series, x_name, y_name, chart_id, title=""):
    """Generate a scatter plot."""
    fig = {
        "data": [{
            "type": "scatter",
            "x": x_series.values.tolist(),
            "y": y_series.values.tolist(),
            "mode": "markers",
            "marker": {"color": COLORS["primary"], "opacity": 0.5, "size": 4},
        }],
        "layout": {
            "title": {"text": title, "font": {"size": 12}},
            "xaxis": {"title": x_name[:20]},
            "yaxis": {"title": y_name[:20]},
            "height": 300,
        }
    }
    return _plotly_div(json.dumps(fig, default=str), chart_id, height=300)


def grouped_box_plot(df, num_col, cat_col, chart_id, title=""):
    """Generate grouped box plots: numeric variable grouped by categorical."""
    colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"],
              COLORS["success"], COLORS["warning"], COLORS["danger"],
              COLORS["info"], "#ec4899", "#14b8a6", "#a855f7"]

    groups = df[cat_col].dropna().unique()[:10]
    fig = {"data": [], "layout": {
        "title": {"text": title, "font": {"size": 14}},
        "showlegend": True,
        "height": 350,
    }}

    for i, group in enumerate(groups):
        mask = df[cat_col] == group
        vals = pd.to_numeric(df.loc[mask, num_col], errors="coerce").dropna()
        fig["data"].append({
            "type": "box",
            "y": vals.values.tolist()[:2000],
            "name": str(group)[:15],
            "marker": {"color": colors[i % len(colors)]},
        })

    return _plotly_div(json.dumps(fig, default=str), chart_id, height=350)


def missing_matrix_chart(matrix_df, chart_id):
    """Generate a nullity matrix visualization."""
    z = matrix_df.values.tolist()
    x_labels = [str(c)[:15] for c in matrix_df.columns]

    fig = {
        "data": [{
            "type": "heatmap",
            "z": z,
            "x": x_labels,
            "colorscale": [[0, "#ef4444"], [1, "#10b981"]],
            "showscale": False,
        }],
        "layout": {
            "title": {"text": "", "font": {"size": 14}},
            "xaxis": {"tickangle": -45},
            "yaxis": {"title": "Row Index", "autorange": "reversed"},
            "height": min(400, max(200, len(z))),
        }
    }
    return _plotly_div(json.dumps(fig, default=str), chart_id,
                       height=min(400, max(200, len(z))))


def control_chart(series, mean, ucl, lcl, chart_id, title=""):
    """Generate a Shewhart control chart."""
    values = series.values.tolist()[:1000]
    x = list(range(len(values)))

    fig = {
        "data": [
            {
                "type": "scatter", "x": x, "y": values,
                "mode": "markers+lines",
                "marker": {"color": COLORS["primary"], "size": 3},
                "line": {"color": COLORS["primary"], "width": 1},
                "name": "Data",
            },
            {
                "type": "scatter", "x": [0, len(values)], "y": [mean, mean],
                "mode": "lines", "line": {"color": COLORS["success"], "dash": "dash", "width": 2},
                "name": f"Mean ({mean:.2f})",
            },
            {
                "type": "scatter", "x": [0, len(values)], "y": [ucl, ucl],
                "mode": "lines", "line": {"color": COLORS["danger"], "dash": "dot", "width": 2},
                "name": f"UCL ({ucl:.2f})",
            },
            {
                "type": "scatter", "x": [0, len(values)], "y": [lcl, lcl],
                "mode": "lines", "line": {"color": COLORS["danger"], "dash": "dot", "width": 2},
                "name": f"LCL ({lcl:.2f})",
            },
        ],
        "layout": {
            "title": {"text": title, "font": {"size": 12}},
            "xaxis": {"title": "Observation Index"},
            "yaxis": {"title": "Value"},
            "height": 300,
            "showlegend": True,
            "legend": {"font": {"size": 10}},
        }
    }
    return _plotly_div(json.dumps(fig, default=str), chart_id, height=300)


def pca_variance_chart(variance_ratio, cumulative, chart_id):
    """Generate PCA explained variance chart."""
    n = len(variance_ratio)
    x = list(range(1, n + 1))

    fig = {
        "data": [
            {
                "type": "bar", "x": x, "y": [v * 100 for v in variance_ratio],
                "marker": {"color": COLORS["primary"], "opacity": 0.7},
                "name": "Individual",
            },
            {
                "type": "scatter", "x": x, "y": [v * 100 for v in cumulative],
                "mode": "lines+markers",
                "line": {"color": COLORS["danger"], "width": 2},
                "marker": {"size": 6},
                "name": "Cumulative",
                "yaxis": "y2",
            },
        ],
        "layout": {
            "title": {"text": "PCA — Variance Explained", "font": {"size": 14}},
            "xaxis": {"title": "Principal Component"},
            "yaxis": {"title": "Individual %"},
            "yaxis2": {
                "title": "Cumulative %", "overlaying": "y", "side": "right",
                "range": [0, 105],
                "gridcolor": "rgba(99,102,241,0.05)",
            },
            "height": 350,
            "showlegend": True,
        }
    }
    return _plotly_div(json.dumps(fig, default=str), chart_id, height=350)


def cluster_scatter(kmeans_result, chart_id):
    """Generate K-Means cluster scatter plot on PCA 2D."""
    pca_data = np.array(kmeans_result["pca_data"])
    labels = np.array(kmeans_result["labels"])
    colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"],
              COLORS["success"], COLORS["warning"], COLORS["danger"]]

    fig = {"data": [], "layout": {
        "title": {"text": f"K-Means Clustering (k={kmeans_result['n_clusters']})", "font": {"size": 14}},
        "xaxis": {"title": "PC1"},
        "yaxis": {"title": "PC2"},
        "height": 400,
        "showlegend": True,
    }}

    for cluster_id in range(kmeans_result["n_clusters"]):
        mask = labels == cluster_id
        if mask.sum() > 0:
            cluster_data = pca_data[mask]
            fig["data"].append({
                "type": "scatter",
                "x": cluster_data[:, 0].tolist(),
                "y": cluster_data[:, 1].tolist(),
                "mode": "markers",
                "marker": {"color": colors[cluster_id % len(colors)], "size": 4, "opacity": 0.6},
                "name": f"Cluster {cluster_id}",
            })

    return _plotly_div(json.dumps(fig, default=str), chart_id, height=400)


def qq_plot(series, chart_id, title=""):
    """Generate a QQ plot."""
    from scipy import stats as scipy_stats
    clean = series.dropna().values
    if len(clean) < 3:
        return '<p class="empty-message">Not enough data for QQ plot.</p>'

    (osm, osr), (slope, intercept, r) = scipy_stats.probplot(clean, dist="norm")

    fig = {
        "data": [
            {
                "type": "scatter", "x": osm.tolist(), "y": osr.tolist(),
                "mode": "markers",
                "marker": {"color": COLORS["primary"], "size": 4, "opacity": 0.5},
                "name": "Data",
            },
            {
                "type": "scatter",
                "x": [min(osm), max(osm)],
                "y": [slope * min(osm) + intercept, slope * max(osm) + intercept],
                "mode": "lines",
                "line": {"color": COLORS["danger"], "dash": "dash"},
                "name": "Normal Line",
            },
        ],
        "layout": {
            "title": {"text": title, "font": {"size": 12}},
            "xaxis": {"title": "Theoretical Quantiles"},
            "yaxis": {"title": "Sample Quantiles"},
            "height": 300,
        }
    }
    return _plotly_div(json.dumps(fig, default=str), chart_id, height=300)


def violin_plot(series, chart_id, title=""):
    """Generate a violin plot."""
    fig = {
        "data": [{
            "type": "violin",
            "y": series.values.tolist()[:5000],
            "box": {"visible": True},
            "meanline": {"visible": True},
            "fillcolor": f"{COLORS['secondary']}60",
            "line": {"color": COLORS["secondary"]},
            "name": "",
        }],
        "layout": {
            "title": {"text": title, "font": {"size": 14}},
            "height": 300,
        }
    }
    return _plotly_div(json.dumps(fig, default=str), chart_id, height=300)


def kde_plot(series, chart_id, title=""):
    """Generate a Kernel Density Estimate (KDE) plot."""
    from scipy import stats as scipy_stats
    clean = series.dropna()
    if len(clean) < 3:
        return '<p class="empty-message">Not enough data for KDE plot.</p>'

    try:
        values = clean.values.astype(float)
        kde = scipy_stats.gaussian_kde(values)
        x_min, x_max = values.min(), values.max()
        x_grid = np.linspace(x_min, x_max, 200)
        y_grid = kde(x_grid)

        fig = {
            "data": [{
                "type": "scatter",
                "x": x_grid.tolist(),
                "y": y_grid.tolist(),
                "mode": "lines",
                "line": {"color": COLORS["primary"], "width": 2.5},
                "fill": "tozeroy",
                "fillcolor": f"{COLORS['primary']}20",
                "name": "KDE",
            }],
            "layout": {
                "title": {"text": title, "font": {"size": 14}},
                "xaxis": {"title": ""},
                "yaxis": {"title": "Density"},
                "height": 300,
            }
        }
        return _plotly_div(json.dumps(fig, default=str), chart_id, height=300)
    except Exception as e:
        return f'<p class="error-message">Could not generate KDE: {e}</p>'


def _json_default(obj):
    """Handle non-serializable types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if pd.isna(obj):
        return None
    return str(obj)
