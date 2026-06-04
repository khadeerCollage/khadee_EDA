"""
Khadee EDA Configuration
========================
Default settings, thresholds, and constants for the profiling engine.
"""


# ── Section Registry ──────────────────────────────────────────────────────────
ALL_SECTIONS = [
    "overview",
    "variables",
    "distributions",
    "correlations",
    "missing",
    "outliers",
    "interactions",
    "advanced_stats",
    "model_readiness",
    "sample",
]

SECTION_LABELS = {
    "overview": "🏠 Dataset Overview",
    "variables": "📊 Variables",
    "distributions": "📈 Distributions",
    "correlations": "🔗 Correlations",
    "missing": "❓ Missing Values",
    "outliers": "🎯 Outlier Detection",
    "interactions": "🔄 Interactions",
    "advanced_stats": "📐 Advanced Statistics",
    "model_readiness": "🤖 Model Readiness",
    "sample": "📋 Sample & Duplicates",
}

# ── Technique Registry ────────────────────────────────────────────────────────
ALL_TECHNIQUES = ["us", "india", "japan", "china"]


# ── Type Detection Thresholds ─────────────────────────────────────────────────
CATEGORICAL_THRESHOLD = 20          # Max unique values to consider categorical
CATEGORICAL_RATIO_THRESHOLD = 0.05  # If unique/total < 5%, likely categorical
HIGH_CARDINALITY_THRESHOLD = 50     # Warn if categorical has > 50 unique values
BOOLEAN_VALUES = {True, False, 0, 1, "true", "false", "yes", "no", "True", "False", "Yes", "No", "TRUE", "FALSE", "YES", "NO", "0", "1"}

# ── Missing Values Thresholds ─────────────────────────────────────────────────
HIGH_MISSING_THRESHOLD = 0.50   # Warn if > 50% missing
MEDIUM_MISSING_THRESHOLD = 0.10 # Note if > 10% missing

# ── Correlation Thresholds ────────────────────────────────────────────────────
HIGH_CORRELATION_THRESHOLD = 0.80  # Warn if |r| > 0.80
CORRELATION_METHODS = ["pearson", "spearman", "kendall", "cramers_v"]

# ── Outlier Thresholds ────────────────────────────────────────────────────────
IQR_MULTIPLIER = 1.5
ZSCORE_THRESHOLD = 3.0
MODIFIED_ZSCORE_THRESHOLD = 3.5

# ── Performance Thresholds ────────────────────────────────────────────────────
LARGE_DATASET_ROWS = 500_000  # Auto-sample above this
SAMPLE_SIZE = 100_000         # Sample size for heavy computations
MAX_INTERACTION_PAIRS = 10    # Max pairs for interaction plots

# ── Chart Defaults ────────────────────────────────────────────────────────────
HISTOGRAM_BINS = 50
MAX_CATEGORIES_DISPLAY = 15
CHART_HEIGHT = 350
CHART_WIDTH = None  # Auto

# ── Color Palette ─────────────────────────────────────────────────────────────
COLORS = {
    "primary": "#3b82f6",      # Electric blue
    "secondary": "#8b5cf6",    # Violet
    "accent": "#06b6d4",       # Cyan
    "success": "#10b981",      # Emerald
    "warning": "#f59e0b",      # Amber
    "danger": "#ef4444",       # Red
    "info": "#6366f1",         # Indigo
    "bg_dark": "#0a0a1a",      # Deep dark background
    "bg_card": "rgba(30, 30, 60, 0.6)",  # Card background
    "text_primary": "#e2e8f0",  # Light text
    "text_secondary": "#94a3b8", # Muted text
    "border": "rgba(99, 102, 241, 0.2)",  # Subtle border
}

PLOTLY_COLORSCALE = [
    [0.0, "#3b82f6"],
    [0.25, "#06b6d4"],
    [0.5, "#10b981"],
    [0.75, "#f59e0b"],
    [1.0, "#ef4444"],
]

PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "#e2e8f0", "family": "Inter, sans-serif"},
        "xaxis": {
            "gridcolor": "rgba(99, 102, 241, 0.1)",
            "zerolinecolor": "rgba(99, 102, 241, 0.2)",
        },
        "yaxis": {
            "gridcolor": "rgba(99, 102, 241, 0.1)",
            "zerolinecolor": "rgba(99, 102, 241, 0.2)",
        },
        "margin": {"l": 50, "r": 30, "t": 40, "b": 40},
    }
}

# ── Model Readiness Weights ──────────────────────────────────────────────────
READINESS_CHECKS = {
    "no_high_missing": {"weight": 15, "label": "No high-missing columns (>50%)"},
    "no_constant_features": {"weight": 10, "label": "No constant/zero-variance features"},
    "no_excessive_duplicates": {"weight": 10, "label": "No excessive duplicate rows (>10%)"},
    "no_multicollinearity": {"weight": 15, "label": "No extreme multicollinearity (|r|>0.95)"},
    "sufficient_rows": {"weight": 10, "label": "Sufficient sample size (>100 rows)"},
    "balanced_types": {"weight": 10, "label": "Balanced feature types"},
    "low_outlier_ratio": {"weight": 10, "label": "Low outlier ratio (<5% per column)"},
    "no_id_columns": {"weight": 10, "label": "No suspected ID columns"},
    "reasonable_cardinality": {"weight": 10, "label": "Reasonable cardinality in categoricals"},
}
