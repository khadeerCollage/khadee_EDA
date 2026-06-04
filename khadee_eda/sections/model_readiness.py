"""
Khadee EDA — Section 9: Model Readiness Assessment
====================================================
Pre-modeling checklist, readiness score, preprocessing and model suggestions.
"""

import numpy as np
import pandas as pd

from ..type_detector import ColumnType, get_columns_by_type
from ..config import READINESS_CHECKS
from ..utils import (
    format_number, format_percentage, section_header, safe_str,
    stat_card, check_item, progress_bar,
)


def generate(df, type_map):
    """Generate the Model Readiness section HTML."""
    html = section_header("section-model-readiness", " Model Readiness Assessment",
                          "Pre-modeling checklist and recommendations for AI/ML engineers")

    checks = _run_checks(df, type_map)

    # Calculate readiness score
    total_weight = sum(c["weight"] for c in READINESS_CHECKS.values())
    earned = sum(
        READINESS_CHECKS[k]["weight"]
        for k, v in checks.items()
        if v["passed"]
    )
    score = (earned / total_weight * 100) if total_weight > 0 else 0

    # Score color
    if score >= 80:
        score_color = "#10b981"
        score_label = "Excellent"
    elif score >= 60:
        score_color = "#f59e0b"
        score_label = "Good"
    elif score >= 40:
        score_color = "#f97316"
        score_label = "Fair"
    else:
        score_color = "#ef4444"
        score_label = "Needs Work"

    # ── Score Card ──
    html += '<div class="readiness-score-card">'
    html += f'<div class="score-circle" style="border-color: {score_color}">'
    html += f'<span class="score-value" style="color: {score_color}">{score:.0f}%</span>'
    html += f'<span class="score-label">{score_label}</span>'
    html += '</div>'
    html += progress_bar(score, 100, score_color, "Model Readiness Score")
    html += '</div>'

    # ── Checklist ──
    html += '<div class="card">'
    html += '<h3 class="card-title"> Readiness Checklist</h3>'
    html += '<div class="checklist">'
    for key, check_info in READINESS_CHECKS.items():
        result = checks.get(key, {"passed": False, "detail": "Not evaluated"})
        html += '<div class="check-row">'
        html += check_item(result["passed"], check_info["label"])
        if result.get("detail"):
            html += f'<span class="check-detail">{safe_str(result["detail"])}</span>'
        html += '</div>'
    html += '</div></div>'

    # ── Preprocessing Recommendations ──
    html += '<div class="card">'
    html += '<h3 class="card-title"> Recommended Preprocessing Pipeline</h3>'
    html += _preprocessing_recommendations(df, type_map)
    html += '</div>'

    # ── Model Suggestions ──
    html += '<div class="card">'
    html += '<h3 class="card-title"> Model Suggestions</h3>'
    html += _model_suggestions(df, type_map)
    html += '</div>'

    return html


def _run_checks(df, type_map):
    """Run all readiness checks."""
    n_rows, n_cols = df.shape
    results = {}

    # No high missing columns
    high_miss_cols = [c for c in df.columns if df[c].isna().sum() / n_rows > 0.5] if n_rows > 0 else []
    results["no_high_missing"] = {
        "passed": len(high_miss_cols) == 0,
        "detail": f"{len(high_miss_cols)} columns with >50% missing" if high_miss_cols else "All columns have acceptable missing rates",
    }

    # No constant features
    constant_cols = get_columns_by_type(type_map, ColumnType.CONSTANT)
    results["no_constant_features"] = {
        "passed": len(constant_cols) == 0,
        "detail": f"{len(constant_cols)} constant columns found" if constant_cols else "No constant columns",
    }

    # No excessive duplicates
    n_dup = df.duplicated().sum()
    dup_pct = n_dup / n_rows if n_rows > 0 else 0
    results["no_excessive_duplicates"] = {
        "passed": dup_pct <= 0.10,
        "detail": f"{format_number(n_dup)} duplicates ({format_percentage(dup_pct)})",
    }

    # No extreme multicollinearity
    numeric_cols = get_columns_by_type(type_map, ColumnType.NUMERIC)
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().abs()
        np.fill_diagonal(corr.values, 0)
        max_corr = corr.max().max()
        results["no_multicollinearity"] = {
            "passed": max_corr <= 0.95,
            "detail": f"Max correlation: {format_number(max_corr, 3)}",
        }
    else:
        results["no_multicollinearity"] = {"passed": True, "detail": "Not enough numeric columns to check"}

    # Sufficient rows
    results["sufficient_rows"] = {
        "passed": n_rows >= 100,
        "detail": f"{format_number(n_rows)} rows available",
    }

    # Balanced types
    n_num = len(get_columns_by_type(type_map, ColumnType.NUMERIC))
    n_cat = len(get_columns_by_type(type_map, ColumnType.CATEGORICAL))
    results["balanced_types"] = {
        "passed": n_num > 0 or n_cat > 0,
        "detail": f"{n_num} numeric, {n_cat} categorical features",
    }

    # Low outlier ratio
    if numeric_cols:
        high_outlier_cols = 0
        for col in numeric_cols:
            clean = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(clean) > 0:
                q1, q3 = clean.quantile([0.25, 0.75])
                iqr = q3 - q1
                outlier_pct = ((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).mean()
                if outlier_pct > 0.05:
                    high_outlier_cols += 1
        results["low_outlier_ratio"] = {
            "passed": high_outlier_cols == 0,
            "detail": f"{high_outlier_cols} columns with >5% outliers" if high_outlier_cols > 0 else "All columns have acceptable outlier rates",
        }
    else:
        results["low_outlier_ratio"] = {"passed": True, "detail": "No numeric columns to check"}

    # No ID columns
    id_cols = get_columns_by_type(type_map, ColumnType.UNIQUE_ID)
    results["no_id_columns"] = {
        "passed": len(id_cols) == 0,
        "detail": f"{len(id_cols)} suspected ID columns" if id_cols else "No ID columns detected",
    }

    # Reasonable cardinality
    cat_cols = get_columns_by_type(type_map, ColumnType.CATEGORICAL)
    high_card = [c for c in cat_cols if df[c].nunique() > 50]
    results["reasonable_cardinality"] = {
        "passed": len(high_card) == 0,
        "detail": f"{len(high_card)} high-cardinality categoricals" if high_card else "All categoricals have reasonable cardinality",
    }

    return results


def _preprocessing_recommendations(df, type_map):
    """Generate preprocessing pipeline recommendations."""
    html = '<div class="pipeline-steps">'

    numeric_cols = get_columns_by_type(type_map, ColumnType.NUMERIC)
    categorical_cols = get_columns_by_type(type_map, ColumnType.CATEGORICAL)

    steps = []

    # Missing values
    if df.isna().sum().sum() > 0:
        code = (
            "# Using Khadee Clean (Recommended):\n"
            "import khadee_eda as ke\n"
            "df = ke.clean.clean_missing(df, strategy=\"median\")\n\n"
            "# Or using standard Pandas:\n"
            "# df = df.fillna(df.median(numeric_only=True))"
        )
        steps.append((
            "1️. Handle Missing Values",
            "Use median imputation for numeric, mode for categorical, or consider advanced methods like KNN imputation.",
            code
        ))

    # Remove ID columns
    id_cols = get_columns_by_type(type_map, ColumnType.UNIQUE_ID)
    if id_cols:
        col_list = ", ".join(f"'{c}'" for c in id_cols[:5])
        code = (
            f"# Drop high-cardinality identifier columns:\n"
            f"df = df.drop(columns=[{col_list}])"
        )
        steps.append((
            "2️. Remove ID Columns",
            f"Drop identifier columns: {', '.join(id_cols[:5])}",
            code
        ))

    # Remove constants
    const_cols = get_columns_by_type(type_map, ColumnType.CONSTANT)
    if const_cols:
        col_list = ", ".join(f"'{c}'" for c in const_cols[:5])
        code = (
            f"# Drop constant zero-variance features:\n"
            f"df = df.drop(columns=[{col_list}])"
        )
        steps.append((
            "3️. Remove Constant Features",
            f"Drop constant columns: {', '.join(const_cols[:5])}",
            code
        ))

    # Encoding
    if categorical_cols:
        col_list = ", ".join(f"'{c}'" for c in categorical_cols[:5])
        code = (
            f"# One-Hot encode categorical columns:\n"
            f"import pandas as pd\n"
            f"df = pd.get_dummies(df, columns=[{col_list}], drop_first=True)"
        )
        steps.append((
            "4️. Encode Categoricals",
            f"One-Hot encode low-cardinality, Label/Target encode high-cardinality ({len(categorical_cols)} columns)",
            code
        ))

    # Scaling
    if numeric_cols:
        col_list = ", ".join(f"'{c}'" for c in numeric_cols[:5])
        code = (
            f"# Scale numeric features:\n"
            f"from sklearn.preprocessing import StandardScaler\n"
            f"scaler = StandardScaler()\n"
            f"df[[{col_list}]] = scaler.fit_transform(df[[{col_list}]])"
        )
        steps.append((
            "5️. Scale Numeric Features",
            f"StandardScaler or RobustScaler for {len(numeric_cols)} numeric columns",
            code
        ))

    # Skewness transform
    skewed = []
    for col in numeric_cols:
        clean = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(clean) > 3 and abs(float(clean.skew())) > 1.0:
            skewed.append(col)
    if skewed:
        col_list = ", ".join(f"'{c}'" for c in skewed[:5])
        code = (
            f"# Transform skewed columns to stabilize variance:\n"
            f"import numpy as np\n"
            f"for col in [{col_list}]:\n"
            f"    df[col] = np.log1p(df[col])"
        )
        steps.append((
            "6️. Transform Skewed Features",
            f"Apply log/sqrt transform to {len(skewed)} skewed columns",
            code
        ))

    if not steps:
        code = (
            "# Data is clean and ready for machine learning!\n"
            "# No preprocessing required."
        )
        steps.append((
            " Data Looks Clean",
            "No major preprocessing needed. Proceed to feature engineering.",
            code
        ))

    for title, desc, code_snippet in steps:
        html += f'<div class="pipeline-step">'
        html += f'<strong>{title}</strong>'
        html += f'<p>{safe_str(desc)}</p>'
        html += f'<div class="pipeline-code-block">'
        html += f'<div class="code-header"><span>Python Pipeline Code</span>'
        html += f'<button class="code-copy-btn" onclick="navigator.clipboard.writeText(this.parentElement.nextElementSibling.innerText); this.innerText=\'Copied!\'; setTimeout(() => this.innerText=\'Copy\', 2000)">Copy</button>'
        html += f'</div>'
        html += f'<pre><code>{safe_str(code_snippet)}</code></pre>'
        html += f'</div></div>'

    html += '</div>'
    return html


def _model_suggestions(df, type_map):
    """Suggest model types based on data characteristics."""
    n_rows, n_cols = df.shape
    numeric_cols = get_columns_by_type(type_map, ColumnType.NUMERIC)
    categorical_cols = get_columns_by_type(type_map, ColumnType.CATEGORICAL)

    html = '<div class="model-suggestions">'

    suggestions = []

    # Based on data size and type mix
    if n_rows < 1000:
        suggestions.append((" Small Dataset", "Consider simpler models: Logistic Regression, SVM, Decision Trees. Avoid deep learning."))
    elif n_rows < 50000:
        suggestions.append((" Medium Dataset", "Good for: Random Forest, XGBoost, LightGBM, SVM."))
    else:
        suggestions.append((" Large Dataset", "Can handle: Deep Learning, XGBoost, LightGBM, CatBoost."))

    if len(categorical_cols) > len(numeric_cols):
        suggestions.append((" Mostly Categorical", "Consider: CatBoost (native categorical support), LightGBM, or Neural Networks with embeddings."))
    elif len(numeric_cols) > 0 and len(categorical_cols) == 0:
        suggestions.append((" All Numeric", "Good for: Linear models, SVM, Neural Networks. Consider feature engineering."))
    else:
        suggestions.append((" Mixed Types", "Best for: Tree-based models (XGBoost, LightGBM, Random Forest) that handle mixed types well."))

    if n_cols > 50:
        suggestions.append((" High Dimensionality", "Consider: PCA for dimensionality reduction, feature selection (mutual information, L1 regularization)."))

    for title, desc in suggestions:
        html += f'<div class="suggestion-card"><strong>{title}</strong><p>{safe_str(desc)}</p></div>'

    html += '</div>'
    return html
