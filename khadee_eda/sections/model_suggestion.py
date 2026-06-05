"""
Khadee EDA — Section: Model Suggestion
========================================
Analyzes dataset characteristics to suggest the most appropriate ML models.
"""

from ..utils import section_header, format_number
from ..type_detector import ColumnType

def generate(df, type_map):
    """Generate the Model Suggestion section HTML."""
    html = section_header("section-model-suggestion", " Model Suggestion",
                          "Algorithmic recommendations based on dataset heuristics")

    n_rows, n_cols = df.shape
    
    # Calculate dataset characteristics
    missing_ratio = df.isna().sum().sum() / (n_rows * n_cols)
    has_missing = missing_ratio > 0.01

    num_cols = sum(1 for t in type_map.values() if t == ColumnType.NUMERIC)
    cat_cols = sum(1 for t in type_map.values() if t == ColumnType.CATEGORICAL)
    text_cols = sum(1 for t in type_map.values() if t == ColumnType.TEXT)
    
    has_text = text_cols > 0
    high_cardinality = cat_cols > (n_cols * 0.5)

    # ── Decision Tree ──
    recommendations = []

    # 1. LightGBM / XGBoost (The standard robust tabular choices)
    if n_rows > 10000:
        reasons = [
            f"Scales exceptionally well to large datasets ({format_number(n_rows)} rows).",
            "Captures complex non-linear relationships without manual feature engineering."
        ]
        if has_missing:
            reasons.append("Native handling of missing values (no imputation pipeline needed).")
            
        recommendations.append({
            "model": "LightGBM / XGBoost",
            "badge": "badge-pass",
            "type": "Gradient Boosting",
            "reasons": reasons
        })
    
    # 2. CatBoost (If lots of categorical features)
    if cat_cols >= 2 and n_rows > 1000:
        recommendations.append({
            "model": "CatBoost",
            "badge": "badge-info",
            "type": "Gradient Boosting",
            "reasons": [
                f"Excellent native handling of categorical features ({cat_cols} detected).",
                "Requires minimal preprocessing and reduces the need for heavy One-Hot Encoding.",
                "Highly resistant to overfitting on default parameters compared to other boosting libraries."
            ]
        })

    # 3. Random Forest (Small/Medium data, robust baseline)
    if n_rows <= 15000 and not has_text:
        recommendations.append({
            "model": "Random Forest",
            "badge": "badge-warning",
            "type": "Ensemble Tree",
            "reasons": [
                f"Dataset is relatively small ({format_number(n_rows)} rows), reducing the risk of deep boosting algorithms overfitting.",
                "Extremely robust to outliers and requires almost no hyperparameter tuning.",
                "Provides out-of-the-box feature importance for immediate interpretability."
            ]
        })

    # 4. Linear / Ridge / Logistic Regression (If simple/small)
    if n_cols > n_rows or (num_cols > cat_cols and n_rows < 5000):
        recommendations.append({
            "model": "Ridge / Lasso Regression",
            "badge": "badge-muted",
            "type": "Linear Model",
            "reasons": [
                "High interpretability and mathematically stable on small or extremely wide datasets.",
                "L1/L2 Regularization will prevent overfitting if numeric features are highly correlated."
            ]
        })
        
    # 5. NLP / Transformers (If text data is present)
    if has_text:
        recommendations.append({
            "model": "Transformers (BERT) + Tabular Model",
            "badge": "badge-danger",
            "type": "NLP + Tabular",
            "reasons": [
                f"Free-form text columns detected ({text_cols} columns).",
                "Use HuggingFace Transformers to extract embeddings from the text, and concatenate them with the tabular features.",
                "Pass the combined embeddings into a fast tabular model like LightGBM for the final prediction."
            ]
        })

    # Fallback if nothing specific triggered
    if not recommendations:
        recommendations.append({
            "model": "XGBoost",
            "badge": "badge-pass",
            "type": "Gradient Boosting",
            "reasons": [
                "Industry standard algorithm for almost all generic tabular datasets.",
                "Robust to outliers, non-linear patterns, and highly performant."
            ]
        })

    html += '<div class="card">'
    html += '<h3 class="card-title"> Algorithmic Recommendations</h3>'
    html += '<p class="card-desc">Based on the physical characteristics of this dataset, these are the recommended machine learning approaches.</p>'
    
    html += '<div class="recommendation-grid" style="display: grid; gap: 1.5rem; margin-top: 1rem;">'
    
    for rec in recommendations:
        html += f'<div class="suggestion-card" style="padding: 1.5rem; background: rgba(30, 30, 60, 0.4); border-radius: 8px; border-left: 4px solid var(--accent);">'
        html += f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">'
        html += f'<h4 style="margin: 0; font-size: 1.2rem; color: #fff;">{rec["model"]}</h4>'
        html += f'<span class="{rec["badge"]}">{rec["type"]}</span>'
        html += '</div>'
        html += '<ul style="margin: 0; padding-left: 1.5rem; color: #cbd5e1; line-height: 1.6;">'
        for reason in rec["reasons"]:
            html += f'<li>{reason}</li>'
        html += '</ul>'
        html += '</div>'
        
    html += '</div></div>'
    
    return html
