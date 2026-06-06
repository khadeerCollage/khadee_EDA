"""
K-EDA — Section 11: Feature Engineering Suggestions
=========================================================
Automated recommendations and copy-pasteable Python code snippets for data transformations.
"""

import numpy as np
import pandas as pd
import warnings

from ..type_detector import ColumnType, get_columns_by_type
from ..utils import section_header, safe_str

def generate(df, type_map):
    """Generate the Feature Engineering section HTML."""
    html = section_header(
        "section-feature-engineering",
        "🛠️ Feature Engineering Suggestions",
        "Automated recommendations and production-ready Python recipes to transform your dataset for modeling."
    )

    # Smart sampling for large datasets to keep calculations fast
    if len(df) > 50000:
        df_sampled = df.sample(50000, random_state=42)
    else:
        df_sampled = df

    recommendations = []

    # 1. Math Transformations (Skewness)
    skew_recs = _detect_skewness(df_sampled, type_map)
    if skew_recs:
        recommendations.append(skew_recs)

    # 2. Outlier Treatments (Robust Scaling vs Standard Scaling)
    outlier_recs = _detect_outlier_scaling(df_sampled, type_map)
    if outlier_recs:
        recommendations.append(outlier_recs)

    # 3. Categorical Encodings
    encoding_recs = _detect_categorical_encoding(df_sampled, type_map)
    if encoding_recs:
        recommendations.append(encoding_recs)

    # 4. Datetime Feature Extraction
    datetime_recs = _detect_datetime_extraction(df_sampled, type_map)
    if datetime_recs:
        recommendations.append(datetime_recs)

    # 5. Text Preprocessing
    text_recs = _detect_text_preprocessing(df_sampled, type_map)
    if text_recs:
        recommendations.append(text_recs)

    # 6. Bivariate / Correlation Interaction
    interaction_recs = _detect_interactions(df_sampled, type_map)
    if interaction_recs:
        recommendations.append(interaction_recs)

    # Build the HTML output
    html += '<div class="pipeline-steps">'
    
    if not recommendations:
        # Fallback if no specific feature engineering suggestions are found
        code_snippet = (
            "# Your dataset appears to be already clean and well-structured!\n"
            "# No major feature engineering recommendations were triggered.\n"
            "# You can proceed to define your features (X) and target (y) for training:\n"
            "X = df.drop(columns=['target_column_name'])\n"
            "y = df['target_column_name']"
        )
        html += '<div class="pipeline-step">'
        html += '<strong>✨ No Specific Recommendations Triggered</strong>'
        html += '<p>The columns in your dataset are well-behaved. Consider standard model fitting directly.</p>'
        html += '<div class="pipeline-code-block">'
        html += '<div class="code-header"><span>Python Baseline Code</span>'
        html += '<button class="code-copy-btn" onclick="navigator.clipboard.writeText(this.parentElement.nextElementSibling.innerText); this.innerText=\'Copied!\'; setTimeout(() => this.innerText=\'Copy\', 2000)">Copy</button>'
        html += '</div>'
        html += f'<pre><code>{safe_str(code_snippet)}</code></pre>'
        html += '</div></div>'
    else:
        # Generate the master pipeline and insert it first
        master_title, master_desc, master_code = _generate_master_pipeline(recommendations)
        html += '<div class="pipeline-step" style="border-left-color: var(--success);">'
        html += f'<strong>{master_title}</strong>'
        html += f'<p>{master_desc}</p>'
        html += '<div class="pipeline-code-block">'
        html += '<div class="code-header"><span>Python Unified Master Pipeline</span>'
        html += '<button class="code-copy-btn" onclick="navigator.clipboard.writeText(this.parentElement.nextElementSibling.innerText); this.innerText=\'Copied!\'; setTimeout(() => this.innerText=\'Copy\', 2000)">Copy</button>'
        html += '</div>'
        html += f'<pre><code>{safe_str(master_code)}</code></pre>'
        html += '</div></div>'
        
        for title, desc, code_snippet in recommendations:
            html += '<div class="pipeline-step">'
            html += f'<strong>{title}</strong>'
            html += f'<p>{desc}</p>'
            html += '<div class="pipeline-code-block">'
            html += '<div class="code-header"><span>Python Pipeline Recipe</span>'
            html += '<button class="code-copy-btn" onclick="navigator.clipboard.writeText(this.parentElement.nextElementSibling.innerText); this.innerText=\'Copied!\'; setTimeout(() => this.innerText=\'Copy\', 2000)">Copy</button>'
            html += '</div>'
            html += f'<pre><code>{safe_str(code_snippet)}</code></pre>'
            html += '</div></div>'

    html += '</div>'
    return html

def _detect_skewness(df, type_map):
    numeric_cols = get_columns_by_type(type_map, ColumnType.NUMERIC)
    if not numeric_cols:
        return None

    highly_skewed_pos = []
    highly_skewed_any = []

    for col in numeric_cols:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals) > 10:
            try:
                skew = vals.skew()
                if abs(skew) > 1.0:
                    min_val = vals.min()
                    if min_val >= 0:
                        highly_skewed_pos.append(col)
                    else:
                        highly_skewed_any.append(col)
            except Exception:
                pass

    if not highly_skewed_pos and not highly_skewed_any:
        return None

    desc_parts = []
    code_lines = ["import numpy as np", "import pandas as pd"]
    
    if highly_skewed_pos:
        desc_parts.append(
            f"The following positive-valued columns have high skewness (|skew| > 1.0): "
            f"<strong>{', '.join(highly_skewed_pos[:5])}</strong>{'...' if len(highly_skewed_pos) > 5 else ''}. "
            f"Applying a log transformation will stabilize variance and make the distributions more normal."
        )
        code_lines.append("\n# Log Transform [np.log1p] for skewed positive features:")
        for col in highly_skewed_pos[:5]:
            code_lines.append(f"df['{col}'] = np.log1p(df['{col}'])")

    if highly_skewed_any:
        desc_parts.append(
            f"The following columns contain negative values and have high skewness: "
            f"<strong>{', '.join(highly_skewed_any[:5])}</strong>{'...' if len(highly_skewed_any) > 5 else ''}. "
            f"A PowerTransformer (Yeo-Johnson) is recommended."
        )
        code_lines.append("\n# PowerTransformer (Yeo-Johnson) for skewed features with negative values:")
        code_lines.append("from sklearn.preprocessing import PowerTransformer")
        code_lines.append("pt = PowerTransformer(method='yeo-johnson')")
        cols_to_pt = ", ".join(f"'{c}'" for c in highly_skewed_any[:5])
        code_lines.append(f"df[[{cols_to_pt}]] = pt.fit_transform(df[[{cols_to_pt}]])")

    return (
        "📈 Transform Skewed Numeric Features",
        " ".join(desc_parts),
        "\n".join(code_lines)
    )

def _detect_outlier_scaling(df, type_map):
    numeric_cols = get_columns_by_type(type_map, ColumnType.NUMERIC)
    if not numeric_cols:
        return None

    robust_cols = []
    standard_cols = []

    for col in numeric_cols:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals) > 10:
            try:
                q1, q3 = vals.quantile([0.25, 0.75])
                iqr = q3 - q1
                if iqr > 0:
                    outlier_mask = (vals < q1 - 1.5 * iqr) | (vals > q3 + 1.5 * iqr)
                    outlier_ratio = outlier_mask.mean()
                    if outlier_ratio > 0.05:
                        robust_cols.append(col)
                    else:
                        standard_cols.append(col)
                else:
                    standard_cols.append(col)
            except Exception:
                standard_cols.append(col)
        else:
            standard_cols.append(col)

    if not robust_cols:
        # No heavy outliers, suggest standard scaler
        code = (
            "from sklearn.preprocessing import StandardScaler\n\n"
            f"# Standardize numeric features (low outliers):\n"
            f"scaler = StandardScaler()\n"
            f"numeric_cols = {list(numeric_cols[:5])}\n"
            f"df[numeric_cols] = scaler.fit_transform(df[numeric_cols])"
        )
        return (
            "⚖️ Scale Numeric Features (Standard)",
            "Numeric features have low outlier ratios (<5%). StandardScaler is recommended to keep features on the same scale.",
            code
        )

    # We have columns with high outlier ratios!
    desc = (
        f"Columns like <strong>{', '.join(robust_cols[:5])}</strong>{'...' if len(robust_cols) > 5 else ''} "
        f"have high outlier ratios (>5%). Using <code>RobustScaler</code> is recommended for these columns, "
        f"as it uses the median and interquartile range (IQR) to scale, making it robust to extreme outlier values."
    )

    code_lines = [
        "from sklearn.preprocessing import RobustScaler, StandardScaler",
        "import pandas as pd",
        ""
    ]
    if robust_cols:
        cols_rob = ", ".join(f"'{c}'" for c in robust_cols[:5])
        code_lines.append(f"# Scale outlier-heavy columns with RobustScaler:")
        code_lines.append(f"robust_scaler = RobustScaler()")
        code_lines.append(f"df[[{cols_rob}]] = robust_scaler.fit_transform(df[[{cols_rob}]])\n")
    
    other_std_cols = [c for c in standard_cols if c not in robust_cols]
    if other_std_cols:
        cols_std = ", ".join(f"'{c}'" for c in other_std_cols[:5])
        code_lines.append(f"# Scale well-behaved columns with StandardScaler:")
        code_lines.append(f"std_scaler = StandardScaler()")
        code_lines.append(f"df[[{cols_std}]] = std_scaler.fit_transform(df[[{cols_std}]])")

    return (
        "⚖️ Scale Numeric Features (Robust & Standard)",
        desc,
        "\n".join(code_lines)
    )

def _detect_categorical_encoding(df, type_map):
    cat_cols = get_columns_by_type(type_map, ColumnType.CATEGORICAL)
    if not cat_cols:
        return None

    low_card_cols = []
    high_card_cols = []

    for col in cat_cols:
        n_uniq = df[col].nunique()
        if n_uniq <= 10:
            low_card_cols.append((col, n_uniq))
        else:
            high_card_cols.append((col, n_uniq))

    desc_parts = []
    code_lines = ["import pandas as pd", ""]

    if low_card_cols:
        col_names = [c[0] for c in low_card_cols]
        desc_parts.append(
            f"Low-cardinality categorical columns (<=10 categories): <strong>{', '.join(col_names[:5])}</strong>. "
            f"One-Hot encoding is recommended."
        )
        col_list_str = ", ".join(f"'{c}'" for c in col_names[:5])
        code_lines.append(f"# One-Hot Encoding for low-cardinality categories:")
        code_lines.append(f"df = pd.get_dummies(df, columns=[{col_list_str}], drop_first=True)\n")

    if high_card_cols:
        col_names = [c[0] for c in high_card_cols]
        desc_parts.append(
            f"High-cardinality categorical columns (>10 categories): "
            f"<strong>{', '.join([f'{c[0]} ({c[1]} categories)' for c in high_card_cols[:5]])}</strong>. "
            f"One-Hot encoding these would explode dimensionality. Consider **Target Encoding** or **Frequency (Count) Encoding**."
        )
        code_lines.append(f"# Frequency (Count) Encoding for high-cardinality categories:")
        for col in col_names[:3]:
            code_lines.append(f"freq_map = df['{col}'].value_counts() / len(df)")
            code_lines.append(f"df['{col}_freq'] = df['{col}'].map(freq_map)")
        
        code_lines.append("\n# Target Encoding (assuming target column exists):")
        code_lines.append("# For each category, compute average target value (with optional smoothing)")
        for col in col_names[:2]:
            code_lines.append(f"# df['{col}_encoded'] = df.groupby('{col}')['target_col'].transform('mean')")

    return (
        "🏷️ Encode Categorical Features",
        " ".join(desc_parts),
        "\n".join(code_lines)
    )

def _detect_datetime_extraction(df, type_map):
    dt_cols = get_columns_by_type(type_map, ColumnType.DATETIME)
    if not dt_cols:
        return None

    desc = (
        f"Detected datetime columns: <strong>{', '.join(dt_cols[:5])}</strong>. "
        f"Machine learning algorithms cannot process raw datetime formats directly. We should extract semantic temporal features."
    )

    code_lines = ["import pandas as pd", ""]
    for col in dt_cols[:3]:
        code_lines.append(f"# Ensure column is parsed as datetime:")
        code_lines.append(f"df['{col}'] = pd.to_datetime(df['{col}'], errors='coerce')")
        code_lines.append(f"# Extract temporal components:")
        code_lines.append(f"df['{col}_year'] = df['{col}'].dt.year")
        code_lines.append(f"df['{col}_month'] = df['{col}'].dt.month")
        code_lines.append(f"df['{col}_day'] = df['{col}'].dt.day")
        code_lines.append(f"df['{col}_hour'] = df['{col}'].dt.hour")
        code_lines.append(f"df['{col}_dayofweek'] = df['{col}'].dt.dayofweek")
        code_lines.append(f"df['{col}_is_weekend'] = df['{col}'].dt.dayofweek.isin([5, 6]).astype(int)\n")

    return (
        "📅 Extract Datetime Components",
        desc,
        "\n".join(code_lines)
    )

def _detect_text_preprocessing(df, type_map):
    text_cols = get_columns_by_type(type_map, ColumnType.TEXT)
    if not text_cols:
        return None

    desc = (
        f"Detected free-form text columns: <strong>{', '.join(text_cols[:5])}</strong>. "
        f"Consider converting text fields to TF-IDF representations or embedding matrices."
    )

    code_lines = [
        "from sklearn.feature_extraction.text import TfidfVectorizer",
        "import pandas as pd",
        ""
    ]
    for col in text_cols[:2]:
        code_lines.append(f"# Apply TF-IDF Vectorization on '{col}':")
        code_lines.append(f"tfidf = TfidfVectorizer(max_features=100, stop_words='english')")
        code_lines.append(f"tfidf_feats = tfidf.fit_transform(df['{col}'].fillna(''))")
        code_lines.append(f"tfidf_df = pd.DataFrame(tfidf_feats.toarray(), columns=[f'{col}_tfidf_{{i}}' for i in range(tfidf_feats.shape[1])])")
        code_lines.append(f"df = pd.concat([df.drop(columns=['{col}']), tfidf_df], axis=1)\n")

    return (
        "📝 Preprocess Free-Text Features",
        desc,
        "\n".join(code_lines)
    )

def _detect_interactions(df, type_map):
    numeric_cols = get_columns_by_type(type_map, ColumnType.NUMERIC)
    if len(numeric_cols) < 2:
        return None

    # Compute correlation matrix to find highly correlated pairs
    # Ignore columns with all NaN
    df_corr = df[numeric_cols]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        corr = df_corr.corr().abs()
    
    np.fill_diagonal(corr.values, 0)
    
    high_corr_pairs = []
    seen = set()
    for col1 in corr.columns:
        for col2 in corr.index:
            if col1 == col2:
                continue
            r = corr.loc[col1, col2]
            if r > 0.6 and r < 0.98:  # if r > 0.98, they are duplicate/redundant
                pair = tuple(sorted([col1, col2]))
                if pair not in seen:
                    seen.add(pair)
                    high_corr_pairs.append((col1, col2, r))

    if not high_corr_pairs:
        return None

    # Sort by correlation descending
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)

    desc_parts = []
    code_lines = ["# Create interaction features for highly correlated pairs:", ""]

    pair_descriptions = []
    for col1, col2, r in high_corr_pairs[:4]:
        pair_descriptions.append(f"<code>{col1}</code> & <code>{col2}</code> (r={r:.2f})")
        # Add code to multiply or divide
        code_lines.append(f"# Interaction terms for '{col1}' and '{col2}':")
        code_lines.append(f"df['{col1}_x_{col2}'] = df['{col1}'] * df['{col2}']")
        # Check if 0 in denominator
        code_lines.append(f"df['{col1}_ratio_{col2}'] = df['{col1}'] / (df['{col2}'] + 1e-8)\n")

    desc = (
        f"The following feature pairs have strong linear correlations (r > 0.6): "
        f"{', '.join(pair_descriptions)}. "
        f"Creating non-linear interaction terms (ratios and products) can capture valuable relationship joint effects for linear/distance models."
    )

    return (
        "🔄 Generate Bivariate Interaction Features",
        desc,
        "\n".join(code_lines)
    )

def _generate_master_pipeline(recommendations):
    """Compile all individual snippets into a single cohesive pipeline function."""
    lines = [
        "import numpy as np",
        "import pandas as pd",
        "from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler",
        "from sklearn.feature_extraction.text import TfidfVectorizer",
        "",
        "def engineer_features(df_input):",
        "    \"\"\"Automatically generated feature engineering pipeline.\"\"\"",
        "    df = df_input.copy()",
        ""
    ]
    
    for title, desc, code in recommendations:
        code_lines = code.split("\n")
        # filter out standalone import statements
        filtered = [l for l in code_lines if not l.startswith("import ") and not l.startswith("from ")]
        
        # Remove emoji from title for the python code comment
        clean_title = title.split(" ", 1)[1] if " " in title else title
        
        lines.append(f"    # --- {clean_title} ---")
        for line in filtered:
            if line.strip() == "":
                lines.append("")
            else:
                lines.append(f"    {line}")
        lines.append("")
        
    lines.append("    return df")
    
    return (
        "🌟 Unified Master Pipeline",
        "Based on the overall analysis of your dataset, here is the complete, consolidated feature engineering function you can copy and deploy directly into your codebase. It applies all the recommended transformations safely.",
        "\n".join(lines)
    )
