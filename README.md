# 🔬 Khadee EDA — Deep Insights Data Profiling & Cleaning

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Package Size](https://img.shields.io/badge/wheel--size-~85_KB-blueviolet.svg)]()

`khadee-eda` is a next-generation, high-performance exploratory data analysis (EDA) and data cleaning library. It generates **stunning, glassmorphism-themed interactive HTML profiling reports** from any dataset and provides a robust, lightweight suite of cleaning tools equivalent to `dataprep.clean`.

---

## ⚡ Quick Start

### 1. Generating a Profiling Report
Auto-detects and loads data from CSV, Excel, JSON, Parquet, SQLite, and 10+ other formats.

```python
from khadee_eda import ProfileReport

# Method A: Direct one-liner from file path
report = ProfileReport("train.csv", title="E-Commerce Analysis")
report.to_html("report.html")

# Method B: From an existing Pandas DataFrame
import pandas as pd
df = pd.read_csv("train.csv")
report = ProfileReport(df, title="Customer Profiles")
report.to_html("report.html")
```

### 2. High-Performance Data Cleaning (`khadee_eda.clean`)
Direct, unified API for cleaning, standardizing, and preparing data (a lightweight alternative to `dataprep.clean`).

```python
from khadee_eda import clean

# Clean column headers (standardize to snake_case, PascalCase, camelCase, etc.)
df_clean = clean.clean_headers(df, case="snake")

# Impute missing values with mean, median, mode, or constant value
df_clean = clean.clean_missing(df_clean, columns=["age", "income"], strategy="median")

# Handle outliers by clipping (winsorization) or dropping rows
df_clean = clean.clean_outliers(df_clean, columns=["fare"], method="iqr", strategy="clip")

# Normalize and clean text columns (strip spaces, lowercase, remove special characters)
df_clean = clean.clean_text(df_clean, columns=["product_desc"], lowercase=True, remove_special=True)

# Remove duplicate rows
df_clean = clean.clean_duplicates(df_clean, columns=["user_id"])

# Run a complete, standard cleanup pass
df_clean = clean.clean_df(df)
```

---

## 📊 10 Structured Analysis Sections

Each HTML report is divided into 10 structured, deeply interactive sections:

1. **🏠 Overview**: High-level dataset shapes, reproduction metadata, alerts (missing cells, zero values, extreme correlations, duplicates), and detected data types.
2. **📊 Variables (Interactive Dropdown Explorer)**: Detailed statistics per variable (quantiles, descriptives, frequencies, categories). Includes a custom select dropdown menu to show/hide column details and dynamically resize Plotly visualizations.
3. **📈 Distributions**: Visual analysis of distributions via histogram grids, kernel density estimations (KDE), skewness, kurtosis, and normality tests.
4. **🔗 Correlations**: Pairwise comparison matrices using **Pearson**, **Spearman**, **Kendall**, and **Cramér's V** metrics represented as interactive heatmaps.
5. **❓ Missing Values**: Visual representation of missing data via matrices, counts, and imputation recommendations.
6. **🎯 Outliers**: Deep outlier diagnostic detailing detection using IQR, Z-score, Median Absolute Deviation (MAD), and Isolation Forest.
7. **🔄 Interactions**: Interactive bivariate scatter plots and grouped box plots.
8. **📐 Advanced Stats (Global AI Hub Methodologies)**: Unique statistical and machine learning frameworks tailored after analytical cultures across the globe (see below).
9. **🤖 Model Readiness**: Preprocessing checklists, ML model suitability rankings, and code recommendation generators.
10. **📋 Sample**: Interactive data table viewer showing the head, tail, duplicates, and data dictionary.

---

## 🌍 Global EDA Techniques

The **Advanced Stats** section includes 4 distinct regional analytical philosophies:

* **🇺🇸 US (ML-Readiness & Feature Engineering)**: Identifies feature importance, flags target leakage, and proposes engineered features.
* **🇮🇳 India (Statistical Foundations & Hypothesis Testing)**: Evaluates confidence intervals, conducts hypothesis testing, and fits target distributions.
* **🇯🇵 Japan (Quality Control & Process Analytics — Kaizen)**: Implements Shewhart control charts, calculates Process Capability Indexes ($C_p$/$C_{pk}$), checks stability indicators, and generates Pareto charts.
* **🇨🇳 China (Large-Scale Pattern Recognition)**: Generates PCA projections, evaluates Hopkins clustering statistics, provides K-Means elbow curves, and profiles data density.

---

## 📂 Supported Formats

No need to write separate loading code. `khadee-eda` automatically detects your dataset extension and uses optimized engines to parse it:

| Format | Extensions | Parser |
| :--- | :--- | :--- |
| **CSV / TSV** | `.csv`, `.tsv`, `.txt` | Pandas optimized parser with latin-1 fallback |
| **Excel** | `.xlsx`, `.xls`, `.xlsm`, `.xlsb` | openpyxl / xlrd engine |
| **JSON** | `.json` | Standard and JSON-lines parsed dynamically |
| **Parquet / Feather** | `.parquet`, `.feather` | PyArrow engine |
| **SQLite** | `.db`, `.sqlite`, `.sqlite3` | Built-in SQLite connection reader |
| **Pickle** | `.pkl`, `.pickle` | Standard Python pickle serializer |
| **Others** | `.h5`, `.hdf5`, `.xml`, `.dta`, `.sas7bdat`, `.sav` | Supporting PyTables, XML, Stata, SAS, SPSS |

---

## 💾 Package Footprint & Download Size

Unlike heavier packages that bundle thick C++ binaries, `khadee-eda` is designed to be **incredibly lightweight** and fast to download.

### 1. Download Size (Pip / UV)
* **Wheel Size (`.whl`)**: **~85 KB**
* **Source Distribution (`.tar.gz`)**: **~90 KB**
* **Package Source Size**: **~170 KB** (Clean, pure Python logic + minimal glassmorphism style assets)

### 2. Dependency Size
If your machine already has standard data science packages (like `pandas`, `numpy`, `scipy`) cached, the installation completes instantly (~85 KB download). If installing into a blank virtual environment, pip/uv will download the scientific stack:

| Dependency | Purpose | Download Size (Approx.) |
| :--- | :--- | :--- |
| **pandas** | Data manipulation & structure | ~12 - 15 MB |
| **numpy** | Array computations | ~14 - 18 MB |
| **scipy** | Advanced statistics & tests | ~35 - 40 MB |
| **scikit-learn** | Machine learning engines & PCA | ~7 - 9 MB |
| **plotly** | Dynamic SVG visualizations | ~7 - 8 MB |
| **pyarrow** | High-performance Parquet storage | ~30 - 35 MB |
| **openpyxl** | Excel read/write compatibility | ~2 - 3 MB |
| **jinja2** | HTML templating engine | ~0.2 MB |
| **Total Dependencies** | Full Scientific Stack | **~110 - 130 MB** |

---

## ⚙️ Selective Reports

Save compute time and reduce HTML sizes for large datasets by only rendering the sections or techniques you need:

```python
from khadee_eda import ProfileReport

# Profile only Specific Sections
report = ProfileReport(
    "dataset.csv", 
    sections=["overview", "variables", "model_readiness"]
)

# Render only Specific Global Techniques
report = ProfileReport(
    "dataset.csv",
    techniques=["japan", "us"]
)
```

---

## 💎 Design & Visual Performance Excellence

* **Glassmorphism Dark Theme**: Standard EDA reports often look like boring 2010 tables. `khadee-eda` features a high-end, dark glassmorphism dashboard with neon accents, dynamic hover states, and smooth CSS micro-animations.
* **Instant PDF Export**: Features a beautiful floating "Download PDF" button that triggers browser printing. The custom media print styles automatically expand all hidden column cards, expand all tabs, hide navigational elements/dropdowns, and switch to a crisp ink-saving light template for a clean, professional corporate report.
* **WebGL Crash Mitigation**: Rendering dozens of ScatterGL plots on a single page causes modern browsers to exceed their WebGL context limit, crash, and display blank charts. `khadee-eda` compiles Scatter plots to optimized vector SVG path strings, ensuring **100% chart rendering reliability** without sacrificing interactive zoom or hover features.
* **Smart Dropdown Selectors**: Instead of scrolling endlessly through dozens of columns, the report includes a dynamic select element to view one variable card at a time, instantly resizing the embedded Plotly chart to prevent layout distortions.
* **Copyable Preprocessing Recommender**: When the library suggests cleaning operations (e.g., standardizing headers or imputing values), it displays a syntax-highlighted code block with a one-click copy button, generating context-aware code ready for your pipeline.

---

## 📦 Installation

To install `khadee-eda` in development mode locally:

```bash
git clone https://github.com/khadee/khadee-eda.git
cd khadee_EDA
pip install -e .
```

To install directly using `uv` (recommended for extreme speed):

```bash
uv pip install -e .
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
