"""
K-EDA — Basic Usage Example
==================================
"""
from k_eda import ProfileReport

# ── Example 1: From CSV file ──
# report = ProfileReport("train.csv", title="Train Dataset Analysis")
# report.to_html("train_report.html")

# ── Example 2: From DataFrame ──
import pandas as pd

# Create a sample dataset for testing
import numpy as np

np.random.seed(42)
n = 1000

df = pd.DataFrame({
    "id": range(1, n + 1),
    "age": np.random.normal(35, 10, n).astype(int),
    "salary": np.random.lognormal(10.5, 0.8, n).round(2),
    "department": np.random.choice(["Engineering", "Sales", "Marketing", "HR", "Finance"], n),
    "satisfaction": np.random.uniform(1, 10, n).round(1),
    "years_exp": np.random.exponential(5, n).round(1),
    "is_manager": np.random.choice([True, False], n, p=[0.2, 0.8]),
    "rating": np.random.choice(["A", "B", "C", "D"], n, p=[0.1, 0.3, 0.4, 0.2]),
    "bonus": np.random.normal(5000, 2000, n).round(2),
    "projects": np.random.poisson(3, n),
})

# Add some missing values
df.loc[np.random.choice(n, 50, replace=False), "salary"] = np.nan
df.loc[np.random.choice(n, 30, replace=False), "satisfaction"] = np.nan
df.loc[np.random.choice(n, 20, replace=False), "department"] = np.nan

# Add some duplicates
df = pd.concat([df, df.sample(15, random_state=42)], ignore_index=True)

print("=" * 60)
print("  K-EDA — Generating Full Report")
print("=" * 60)

# Generate full report (ALL sections, ALL techniques)
report = ProfileReport(df, title="Employee Dataset Analysis")
report.to_html("employee_report.html")

print("\n[+] Done! Open employee_report.html in your browser.")
