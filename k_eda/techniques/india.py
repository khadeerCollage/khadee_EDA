"""
K-EDA — 🇮🇳 India Technique: Statistical Foundation & Hypothesis Testing
================================================================================
T-tests, Chi-square, ANOVA, confidence intervals, effect sizes, distribution fitting.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from ..type_detector import ColumnType
from ..utils import format_number, format_percentage, safe_str


def generate(df, type_map):
    """Generate India technique analysis HTML."""
    numeric_cols = [c for c, t in type_map.items()
                    if t == ColumnType.NUMERIC and df[c].notna().sum() > 3]

    html = ""

    # ── Confidence Intervals ──
    html += '<div class="card">'
    html += '<h4 class="card-title">📊 95% Confidence Intervals for Numeric Means</h4>'

    if numeric_cols:
        html += '<div class="table-responsive"><table class="styled-table compact">'
        html += '<thead><tr><th>Feature</th><th>Mean</th><th>Std Error</th><th>95% CI Lower</th><th>95% CI Upper</th><th>CI Width</th></tr></thead>'
        html += '<tbody>'

        for col in numeric_cols[:20]:
            clean = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(clean) < 3:
                continue
            mean = float(clean.mean())
            se = float(clean.std() / np.sqrt(len(clean)))
            ci_lower = mean - 1.96 * se
            ci_upper = mean + 1.96 * se
            ci_width = ci_upper - ci_lower

            html += f'<tr><td class="col-name">{safe_str(col)}</td>'
            html += f'<td>{format_number(mean)}</td>'
            html += f'<td>{format_number(se, 4)}</td>'
            html += f'<td>{format_number(ci_lower)}</td>'
            html += f'<td>{format_number(ci_upper)}</td>'
            html += f'<td>{format_number(ci_width)}</td></tr>'

        html += '</tbody></table></div>'
    else:
        html += '<p class="empty-message">No numeric columns for confidence interval analysis.</p>'
    html += '</div>'

    # ── Distribution Fitting ──
    html += '<div class="card">'
    html += '<h4 class="card-title">📈 Best-Fit Distribution per Feature</h4>'
    html += '<p class="card-desc">Testing Normal, Log-Normal, Exponential, and Uniform distributions.</p>'

    if numeric_cols:
        distributions = [
            ("Normal", scipy_stats.norm),
            ("Log-Normal", scipy_stats.lognorm),
            ("Exponential", scipy_stats.expon),
            ("Uniform", scipy_stats.uniform),
        ]

        html += '<div class="table-responsive"><table class="styled-table compact">'
        html += '<thead><tr><th>Feature</th><th>Best Fit</th><th>KS Statistic</th><th>p-value</th></tr></thead>'
        html += '<tbody>'

        for col in numeric_cols[:15]:
            clean = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(clean) < 10:
                continue

            sample = clean.sample(min(1000, len(clean)), random_state=42)
            best_name = "Unknown"
            best_ks = 1.0
            best_p = 0.0

            for name, dist in distributions:
                try:
                    if name == "Log-Normal" and sample.min() <= 0:
                        continue
                    params = dist.fit(sample)
                    ks_stat, p_val = scipy_stats.kstest(sample, dist.cdf, args=params)
                    if ks_stat < best_ks:
                        best_ks = float(ks_stat)
                        best_p = float(p_val)
                        best_name = name
                except Exception:
                    continue

            html += f'<tr><td class="col-name">{safe_str(col)}</td>'
            html += f'<td><strong>{best_name}</strong></td>'
            html += f'<td>{format_number(best_ks, 4)}</td>'
            html += f'<td>{format_number(best_p, 4)}</td></tr>'

        html += '</tbody></table></div>'
    html += '</div>'

    # ── Hypothesis Testing Summary ──
    html += '<div class="card">'
    html += '<h4 class="card-title">🧪 Hypothesis Testing Panel</h4>'
    html += '<p class="card-desc">One-sample t-test (H₀: μ = 0) for each numeric feature.</p>'

    if numeric_cols:
        html += '<div class="table-responsive"><table class="styled-table compact">'
        html += '<thead><tr><th>Feature</th><th>t-statistic</th><th>p-value</th><th>Significant?</th><th>Effect Size (Cohen\'s d)</th></tr></thead>'
        html += '<tbody>'

        for col in numeric_cols[:20]:
            clean = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(clean) < 3:
                continue

            try:
                t_stat, p_val = scipy_stats.ttest_1samp(clean, 0)
                cohens_d = float(clean.mean() / clean.std()) if clean.std() > 0 else 0

                sig = p_val < 0.05
                sig_badge = '<span class="badge-pass">Yes (p<0.05)</span>' if sig else '<span class="badge-fail">No</span>'

                # Effect size interpretation
                d_abs = abs(cohens_d)
                if d_abs < 0.2:
                    d_label = "Negligible"
                elif d_abs < 0.5:
                    d_label = "Small"
                elif d_abs < 0.8:
                    d_label = "Medium"
                else:
                    d_label = "Large"

                html += f'<tr><td class="col-name">{safe_str(col)}</td>'
                html += f'<td>{format_number(float(t_stat))}</td>'
                html += f'<td>{format_number(float(p_val), 4)}</td>'
                html += f'<td>{sig_badge}</td>'
                html += f'<td>{format_number(cohens_d)} ({d_label})</td></tr>'
            except Exception:
                continue

        html += '</tbody></table></div>'
    html += '</div>'

    return html


def analyze(df):
    """Standalone India technique analysis."""
    from ..type_detector import detect_types
    type_map = detect_types(df)
    numeric_cols = [c for c, t in type_map.items() if t == ColumnType.NUMERIC]

    results = {"confidence_intervals": {}, "hypothesis_tests": {}}

    for col in numeric_cols:
        clean = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(clean) < 3:
            continue
        mean = float(clean.mean())
        se = float(clean.std() / np.sqrt(len(clean)))
        results["confidence_intervals"][col] = {
            "mean": mean, "se": se,
            "ci_lower": mean - 1.96 * se, "ci_upper": mean + 1.96 * se,
        }

    return results
