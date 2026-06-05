"""
Khadee EDA — Deep Insights Data Profiling
==========================================

A comprehensive EDA module that generates stunning HTML profiling reports
from any dataset format. Supports CSV, Excel, JSON, Parquet, and 10+ more formats.

Usage
-----
    from khadee_eda import ProfileReport

    # From file (any format — auto-detected)
    report = ProfileReport("train.csv", title="My EDA Report")
    report = ProfileReport("data.xlsx", title="Excel Analysis")

    # From DataFrame
    import pandas as pd
    df = pd.read_csv("train.csv")
    report = ProfileReport(df, title="My EDA Report")

    # Generate HTML report
    report.to_html("report.html")

    # Selective sections
    report = ProfileReport(df, sections=["overview", "variables", "correlations"])

    # Selective techniques
    report = ProfileReport(df, techniques=["us", "japan"])

Sub-modules
-----------
    from khadee_eda.techniques import us, india, japan, china
    from khadee_eda.engines import stats_engine, correlation_engine, missing_engine, outlier_engine
"""

__version__ = "1.0.2"
__author__ = "Khadee"

import sys
import time
import warnings

import pandas as pd


def _print(msg):
    """Print with UTF-8 encoding fallback for Windows consoles."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))

from .config import ALL_SECTIONS, ALL_TECHNIQUES
from .loader import load_dataset
from .type_detector import detect_types
from .renderers.html_renderer import render_html
from . import clean


class ProfileReport:
    """
    Generate a comprehensive EDA profiling report.

    Parameters
    ----------
    source : str or pd.DataFrame
        File path (auto-detects format from extension) or pandas DataFrame.
    title : str, optional
        Report title. Default: "Khadee EDA Report".
    sections : list, optional
        List of section IDs to include. Default: all 10 sections.
        Options: overview, variables, distributions, correlations, missing,
                 outliers, interactions, advanced_stats, model_readiness, sample
    techniques : list, optional
        List of technique IDs for the Advanced Statistics section.
        Default: all 4 techniques.
        Options: us, india, japan, china
    **kwargs : dict
        Extra arguments passed to the file reader (e.g., sheet_name for Excel).

    Examples
    --------
    >>> from khadee_eda import ProfileReport
    >>> report = ProfileReport("train.csv", title="Profiling Report")
    >>> report.to_html("report.html")
    """

    def __init__(self, source, title="Khadee EDA Report", sections=None,
                 techniques=None, **kwargs):
        self.title = title
        self.sections = sections or ALL_SECTIONS
        self.techniques = techniques or ALL_TECHNIQUES
        self._start_time = time.time()

        # Validate sections
        for s in self.sections:
            if s not in ALL_SECTIONS:
                raise ValueError(
                    f"Unknown section: '{s}'. Available: {ALL_SECTIONS}"
                )

        # Validate techniques
        for t in self.techniques:
            if t not in ALL_TECHNIQUES:
                raise ValueError(
                    f"Unknown technique: '{t}'. Available: {ALL_TECHNIQUES}"
                )

        # Load data
        _print("[*] Khadee EDA -- Loading dataset...")
        self.df, self.metadata = load_dataset(source, **kwargs)
        _print(f"   [+] Loaded: {self.df.shape[0]:,} rows x {self.df.shape[1]:,} columns")

        # Detect types
        _print("   [*] Detecting column types...")
        self.type_map = detect_types(self.df)

        # Pre-compute report
        _print("   [*] Analyzing data...")
        self._sections_html = self._generate_sections()

        elapsed = time.time() - self._start_time
        _print(f"   [+] Analysis complete in {elapsed:.2f}s")

    def _generate_sections(self):
        """Generate HTML for all requested sections."""
        from .sections import (
            overview, variables, distributions, correlations,
            missing, outliers, interactions, advanced_stats,
            model_readiness, model_suggestion, sample,
        )

        section_generators = {
            "overview": lambda: overview.generate(
                self.df, self.type_map, self.metadata, self._start_time
            ),
            "variables": lambda: variables.generate(self.df, self.type_map),
            "distributions": lambda: distributions.generate(self.df, self.type_map),
            "correlations": lambda: correlations.generate(self.df, self.type_map),
            "missing": lambda: missing.generate(self.df, self.type_map),
            "outliers": lambda: outliers.generate(self.df, self.type_map),
            "interactions": lambda: interactions.generate(self.df, self.type_map),
            "advanced_stats": lambda: advanced_stats.generate(
                self.df, self.type_map, self.techniques
            ),
            "model_readiness": lambda: model_readiness.generate(self.df, self.type_map),
            "model_suggestion": lambda: model_suggestion.generate(self.df, self.type_map),
            "sample": lambda: sample.generate(self.df, self.type_map),
        }

        results = {}
        for section_id in self.sections:
            gen = section_generators.get(section_id)
            if gen:
                try:
                    results[section_id] = gen()
                    _print(f"   [+] {section_id}")
                except Exception as e:
                    warnings.warn(f"Error generating section '{section_id}': {e}")
                    results[section_id] = (
                        f'<div class="card"><h3 class="card-title">⚠️ Error in {section_id}</h3>'
                        f'<p class="error-message">{str(e)}</p></div>'
                    )

        return results

    def to_html(self, output_path="report.html"):
        """
        Generate and save the HTML report.

        Parameters
        ----------
        output_path : str
            Path to save the HTML report.
        """
        _print("   [*] Generating HTML report...")

        html = render_html(self.title, self._sections_html, self.sections)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        _print(f"   [+] Report saved to: {output_path}")
        return output_path

    def to_html_string(self):
        """Return the HTML report as a string."""
        return render_html(self.title, self._sections_html, self.sections)

    def __repr__(self):
        return (
            f"ProfileReport("
            f"rows={self.df.shape[0]:,}, cols={self.df.shape[1]:,}, "
            f"sections={len(self.sections)}, techniques={len(self.techniques)})"
        )
