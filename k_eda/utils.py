"""
K-EDA — Utility Functions
==============================
Helpers for formatting, HTML generation, and data manipulation.
"""

import html


def format_number(value, decimals=2):
    """Format a number with commas and decimals."""
    if value is None:
        return "N/A"
    try:
        if isinstance(value, int) or (isinstance(value, float) and value == int(value)):
            return f"{int(value):,}"
        return f"{value:,.{decimals}f}"
    except (ValueError, TypeError, OverflowError):
        return str(value)


def format_percentage(value, decimals=1):
    """Format a value as percentage."""
    if value is None:
        return "N/A"
    try:
        return f"{value * 100:.{decimals}f}%"
    except (ValueError, TypeError):
        return str(value)


def format_memory(bytes_val):
    """Convert bytes to human-readable memory format."""
    if bytes_val is None:
        return "N/A"
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} TB"


def safe_str(value):
    """Safely convert any value to an HTML-safe string."""
    return html.escape(str(value))


def truncate(text, max_len=50):
    """Truncate text to a maximum length."""
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def make_id(text):
    """Convert text to a valid HTML id."""
    return text.lower().replace(" ", "-").replace("_", "-")


def alert_badge(level, text):
    """Generate an HTML alert badge."""
    colors = {
        "danger": "#ef4444",
        "warning": "#f59e0b",
        "info": "#3b82f6",
        "success": "#10b981",
    }
    color = colors.get(level, "#94a3b8")
    return (
        f'<span class="alert-badge" style="background: {color}20; '
        f'color: {color}; border: 1px solid {color}40;">{safe_str(text)}</span>'
    )


def stat_card(label, value, icon="", color="#3b82f6"):
    """Generate an HTML stat card."""
    return f"""
    <div class="stat-card">
        <div class="stat-icon" style="color: {color}">{icon}</div>
        <div class="stat-value">{safe_str(str(value))}</div>
        <div class="stat-label">{safe_str(label)}</div>
    </div>
    """


def section_header(section_id, title, description=""):
    """Generate a section header HTML."""
    desc_html = f'<p class="section-description">{safe_str(description)}</p>' if description else ""
    return f"""
    <div class="section-header" id="{section_id}">
        <h2 class="section-title">{title}</h2>
        {desc_html}
    </div>
    """


def collapsible(title, content, open_default=False):
    """Generate a collapsible/accordion HTML block."""
    open_attr = "open" if open_default else ""
    return f"""
    <details class="collapsible" {open_attr}>
        <summary class="collapsible-header">{title}</summary>
        <div class="collapsible-body">
            {content}
        </div>
    </details>
    """


def progress_bar(value, max_val=100, color="#3b82f6", label=""):
    """Generate an HTML progress bar."""
    pct = min(100, max(0, (value / max_val * 100) if max_val > 0 else 0))
    return f"""
    <div class="progress-container">
        <div class="progress-label">{safe_str(label)}</div>
        <div class="progress-bar-bg">
            <div class="progress-bar-fill" style="width: {pct:.1f}%; background: {color};">
                {pct:.1f}%
            </div>
        </div>
    </div>
    """


def check_item(passed, label):
    """Generate a readiness check item."""
    icon = "✅" if passed else "❌"
    cls = "check-pass" if passed else "check-fail"
    return f'<div class="check-item {cls}">{icon} {safe_str(label)}</div>'


def optimize_dataframe(df):
    """
    Optimizes memory footprint of a pandas DataFrame by downcasting numeric columns
    and converting low-cardinality object/string columns to category.
    """
    import pandas as pd
    df = df.copy()
    
    # Track initial memory for logging if needed
    for col in df.columns:
        # Ignore empty/all-NaN columns
        if df[col].isna().all():
            continue
            
        col_type = df[col].dtype
        
        # Optimize integers
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        # Optimize floats
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast='float')
        # Optimize strings / objects
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            try:
                n_rows = len(df[col])
                n_unique = df[col].nunique()
                # If unique values < 100 or unique values make up less than 5% of dataset, cast to category
                if n_unique < 100 or (n_rows > 0 and n_unique / n_rows < 0.05):
                    df[col] = df[col].astype('category')
            except Exception:
                pass
                
    return df
