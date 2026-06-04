"""
Khadee EDA — Section 8: Advanced Statistics
=============================================
Dispatches to global technique modules: US, India, Japan, China.
"""

from ..utils import section_header
from ..config import ALL_TECHNIQUES


def generate(df, type_map, techniques=None):
    """Generate the Advanced Statistics section HTML."""
    if techniques is None:
        techniques = ALL_TECHNIQUES

    html = section_header("section-advanced-stats", " Advanced Statistics",
                          "Global EDA techniques")

    from ..techniques import us, india, japan, china

    technique_modules = {
        "us": us,
        "india": india,
        "japan": japan,
        "china": china,
    }

    labels = {
        "us": "🇺🇸 US",
        "india": "🇮🇳 India",
        "japan": "🇯🇵 Japan",
        "china": "🇨🇳 China",
    }

    for tech in techniques:
        if tech in technique_modules:
            module = technique_modules[tech]
            label = labels.get(tech, tech.title())
            html += f'<div class="technique-section">'
            html += f'<h3 class="technique-title">{label}</h3>'
            try:
                html += module.generate(df, type_map)
            except Exception as e:
                html += f'<div class="card"><p class="error-message">Error in {tech} analysis: {str(e)}</p></div>'
            html += '</div>'

    return html
