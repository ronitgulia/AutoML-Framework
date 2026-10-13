"""
mentorml.reporting.html_report
-------------------------------
Phase 8: Interactive HTML Report Generation.

Generates self-contained HTML reports summarizing dataset analysis,
quality checks, preprocessing transformations, model evaluation rankings,
feature importances, and business narratives.
"""

from __future__ import annotations

import html
import logging
import os
from datetime import datetime
from typing import Any

from mentorml.core.decision import DecisionLog, Severity

logger = logging.getLogger(__name__)


class HTMLReportGenerator:
    """
    Self-contained HTML report builder for AutoML pipeline runs.

    Compiles dataset quality findings, preprocessing choices, model rankings,
    feature importance explanations, and decision logs into an interactive,
    responsive HTML dashboard.

    Parameters
    ----------
    project_name : str
        Human-readable title for the generated report.
    """

    def __init__(self, project_name: str = "MentorML Project") -> None:
        self.project_name = project_name

    def _render_styles(self) -> str:
        """Return embedded modern CSS design tokens and layout styling."""
        return """
        :root {
            --bg-primary: #0f172a;
            --bg-card: #1e293b;
            --border-color: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent: #38bdf8;
            --accent-hover: #0ea5e9;
            --success: #34d399;
            --warning: #fbbf24;
            --danger: #f87171;
            --font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: var(--font-family);
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem 1rem;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header {
            margin-bottom: 2rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid var(--border-color);
        }
        .header h1 { font-size: 2.25rem; font-weight: 700; color: var(--accent); }
        .header p { color: var(--text-secondary); margin-top: 0.5rem; }
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 0.75rem;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }
        .card h2 {
            font-size: 1.35rem;
            color: var(--text-primary);
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        """
