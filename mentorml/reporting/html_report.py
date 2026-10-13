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
        .grid-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }
        .stat-box {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-color);
            padding: 1rem;
            border-radius: 0.5rem;
        }
        .stat-label { font-size: 0.85rem; color: var(--text-secondary); text-transform: uppercase; }
        .stat-value { font-size: 1.5rem; font-weight: 700; color: var(--accent); margin-top: 0.25rem; }
        .badge {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .badge-warning { background: rgba(251, 191, 36, 0.2); color: var(--warning); }
        .badge-danger { background: rgba(248, 113, 113, 0.2); color: var(--danger); }
        .badge-info { background: rgba(56, 189, 248, 0.2); color: var(--accent); }
        """

    def _render_header(self) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"""
        <div class="header">
            <h1>{html.escape(self.project_name)}</h1>
            <p>Automated Machine Learning & Pipeline Audit Report  Generated on {timestamp}</p>
        </div>
        """

    def _render_dataset_overview(self, analysis_result: dict[str, Any]) -> str:
        n_rows = analysis_result.get("n_rows", analysis_result.get("rows", "N/A"))
        n_cols = analysis_result.get("n_columns", analysis_result.get("columns", "N/A"))
        task_type = analysis_result.get("task_type", "auto-inferred")

        return f"""
        <div class="card">
            <h2>Dataset Profile & Overview</h2>
            <div class="grid-stats">
                <div class="stat-box">
                    <div class="stat-label">Total Rows</div>
                    <div class="stat-value">{n_rows}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Total Columns</div>
                    <div class="stat-value">{n_cols}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Task Type</div>
                    <div class="stat-value" style="font-size: 1.1rem; text-transform: capitalize;">{task_type}</div>
                </div>
            </div>
        </div>
        """

    def _render_quality_section(self, quality_report: dict[str, Any]) -> str:
        issues = quality_report.get("issues", [])
        if not issues:
            issues_content = '<p style="color: var(--success);">No critical data quality issues identified. Dataset is well-structured.</p>'
        else:
            cards = []
            for item in issues:
                col = item.get("column", "Dataset-Level")
                desc = item.get("detail", item.get("issue_type", "Quality Warning"))
                severity = str(item.get("severity", "WARNING")).upper()
                badge_class = "badge-danger" if "CRITICAL" in severity or "HIGH" in severity else "badge-warning"
                cards.append(f"""
                <div style="margin-top: 0.75rem; padding: 0.75rem; background: rgba(0,0,0,0.2); border-left: 4px solid var(--accent); border-radius: 4px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong>{html.escape(str(col))}</strong>
                        <span class="badge {badge_class}">{html.escape(severity)}</span>
                    </div>
                    <p style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 0.25rem;">{html.escape(str(desc))}</p>
                </div>
                """)
            issues_content = "".join(cards)

        return f"""
        <div class="card">
            <h2>Phase 2  Data Quality Audit</h2>
            {issues_content}
        </div>
        """
