"""
mentorml.narrative.narrator
-----------------------------
Phase 7: Business Insight Narrator

Translates the ``DecisionLog`` and ``AnalysisResult`` into a structured,
plain-English Markdown narrative that a non-technical stakeholder can read.

The narrator uses rule-based templates (no LLM required).  The narrative
covers:
1. Executive Summary
2. Dataset Overview
3. Data Quality Findings
4. Preprocessing Choices
5. Feature Engineering
6. Model Selection Rationale
7. Top Predictive Features
8. Key Recommendations
"""

from __future__ import annotations

import logging
from typing import Any

from mentorml.core.decision import DecisionLog, DecisionRecord, Severity

logger = logging.getLogger(__name__)


class BusinessNarrator:
    """
    Phase 7: Rule-based business narrative generator.

    Reads the ``DecisionLog`` and ``AnalysisResult`` (plus optional
    ``QualityReport`` and explainability result) and produces a Markdown
    document structured for business stakeholders.

    Design rationale (ADR-005)
    --------------------------
    - No API key required — works fully offline.
    - Reproducible: same inputs → same output.
    - LLM backend can be layered on top in a future optional extension.

    Parameters
    ----------
    project_name : str
        Name of the ML project (used in the report title).

    Examples
    --------
    ::

        narrator = BusinessNarrator(project_name="Customer Churn Model")
        narrative = narrator.narrate(
            log, analysis_result, quality_report, explanation_result
        )
        print(narrative)
    """

    def __init__(self, project_name: str = "ML Project") -> None:
        self.project_name = project_name

    def narrate(
        self,
        log: DecisionLog,
        analysis_result: dict[str, Any] | None = None,
        quality_report: dict[str, Any] | None = None,
        explanation_result: dict[str, Any] | None = None,
        model_selection_result: dict[str, Any] | None = None,
    ) -> str:
        """
        Generate a Markdown narrative from the decision log and results.

        Parameters
        ----------
        log : DecisionLog
            The full decision log accumulated by the pipeline.
        analysis_result : dict | None
            Output of ``DatasetAnalyzer.analyze()``.
        quality_report : dict | None
            Output of ``DataQualityAssessor.analyze()``.
        explanation_result : dict | None
            Output of ``SHAPExplainer.explain()``.
        model_selection_result : dict | None
            Output of ``ModelSelector.select()``.

        Returns
        -------
        str
            Structured Markdown narrative.
        """
        sections: list[str] = []

        sections.append(self._render_header())
        sections.append(self._render_executive_summary(
            log, analysis_result, quality_report, model_selection_result
        ))
        sections.append(self._render_dataset_overview(analysis_result))
        sections.append(self._render_quality_section(quality_report))
        sections.append(self._render_preprocessing_section(log))
        sections.append(self._render_feature_section(log, explanation_result))
        sections.append(self._render_model_section(model_selection_result))
        sections.append(self._render_recommendations(log, quality_report))
        sections.append(self._render_decision_log_summary(log))

        return "\n\n".join(s for s in sections if s.strip())

    # ------------------------------------------------------------------
    # Section renderers
    # ------------------------------------------------------------------

    def _render_header(self) -> str:
        return f"# 📊 {self.project_name} — mentorml Analysis Report"

    def _render_executive_summary(
        self,
        log: DecisionLog,
        analysis_result: dict[str, Any] | None,
        quality_report: dict[str, Any] | None,
        model_selection_result: dict[str, Any] | None,
    ) -> str:
        lines = ["## 🎯 Executive Summary", ""]

        n_rows = (analysis_result or {}).get("n_rows", "N/A")
        n_cols = (analysis_result or {}).get("n_cols", "N/A")
        task = (analysis_result or {}).get("target", {}).get("inferred_task_type", "unknown")
        overall_quality = (quality_report or {}).get("overall_severity", "ok")

        lines.append(
            f"This report documents an automated machine learning analysis on a dataset with "
            f"**{n_rows} rows** and **{n_cols} columns** targeting a **{task}** problem."
        )

        quality_emoji = {
            Severity.CRITICAL: "🔴",
            Severity.WARNING: "🟡",
            Severity.INFO: "🟢",
            "ok": "🟢",
        }.get(overall_quality, "⚪")

        lines.append(
            f"\n**Data Quality**: {quality_emoji} {str(overall_quality).upper()}"
        )

        if model_selection_result:
            best_model = model_selection_result.get("best_model_name", "N/A")
            metric = model_selection_result.get("scoring_metric", "score")
            scores = model_selection_result.get("cv_scores", {})
            best_score = scores.get(best_model, 0.0)
            lines.append(
                f"\n**Best Model**: `{best_model}` "
                f"({metric}={best_score:.4f})"
            )

        n_decisions = len(log)
        n_warnings = len(log.filter(Severity.WARNING))
        n_critical = len(log.filter(Severity.CRITICAL))

        lines.append(
            f"\n**Pipeline Decisions**: {n_decisions} total "
            f"({n_warnings} warnings, {n_critical} critical)"
        )
        return "\n".join(lines)

    def _render_dataset_overview(
        self,
        analysis_result: dict[str, Any] | None,
    ) -> str:
        if not analysis_result:
            return ""

        lines = ["## 📋 Dataset Overview", ""]
        n_rows = analysis_result.get("n_rows", 0)
        n_cols = analysis_result.get("n_cols", 0)
        n_dup = analysis_result.get("n_duplicate_rows", 0)
        missing_pct = analysis_result.get("overall_missing_pct", 0.0)

        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Rows | {n_rows:,} |")
        lines.append(f"| Columns | {n_cols} |")
        lines.append(f"| Duplicate rows | {n_dup:,} ({n_dup/max(n_rows,1):.1%}) |")
        lines.append(f"| Overall missing | {missing_pct:.1%} |")

        # Target info
        target = analysis_result.get("target", {})
        if target:
            task = target.get("inferred_task_type", "unknown")
            col = target.get("column", "unknown")
            lines.append(f"| Target column | `{col}` ({task}) |")

        # Column type breakdown
        profiles = analysis_result.get("column_profiles", {})
        if profiles:
            from collections import Counter
            def _get_cat(p: Any) -> str:
                cat = getattr(p, "dtype_category", None) if not isinstance(p, dict) else p.get("dtype_category")
                return cat.value if hasattr(cat, "value") else str(cat or "unknown")

            cats = Counter(_get_cat(p) for p in profiles.values())
            breakdown = ", ".join(f"{v} {k}" for k, v in cats.items())
            lines.append(f"| Column types | {breakdown} |")

        return "\n".join(lines)

    def _render_quality_section(
        self,
        quality_report: dict[str, Any] | None,
    ) -> str:
        if not quality_report:
            return ""

        lines = ["## 🔍 Data Quality Findings", ""]
        issues = quality_report.get("issues", [])

        if not issues:
            lines.append("✅ No significant data quality issues detected.")
            return "\n".join(lines)

        severity_counts = quality_report.get("n_issues_by_severity", {})
        n_crit = severity_counts.get(Severity.CRITICAL, 0)
        n_warn = severity_counts.get(Severity.WARNING, 0)
        n_info = severity_counts.get(Severity.INFO, 0)

        lines.append(
            f"Found **{len(issues)} issues**: "
            f"{n_crit} critical 🔴, {n_warn} warnings 🟡, {n_info} informational 🔵"
        )
        lines.append("")

        emoji_map = {
            Severity.CRITICAL: "🔴",
            Severity.WARNING: "🟡",
            Severity.INFO: "🔵",
        }
        for issue in issues:
            sev = issue.get("severity", "info")
            emoji = emoji_map.get(sev, "⚪")
            col = issue.get("column", "dataset-level")
            lines.append(f"- {emoji} **{col}**: {issue.get('detail', '')}")
            rec = issue.get("recommendation", "")
            if rec:
                lines.append(f"  - 💡 *{rec}*")

        to_drop = quality_report.get("columns_to_drop", [])
        if to_drop:
            lines.append(f"\n**Columns recommended for removal**: {', '.join(f'`{c}`' for c in to_drop)}")

        return "\n".join(lines)

    def _render_preprocessing_section(self, log: DecisionLog) -> str:
        lines = ["## ⚙️ Preprocessing Choices", ""]
        prep_records = [
            r for r in log
            if "ExplainablePreprocessor" in r.component
            and r.action not in ("fit_start", "fit_complete",
                                 "transform_complete", "transform_start")
        ]
        if not prep_records:
            lines.append("*No preprocessing records found.*")
            return "\n".join(lines)

        for record in prep_records:
            lines.append(f"- **{record.action}**: {record.rationale}")
        return "\n".join(lines)

    def _render_feature_section(
        self,
        log: DecisionLog,
        explanation_result: dict[str, Any] | None,
    ) -> str:
        lines = ["## 🔬 Feature Engineering & Importance", ""]

        fe_records = [r for r in log if "FeatureEngineer" in r.component
                      and r.action not in ("fit_start", "fit_complete",
                                           "transform_complete")]
        if fe_records:
            lines.append("### Feature Engineering Decisions")
            for record in fe_records:
                lines.append(f"- **{record.action}**: {record.rationale}")
            lines.append("")

        if explanation_result:
            method = explanation_result.get("method", "unknown")
            lines.append(f"### Top Predictive Features (method: `{method}`)")
            lines.append("")
            lines.append("| Rank | Feature | Importance |")
            lines.append("|------|---------|------------|")
            importances = explanation_result.get("feature_importances", [])[:10]
            for i, item in enumerate(importances, 1):
                feat = item.get("feature", "?")
                imp = item.get("importance", 0.0)
                bar = "█" * max(1, int(imp * 20 / max(
                    importances[0].get("importance", 1), 1e-9
                )))
                lines.append(f"| {i} | `{feat}` | {imp:.6f} {bar} |")

        return "\n".join(lines)

    def _render_model_section(
        self,
        model_selection_result: dict[str, Any] | None,
    ) -> str:
        if not model_selection_result:
            return ""

        lines = ["## 🤖 Model Selection", ""]
        best = model_selection_result.get("best_model_name", "N/A")
        task = model_selection_result.get("task_type", "N/A")
        metric = model_selection_result.get("scoring_metric", "score")
        scores = model_selection_result.get("cv_scores", {})

        lines.append(f"**Task type**: {task}")
        lines.append(f"**Evaluation metric**: `{metric}`")
        lines.append(f"**Winner**: `{best}`")
        lines.append("")
        lines.append("| Model | CV Score |")
        lines.append("|-------|----------|")
        for name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            winner = " ✅" if name == best else ""
            lines.append(f"| `{name}` | {score:.4f}{winner} |")

        return "\n".join(lines)

    def _render_recommendations(
        self,
        log: DecisionLog,
        quality_report: dict[str, Any] | None,
    ) -> str:
        lines = ["## 💡 Key Recommendations", ""]
        recs: list[str] = []

        # From quality report
        if quality_report:
            for issue in quality_report.get("issues", []):
                rec = issue.get("recommendation", "")
                if rec:
                    recs.append(rec)

        # From critical decisions
        critical = log.filter(Severity.CRITICAL)
        for record in critical:
            recs.append(record.rationale)

        if not recs:
            lines.append("✅ No critical recommendations at this time.")
        else:
            for i, rec in enumerate(recs[:10], 1):
                lines.append(f"{i}. {rec}")

        return "\n".join(lines)

    def _render_decision_log_summary(self, log: DecisionLog) -> str:
        lines = ["## 📝 Decision Log Summary", ""]
        components: dict[str, int] = {}
        for record in log:
            comp = record.component.split(".")[0]
            components[comp] = components.get(comp, 0) + 1

        lines.append("| Component | # Decisions |")
        lines.append("|-----------|-------------|")
        for comp, count in sorted(components.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"| `{comp}` | {count} |")

        lines.append("")
        lines.append(
            f"*Total: {len(log)} decisions recorded across "
            f"{len(components)} components.*"
        )
        return "\n".join(lines)