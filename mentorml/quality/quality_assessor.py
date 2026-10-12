"""
mentorml.quality.quality_assessor
----------------------------------
Phase 2: Data Quality Assessor

Performs a deep quality audit of a DataFrame and emits ``DecisionRecord``
objects for every issue found.  The output ``QualityReport`` dict is consumed
by the ``ExplainablePreprocessor`` (Phase 3) to guide imputation and cleaning
strategies.

Components
----------
DataQualityAssessor   ← public facade (implements Analyzable)
  ├── MissingValueInspector  ← per-column missing pattern analysis
  ├── OutlierInspector       ← dataset-wide outlier summary
  └── DuplicateInspector     ← exact and near-duplicate detection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from mentorml.config import MentorConfig
from mentorml.core.decision import DecisionLog, DecisionRecord, Severity
from mentorml.core.exceptions import DataValidationError

logger = logging.getLogger(__name__)

_MIN_ROWS = 10


# ---------------------------------------------------------------------------
# QualityIssue dataclass
# ---------------------------------------------------------------------------


@dataclass
class QualityIssue:
    """
    A single data quality problem detected in the dataset.

    Parameters
    ----------
    column : str | None
        Affected column name, or ``None`` for dataset-level issues.
    issue_type : str
        Short machine-readable tag (e.g. ``"missing_high"``, ``"outliers"``).
    severity : Severity
        How critical this issue is.
    detail : str
        Human-readable explanation of the problem.
    metric : float | None
        Quantitative metric supporting the finding (e.g. missing fraction).
    recommendation : str
        Suggested remediation action.
    """

    column: str | None
    issue_type: str
    severity: Severity
    detail: str
    metric: float | None = None
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain JSON-friendly dict."""
        return {
            "column": self.column,
            "issue_type": self.issue_type,
            "severity": self.severity,
            "detail": self.detail,
            "metric": round(self.metric, 4) if self.metric is not None else None,
            "recommendation": self.recommendation,
        }


# ---------------------------------------------------------------------------
# MissingValueInspector
# ---------------------------------------------------------------------------


class MissingValueInspector:
    """
    Analyses missing value patterns for every column in a DataFrame.

    Uses the ``missing_threshold`` from ``MentorConfig`` to decide severity:

    - ``missing_pct >= missing_threshold``   → CRITICAL (drop candidate)
    - ``0.05 <= missing_pct < threshold``    → WARNING  (imputation needed)
    - ``0 < missing_pct < 0.05``             → INFO     (minor, impute safely)
    """

    @staticmethod
    def inspect(
        df: pd.DataFrame,
        config: MentorConfig,
        log: DecisionLog,
    ) -> list[QualityIssue]:
        """
        Inspect all columns for missing values.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.
        config : MentorConfig
            Configuration (uses ``missing_threshold``).
        log : DecisionLog
            Decision log to append records to.

        Returns
        -------
        list[QualityIssue]
            One issue per column that has any missing values.
        """
        issues: list[QualityIssue] = []
        n_rows = len(df)

        for col in df.columns:
            n_missing = int(df[col].isna().sum())
            if n_missing == 0:
                continue

            missing_pct = n_missing / n_rows

            if missing_pct >= config.missing_threshold:
                severity = Severity.CRITICAL
                issue_type = "missing_critical"
                recommendation = (
                    f"Drop column '{col}' — {missing_pct:.1%} missing exceeds "
                    f"the configured threshold of {config.missing_threshold:.1%}."
                )
            elif missing_pct >= 0.05:
                severity = Severity.WARNING
                issue_type = "missing_high"
                recommendation = (
                    f"Impute '{col}' — {missing_pct:.1%} missing values detected. "
                    "Use median (numeric) or mode (categorical) imputation."
                )
            else:
                severity = Severity.INFO
                issue_type = "missing_low"
                recommendation = (
                    f"Impute '{col}' — {missing_pct:.1%} missing values "
                    "(minor; safe to impute with median/mode)."
                )

            detail = (
                f"Column '{col}' has {n_missing}/{n_rows} "
                f"({missing_pct:.1%}) missing values."
            )
            issue = QualityIssue(
                column=col,
                issue_type=issue_type,
                severity=severity,
                detail=detail,
                metric=missing_pct,
                recommendation=recommendation,
            )
            issues.append(issue)

            log.append(
                DecisionRecord(
                    component="DataQualityAssessor.MissingValueInspector",
                    action=f"missing_values_detected:{col}",
                    rationale=detail,
                    severity=severity,
                    data={
                        "column": col,
                        "n_missing": n_missing,
                        "missing_pct": round(missing_pct, 4),
                        "recommendation": recommendation,
                    },
                )
            )

        return issues


# ---------------------------------------------------------------------------
# OutlierInspector
# ---------------------------------------------------------------------------

