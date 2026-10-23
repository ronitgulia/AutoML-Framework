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
        """Serialise to a plain JSON-friendly dict.

        All enum values are converted to their string `.value` so the
        returned dict passes `json.dumps()` without a custom encoder.
        """
        return {
            "column": self.column,
            "issue_type": self.issue_type,
            "severity": self.severity.value,   # str, not Severity enum
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


class OutlierInspector:
    """
    Identifies columns with a high proportion of statistical outliers.

    Uses the IQR method: values outside ``[Q1 - 1.5·IQR, Q3 + 1.5·IQR]``
    are flagged as outliers.

    A column with ``outlier_pct >= 0.10`` is flagged as WARNING.
    """

    _OUTLIER_WARNING_THRESHOLD = 0.10

    @classmethod
    def inspect(
        cls,
        df: pd.DataFrame,
        log: DecisionLog,
    ) -> list[QualityIssue]:
        """
        Inspect all numeric columns for outlier prevalence.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.
        log : DecisionLog
            Decision log to append records to.

        Returns
        -------
        list[QualityIssue]
            One issue per numeric column with notable outliers.
        """
        issues: list[QualityIssue] = []
        numeric_cols = df.select_dtypes(include="number").columns

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 4:
                continue

            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            if iqr == 0:
                continue

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            n_outliers = int(((series < lower) | (series > upper)).sum())
            outlier_pct = n_outliers / len(series)

            if outlier_pct < cls._OUTLIER_WARNING_THRESHOLD:
                continue

            detail = (
                f"Column '{col}' has {n_outliers} outliers "
                f"({outlier_pct:.1%} of non-missing values) "
                f"outside IQR fence [{lower:.3g}, {upper:.3g}]."
            )
            recommendation = (
                f"Consider capping '{col}' at the IQR fence or applying "
                "a log/sqrt transform to reduce outlier influence."
            )
            issue = QualityIssue(
                column=col,
                issue_type="outliers_high",
                severity=Severity.WARNING,
                detail=detail,
                metric=outlier_pct,
                recommendation=recommendation,
            )
            issues.append(issue)

            log.append(
                DecisionRecord(
                    component="DataQualityAssessor.OutlierInspector",
                    action=f"outliers_detected:{col}",
                    rationale=detail,
                    severity=Severity.WARNING,
                    data={
                        "column": col,
                        "n_outliers": n_outliers,
                        "outlier_pct": round(outlier_pct, 4),
                        "iqr_lower": round(lower, 4),
                        "iqr_upper": round(upper, 4),
                    },
                )
            )

        return issues


# ---------------------------------------------------------------------------
# DuplicateInspector
# ---------------------------------------------------------------------------


class DuplicateInspector:
    """
    Detects exact duplicate rows in the dataset.

    Exact duplicates are rows where every column value is identical.  They
    inflate sample size without adding information and can bias model training.
    """

    @staticmethod
    def inspect(
        df: pd.DataFrame,
        log: DecisionLog,
    ) -> list[QualityIssue]:
        """
        Inspect dataset for exact duplicate rows.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.
        log : DecisionLog
            Decision log to append records to.

        Returns
        -------
        list[QualityIssue]
            A single-element list if duplicates are found, otherwise empty.
        """
        issues: list[QualityIssue] = []
        n_duplicates = int(df.duplicated().sum())

        if n_duplicates == 0:
            return issues

        dup_pct = n_duplicates / len(df)
        severity = Severity.WARNING if dup_pct < 0.05 else Severity.CRITICAL
        detail = (
            f"Dataset has {n_duplicates} exact duplicate rows "
            f"({dup_pct:.1%} of total rows)."
        )
        recommendation = "Drop duplicate rows before training to avoid data leakage."

        issues.append(
            QualityIssue(
                column=None,
                issue_type="duplicate_rows",
                severity=severity,
                detail=detail,
                metric=dup_pct,
                recommendation=recommendation,
            )
        )

        log.append(
            DecisionRecord(
                component="DataQualityAssessor.DuplicateInspector",
                action="duplicate_rows_detected",
                rationale=detail,
                severity=severity,
                data={
                    "n_duplicates": n_duplicates,
                    "duplicate_pct": round(dup_pct, 4),
                },
            )
        )

        return issues


# ---------------------------------------------------------------------------
# DataQualityAssessor — public facade
# ---------------------------------------------------------------------------


class DataQualityAssessor:
    """
    Phase 2: Comprehensive data quality assessment.

    Runs three sub-inspectors (missing, outliers, duplicates) and aggregates
    findings into a ``QualityReport`` dict.  Every finding is accompanied by
    a ``DecisionRecord`` explaining the issue and suggesting a remediation.

    The ``QualityReport`` is consumed by ``ExplainablePreprocessor`` (Phase 3)
    to select imputation and cleaning strategies.

    Parameters
    ----------
    config : MentorConfig
        Global configuration.

    Examples
    --------
    ::

        assessor = DataQualityAssessor(config)
        report = assessor.analyze(df, log)
        print(report["overall_severity"])
        for issue in report["issues"]:
            print(issue["detail"])
    """

    def __init__(self, config: MentorConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Analyzable interface
    # ------------------------------------------------------------------

    def analyze(
        self,
        df: pd.DataFrame,
        log: DecisionLog,
    ) -> dict[str, Any]:
        return self._analyze(df, log)

    assess = analyze

    def _analyze(
        self,
        df: pd.DataFrame,
        log: DecisionLog,
    ) -> dict[str, Any]:
        """
        Run a full quality audit on ``df``.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.  Must not be mutated.
        log : DecisionLog
            Decision log to append quality findings to.

        Returns
        -------
        dict[str, Any]
            A ``QualityReport`` with the following keys:

            - ``"n_rows"`` – int
            - ``"n_cols"`` – int
            - ``"issues"`` – list of issue dicts
            - ``"n_issues_by_severity"`` – dict mapping severity → count
            - ``"overall_severity"`` – worst severity found (or ``"ok"``)
            - ``"columns_to_drop"`` – list of columns recommended for removal
            - ``"columns_to_impute"`` – list of columns needing imputation

        Raises
        ------
        DataValidationError
            If ``df`` is empty.
        """
        if df.empty:
            raise DataValidationError(
                "DataFrame is empty — cannot assess quality.",
                context={"n_rows": 0, "n_cols": 0},
            )

        log.append(
            DecisionRecord(
                component="DataQualityAssessor",
                action="quality_assessment_start",
                rationale=(
                    f"Starting quality assessment on dataset with "
                    f"{len(df)} rows × {len(df.columns)} columns."
                ),
                data={},

                severity=Severity.INFO,
            )
        )

        all_issues: list[QualityIssue] = []

        # --- Run sub-inspectors ---
        all_issues.extend(
            MissingValueInspector.inspect(df, self.config, log)
        )
        all_issues.extend(OutlierInspector.inspect(df, log))
        all_issues.extend(DuplicateInspector.inspect(df, log))

        # --- Aggregate ---
        severity_counts: dict[str, int] = {
            Severity.INFO.value: 0,
            Severity.WARNING.value: 0,
            Severity.CRITICAL.value: 0,
        }
        for issue in all_issues:
            key = issue.severity.value  # always a str — JSON-safe
            severity_counts[key] = severity_counts.get(key, 0) + 1

        if severity_counts[Severity.CRITICAL.value] > 0:
            overall = Severity.CRITICAL.value
        elif severity_counts[Severity.WARNING.value] > 0:
            overall = Severity.WARNING.value
        elif severity_counts[Severity.INFO.value] > 0:
            overall = Severity.INFO.value
        else:
            overall = "ok"

        columns_to_drop = [
            i.column
            for i in all_issues
            if i.issue_type == "missing_critical"
            and i.column is not None
            and i.column != self.config.target_column
        ]
        columns_to_impute = [
            i.column
            for i in all_issues
            if i.issue_type in ("missing_high", "missing_low")
            and i.column is not None
            and i.column != self.config.target_column
        ]

        report: dict[str, Any] = {
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "issues": [i.to_dict() for i in all_issues],
            "n_issues_by_severity": severity_counts,
            "overall_severity": overall,
            "columns_to_drop": columns_to_drop,
            "columns_to_impute": columns_to_impute,
        }

        log.append(
            DecisionRecord(
                component="DataQualityAssessor",
                action="quality_assessment_complete",
                rationale=(
                    f"Quality assessment found {len(all_issues)} issues "
                    f"(overall severity: {overall}). "
                    f"Drop candidates: {columns_to_drop}. "
                    f"Impute candidates: {columns_to_impute}."
                ),
                severity=Severity.INFO,
                data={
                    "n_issues": len(all_issues),
                    "overall_severity": overall,
                    "columns_to_drop": columns_to_drop,
                    "columns_to_impute": columns_to_impute,
                },
            )
        )

        return report