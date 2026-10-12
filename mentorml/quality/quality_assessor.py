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

