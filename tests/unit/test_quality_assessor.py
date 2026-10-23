"""
tests.unit.test_quality_assessor
--------------------------------
Unit test suite for DataQualityAssessor and quality sub-inspectors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mentorml.config import MentorConfig
from mentorml.core.decision import DecisionLog, Severity
from mentorml.quality.quality_assessor import (
    DataQualityAssessor,
    DuplicateInspector,
    MissingValueInspector,
    OutlierInspector,
    QualityIssue,
)


def test_quality_issue_serialization():
    """Verify QualityIssue serializes severity to string without json error."""
    issue = QualityIssue(
        column="col_x",
        issue_type="missing_high",
        severity=Severity.WARNING,
        detail="Missing 25% values",
        metric=0.25,
        recommendation="Impute with median",
    )
    d = issue.to_dict()
    assert d["column"] == "col_x"
    assert d["severity"] == "WARNING"
    assert d["metric"] == 0.25


def test_missing_inspector_thresholds():
    """Verify MissingValueInspector tags issues according to config thresholds."""
    df = pd.DataFrame({
        "clean": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "low_missing": [1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan],  # 10% missing -> warning
        "critical_missing": [1, 2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # 80% -> critical
    })
    cfg = MentorConfig(missing_threshold=0.4)
    log = DecisionLog()
    issues = MissingValueInspector.inspect(df, cfg, log)

    cols = {i.column: i for i in issues}
    assert "clean" not in cols
    assert "low_missing" in cols
    assert cols["low_missing"].severity == Severity.WARNING
    assert "critical_missing" in cols
    assert cols["critical_missing"].severity == Severity.CRITICAL


def test_duplicate_inspector():
    """Verify DuplicateInspector detects duplicate rows."""
    df = pd.DataFrame({
        "a": [1, 1, 2],
        "b": [10, 10, 20],
    })
    cfg = MentorConfig()
    log = DecisionLog()
    issues = DuplicateInspector.inspect(df, log)
    assert len(issues) == 1
    assert issues[0].issue_type == "duplicate_rows"
    assert issues[0].metric == pytest.approx(1 / 3)


def test_outlier_inspector():
    """Verify OutlierInspector detects numeric outliers via IQR rule."""
    # 20 normal values between 10 and 20, plus 5 extreme outliers
    vals = list(range(10, 30)) + [1000, 2000, 3000, 4000, 5000]
    df = pd.DataFrame({"metric_col": vals})
    log = DecisionLog()
    issues = OutlierInspector.inspect(df, log)
    assert len(issues) == 1
    assert issues[0].column == "metric_col"
    assert issues[0].severity == Severity.WARNING


def test_assessor_guards_target_column_from_drop():
    """Verify target column with high missingness is NEVER added to columns_to_drop."""
    df = pd.DataFrame({
        "feature_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "target": [1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    })
    cfg = MentorConfig(target_column="target", missing_threshold=0.3)
    log = DecisionLog()
    assessor = DataQualityAssessor(cfg)
    report = assessor.assess(df, log)

    assert "target" not in report["columns_to_drop"]
    assert report["n_rows"] == 10
    assert report["n_cols"] == 2
