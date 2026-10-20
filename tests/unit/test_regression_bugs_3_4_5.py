"""
tests/unit/test_regression_bugs_3_4_5.py
------------------------------------------
Regression tests for Bug #3, #4, and #5 identified in the audit.

Bug #3 - pipeline.py passed `data=` to ModelExporter.export() which
         expects `metadata=`.  This caused a TypeError at runtime.

Bug #4 - DatasetAnalyzer._analyze_duplicates() emitted Severity.WARNING
         for duplicate_pct >= 0.05, where the correct level is CRITICAL.
         Also used strict `>` instead of `>=`, off-by-one at exactly 5%.

Bug #5 - QualityIssue.to_dict() returned the raw Severity enum object
         under the "severity" key, causing json.dumps() to raise TypeError.
         The n_issues_by_severity dict also used enum objects as keys.
"""

from __future__ import annotations

import json
import os
import tempfile

import pandas as pd
import pytest

from mentorml.config import MentorConfig
from mentorml.core.decision import DecisionLog, Severity
from mentorml.quality.quality_assessor import DataQualityAssessor, QualityIssue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age":    list(range(n)),
            "income": [float(i * 1000) for i in range(n)],
            "city":   ["London", "Paris"] * (n // 2),
            "target": [i % 2 for i in range(n)],
        }
    )


def _df_with_dups(total: int, n_dups: int) -> pd.DataFrame:
    base = _make_df(total - n_dups)
    dup_rows = base.iloc[:n_dups].copy()
    return pd.concat([base, dup_rows], ignore_index=True)


# ---------------------------------------------------------------------------
# Bug #3 - ModelExporter.export() kwarg mismatch (data= vs metadata=)
# ---------------------------------------------------------------------------

class TestBug3ExporterKwarg:

    def test_export_accepts_metadata_kwarg(self):
        from mentorml.export.model_exporter import ModelExporter
        from sklearn.linear_model import LogisticRegression
        from mentorml.preprocessing.preprocessor import ExplainablePreprocessor

        log = DecisionLog()
        cfg = MentorConfig(target_column="target")
        df = _make_df(30)
        X = df.drop(columns=["target"])
        y = df["target"]

        prep = ExplainablePreprocessor(cfg)
        prep.fit(X, log)
        X_proc = prep.transform(X, log)

        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X_proc, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = ModelExporter("test").export(
                model=model,
                preprocessor=prep,
                feature_names=list(X_proc.columns),
                log=log,
                output_dir=tmpdir,
                metadata={"task_type": "classification", "best_model": "LogisticRegression"},
            )
            assert os.path.isdir(export_dir)
            with open(os.path.join(export_dir, "manifest.json")) as fh:
                manifest = json.load(fh)
            assert manifest.get("task_type") == "classification"
            assert manifest.get("best_model") == "LogisticRegression"


# ---------------------------------------------------------------------------
# Bug #4 - DatasetAnalyzer duplicate-row severity is inverted
# ---------------------------------------------------------------------------

class TestBug4DuplicateSeverity:

    def test_high_dup_pct_emits_critical(self):
        from mentorml.analysis.dataset_analyzer import DatasetAnalyzer
        df = _df_with_dups(total=20, n_dups=2)   # 10% dups
        log = DecisionLog()
        DatasetAnalyzer(MentorConfig(target_column="target")).analyze(df, log)
        rec = next(r for r in log if r.action == "detect_duplicates")
        assert rec.severity == Severity.CRITICAL

    def test_exact_five_pct_boundary_is_critical(self):
        from mentorml.analysis.dataset_analyzer import DatasetAnalyzer
        df = _df_with_dups(total=20, n_dups=1)   # exactly 5%
        log = DecisionLog()
        DatasetAnalyzer(MentorConfig(target_column="target")).analyze(df, log)
        rec = next(r for r in log if r.action == "detect_duplicates")
        assert rec.severity == Severity.CRITICAL, "5% boundary must be CRITICAL (>= not >)"

    def test_low_dup_pct_emits_warning(self):
        from mentorml.analysis.dataset_analyzer import DatasetAnalyzer
        base = pd.DataFrame({
            "age":    list(range(100)), "income": [float(i) for i in range(100)],
            "city":   ["London"] * 100, "target": [i % 2 for i in range(100)],
        })
        df = pd.concat([base, base.iloc[:1]], ignore_index=True)  # 1% dups
        log = DecisionLog()
        DatasetAnalyzer(MentorConfig(target_column="target")).analyze(df, log)
        rec = next(r for r in log if r.action == "detect_duplicates")
        assert rec.severity == Severity.WARNING

    def test_no_duplicates_emits_info(self):
        from mentorml.analysis.dataset_analyzer import DatasetAnalyzer
        log = DecisionLog()
        DatasetAnalyzer(MentorConfig(target_column="target")).analyze(_make_df(20), log)
        rec = next(r for r in log if r.action == "detect_duplicates")
        assert rec.severity == Severity.INFO

    def test_analyzer_and_inspector_agree_at_boundary(self):
        from mentorml.analysis.dataset_analyzer import DatasetAnalyzer
        from mentorml.quality.quality_assessor import DuplicateInspector
        df = _df_with_dups(total=20, n_dups=1)   # exactly 5%
        log_a, log_b = DecisionLog(), DecisionLog()
        DatasetAnalyzer(MentorConfig(target_column="target")).analyze(df, log_a)
        issues = DuplicateInspector.inspect(df, log_b)
        analyzer_sev = next(r.severity for r in log_a if r.action == "detect_duplicates")
        if issues:
            assert analyzer_sev == issues[0].severity, "Threshold mismatch between analyzer and inspector"


# ---------------------------------------------------------------------------
# Bug #5 - QualityIssue.to_dict() returns raw Severity enum (not .value)
# ---------------------------------------------------------------------------

class TestBug5EnumSerialization:

    def _issue(self, sev: Severity = Severity.WARNING) -> QualityIssue:
        return QualityIssue(
            column="income", issue_type="missing_high", severity=sev,
            detail="40% missing", metric=0.4, recommendation="Drop it.",
        )

    def test_severity_is_string_in_to_dict(self):
        d = self._issue(Severity.WARNING).to_dict()
        assert isinstance(d["severity"], str), f"Expected str, got {type(d['severity'])}"
        assert d["severity"] == Severity.WARNING.value

    @pytest.mark.parametrize("sev", list(Severity))
    def test_to_dict_is_json_serializable(self, sev: Severity):
        json.dumps(self._issue(sev).to_dict())   # must not raise

    def test_full_report_is_json_serializable(self):
        df = pd.DataFrame({
            "age":    [None, 25, 30, 35, 40] * 4,
            "income": [50000.0] * 20,
            "city":   ["London"] * 20,
            "target": [0, 1] * 10,
        })
        report = DataQualityAssessor(MentorConfig(target_column="target")).analyze(df, DecisionLog())
        json.dumps(report)   # must not raise TypeError

    def test_n_issues_by_severity_has_string_keys(self):
        df = pd.DataFrame({
            "age":    [None, 25, 30, 35, 40] * 4,
            "income": [50000.0] * 20,
            "city":   ["London"] * 20,
            "target": [0, 1] * 10,
        })
        report = DataQualityAssessor(MentorConfig(target_column="target")).analyze(df, DecisionLog())
        for key in report["n_issues_by_severity"]:
            assert isinstance(key, str), f"Key must be str, got {type(key)}: {key!r}"

    def test_overall_severity_is_string(self):
        report = DataQualityAssessor(MentorConfig(target_column="target")).analyze(_make_df(20), DecisionLog())
        assert isinstance(report["overall_severity"], str)
