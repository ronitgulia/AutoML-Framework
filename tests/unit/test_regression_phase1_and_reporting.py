"""
tests/unit/test_regression_phase1_and_reporting.py
--------------------------------------------------
Targeted regression test suite verifying fixes across Phase 1 dataset profiling,
target analysis, correlation matrices, and Phase 8 HTML reporting.
"""

from __future__ import annotations

import json
import math
import numpy as np
import pandas as pd
import pytest

from mentorml.analysis.dataset_analyzer import (
    ColumnProfile,
    ColumnProfiler,
    CorrelationAnalyzer,
    DatasetAnalyzer,
    DtypeCategory,
    TargetAnalyzer,
)
from mentorml.config import MentorConfig
from mentorml.core.decision import DecisionLog, Severity
from mentorml.reporting.html_report import HTMLReportGenerator


class TestDtypeInferenceFixes:
    def test_all_null_numeric_is_numeric_not_boolean(self):
        series = pd.Series([np.nan, np.nan, np.nan], dtype=float, name="empty_num")
        cat = ColumnProfiler._infer_dtype_category(series, 0, 0.0, MentorConfig())
        assert cat == DtypeCategory.NUMERIC

    def test_string_dtype_is_categorical_or_text(self):
        series = pd.Series(["apple", "banana", "cherry"] * 5, dtype="string", name="fruits")
        cat = ColumnProfiler._infer_dtype_category(series, 3, 3 / 15, MentorConfig())
        assert cat in (DtypeCategory.CATEGORICAL, DtypeCategory.TEXT)


class TestNumericStatsAndProfileSerialization:
    def test_single_value_std_is_not_nan(self):
        series = pd.Series([42.0], name="single_val")
        profile = ColumnProfile(
            name="single_val",
            dtype="float64",
            dtype_category=DtypeCategory.NUMERIC,
            n_rows=1,
            n_missing=0,
            missing_pct=0.0,
            n_unique=1,
            cardinality_ratio=1.0,
            is_constant=True,
            is_id_like=False,
        )
        ColumnProfiler._add_numeric_stats(profile, series)
        assert profile.std is not None
        assert not math.isnan(profile.std)
        # Verify JSON serializability
        d = profile.to_dict()
        assert d["std"] == 0.0 or d["std"] is None
        assert "iqr" in d
        assert json.dumps(d) is not None

    def test_iqr_field_included_in_to_dict(self):
        profile = ColumnProfile(
            name="test_col",
            dtype="float64",
            dtype_category=DtypeCategory.NUMERIC,
            n_rows=10,
            n_missing=0,
            missing_pct=0.0,
            n_unique=5,
            cardinality_ratio=0.5,
            is_constant=False,
            is_id_like=False,
            iqr=5.2,
        )
        d = profile.to_dict()
        assert "iqr" in d
        assert d["iqr"] == 5.2


class TestNumericIdDetection:
    def test_integer_id_column_flagged_as_id_like(self):
        # 100 unique sequential integers
        series = pd.Series(list(range(1, 101)), name="user_id", dtype=int)
        profile = ColumnProfiler.profile(series, n_rows=100, config=MentorConfig())
        assert profile.is_id_like is True

    def test_continuous_measurement_not_flagged_as_id_like(self):
        rng = np.random.default_rng(123)
        series = pd.Series(rng.normal(50.0, 10.0, 100), name="temperature")
        profile = ColumnProfiler.profile(series, n_rows=100, config=MentorConfig())
        assert profile.is_id_like is False


class TestTargetAnalyzerFixes:
    def test_empty_target_does_not_crash(self):
        df = pd.DataFrame({"target": [np.nan] * 20, "x": range(20)})
        log = DecisionLog()
        res = TargetAnalyzer.analyze(df, "target", MentorConfig(), log)
        assert res["n_classes"] == 0
        assert res["imbalance_ratio"] is None
        assert any(r.severity == Severity.CRITICAL for r in log)

    def test_single_class_target_emits_critical(self):
        df = pd.DataFrame({"target": [1] * 30, "x": range(30)})
        log = DecisionLog()
        res = TargetAnalyzer.analyze(df, "target", MentorConfig(), log)
        assert res["n_classes"] == 1
        assert any(r.severity == Severity.CRITICAL and r.action == "target_single_class" for r in log)

    def test_multiclass_balanced_does_not_flag_warning(self):
        # 10 classes with 10 instances each -> 100% balanced
        df = pd.DataFrame({"target": [i % 10 for i in range(100)], "x": range(100)})
        log = DecisionLog()
        res = TargetAnalyzer.analyze(df, "target", MentorConfig(), log)
        assert res["n_classes"] == 10
        # Should be INFO, not WARNING
        balance_records = [r for r in log if r.action == "analyze_class_balance"]
        assert len(balance_records) == 1
        assert balance_records[0].severity == Severity.INFO

    def test_imbalance_ratio_json_serializable(self):
        df = pd.DataFrame({"target": [0] * 50 + [1] * 50, "x": range(100)})
        log = DecisionLog()
        res = TargetAnalyzer.analyze(df, "target", MentorConfig(), log)
        assert json.dumps(res) is not None


class TestCorrelationAnalyzerFixes:
    def test_constant_numeric_in_correlation_does_not_emit_nan_in_dict(self):
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "const": [5.0] * 10,
        })
        log = DecisionLog()
        res = CorrelationAnalyzer.analyze(df, MentorConfig(), log)
        # JSON dump must not raise ValueError for NaN
        assert json.dumps(res) is not None
        # const column correlations should be None, not NaN
        assert res["correlation_matrix"]["const"]["a"] is None


class TestDatasetAnalyzerFacadeFixes:
    def test_target_column_not_in_candidate_drop_lists(self):
        df = pd.DataFrame({
            "feature": range(20),
            "target": [np.nan] * 18 + [1.0, 0.0],  # 90% missing
        })
        config = MentorConfig(target_column="target", missing_threshold=0.5)
        log = DecisionLog()
        analyzer = DatasetAnalyzer(config)
        res = analyzer.analyze(df, log)
        assert "target" not in res["flagged_columns"]["high_missing"]

    def test_n_columns_flagged_avoids_double_counting(self):
        df = pd.DataFrame({
            "valid": range(20),
            "all_null": [np.nan] * 20,  # is_constant=True AND missing_pct=1.0
        })
        config = MentorConfig(missing_threshold=0.5)
        log = DecisionLog()
        analyzer = DatasetAnalyzer(config)
        res = analyzer.analyze(df, log)
        # all_null is in constant and high_missing, but total flagged should be 1
        assert res["n_columns_flagged"] == 1


class TestHTMLReportOverviewFix:
    def test_dataset_overview_renders_n_cols_from_analyzer_output(self):
        df = pd.DataFrame({
            "a": range(20),
            "b": range(20),
            "target": [0, 1] * 10,
        })
        config = MentorConfig(target_column="target")
        log = DecisionLog()
        analyzer = DatasetAnalyzer(config)
        analysis_result = analyzer.analyze(df, log)

        reporter = HTMLReportGenerator(project_name="Test Project")
        html_out = reporter._render_dataset_overview(analysis_result)

        assert "Total Columns" in html_out
        assert ">3<" in html_out  # 3 columns
        assert "N/A" not in html_out
        assert "classification" in html_out.lower()
