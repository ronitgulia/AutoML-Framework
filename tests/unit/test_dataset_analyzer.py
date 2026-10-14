"""
tests/unit/test_dataset_analyzer.py
-------------------------------------
Unit tests for Phase 1: DatasetAnalyzer, ColumnProfiler, TargetAnalyzer,
CorrelationAnalyzer, ColumnProfile, and DtypeCategory.

Test philosophy
---------------
- Each internal sub-analyzer is tested independently before testing the
  public ``DatasetAnalyzer`` facade.
- We test edge cases and boundary conditions, not just happy paths.
- We verify that the correct ``DecisionRecord`` objects are emitted —
  not just that output dicts have the right shape.
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

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
from mentorml.core.exceptions import DataValidationError, InsufficientDataError


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> MentorConfig:
    return MentorConfig()


@pytest.fixture
def config_with_target() -> MentorConfig:
    return MentorConfig(target_column="target", task_type="classification")


@pytest.fixture
def log() -> DecisionLog:
    return DecisionLog()


@pytest.fixture
def titanic_like() -> pd.DataFrame:
    """Realistic-ish DataFrame with numeric, categorical, missing, skewed columns."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "passenger_id": [f"P{i:04d}" for i in range(1, n + 1)],  # string ID-like
        "survived":     rng.integers(0, 2, n),        # binary target
        "pclass":       rng.choice([1, 2, 3], n),     # low-cardinality numeric
        "age":          np.where(rng.random(n) < 0.2, np.nan, rng.normal(30, 12, n)),  # 20% missing
        "fare":         rng.exponential(50, n),        # right-skewed
        "embarked":     rng.choice(["S", "C", "Q", None], n),  # categorical w/ missing
        "constant_col": [1] * n,                      # constant
        "sex":          rng.choice(["male", "female"], n),
    })


@pytest.fixture
def small_balanced_df() -> pd.DataFrame:
    """Clean, small DataFrame for basic happy-path tests."""
    return pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "b": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        "cat": ["x", "y", "x", "y", "x", "y", "x", "y", "x", "y"],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })


# ===========================================================================
# DtypeCategory
# ===========================================================================


class TestDtypeCategory:
    def test_values_are_strings(self) -> None:
        import json
        for cat in DtypeCategory:
            assert json.dumps(cat) is not None  # JSON-serialisable

    def test_numeric_value(self) -> None:
        assert DtypeCategory.NUMERIC.value == "numeric"

    def test_categorical_value(self) -> None:
        assert DtypeCategory.CATEGORICAL.value == "categorical"


# ===========================================================================
# ColumnProfiler
# ===========================================================================


class TestColumnProfiler:
    """Tests for ColumnProfiler._infer_dtype_category and .profile()."""

    def test_numeric_column(self, default_config: MentorConfig) -> None:
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="val")
        p = ColumnProfiler.profile(s, n_rows=5, config=default_config)
        assert p.dtype_category == DtypeCategory.NUMERIC
        assert p.mean is not None
        assert p.skewness is not None

    def test_categorical_column(self, default_config: MentorConfig) -> None:
        s = pd.Series(["a", "b", "a", "c", "b"] * 10, name="cat")
        p = ColumnProfiler.profile(s, n_rows=50, config=default_config)
        assert p.dtype_category == DtypeCategory.CATEGORICAL
        assert p.mean is None  # no numeric stats

    def test_boolean_dtype(self, default_config: MentorConfig) -> None:
        s = pd.Series([True, False, True, True, False], name="flag")
        p = ColumnProfiler.profile(s, n_rows=5, config=default_config)
        assert p.dtype_category == DtypeCategory.BOOLEAN

    def test_binary_int_is_boolean(self, default_config: MentorConfig) -> None:
        s = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], name="y")
        p = ColumnProfiler.profile(s, n_rows=10, config=default_config)
        assert p.dtype_category == DtypeCategory.BOOLEAN

    def test_datetime_column(self, default_config: MentorConfig) -> None:
        s = pd.Series(pd.date_range("2023-01-01", periods=5), name="ts")
        p = ColumnProfiler.profile(s, n_rows=5, config=default_config)
        assert p.dtype_category == DtypeCategory.DATETIME
        assert p.mean is None

    def test_missing_values_counted(self, default_config: MentorConfig) -> None:
        s = pd.Series([1.0, None, 3.0, None, 5.0], name="x")
        p = ColumnProfiler.profile(s, n_rows=5, config=default_config)
        assert p.n_missing == 2
        assert abs(p.missing_pct - 0.4) < 1e-6

    def test_constant_column(self, default_config: MentorConfig) -> None:
        s = pd.Series([7] * 20, name="const")
        p = ColumnProfiler.profile(s, n_rows=20, config=default_config)
        assert p.is_constant is True
        assert p.is_id_like is False

    def test_id_like_column(self, default_config: MentorConfig) -> None:
        # String ID columns (e.g., UUIDs, slugs) should be flagged as ID-like
        s = pd.Series([f"id_{i}" for i in range(100)], name="user_id")
        p = ColumnProfiler.profile(s, n_rows=100, config=default_config)
        assert p.is_id_like is True
        assert p.cardinality_ratio == pytest.approx(1.0)

    def test_outlier_detection(self, default_config: MentorConfig) -> None:
        # Normal data with two extreme outliers
        base = list(range(1, 51))
        outliers = [500, -500]
        s = pd.Series(base + outliers, name="outliers")
        p = ColumnProfiler.profile(s, n_rows=52, config=default_config)
        assert p.n_outliers == 2
        assert p.outlier_pct is not None and p.outlier_pct > 0

    def test_skewness_right(self, default_config: MentorConfig) -> None:
        rng = np.random.default_rng(0)
        s = pd.Series(rng.exponential(1, 200), name="exp")
        p = ColumnProfiler.profile(s, n_rows=200, config=default_config)
        assert p.skewness is not None
        assert p.skewness > 0  # exponential is right-skewed

    def test_top_values_capped_at_10(self, default_config: MentorConfig) -> None:
        s = pd.Series(list(range(100)), name="many_unique")
        p = ColumnProfiler.profile(s, n_rows=100, config=default_config)
        assert len(p.top_values) <= 10

    def test_to_dict_is_serialisable(self, default_config: MentorConfig) -> None:
        import json
        s = pd.Series([1.0, 2.0, None, 4.0, 5.0] * 4, name="v")
        p = ColumnProfiler.profile(s, n_rows=20, config=default_config)
        d = p.to_dict()
        json.dumps(d)  # must not raise

    def test_zero_iqr_no_outliers(self, default_config: MentorConfig) -> None:
        """Constant numeric column → IQR=0 → no outliers flagged."""
        s = pd.Series([5.0] * 20, name="c")
        p = ColumnProfiler.profile(s, n_rows=20, config=default_config)
        assert p.n_outliers == 0


# ===========================================================================
# TargetAnalyzer
# ===========================================================================


class TestTargetAnalyzer:
    """Tests for task type inference and class imbalance detection."""

    def test_binary_int_target_is_classification(self, default_config: MentorConfig, log: DecisionLog) -> None:
        df = pd.DataFrame({"x": [1.0] * 50, "y": [0, 1] * 25})
        result = TargetAnalyzer.analyze(df, "y", default_config, log)
        assert result["task_type"] == "classification"

    def test_continuous_numeric_is_regression(self, default_config: MentorConfig, log: DecisionLog) -> None:
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"x": rng.random(200), "price": rng.normal(100, 20, 200)})
        result = TargetAnalyzer.analyze(df, "price", default_config, log)
        assert result["task_type"] == "regression"

    def test_string_target_is_classification(self, default_config: MentorConfig, log: DecisionLog) -> None:
        df = pd.DataFrame({"x": range(50), "label": ["cat", "dog"] * 25})
        result = TargetAnalyzer.analyze(df, "label", default_config, log)
        assert result["task_type"] == "classification"

    def test_explicit_config_task_type_respected(self, log: DecisionLog) -> None:
        config = MentorConfig(target_column="y", task_type="regression")
        df = pd.DataFrame({"y": [0, 1] * 50})
        result = TargetAnalyzer.analyze(df, "y", config, log)
        assert result["task_type"] == "regression"
        # Should emit a record about using explicit config
        records = log.filter_by_component("TargetAnalyzer")
        assert any("explicit" in r.rationale.lower() or "explicitly" in r.rationale.lower() for r in records)

    def test_severe_imbalance_emits_critical(self, default_config: MentorConfig, log: DecisionLog) -> None:
        # 98% class 0, 2% class 1
        targets = [0] * 98 + [1] * 2
        df = pd.DataFrame({"x": range(100), "y": targets})
        TargetAnalyzer.analyze(df, "y", default_config, log)
        critical = log.filter(Severity.CRITICAL)
        assert any("balance" in r.action for r in critical)

    def test_moderate_imbalance_emits_warning(self, default_config: MentorConfig, log: DecisionLog) -> None:
        targets = [0] * 85 + [1] * 15
        df = pd.DataFrame({"x": range(100), "y": targets})
        TargetAnalyzer.analyze(df, "y", default_config, log)
        warnings = log.filter(Severity.WARNING)
        assert any("balance" in r.action for r in warnings)

    def test_balanced_target_emits_info_only(self, default_config: MentorConfig, log: DecisionLog) -> None:
        targets = [0] * 50 + [1] * 50
        df = pd.DataFrame({"x": range(100), "y": targets})
        TargetAnalyzer.analyze(df, "y", default_config, log)
        # No critical or warning for balance
        assert not any(r.action == "analyze_class_balance" and r.severity == Severity.CRITICAL for r in log)
        assert not any(r.action == "analyze_class_balance" and r.severity == Severity.WARNING for r in log)

    def test_class_counts_in_result(self, default_config: MentorConfig, log: DecisionLog) -> None:
        targets = [0] * 60 + [1] * 40
        df = pd.DataFrame({"x": range(100), "y": targets})
        result = TargetAnalyzer.analyze(df, "y", default_config, log)
        assert result["class_counts"] is not None
        assert result["n_classes"] == 2

    def test_imbalance_ratio_calculated(self, default_config: MentorConfig, log: DecisionLog) -> None:
        targets = [0] * 80 + [1] * 20
        df = pd.DataFrame({"x": range(100), "y": targets})
        result = TargetAnalyzer.analyze(df, "y", default_config, log)
        assert result["imbalance_ratio"] == pytest.approx(4.0)


# ===========================================================================
# CorrelationAnalyzer
# ===========================================================================


class TestCorrelationAnalyzer:
    """Tests for pairwise correlation analysis."""

    def test_high_correlation_detected(self, log: DecisionLog) -> None:
        config = MentorConfig(correlation_threshold=0.9)
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0] * 10,
            "b": [1.1, 2.1, 3.1, 4.1, 5.1] * 10,  # nearly identical to a
        })
        result = CorrelationAnalyzer.analyze(df, config, log)
        assert len(result["high_correlation_pairs"]) >= 1
        assert any("a" in p["col_a"] or "a" in p["col_b"] for p in result["high_correlation_pairs"])

    def test_low_correlation_not_flagged(self, log: DecisionLog) -> None:
        config = MentorConfig(correlation_threshold=0.95)
        rng = np.random.default_rng(7)
        df = pd.DataFrame({
            "a": rng.random(100),
            "b": rng.random(100),  # independent → low correlation
        })
        result = CorrelationAnalyzer.analyze(df, config, log)
        # Very unlikely to exceed 0.95 with random data
        assert len(result["high_correlation_pairs"]) == 0

    def test_single_numeric_column_skips(self, log: DecisionLog) -> None:
        config = MentorConfig()
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "cat": ["x", "y"] * 2 + ["x"]})
        result = CorrelationAnalyzer.analyze(df, config, log)
        assert result["high_correlation_pairs"] == []
        # Should emit a skip record
        records = log.filter_by_component("CorrelationAnalyzer")
        assert any("skip" in r.action for r in records)

    def test_correlation_matrix_in_result(self, log: DecisionLog) -> None:
        config = MentorConfig()
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0] * 4,
            "b": [2.0, 3.0, 4.0, 5.0, 6.0] * 4,
        })
        result = CorrelationAnalyzer.analyze(df, config, log)
        assert "a" in result["correlation_matrix"]

    def test_target_excluded_from_correlation(self, log: DecisionLog) -> None:
        config = MentorConfig(correlation_threshold=0.9)
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0] * 10,
            "b": [1.1, 2.1, 3.1, 4.1, 5.1] * 10,
            "target": [0, 1] * 25,
        })
        result = CorrelationAnalyzer.analyze(df, config, log, target_col="target")
        # target should not appear as a flagged feature
        for pair in result["high_correlation_pairs"]:
            assert pair["col_a"] != "target"
            assert pair["col_b"] != "target"


# ===========================================================================
# DatasetAnalyzer (public facade)
# ===========================================================================


class TestDatasetAnalyzerValidation:
    """Tests for input validation guards."""

    def test_empty_df_raises(self, default_config: MentorConfig, log: DecisionLog) -> None:
        analyzer = DatasetAnalyzer(default_config)
        with pytest.raises(DataValidationError):
            analyzer.analyze(pd.DataFrame(), log)

    def test_too_few_rows_raises(self, default_config: MentorConfig, log: DecisionLog) -> None:
        analyzer = DatasetAnalyzer(default_config)
        small = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(InsufficientDataError):
            analyzer.analyze(small, log)


class TestDatasetAnalyzerResults:
    """Tests for the structured result dict returned by analyze()."""

    def test_result_has_required_keys(
        self, titanic_like: pd.DataFrame, log: DecisionLog
    ) -> None:
        config = MentorConfig(target_column="survived")
        analyzer = DatasetAnalyzer(config)
        result = analyzer.analyze(titanic_like, log)
        required = {
            "n_rows", "n_cols", "memory_mb",
            "n_duplicates", "duplicate_pct",
            "column_profiles", "target", "correlation",
            "flagged_columns", "n_columns_flagged",
        }
        assert required.issubset(result.keys())

    def test_n_rows_correct(
        self, titanic_like: pd.DataFrame, log: DecisionLog
    ) -> None:
        config = MentorConfig(target_column="survived")
        analyzer = DatasetAnalyzer(config)
        result = analyzer.analyze(titanic_like, log)
        assert result["n_rows"] == len(titanic_like)

    def test_column_profiles_all_columns_present(
        self, titanic_like: pd.DataFrame, log: DecisionLog
    ) -> None:
        config = MentorConfig(target_column="survived")
        analyzer = DatasetAnalyzer(config)
        result = analyzer.analyze(titanic_like, log)
        assert set(result["column_profiles"].keys()) == set(titanic_like.columns)

    def test_constant_column_flagged(
        self, titanic_like: pd.DataFrame, log: DecisionLog
    ) -> None:
        config = MentorConfig(target_column="survived")
        analyzer = DatasetAnalyzer(config)
        result = analyzer.analyze(titanic_like, log)
        assert "constant_col" in result["flagged_columns"]["constant"]

    def test_id_like_column_flagged(
        self, titanic_like: pd.DataFrame, log: DecisionLog
    ) -> None:
        config = MentorConfig(target_column="survived")
        analyzer = DatasetAnalyzer(config)
        result = analyzer.analyze(titanic_like, log)
        assert "passenger_id" in result["flagged_columns"]["id_like"]

    def test_target_result_present(
        self, titanic_like: pd.DataFrame, log: DecisionLog
    ) -> None:
        config = MentorConfig(target_column="survived")
        analyzer = DatasetAnalyzer(config)
        result = analyzer.analyze(titanic_like, log)
        assert result["target"] is not None
        assert result["target"]["task_type"] == "classification"

    def test_missing_target_column_emits_warning(
        self, titanic_like: pd.DataFrame, log: DecisionLog
    ) -> None:
        config = MentorConfig(target_column="nonexistent_col")
        analyzer = DatasetAnalyzer(config)
        result = analyzer.analyze(titanic_like, log)
        assert result["target"] is None
        # Should have emitted a warning about missing target
        warnings = [r for r in log if r.severity == Severity.WARNING and "target" in r.action]
        assert len(warnings) >= 1

    def test_log_has_records_after_analyze(
        self, small_balanced_df: pd.DataFrame, log: DecisionLog
    ) -> None:
        config = MentorConfig(target_column="target")
        analyzer = DatasetAnalyzer(config)
        analyzer.analyze(small_balanced_df, log)
        assert len(log) > 0

    def test_analyze_emits_start_and_complete_records(
        self, small_balanced_df: pd.DataFrame, log: DecisionLog
    ) -> None:
        config = MentorConfig(target_column="target")
        analyzer = DatasetAnalyzer(config)
        analyzer.analyze(small_balanced_df, log)
        actions = {r.action for r in log}
        assert "start_analysis" in actions
        assert "analysis_complete" in actions

    def test_duplicate_detection(self, log: DecisionLog) -> None:
        df = pd.DataFrame({
            "a": [1, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "b": [1, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        })
        config = MentorConfig()
        analyzer = DatasetAnalyzer(config)
        result = analyzer.analyze(df, log)
        assert result["n_duplicates"] == 1

    def test_protocol_compliance(self, default_config: MentorConfig) -> None:
        """DatasetAnalyzer must satisfy the Analyzable protocol."""
        from mentorml.core.protocols import Analyzable
        analyzer = DatasetAnalyzer(default_config)
        assert isinstance(analyzer, Analyzable)