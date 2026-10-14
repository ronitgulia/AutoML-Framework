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


