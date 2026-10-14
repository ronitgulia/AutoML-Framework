"""
tests/conftest.py
-----------------
Shared pytest fixtures available to all test modules.

Fixtures here are auto-discovered by pytest — no imports needed in test files.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mentorml.config import MentorConfig
from mentorml.core.decision import DecisionLog, DecisionRecord, Severity


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> MentorConfig:
    """Return a ``MentorConfig`` with all defaults."""
    return MentorConfig()


@pytest.fixture
def classification_config() -> MentorConfig:
    """Return a ``MentorConfig`` configured for a binary classification task."""
    return MentorConfig(target_column="target", task_type="classification")


@pytest.fixture
def regression_config() -> MentorConfig:
    """Return a ``MentorConfig`` configured for a regression task."""
    return MentorConfig(target_column="price", task_type="regression")


# ---------------------------------------------------------------------------
# Decision layer fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_log() -> DecisionLog:
    """Return a fresh, empty ``DecisionLog``."""
    return DecisionLog()


@pytest.fixture
def sample_record() -> DecisionRecord:
    """Return a minimal ``DecisionRecord`` for testing."""
    return DecisionRecord(
        component="TestComponent",
        action="test_action",
        rationale="This is a test rationale explaining the decision.",
        data={"key": "value", "count": 42},
        severity=Severity.INFO,
    )


@pytest.fixture
def populated_log(sample_record: DecisionRecord) -> DecisionLog:
    """Return a ``DecisionLog`` pre-populated with one record of each severity."""
    log = DecisionLog()
    log.append(sample_record)
    log.append(
        DecisionRecord(
            component="TestComponent",
            action="warn_action",
            rationale="A warning-level decision.",
            data={},
            severity=Severity.WARNING,
        )
    )
    log.append(
        DecisionRecord(
            component="OtherComponent",
            action="critical_action",
            rationale="A critical-level decision.",
            data={"severity": "high"},
            severity=Severity.CRITICAL,
        )
    )
    return log


# ---------------------------------------------------------------------------
# DataFrame fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_clean_df() -> pd.DataFrame:
    """
    Return a small, clean DataFrame with no issues.

    Useful as a baseline for analysis tests.
    """
    return pd.DataFrame(
        {
            "age": [25, 32, 45, 28, 37],
            "income": [50000.0, 72000.0, 91000.0, 43000.0, 65000.0],
            "city": ["London", "Paris", "London", "Berlin", "Paris"],
            "target": [0, 1, 1, 0, 1],
        }
    )


@pytest.fixture
def df_with_missing() -> pd.DataFrame:
    """Return a DataFrame with intentional missing values."""
    return pd.DataFrame(
        {
            "age": [25, None, 45, 28, None],
            "income": [50000.0, 72000.0, None, None, 65000.0],
            "city": ["London", "Paris", None, "Berlin", "Paris"],
            "target": [0, 1, 1, 0, 1],
        }
    )


@pytest.fixture
def df_empty() -> pd.DataFrame:
    """Return a completely empty DataFrame."""
    return pd.DataFrame()