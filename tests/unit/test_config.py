"""
tests.unit.test_config
----------------------
Unit test suite for MentorConfig validation and parameter defaults.
"""

from __future__ import annotations

import pytest

from mentorml.config import MentorConfig
from mentorml.core.exceptions import ConfigurationError


def test_config_defaults():
    """Verify default values meet framework expectations."""
    cfg = MentorConfig()
    assert cfg.target_column is None
    assert cfg.task_type is None
    assert cfg.missing_threshold == 0.4
    assert cfg.cardinality_threshold == 50
    assert cfg.correlation_threshold == 0.95
    assert cfg.variance_threshold == 0.01
    assert cfg.test_size == 0.2
    assert cfg.cv_folds == 5
    assert cfg.random_state == 42
    assert cfg.n_jobs == -1
    assert cfg.verbose is True
    assert cfg.report_output_dir == "."
    assert cfg.export_output_dir == "."


def test_config_custom_valid():
    """Verify valid custom parameters are stored properly."""
    cfg = MentorConfig(
        target_column="target",
        task_type="classification",
        missing_threshold=0.5,
        cardinality_threshold=20,
        correlation_threshold=0.90,
        variance_threshold=0.05,
        test_size=0.3,
        cv_folds=3,
        random_state=123,
    )
    assert cfg.target_column == "target"
    assert cfg.task_type == "classification"
    assert cfg.missing_threshold == 0.5
    assert cfg.cv_folds == 3


@pytest.mark.parametrize("missing_threshold", [-0.1, 1.1])
def test_config_invalid_missing_threshold(missing_threshold):
    """Verify missing_threshold range check [0.0, 1.0]."""
    with pytest.raises(ConfigurationError):
        MentorConfig(missing_threshold=missing_threshold)


@pytest.mark.parametrize("corr_threshold", [0.0, -0.2, 1.5])
def test_config_invalid_correlation_threshold(corr_threshold):
    """Verify correlation_threshold range check (0.0, 1.0]."""
    with pytest.raises(ConfigurationError):
        MentorConfig(correlation_threshold=corr_threshold)


@pytest.mark.parametrize("test_size", [0.0, 1.0, -0.5, 1.5])
def test_config_invalid_test_size(test_size):
    """Verify test_size range check (0.0, 1.0)."""
    with pytest.raises(ConfigurationError):
        MentorConfig(test_size=test_size)


@pytest.mark.parametrize("cv_folds", [0, 1, -2])
def test_config_invalid_cv_folds(cv_folds):
    """Verify cv_folds must be at least 2."""
    with pytest.raises(ConfigurationError):
        MentorConfig(cv_folds=cv_folds)


@pytest.mark.parametrize("cardinality", [0, -5])
def test_config_invalid_cardinality_threshold(cardinality):
    """Verify cardinality_threshold must be >= 1."""
    with pytest.raises(ConfigurationError):
        MentorConfig(cardinality_threshold=cardinality)


def test_config_invalid_task_type():
    """Verify task_type must be 'classification', 'regression', or None."""
    with pytest.raises(ConfigurationError):
        MentorConfig(task_type="invalid_task")


def test_config_summary():
    """Verify summary() produces readable text containing fields."""
    cfg = MentorConfig(target_column="outcome")
    summary_text = cfg.summary()
    assert "MentorConfig:" in summary_text
    assert "outcome" in summary_text
    assert "missing_threshold" in summary_text
