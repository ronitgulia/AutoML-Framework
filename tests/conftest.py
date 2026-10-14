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


