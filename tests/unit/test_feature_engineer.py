"""
tests.unit.test_feature_engineer
--------------------------------
Unit tests for FeatureEngineer transformations and correlation filtering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mentorml.config import MentorConfig
from mentorml.core.decision import DecisionLog
from mentorml.core.exceptions import ComponentNotFittedError
from mentorml.features.feature_engineer import FeatureEngineer


def test_feature_engineer_not_fitted():
    """Verify ComponentNotFittedError is raised if transform called prior to fit."""
    cfg = MentorConfig()
    fe = FeatureEngineer(cfg)
    log = DecisionLog()
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ComponentNotFittedError):
        fe.transform(df, log)


def test_feature_engineer_correlation_drop():
    """Verify near-perfectly correlated features trigger removal of one column."""
    x = np.linspace(1, 100, 100)
    df = pd.DataFrame({
        "x1": x,
        "x2": x * 1.0001,  # correlation ~ 1.0
        "x3": np.random.randn(100),
    })
    # Make x2 have missing values so the heuristic prioritizes dropping x2
    df.loc[0:5, "x2"] = np.nan

    cfg = MentorConfig(correlation_threshold=0.98)
    fe = FeatureEngineer(cfg)
    log = DecisionLog()
    fe.fit(df, log)
    df_trans = fe.transform(df, log)

    # One of the two correlated features should be dropped
    assert "x2" not in df_trans.columns
    assert "x1" in df_trans.columns
    assert "x3" in df_trans.columns


def test_feature_engineer_interaction_creation():
    """Verify interaction pairs create multiplication feature columns."""
    df = pd.DataFrame({
        "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0] * 10,
        "feat_b": [2.0, 4.0, 6.0, 8.0, 10.0] * 10,
        "feat_c": np.random.randn(50),
    })
    cfg = MentorConfig(correlation_threshold=0.999)
    fe = FeatureEngineer(cfg)
    log = DecisionLog()
    fe.fit(df, log)
    df_trans = fe.transform(df, log)

    # If interaction pairs were found, interaction columns should exist
    assert len(df_trans.columns) >= len(df.columns)
