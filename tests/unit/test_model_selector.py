"""
tests.unit.test_model_selector
------------------------------
Unit tests for ModelSelector candidate evaluation, multiclass support, and target encoding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mentorml.config import MentorConfig
from mentorml.core.decision import DecisionLog
from mentorml.modeling.model_selector import ModelSelector


def test_model_selector_binary_classification():
    """Verify binary classification model selection and leaderboard structure."""
    np.random.seed(42)
    n = 60
    X = pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.uniform(0, 10, n),
    })
    y = pd.Series(np.random.choice([0, 1], n))

    cfg = MentorConfig(cv_folds=2, random_state=42)
    selector = ModelSelector(cfg)
    log = DecisionLog()
    res = selector.select(X, y, log, task_type="classification")

    assert res["best_model"] is not None
    assert res["best_model_name"] in ["LogisticRegression", "RandomForestClassifier", "GradientBoostingClassifier"]
    assert "best_score" in res
    assert "metric_name" in res
    assert "leaderboard" in res
    assert len(res["leaderboard"]) > 0


def test_model_selector_multiclass_and_target_encoding():
    """Verify string targets are label-encoded and multiclass ROC-AUC does not crash."""
    np.random.seed(42)
    n = 60
    X = pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
    })
    y = pd.Series(np.random.choice(["cat", "dog", "bird"], n))

    cfg = MentorConfig(cv_folds=2, random_state=42)
    selector = ModelSelector(cfg)
    log = DecisionLog()
    res = selector.select(X, y, log, task_type="classification")

    assert res["best_model"] is not None
    assert res.get("target_encoder") is not None
    assert set(res["target_encoder"].classes_) == {"bird", "cat", "dog"}
    assert res["metric_name"] in ["roc_auc_ovr", "f1_weighted"]


def test_model_selector_regression():
    """Verify regression selection returns fitted regressor and r2 metric."""
    np.random.seed(42)
    n = 60
    X = pd.DataFrame({
        "feat_a": np.random.randn(n),
        "feat_b": np.random.uniform(1, 10, n),
    })
    y = pd.Series(2.0 * X["feat_a"] + 0.5 * X["feat_b"] + np.random.randn(n) * 0.1)

    cfg = MentorConfig(cv_folds=2, random_state=42)
    selector = ModelSelector(cfg)
    log = DecisionLog()
    res = selector.select(X, y, log, task_type="regression")

    assert res["best_model"] is not None
    assert res["task_type"] == "regression"
    assert res["metric_name"] == "neg_root_mean_squared_error"
