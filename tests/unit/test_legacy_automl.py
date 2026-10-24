"""
tests.unit.test_legacy_automl
-----------------------------
Unit tests for the legacy automl package (automl.preprocessing and automl.models).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from automl.models.selector import ModelSelector
from automl.preprocessing.preprocessor import DataPreprocessor


def test_legacy_preprocessor_categorical_only():
    """Verify legacy DataPreprocessor does not crash on categorical-only dataframe."""
    df = pd.DataFrame({
        "cat_a": ["low", "medium", "high", "low", "medium"],
        "cat_b": ["yes", "no", "yes", "no", "yes"],
    })
    prep = DataPreprocessor(cat_encoding="onehot")
    prep.fit(df)
    transformed = prep.transform(df)
    assert transformed.shape[0] == 5
    assert transformed.shape[1] > 0


def test_legacy_preprocessor_numeric_only():
    """Verify legacy DataPreprocessor handles numeric-only dataframe without errors."""
    df = pd.DataFrame({
        "num_a": [1.0, 2.0, np.nan, 4.0, 5.0],
        "num_b": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    prep = DataPreprocessor(impute_strategy="mean", scale=True)
    prep.fit(df)
    transformed = prep.transform(df)
    assert transformed.shape == (5, 2)
    assert not np.isnan(transformed).any()


def test_legacy_model_selector_best_model_is_fitted():
    """Verify selector.best_model is fitted on full data and ready for inference."""
    np.random.seed(42)
    X = np.random.randn(40, 4)
    y = np.random.choice([0, 1], size=40)

    selector = ModelSelector(task="classification", metric="accuracy", cv=2, n_jobs=1)
    results = selector.fit(X, y)

    assert len(results) > 0
    best = selector.best_model
    assert best is not None
    # Must be able to predict without raising NotFittedError
    preds = best.predict(X)
    assert len(preds) == 40


def test_legacy_model_selector_multiclass_handling():
    """Verify selector handles multiclass classification without roc_auc crash."""
    np.random.seed(42)
    X = np.random.randn(60, 4)
    y = np.random.choice([0, 1, 2], size=60)

    selector = ModelSelector(task="classification", metric="roc_auc", cv=2, n_jobs=1)
    results = selector.fit(X, y)
    assert len(results) > 0
    assert selector.best_model is not None
