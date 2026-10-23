"""
tests.unit.test_preprocessor
----------------------------
Unit tests for ExplainablePreprocessor fitting, encoding, scaling, and imputation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mentorml.config import MentorConfig
from mentorml.core.decision import DecisionLog
from mentorml.core.exceptions import ComponentNotFittedError
from mentorml.preprocessing.preprocessor import ExplainablePreprocessor


def test_preprocessor_not_fitted_error():
    """Verify ComponentNotFittedError is raised if transform called prior to fit."""
    cfg = MentorConfig()
    prep = ExplainablePreprocessor(cfg)
    log = DecisionLog()
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ComponentNotFittedError):
        prep.transform(df, log)


def test_preprocessor_string_dtype_encoding():
    """Verify pandas 2.0+ string dtypes are correctly categorized and encoded."""
    df = pd.DataFrame({
        "city": pd.Series(["Berlin", "Tokyo", "London", "Berlin", "Tokyo"], dtype="string"),
        "score": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    cfg = MentorConfig(cardinality_threshold=10)
    prep = ExplainablePreprocessor(cfg)
    log = DecisionLog()
    prep.fit(df, log)
    df_clean = prep.transform(df, log)

    # City should be one-hot encoded
    assert any("city__" in col for col in df_clean.columns)
    assert "score" in df_clean.columns


def test_preprocessor_imputation_and_scaling():
    """Verify numeric columns are imputed and scaled properly."""
    df = pd.DataFrame({
        "val": [10.0, 20.0, np.nan, 40.0, 50.0],
    })
    cfg = MentorConfig()
    prep = ExplainablePreprocessor(cfg)
    log = DecisionLog()
    prep.fit(df, log)
    df_clean = prep.transform(df, log)

    assert not df_clean["val"].isna().any()
    # Scaled values have approximately zero mean
    assert abs(df_clean["val"].mean()) < 1e-6


def test_preprocessor_drops_flagged_columns_except_target():
    """Verify preprocessor drops flagged columns but never drops target_column."""
    df = pd.DataFrame({
        "drop_me": [np.nan] * 10,
        "target": [0, 1] * 5,
        "keep_me": range(10),
    })
    cfg = MentorConfig(target_column="target")
    prep = ExplainablePreprocessor(cfg)
    log = DecisionLog()
    quality_report = {"columns_to_drop": ["drop_me", "target"]}
    prep.fit(df, log, quality_report=quality_report)
    df_clean = prep.transform(df, log)

    assert "drop_me" not in df_clean.columns
    assert "target" in df_clean.columns
    assert "keep_me" in df_clean.columns
