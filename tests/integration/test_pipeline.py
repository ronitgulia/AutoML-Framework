"""
tests.integration.test_pipeline
--------------------------------
Integration test suite for MentorPipeline execution lifecycle.
"""

from __future__ import annotations

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from mentorml.config import MentorConfig
from mentorml.pipeline import MentorPipeline, PipelineResult
from mentorml.reporting.html_report import HTMLReportGenerator


@pytest.fixture
def sample_classification_data():
    """Generate a clean synthetic dataset for binary classification."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "num_feature_1": np.random.randn(n),
        "num_feature_2": np.random.uniform(10, 50, n),
        "cat_feature": np.random.choice(["type_a", "type_b", "type_c"], n),
        "target": np.random.choice([0, 1], n),
    })
    return df


def test_pipeline_classification_initialization(sample_classification_data):
    """Verify that MentorPipeline initializes correctly with classification configuration."""
    config = MentorConfig(
        target_column="target",
        task_type="classification",
        cv_folds=2,
    )
    pipeline = MentorPipeline(config=config, project_name="Test Classification")
    assert pipeline is not None
    assert pipeline.project_name == "Test Classification"
    assert pipeline.config.target_column == "target"
