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


@pytest.fixture
def sample_regression_data():
    """Generate a clean synthetic dataset for regression."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "feature_x": np.random.randn(n),
        "feature_y": np.random.uniform(1, 100, n),
        "category": np.random.choice(["cat1", "cat2"], n),
        "target": np.random.uniform(10.0, 50.0, n),
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


def test_pipeline_regression_configuration(sample_regression_data):
    """Verify regression configuration and initialization."""
    config = MentorConfig(
        target_column="target",
        task_type="regression",
        cv_folds=2,
    )
    pipeline = MentorPipeline(config=config, project_name="Test Regression")
    assert pipeline.config.task_type == "regression"
    assert pipeline.config.target_column == "target"


def test_html_report_generation():
    """Verify that HTMLReportGenerator creates a self-contained report."""
    reporter = HTMLReportGenerator(project_name="Integration Test Project")
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = reporter.generate(
            log=None,
            output_dir=tmp_dir,
            analysis_result={"n_rows": 100, "n_columns": 5},
            quality_report={"issues": []},
            model_selection_result={"best_model": "RandomForestClassifier", "best_score": 0.88},
            narrative="Integration test narrative.",
        )
        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8") as f:
            html_text = f.read()
        assert "Integration Test Project" in html_text
        assert "RandomForestClassifier" in html_text


def test_pipeline_result_summary():
    """Verify PipelineResult summary text output."""
    from mentorml.core.decision import DecisionLog
    log = DecisionLog()
    result = PipelineResult(
        model=None,
        model_name="MockModel",
        task_type="classification",
        log=log,
        feature_names=["f1", "f2"],
    )
    summary = result.summary()
    assert "MockModel" in summary
    assert "classification" in summary
