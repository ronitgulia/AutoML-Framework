"""
tests.unit.test_phases_6_to_9
-----------------------------
Unit tests for SHAPExplainer, BusinessNarrator, HTMLReportGenerator, and ModelExporter.
"""

from __future__ import annotations

import json
import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from mentorml.config import MentorConfig
from mentorml.core.decision import DecisionLog, DecisionRecord, Severity
from mentorml.explainability.shap_explainer import SHAPExplainer
from mentorml.export.model_exporter import ModelExporter
from mentorml.narrative.narrator import BusinessNarrator
from mentorml.preprocessing.preprocessor import ExplainablePreprocessor
from mentorml.reporting.html_report import HTMLReportGenerator


def test_shap_explainer_basic():
    """Verify SHAPExplainer calculates mean absolute feature importance."""
    np.random.seed(42)
    n = 40
    X = pd.DataFrame({
        "num1": np.random.randn(n),
        "num2": np.random.randn(n),
    })
    y = pd.Series(np.random.choice([0, 1], n))

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    cfg = MentorConfig()
    explainer = SHAPExplainer(cfg)
    log = DecisionLog()
    result = explainer.explain(model, X, log, y=y)

    assert "feature_importances" in result
    assert "top_features" in result
    assert len(result["feature_importances"]) == 2


def test_business_narrator_narrates_all_phases():
    """Verify BusinessNarrator produces comprehensive markdown without crashing."""
    log = DecisionLog()
    log.append(DecisionRecord(
        component="TestComp",
        action="test_action",
        rationale="Testing narrator",
        data={},
        severity=Severity.INFO,
    ))

    narrator = BusinessNarrator(project_name="Narrator Test")
    text = narrator.narrate(
        log,
        analysis_result={"n_rows": 50, "n_columns": 4, "task_type": "classification"},
        quality_report={"issues": [], "overall_severity": "info"},
        model_selection_result={"best_model_name": "RandomForestClassifier", "best_score": 0.92, "scoring_metric": "roc_auc"},
        explanation_result={"top_features": [("feature_a", 0.45), ("feature_b", 0.20)]},
    )

    assert "Narrator Test" in text
    assert "RandomForestClassifier" in text
    assert "feature_a" in text


def test_html_report_generator_feature_importance():
    """Verify HTML report includes rendered feature importance chart."""
    reporter = HTMLReportGenerator(project_name="HTML Test")
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = reporter.generate(
            log=None,
            output_dir=tmp_dir,
            analysis_result={"n_rows": 100, "n_columns": 5},
            quality_report={"issues": []},
            model_selection_result={"best_model": "LogisticRegression", "best_score": 0.85},
            explanation_result={"feature_importance": {"age": 0.5, "income": 0.3}},
            narrative="Test summary narrative.",
        )
        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8") as f:
            html_text = f.read()

        assert "Phase 6" in html_text
        assert "Global Feature Importance (SHAP)" in html_text
        assert "age" in html_text
        assert "income" in html_text


def test_model_exporter_files():
    """Verify ModelExporter exports joblib models, feature names, decisions, and predict.py."""
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit([[1, 2], [3, 4]], [0, 1])
    cfg = MentorConfig()
    prep = ExplainablePreprocessor(cfg)
    log = DecisionLog()
    prep.fit(pd.DataFrame([[1, 2], [3, 4]], columns=["f1", "f2"]), log)

    exporter = ModelExporter(project_name="ExportProject")
    with tempfile.TemporaryDirectory() as tmp_dir:
        export_dir = exporter.export(
            model=model,
            preprocessor=prep,
            feature_names=["f1", "f2"],
            log=log,
            output_dir=tmp_dir,
        )

        assert os.path.exists(os.path.join(export_dir, "model.joblib"))
        assert os.path.exists(os.path.join(export_dir, "preprocessor.joblib"))
        assert os.path.exists(os.path.join(export_dir, "feature_names.json"))
        assert os.path.exists(os.path.join(export_dir, "predict.py"))
        assert os.path.exists(os.path.join(export_dir, "manifest.json"))

        with open(os.path.join(export_dir, "manifest.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["project_name"] == "exportproject"
        assert meta["n_features"] == 2
