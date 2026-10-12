"""
mentorml.pipeline
-----------------
MentorPipeline — the top-level orchestrator.

A single ``fit(df)`` call chains all 9 phases in order:

1. DatasetAnalyzer       — analysis
2. DataQualityAssessor   — quality audit
3. ExplainablePreprocessor.fit() — learn transforms
4. ExplainablePreprocessor.transform() — clean data
5. FeatureEngineer.fit() — learn features
6. FeatureEngineer.transform() — engineered features
7. ModelSelector.select() — cross-validate and pick best model
8. SHAPExplainer.explain() — feature importance
9. BusinessNarrator.narrate() — plain-English narrative
10. HTMLReportGenerator.generate() — interactive HTML report
11. ModelExporter.export() — save all artefacts

Every decision is captured in a shared ``DecisionLog``.

Usage
-----
::

    from mentorml import MentorPipeline, MentorConfig

    config = MentorConfig(target_column="churn", task_type="classification")
    pipeline = MentorPipeline(config, project_name="Churn Model")
    result = pipeline.fit(df)

    print(result.narrative)
    print(f"Report: {result.report_path}")
    print(f"Export: {result.export_dir}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from mentorml.analysis.dataset_analyzer import DatasetAnalyzer
from mentorml.config import MentorConfig
from mentorml.core.decision import DecisionLog, DecisionRecord, Severity
from mentorml.explainability.shap_explainer import SHAPExplainer
from mentorml.export.model_exporter import ModelExporter
from mentorml.features.feature_engineer import FeatureEngineer
from mentorml.modeling.model_selector import ModelSelector
from mentorml.narrative.narrator import BusinessNarrator
from mentorml.preprocessing.preprocessor import ExplainablePreprocessor
from mentorml.quality.quality_assessor import DataQualityAssessor
from mentorml.reporting.html_report import HTMLReportGenerator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PipelineResult
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """
    Container for all outputs of a completed ``MentorPipeline.fit()`` run.

    Attributes
    ----------
    model : Any
        The fitted best estimator (sklearn-compatible).
    model_name : str
        Name of the best model class (e.g. ``"GradientBoostingClassifier"``).
    task_type : str
        Inferred or configured task type (``"classification"`` or ``"regression"``).
    log : DecisionLog
        Full pipeline decision log.
    analysis_result : dict
        Output of Phase 1 (DatasetAnalyzer).
    quality_report : dict
        Output of Phase 2 (DataQualityAssessor).
    model_selection_result : dict
        Output of Phase 5 (ModelSelector).
    explanation_result : dict
        Output of Phase 6 (SHAPExplainer).
    narrative : str
        Plain-English Markdown narrative from Phase 7.
    report_path : str | None
        Absolute path to the generated HTML report (Phase 8).
    export_dir : str | None
        Absolute path to the exported artefacts directory (Phase 9).
    feature_names : list[str]
        Column names of the final feature matrix.
    preprocessor : ExplainablePreprocessor
        The fitted preprocessor (for reuse on new data).
    feature_engineer : FeatureEngineer
        The fitted feature engineer (for reuse on new data).
    """

    model: Any
    model_name: str
    task_type: str
    log: DecisionLog
    analysis_result: dict[str, Any] = field(default_factory=dict)
    quality_report: dict[str, Any] = field(default_factory=dict)
    model_selection_result: dict[str, Any] = field(default_factory=dict)
    explanation_result: dict[str, Any] = field(default_factory=dict)
    narrative: str = ""
    report_path: str | None = None
    export_dir: str | None = None
    feature_names: list[str] = field(default_factory=list)
    preprocessor: ExplainablePreprocessor | None = None
    feature_engineer: FeatureEngineer | None = None

    def summary(self) -> str:
        """Return a concise text summary of the pipeline run."""
        lines = [
            "=" * 60,
            "MentorPipeline — Run Complete",
            "=" * 60,
            f"  Task type   : {self.task_type}",
            f"  Best model  : {self.model_name}",
            f"  Features    : {len(self.feature_names)}",
            f"  Decisions   : {len(self.log)}",
            f"  Report      : {self.report_path or 'not generated'}",
            f"  Export dir  : {self.export_dir or 'not exported'}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# MentorPipeline
# ---------------------------------------------------------------------------

