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


class MentorPipeline:
    """
    Top-level orchestrator for the mentorml AutoML pipeline.

    Chains all 9 phases in a single ``fit(df)`` call, returns a rich
    ``PipelineResult`` containing the model, narrative, report and exports.

    Parameters
    ----------
    config : MentorConfig
        Global configuration.  Controls thresholds, CV folds, paths, etc.
    project_name : str
        Human-readable project name — used in the narrative and report title.
    generate_report : bool
        If ``True`` (default), generate the interactive HTML report (Phase 8).
    export_artefacts : bool
        If ``True`` (default), export all artefacts to disk (Phase 9).

    Examples
    --------
    ::

        from mentorml import MentorPipeline, MentorConfig

        config = MentorConfig(
            target_column="churn",
            task_type="classification",
            report_output_dir="reports/",
            export_output_dir="exports/",
        )
        pipeline = MentorPipeline(config, project_name="Churn Model")
        result = pipeline.fit(df)
        print(result.summary())
    """

    def __init__(
        self,
        config: MentorConfig,
        project_name: str = "ML Project",
        generate_report: bool = True,
        export_artefacts: bool = True,
    ) -> None:
        self.config = config
        self.project_name = project_name
        self.generate_report = generate_report
        self.export_artefacts = export_artefacts

    def fit(self, df: pd.DataFrame) -> PipelineResult:
        """
        Run the full mentorml pipeline on ``df``.

        Parameters
        ----------
        df : pd.DataFrame
            Raw input data, including the target column.

        Returns
        -------
        PipelineResult
            All pipeline outputs.
        """
        log = DecisionLog()

        log.append(
            DecisionRecord(
                component="MentorPipeline",
                action="pipeline_start",
                rationale=(
                    f"Starting MentorPipeline for project '{self.project_name}'. "
                    f"Input shape: {df.shape}."
                ),
                severity=Severity.INFO,
                data={"project": self.project_name, "shape": list(df.shape)},
            )
        )

        # ----------------------------------------------------------------
        # Phase 1: Dataset Analysis
        # ----------------------------------------------------------------
        analyzer = DatasetAnalyzer(self.config)
        analysis_result = analyzer.analyze(df, log)
        task_type = analysis_result.get("target", {}).get(
            "inferred_task_type", self.config.task_type or "classification"
        )

        # ----------------------------------------------------------------
        # Phase 2: Data Quality Assessment
        # ----------------------------------------------------------------
        assessor = DataQualityAssessor(self.config)
        quality_report = assessor.analyze(df, log)

        # ----------------------------------------------------------------
        # Separate features and target
        # ----------------------------------------------------------------
        target_col = self.config.target_column or analysis_result.get(
            "target", {}
        ).get("column")

        if target_col and target_col in df.columns:
            X_raw = df.drop(columns=[target_col])
            y = df[target_col]
        else:
            X_raw = df.copy()
            y = pd.Series(dtype="float64")
            log.append(
                DecisionRecord(
                    component="MentorPipeline",
                    action="target_column_missing",
                    rationale=(
                        f"Target column '{target_col}' not found. "
                        "Pipeline will preprocess without a target."
                    ),
                    data={},

                    severity=Severity.WARNING,
                )
            )

        # ----------------------------------------------------------------
        # Phase 3: Explainable Preprocessing
        # ----------------------------------------------------------------
        preprocessor = ExplainablePreprocessor(self.config)
        preprocessor.fit(X_raw, log, analysis_result, quality_report)
        X_clean = preprocessor.transform(X_raw, log)

        # ----------------------------------------------------------------
        # Phase 4: Feature Engineering
        # ----------------------------------------------------------------
        fe = FeatureEngineer(self.config)
        fe.fit(X_clean, log, analysis_result)
        X_engineered = fe.transform(X_clean, log)

        feature_names = list(X_engineered.columns)

        # ----------------------------------------------------------------
        # Phase 5: Model Selection
        # ----------------------------------------------------------------
        model_selection_result: dict[str, Any] = {}
        fitted_model: Any = None
        best_model_name = "N/A"

        if len(y) > 0 and len(X_engineered) > 0:
            X_num = X_engineered.select_dtypes(include="number").fillna(0)

            if len(X_num) >= 10:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_num,
                    y,
                    test_size=self.config.test_size,
                    random_state=self.config.random_state,
                )
                selector = ModelSelector(self.config)
                model_selection_result = selector.select(
                    X_tr, y_tr, log, task_type=task_type
                )
                fitted_model = model_selection_result["best_model"]
                best_model_name = model_selection_result["best_model_name"]
            else:
                log.append(
                    DecisionRecord(
                        component="MentorPipeline",
                        action="model_selection_skipped",
                        rationale="Too few rows for cross-validation; skipping model selection.",
                        data={},

                        severity=Severity.WARNING,
                    )
                )
                X_te = X_engineered.select_dtypes(include="number").fillna(0)
                y_te = y
        else:
            X_te = X_engineered.select_dtypes(include="number").fillna(0)
            y_te = y

        # ----------------------------------------------------------------
        # Phase 6: SHAP Explainability
        # ----------------------------------------------------------------
        explanation_result: dict[str, Any] = {}
        if fitted_model is not None and len(X_te) > 0:
            explainer = SHAPExplainer(self.config)
            try:
                explanation_result = explainer.explain(
                    fitted_model, X_te, log, y=y_te if len(y_te) > 0 else None
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Explainability failed: %s", exc)
                log.append(
                    DecisionRecord(
                        component="MentorPipeline",
                        action="explainability_failed",
                        rationale=f"Explainability step failed: {exc}",
                        data={},

                        severity=Severity.WARNING,
                    )
                )

        # ----------------------------------------------------------------
        # Phase 7: Business Narration
        # ----------------------------------------------------------------
        narrator = BusinessNarrator(project_name=self.project_name)
        narrative = narrator.narrate(
            log,
            analysis_result=analysis_result,
            quality_report=quality_report,
            explanation_result=explanation_result,
            model_selection_result=model_selection_result or None,
        )

        # ----------------------------------------------------------------
        # Phase 8: HTML Report
        # ----------------------------------------------------------------
        report_path: str | None = None
        if self.generate_report:
            reporter = HTMLReportGenerator(project_name=self.project_name)
            try:
                report_path = reporter.generate(
                    log,
                    output_dir=self.config.report_output_dir,
                    analysis_result=analysis_result,
                    quality_report=quality_report,
                    explanation_result=explanation_result,
                    model_selection_result=model_selection_result or None,
                    narrative=narrative,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Report generation failed: %s", exc)

        # ----------------------------------------------------------------
        # Phase 9: Export
        # ----------------------------------------------------------------
        export_dir: str | None = None
        if self.export_artefacts and fitted_model is not None:
            exporter = ModelExporter(project_name=self.project_name)
            try:
                export_dir = exporter.export(
                    model=fitted_model,
                    preprocessor=preprocessor,
                    feature_names=feature_names,
                    log=log,
                    output_dir=self.config.export_output_dir,
                    report_path=report_path,
                    data={
                        "task_type": task_type,
                        "best_model": best_model_name,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Export failed: %s", exc)

        # ----------------------------------------------------------------
        # Done
        # ----------------------------------------------------------------
        log.append(
            DecisionRecord(
                component="MentorPipeline",
                action="pipeline_complete",
                rationale=(
                    f"Pipeline complete. "
                    f"Best model: {best_model_name}. "
                    f"Total decisions: {len(log)}."
                ),
                severity=Severity.INFO,
                data={
                    "best_model": best_model_name,
                    "task_type": task_type,
                    "n_decisions": len(log),
                    "report_path": report_path,
                    "export_dir": export_dir,
                },
            )
        )

        return PipelineResult(
            model=fitted_model,
            model_name=best_model_name,
            task_type=task_type,
            log=log,
            analysis_result=analysis_result,
            quality_report=quality_report,
            model_selection_result=model_selection_result,
            explanation_result=explanation_result,
            narrative=narrative,
            report_path=report_path,
            export_dir=export_dir,
            feature_names=feature_names,
            preprocessor=preprocessor,
            feature_engineer=fe,
        )