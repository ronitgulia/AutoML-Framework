"""
mentorml — AI Data Scientist Copilot
=====================================

mentorml is a production-quality Python package that reasons and narrates
like a senior data scientist.  Unlike generic AutoML tools that optimise
metrics silently, mentorml explains the *why* behind every decision it makes.

Quick start
-----------
::

    import pandas as pd
    from mentorml import MentorConfig, DatasetAnalyzer
    from mentorml.core import DecisionLog

    config = MentorConfig(target_column="churn")
    log = DecisionLog()

    analyzer = DatasetAnalyzer(config)
    results = analyzer.analyze(df, log)

Public API
----------
The stable public API is everything exported from this module.
Sub-module internals are considered private and may change between versions.
"""

from __future__ import annotations

import logging

from mentorml.config import MentorConfig
from mentorml.core import (
    Analyzable,
    ColumnNotFoundError,
    ComponentNotFittedError,
    ConfigurationError,
    DataValidationError,
    DecisionLog,
    DecisionRecord,
    ExplainabilityError,
    ExplainerProtocol,
    ExportError,
    Fittable,
    InsufficientDataError,
    MentorMLError,
    ModelSelectionError,
    ModelSelectorProtocol,
    NarratorProtocol,
    ReportGenerationError,
    ReportGeneratorProtocol,
    Severity,
)
from mentorml.analysis import (
    ColumnProfile,
    DatasetAnalyzer,
    DtypeCategory,
)
from mentorml.quality import DataQualityAssessor, QualityIssue
from mentorml.preprocessing import ExplainablePreprocessor
from mentorml.features import FeatureEngineer
from mentorml.modeling import ModelSelector
from mentorml.explainability import SHAPExplainer
from mentorml.narrative import BusinessNarrator
from mentorml.reporting import HTMLReportGenerator
from mentorml.export import ModelExporter
from mentorml.pipeline import MentorPipeline, PipelineResult

# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------

__version__ = "0.1.0"
__author__ = "mentorml contributors"
__license__ = "MIT"

# ---------------------------------------------------------------------------
# Logging setup
#
# We configure a NullHandler by default so that library code never emits
# output on its own.  The *application* that uses mentorml is responsible
# for configuring log handlers.  This is the standard library convention
# (see https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library).
# ---------------------------------------------------------------------------

logging.getLogger(__name__).addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

__all__ = [
    # Metadata
    "__version__",
    # Configuration
    "MentorConfig",
    # Decision layer
    "DecisionLog",
    "DecisionRecord",
    "Severity",
    # Exceptions
    "MentorMLError",
    "DataValidationError",
    "InsufficientDataError",
    "ColumnNotFoundError",
    "ConfigurationError",
    "ComponentNotFittedError",
    "ModelSelectionError",
    "ExplainabilityError",
    "ReportGenerationError",
    "ExportError",
    # Protocols
    "Analyzable",
    "Fittable",
    "ModelSelectorProtocol",
    "ExplainerProtocol",
    "NarratorProtocol",
    "ReportGeneratorProtocol",
    # Phase 1 — Analysis
    "DatasetAnalyzer",
    "ColumnProfile",
    "DtypeCategory",
    # Phase 2 — Quality
    "DataQualityAssessor",
    "QualityIssue",
    # Phase 3 — Preprocessing
    "ExplainablePreprocessor",
    # Phase 4 — Features
    "FeatureEngineer",
    # Phase 5 — Modeling
    "ModelSelector",
    # Phase 6 — Explainability
    "SHAPExplainer",
    # Phase 7 — Narration
    "BusinessNarrator",
    # Phase 8 — Reporting
    "HTMLReportGenerator",
    # Phase 9 — Export
    "ModelExporter",
    # Pipeline
    "MentorPipeline",
    "PipelineResult",
]