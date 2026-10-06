"""
mentorml.core
-------------
Core abstractions — exported for convenience.
"""

from mentorml.core.decision import DecisionLog, DecisionRecord, Severity
from mentorml.core.exceptions import (
    ColumnNotFoundError,
    ComponentNotFittedError,
    ConfigurationError,
    DataValidationError,
    ExplainabilityError,
    ExportError,
    InsufficientDataError,
    MentorMLError,
    ModelSelectionError,
    ReportGenerationError,
)
from mentorml.core.protocols import (
    Analyzable,
    ExplainerProtocol,
    Fittable,
    ModelSelectorProtocol,
    NarratorProtocol,
    ReportGeneratorProtocol,
)

__all__ = [
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
]