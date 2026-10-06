"""
mentorml.core.exceptions
------------------------
Custom exception hierarchy for the mentorml package.

Design rationale
~~~~~~~~~~~~~~~~
We define a strict hierarchy rooted at ``MentorMLError`` so that callers can
catch package-level errors with a single ``except MentorMLError`` clause while
still being able to discriminate between specific failure modes when needed.

Every exception carries a human-readable ``message`` and an optional
``context`` dict for structured debugging data — making errors as
informative to developers as the ``DecisionRecord`` is to end users.
"""

from __future__ import annotations

from typing import Any


class MentorMLError(Exception):
    """
    Root exception for all mentorml errors.

    All custom exceptions in this package inherit from this class so that
    callers can catch all mentorml errors with a single clause::

        try:
            advisor.analyze(df)
        except MentorMLError as e:
            print(e.message, e.context)

    Parameters
    ----------
    message : str
        Human-readable description of the error.
    context : dict[str, Any] | None
        Optional structured metadata (column names, thresholds, etc.) to aid
        debugging. Defaults to an empty dict.
    """

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        self.message = message
        self.context: dict[str, Any] = context or {}
        super().__init__(message)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, context={self.context!r})"


# ---------------------------------------------------------------------------
# Data-related errors
# ---------------------------------------------------------------------------


class DataValidationError(MentorMLError):
    """
    Raised when input data fails a structural or semantic validation check.

    Examples
    --------
    - DataFrame is empty.
    - Required target column is missing.
    - All values in a column are null.
    """


class InsufficientDataError(MentorMLError):
    """
    Raised when the dataset is too small for a requested operation.

    Examples
    --------
    - Fewer rows than folds requested for cross-validation.
    - Not enough samples to compute reliable statistics.
    """


class ColumnNotFoundError(MentorMLError):
    """
    Raised when a referenced column does not exist in the DataFrame.

    Parameters
    ----------
    column : str
        Name of the missing column.
    available : list[str]
        Columns actually present in the DataFrame, included in context for
        easy debugging.
    """

    def __init__(self, column: str, available: list[str]) -> None:
        super().__init__(
            message=f"Column '{column}' not found in DataFrame.",
            context={"column": column, "available_columns": available},
        )
        self.column = column
        self.available = available


# ---------------------------------------------------------------------------
# Configuration errors
# ---------------------------------------------------------------------------


class ConfigurationError(MentorMLError):
    """
    Raised when a ``MentorConfig`` value is invalid or contradictory.

    Examples
    --------
    - ``missing_threshold`` outside ``[0.0, 1.0]``.
    - ``random_state`` is not an integer.
    """


# ---------------------------------------------------------------------------
# Pipeline / component errors
# ---------------------------------------------------------------------------


class ComponentNotFittedError(MentorMLError):
    """
    Raised when a stateful component method is called before ``fit()``.

    This mirrors sklearn's ``NotFittedError`` convention so users familiar
    with sklearn feel at home.
    """


class ModelSelectionError(MentorMLError):
    """
    Raised when no viable model candidate can be identified.

    Examples
    --------
    - All candidate models fail during cross-validation.
    - Task type (regression/classification) cannot be inferred.
    """


class ExplainabilityError(MentorMLError):
    """
    Raised when SHAP or another explainability method fails.

    Examples
    --------
    - Model type not supported by the selected explainer.
    - SHAP values contain NaN or Inf.
    """


# ---------------------------------------------------------------------------
# Reporting errors
# ---------------------------------------------------------------------------


class ReportGenerationError(MentorMLError):
    """
    Raised when report rendering or export fails.

    Examples
    --------
    - Output directory is not writable.
    - Template is missing or malformed.
    """


class ExportError(MentorMLError):
    """
    Raised when model serialisation or export fails.

    Examples
    --------
    - Unsupported export format requested.
    - File system permission denied.
    """