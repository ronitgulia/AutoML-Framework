"""
mentorml.core.protocols
-----------------------
Structural typing contracts for all mentorml components.

Design rationale
~~~~~~~~~~~~~~~~
Python's ``typing.Protocol`` (PEP 544) enables *structural subtyping*:
a class satisfies a Protocol simply by having the right methods and
attributes — no explicit inheritance required.

Why this matters for mentorml
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We want components to be:

1. **Independently testable** — you can test a custom preprocessor without
   importing the full mentorml pipeline.
2. **Loosely coupled** — the pipeline orchestrator doesn't need to know the
   concrete type of each component, only its interface.
3. **Easily extensible** — users can write their own components (e.g., a
   domain-specific feature engineer) and plug them in without subclassing
   anything from mentorml.

Every protocol here is ``runtime_checkable`` so you can use
``isinstance(obj, Analyzable)`` in tests and runtime guards.

Convention
~~~~~~~~~~
- ``fit(df, log)``  — learns from data; writes decisions to *log*
- ``transform(df, log)`` — applies the learned transformation
- ``analyze(df, log)`` — pure analysis; returns a structured result dict
- All methods return ``Self`` or a typed result; they never return ``None``
  implicitly.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pandas as pd

from mentorml.core.decision import DecisionLog


# ---------------------------------------------------------------------------
# Analyzable
# ---------------------------------------------------------------------------


@runtime_checkable
class Analyzable(Protocol):
    """
    Contract for components that perform read-only analysis of a DataFrame.

    Implementing classes inspect data and emit ``DecisionRecord`` objects
    describing what they found.  They must **not** mutate the input
    DataFrame.

    Examples of implementing classes
    ----------------------------------
    - ``DatasetAnalyzer``
    - ``DataQualityAssessor``
    """

    def analyze(
        self,
        df: pd.DataFrame,
        log: DecisionLog,
    ) -> dict[str, Any]:
        """
        Analyse ``df`` and return a structured result.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.  Must not be mutated.
        log : DecisionLog
            Decision log to append observations to.

        Returns
        -------
        dict[str, Any]
            Structured analysis results.  Schema is component-specific
            and documented in each implementing class.
        """
        ...


# ---------------------------------------------------------------------------
# Fittable
# ---------------------------------------------------------------------------


@runtime_checkable
class Fittable(Protocol):
    """
    Contract for stateful components that learn parameters from data.

    Implementing classes follow the sklearn ``fit / transform`` pattern,
    with the addition of a ``DecisionLog`` parameter so the fitting process
    is fully narrated.

    Examples of implementing classes
    ----------------------------------
    - ``ExplainablePreprocessor``
    - ``FeatureEngineer``
    """

    def fit(
        self,
        df: pd.DataFrame,
        log: DecisionLog,
    ) -> "Fittable":
        """
        Learn transformation parameters from ``df``.

        Must write at least one ``DecisionRecord`` per meaningful choice made
        (e.g., which imputation strategy was selected and why).

        Parameters
        ----------
        df : pd.DataFrame
            Training data.
        log : DecisionLog
            Decision log to append fitting decisions to.

        Returns
        -------
        self
            Returns ``self`` to allow method chaining::

                prep.fit(df, log).transform(df, log)
        """
        ...

    def transform(
        self,
        df: pd.DataFrame,
        log: DecisionLog,
    ) -> pd.DataFrame:
        """
        Apply the learned transformation to ``df``.

        Parameters
        ----------
        df : pd.DataFrame
            Data to transform.  May be training or held-out data.
        log : DecisionLog
            Decision log to append transformation records to.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame.  Shape may differ from input.

        Raises
        ------
        ComponentNotFittedError
            If called before ``fit()``.
        """
        ...


# ---------------------------------------------------------------------------
# ModelSelector
# ---------------------------------------------------------------------------


@runtime_checkable
class ModelSelectorProtocol(Protocol):
    """
    Contract for components that select and tune a model.

    Examples of implementing classes
    ----------------------------------
    - ``ModelSelector``
    """

    def select(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        log: DecisionLog,
    ) -> Any:
        """
        Select the best model for the given task.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target vector.
        log : DecisionLog
            Decision log to append selection rationale to.

        Returns
        -------
        Any
            A fitted sklearn-compatible estimator.
        """
        ...


# ---------------------------------------------------------------------------
# Explainer
# ---------------------------------------------------------------------------


@runtime_checkable
class ExplainerProtocol(Protocol):
    """
    Contract for components that generate model explanations.

    Examples of implementing classes
    ----------------------------------
    - ``SHAPExplainer``
    """

    def explain(
        self,
        model: Any,
        X: pd.DataFrame,
        log: DecisionLog,
    ) -> dict[str, Any]:
        """
        Generate explanations for ``model`` predictions on ``X``.

        Parameters
        ----------
        model : Any
            A fitted sklearn-compatible estimator.
        X : pd.DataFrame
            Feature matrix (typically the held-out test set).
        log : DecisionLog
            Decision log to append explanation metadata to.

        Returns
        -------
        dict[str, Any]
            Explanation artefacts.  Schema is explainer-specific and
            documented in each implementing class.
        """
        ...


# ---------------------------------------------------------------------------
# Narrator
# ---------------------------------------------------------------------------


@runtime_checkable
class NarratorProtocol(Protocol):
    """
    Contract for components that translate the decision log into business
    language.

    Examples of implementing classes
    ----------------------------------
    - ``BusinessNarrator``
    """

    def narrate(
        self,
        log: DecisionLog,
        analysis_results: dict[str, Any],
    ) -> str:
        """
        Generate a plain-English narrative from the decision log.

        Parameters
        ----------
        log : DecisionLog
            The full decision log accumulated by the pipeline.
        analysis_results : dict[str, Any]
            Structured results from the analysis phase.

        Returns
        -------
        str
            A business-friendly narrative (may be Markdown-formatted).
        """
        ...


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------


@runtime_checkable
class ReportGeneratorProtocol(Protocol):
    """
    Contract for components that produce human-readable reports.

    Examples of implementing classes
    ----------------------------------
    - ``HTMLReportGenerator``
    """

    def generate(
        self,
        log: DecisionLog,
        analysis_results: dict[str, Any],
        output_path: str,
    ) -> str:
        """
        Render and write a report to ``output_path``.

        Parameters
        ----------
        log : DecisionLog
            The full decision log.
        analysis_results : dict[str, Any]
            Structured results from the analysis phase.
        output_path : str
            File path where the report should be written.

        Returns
        -------
        str
            Absolute path of the generated report file.
        """
        ...