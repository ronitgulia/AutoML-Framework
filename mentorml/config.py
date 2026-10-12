"""
mentorml.config
---------------
Central configuration for the mentorml package.

All behavioural thresholds and defaults live here.  Nothing in the codebase
should use magic numbers — import ``MentorConfig`` and use its fields.

Design rationale
~~~~~~~~~~~~~~~~
Using a ``dataclass`` for configuration gives us:

- **Immutable defaults**: the class definition is the single source of truth.
- **Explicitness**: every knob has a name, a type, and a docstring.
- **Easy overriding**: users construct ``MentorConfig(missing_threshold=0.3)``
  and pass it to the pipeline — no monkey-patching required.
- **Testability**: tests can create config variants in one line.

Future extension
~~~~~~~~~~~~~~~~
In a later phase we may add Pydantic validation for stricter type coercion
and range checking.  The dataclass design is compatible with that migration.

Usage
~~~~~
::

    from mentorml.config import MentorConfig

    # Default config
    config = MentorConfig()

    # Custom config
    config = MentorConfig(
        target_column="churn",
        missing_threshold=0.3,
        verbose=False,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from mentorml.core.exceptions import ConfigurationError


@dataclass
class MentorConfig:
    """
    Global configuration for a mentorml session.

    All attributes have sensible defaults so that users can get started
    with ``MentorConfig()`` and progressively customise.

    Parameters
    ----------
    target_column : str | None
        Name of the prediction target column.  If ``None``, mentorml
        will attempt to infer it (last column convention).
    task_type : str | None
        One of ``"classification"`` | ``"regression"`` | ``None``.
        If ``None``, mentorml will infer from the target column's dtype.
    missing_threshold : float
        Fraction of missing values above which a column is dropped rather
        than imputed.  Must be in ``[0.0, 1.0]``.  Default: ``0.4``.
    cardinality_threshold : int
        Number of unique values above which a categorical column is
        considered *high-cardinality*.  High-cardinality columns receive
        different encoding treatment.  Default: ``50``.
    correlation_threshold : float
        Pearson correlation above which two features are considered
        near-duplicates and one is flagged for removal.  Must be in
        ``(0.0, 1.0]``.  Default: ``0.95``.
    variance_threshold : float
        Features with variance below this value are considered
        near-zero-variance and flagged for removal.  Default: ``0.01``.
    test_size : float
        Fraction of data held out for evaluation.  Must be in
        ``(0.0, 1.0)``.  Default: ``0.2``.
    cv_folds : int
        Number of cross-validation folds for model selection.
        Default: ``5``.
    random_state : int
        Global random seed for reproducibility.  Default: ``42``.
    n_jobs : int
        Number of parallel jobs for model training (``-1`` = all cores).
        Default: ``-1``.
    verbose : bool
        If ``True``, mentorml logs progress to stdout via the logging
        system.  Default: ``True``.
    report_output_dir : str
        Directory where generated reports are written.  Default: ``"."``
        (current working directory).
    export_output_dir : str
        Directory where exported model artefacts are written.
        Default: ``"."`` (current working directory).
    """

    # Target specification
    target_column: Optional[str] = None
    task_type: Optional[str] = None  # "classification" | "regression" | None

    # Data quality thresholds
    missing_threshold: float = 0.4
    cardinality_threshold: int = 50
    correlation_threshold: float = 0.95
    variance_threshold: float = 0.01

    # Evaluation settings
    test_size: float = 0.2
    cv_folds: int = 5

    # Reproducibility
    random_state: int = 42

    # Performance
    n_jobs: int = -1

    # UX
    verbose: bool = True

    # Output paths
    report_output_dir: str = "."
    export_output_dir: str = "."

    # Internal: populated by __post_init__
    _valid_task_types: tuple[str, ...] = field(
        default=("classification", "regression"),
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Validate configuration values immediately after construction."""
        self._validate()

    def _validate(self) -> None:
        """
        Run all validation checks.

        Raises
        ------
        ConfigurationError
            If any field contains an invalid value.
        """
        if not 0.0 <= self.missing_threshold <= 1.0:
            raise ConfigurationError(
                f"missing_threshold must be in [0.0, 1.0], got {self.missing_threshold}.",
                context={"field": "missing_threshold", "value": self.missing_threshold},
            )

        if not 0.0 < self.correlation_threshold <= 1.0:
            raise ConfigurationError(
                f"correlation_threshold must be in (0.0, 1.0], got {self.correlation_threshold}.",
                context={
                    "field": "correlation_threshold",
                    "value": self.correlation_threshold,
                },
            )

        if not 0.0 < self.test_size < 1.0:
            raise ConfigurationError(
                f"test_size must be in (0.0, 1.0), got {self.test_size}.",
                context={"field": "test_size", "value": self.test_size},
            )

        if self.cv_folds < 2:
            raise ConfigurationError(
                f"cv_folds must be >= 2, got {self.cv_folds}.",
                context={"field": "cv_folds", "value": self.cv_folds},
            )

        if self.cardinality_threshold < 1:
            raise ConfigurationError(
                f"cardinality_threshold must be >= 1, got {self.cardinality_threshold}.",
                context={
                    "field": "cardinality_threshold",
                    "value": self.cardinality_threshold,
                },
            )

        if self.task_type is not None and self.task_type not in self._valid_task_types:
            raise ConfigurationError(
                f"task_type must be one of {self._valid_task_types} or None, "
                f"got '{self.task_type}'.",
                context={"field": "task_type", "value": self.task_type},
            )

    def summary(self) -> str:
        """
        Return a human-readable summary of the current configuration.

        Returns
        -------
        str
            Multi-line string suitable for logging or display.
        """
        lines = ["MentorConfig:"]
        for f_name, f_value in self.__dataclass_fields__.items():  # type: ignore[attr-defined]
            if not f_name.startswith("_"):
                lines.append(f"  {f_name}: {getattr(self, f_name)!r}")
        return "\n".join(lines)