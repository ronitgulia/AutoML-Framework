"""
mentorml.analysis.dataset_analyzer
------------------------------------
Phase 1: Intelligent Dataset Analyzer

The ``DatasetAnalyzer`` is the first component every mentorml pipeline runs.
It performs a deep, multi-level analysis of a raw DataFrame and emits
``DecisionRecord`` objects explaining every significant finding.

Hierarchy
---------
::

    DatasetAnalyzer          ← public facade (implements Analyzable)
      ├── ColumnProfiler     ← profiles a single pd.Series
      ├── StructuralProfiler ← dataset-level structure (rows, dups, memory)
      ├── TargetAnalyzer     ← target column behaviour, task-type inference
      └── CorrelationAnalyzer← pairwise feature correlations

Key data structures
-------------------
- ``DtypeCategory``   — enum: NUMERIC, CATEGORICAL, DATETIME, BOOLEAN, TEXT, UNKNOWN
- ``ColumnProfile``   — per-column dataclass (reused by Phase 3 preprocessor)
- ``AnalysisResult``  — TypedDict returned by ``DatasetAnalyzer.analyze()``
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from mentorml.config import MentorConfig
from mentorml.core.decision import DecisionLog, DecisionRecord, Severity
from mentorml.core.exceptions import DataValidationError, InsufficientDataError

logger = logging.getLogger(__name__)

# Minimum rows required to compute reliable statistics
_MIN_ROWS = 10


# ---------------------------------------------------------------------------
# DtypeCategory enum
# ---------------------------------------------------------------------------


class DtypeCategory(str, Enum):
    """
    Semantic category of a DataFrame column's dtype.

    Using ``str`` as a mixin makes the enum JSON-serialisable without a
    custom encoder.

    Attributes
    ----------
    NUMERIC :
        Continuous or discrete numeric column (int, float).
    CATEGORICAL :
        Low-to-medium cardinality string/object column.
    BOOLEAN :
        Binary column (bool dtype, or int64 with only 0/1 values).
    DATETIME :
        Temporal column (datetime64 dtype).
    TEXT :
        High-cardinality string column whose content looks like free text.
    UNKNOWN :
        Could not be reliably categorised.
    """

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# ColumnProfile dataclass
# ---------------------------------------------------------------------------


@dataclass
class ColumnProfile:
    """
    Complete statistical profile of a single DataFrame column.

    This dataclass is the primary output of ``ColumnProfiler`` and is
    consumed by downstream components (e.g., Phase 3's
    ``ExplainablePreprocessor``) to make informed transformation decisions.

    Parameters
    ----------
    name : str
        Column name.
    dtype : str
        Raw pandas dtype string (e.g. ``"float64"``, ``"object"``).
    dtype_category : DtypeCategory
        Semantic category inferred from dtype and content.
    n_rows : int
        Total number of rows in the parent DataFrame.
    n_missing : int
        Count of null / NaN values.
    missing_pct : float
        Fraction of missing values in ``[0.0, 1.0]``.
    n_unique : int
        Number of unique non-null values.
    cardinality_ratio : float
        ``n_unique / (n_rows - n_missing)``; approaches 1.0 for ID-like columns.
    is_constant : bool
        ``True`` if the column has at most one unique value (zero variance).
    is_id_like : bool
        ``True`` if cardinality_ratio > 0.95 and dtype is NUMERIC or CATEGORICAL.
        ID-like columns are strong candidates for removal.
    top_values : list[tuple[Any, int]]
        Top-10 most frequent values and their counts.

    Numeric-only attributes (``None`` for non-numeric columns)
    -----------------------------------------------------------
    mean : float | None
    std : float | None
    min_val : float | None
    q25 : float | None
    median : float | None
    q75 : float | None
    max_val : float | None
    iqr : float | None
    skewness : float | None
        ``pandas.Series.skew()``.  > 1 → right-skewed; < -1 → left-skewed.
    kurtosis : float | None
        Excess kurtosis (``pandas.Series.kurt()``).  > 3 → heavy-tailed.
    n_outliers : int | None
        Count of values outside ``[Q1 - 1.5·IQR, Q3 + 1.5·IQR]``.
    outlier_pct : float | None
        Fraction of non-missing values that are outliers.
    """

    # Core fields (all dtypes)
    name: str
    dtype: str
    dtype_category: DtypeCategory
    n_rows: int
    n_missing: int
    missing_pct: float
    n_unique: int
    cardinality_ratio: float
    is_constant: bool
    is_id_like: bool
    top_values: list[tuple[Any, int]] = field(default_factory=list)

    # Numeric-only fields
    mean: float | None = None
    std: float | None = None
    min_val: float | None = None
    q25: float | None = None
    median: float | None = None
    q75: float | None = None
    max_val: float | None = None
    iqr: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    n_outliers: int | None = None
    outlier_pct: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise the profile to a plain JSON-friendly dict."""
        return {
            "name": self.name,
            "dtype": self.dtype,
            "dtype_category": self.dtype_category.value,
            "n_rows": self.n_rows,
            "n_missing": self.n_missing,
            "missing_pct": round(self.missing_pct, 4),
            "n_unique": self.n_unique,
            "cardinality_ratio": round(self.cardinality_ratio, 4),
            "is_constant": self.is_constant,
            "is_id_like": self.is_id_like,
            "top_values": self.top_values,
            # Numeric stats — round for readability
            "mean": round(self.mean, 4) if self.mean is not None else None,
            "std": round(self.std, 4) if self.std is not None else None,
            "min_val": round(self.min_val, 4) if self.min_val is not None else None,
            "q25": round(self.q25, 4) if self.q25 is not None else None,
            "median": round(self.median, 4) if self.median is not None else None,
            "q75": round(self.q75, 4) if self.q75 is not None else None,
            "max_val": round(self.max_val, 4) if self.max_val is not None else None,
            "skewness": round(self.skewness, 4) if self.skewness is not None else None,
            "kurtosis": round(self.kurtosis, 4) if self.kurtosis is not None else None,
            "n_outliers": self.n_outliers,
            "outlier_pct": round(self.outlier_pct, 4) if self.outlier_pct is not None else None,
        }


# ---------------------------------------------------------------------------
# ColumnProfiler — internal, single responsibility
# ---------------------------------------------------------------------------


class ColumnProfiler:
    """
    Profiles a single ``pd.Series`` and returns a ``ColumnProfile``.

    This class is intentionally stateless — all logic lives in class methods
    so it can be called without instantiation in tests.

    Design note
    -----------
    ``ColumnProfiler`` is *not* public API.  It is an implementation detail
    of ``DatasetAnalyzer``.  Downstream components import ``ColumnProfile``,
    not ``ColumnProfiler``.
    """

    # Cardinality ratio above which a column is considered ID-like
    _ID_CARDINALITY_THRESHOLD = 0.95
    # Cardinality ratio above which a categorical column is considered free-text
    _TEXT_CARDINALITY_THRESHOLD = 0.50

    @classmethod
    def profile(
        cls,
        series: pd.Series,  # type: ignore[type-arg]
        n_rows: int,
        config: MentorConfig,
    ) -> ColumnProfile:
        """
        Build a ``ColumnProfile`` for ``series``.

        Parameters
        ----------
        series : pd.Series
            The column to profile.
        n_rows : int
            Total rows in the parent DataFrame (used for ratio calculations).
        config : MentorConfig
            Global configuration (``cardinality_threshold`` used here).

        Returns
        -------
        ColumnProfile
            Fully populated profile.
        """
        name = str(series.name)
        dtype = str(series.dtype)
        n_missing = int(series.isna().sum())
        missing_pct = n_missing / n_rows if n_rows > 0 else 0.0

        non_null = series.dropna()
        n_non_null = len(non_null)
        n_unique = int(non_null.nunique())
        cardinality_ratio = n_unique / n_non_null if n_non_null > 0 else 0.0
        is_constant = n_unique <= 1

        dtype_category = cls._infer_dtype_category(
            series, n_unique, cardinality_ratio, config
        )

        is_id_like = (
            cardinality_ratio >= cls._ID_CARDINALITY_THRESHOLD
            and not is_constant
            and dtype_category in (DtypeCategory.CATEGORICAL, DtypeCategory.TEXT)
            and n_non_null > 20  # avoid flagging tiny datasets
        )

        top_values = cls._top_values(non_null)

        profile = ColumnProfile(
            name=name,
            dtype=dtype,
            dtype_category=dtype_category,
            n_rows=n_rows,
            n_missing=n_missing,
            missing_pct=missing_pct,
            n_unique=n_unique,
            cardinality_ratio=cardinality_ratio,
            is_constant=is_constant,
            is_id_like=is_id_like,
            top_values=top_values,
        )

        if dtype_category == DtypeCategory.NUMERIC:
            cls._add_numeric_stats(profile, non_null)

        return profile

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_dtype_category(
        series: pd.Series,  # type: ignore[type-arg]
        n_unique: int,
        cardinality_ratio: float,
        config: MentorConfig,
    ) -> DtypeCategory:
        """Infer the semantic category of a column from its dtype and content."""
        dtype = series.dtype

        if pd.api.types.is_bool_dtype(dtype):
            return DtypeCategory.BOOLEAN

        if pd.api.types.is_datetime64_any_dtype(dtype):
            return DtypeCategory.DATETIME

        if pd.api.types.is_numeric_dtype(dtype):
            # Binary numeric (only 0 and 1) → treat as boolean
            unique_vals = set(series.dropna().unique())
            if unique_vals <= {0, 1, 0.0, 1.0}:
                return DtypeCategory.BOOLEAN
            return DtypeCategory.NUMERIC

        if pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
            # High-cardinality object → likely free text
            if cardinality_ratio >= ColumnProfiler._TEXT_CARDINALITY_THRESHOLD and n_unique > config.cardinality_threshold:
                return DtypeCategory.TEXT
            return DtypeCategory.CATEGORICAL

        return DtypeCategory.UNKNOWN

    @staticmethod
    def _top_values(
        non_null: pd.Series,  # type: ignore[type-arg]
        n: int = 10,
    ) -> list[tuple[Any, int]]:
        """Return the top ``n`` most frequent values as (value, count) pairs."""
        counts = non_null.value_counts().head(n)
        return [(val, int(cnt)) for val, cnt in counts.items()]

    @staticmethod
    def _add_numeric_stats(
        profile: ColumnProfile,
        non_null: pd.Series,  # type: ignore[type-arg]
    ) -> None:
        """Populate numeric statistics fields on ``profile`` in-place."""
        if len(non_null) == 0:
            return

        q25 = float(non_null.quantile(0.25))
        q75 = float(non_null.quantile(0.75))
        iqr = q75 - q25

        profile.mean = float(non_null.mean())
        profile.std = float(non_null.std())
        profile.min_val = float(non_null.min())
        profile.q25 = q25
        profile.median = float(non_null.median())
        profile.q75 = q75
        profile.max_val = float(non_null.max())
        profile.iqr = iqr

        # Skewness and kurtosis require >= 3 observations
        if len(non_null) >= 3:
            skew_val = non_null.skew()
            kurt_val = non_null.kurt()
            profile.skewness = float(skew_val) if not math.isnan(skew_val) else None
            profile.kurtosis = float(kurt_val) if not math.isnan(kurt_val) else None

        # IQR-based outlier detection
        if iqr > 0:
            lower = q25 - 1.5 * iqr
            upper = q75 + 1.5 * iqr
            n_outliers = int(((non_null < lower) | (non_null > upper)).sum())
            profile.n_outliers = n_outliers
            profile.outlier_pct = n_outliers / len(non_null)
        else:
            # IQR=0 → all values equal (or near-zero variance); no outliers
            profile.n_outliers = 0
            profile.outlier_pct = 0.0


# ---------------------------------------------------------------------------
# TargetAnalyzer — internal
# ---------------------------------------------------------------------------


class TargetAnalyzer:
    """
    Analyses the target column and infers task type.

    Task-type inference logic
    -------------------------
    1. If ``config.task_type`` is explicitly set → use it (emit INFO record).
    2. If target dtype is bool or binary int → CLASSIFICATION.
    3. If target has <= ``_CLASSIFICATION_UNIQUE_THRESHOLD`` unique values
       relative to dataset size → CLASSIFICATION.
    4. Otherwise → REGRESSION.

    Class imbalance detection (classification only)
    -----------------------------------------------
    - CRITICAL if minority class < 5% of samples.
    - WARNING if minority class < 20% of samples.
    """

    _CLASSIFICATION_UNIQUE_THRESHOLD = 20  # absolute unique-value ceiling

    @classmethod
    def analyze(
        cls,
        df: pd.DataFrame,
        target_col: str,
        config: MentorConfig,
        log: DecisionLog,
    ) -> dict[str, Any]:
        """
        Analyse the target column and return a target profile dict.

        Parameters
        ----------
        df : pd.DataFrame
            Full dataset.
        target_col : str
            Name of the target column.
        config : MentorConfig
            Global config.
        log : DecisionLog
            Decision log to append findings to.

        Returns
        -------
        dict[str, Any]
            Keys: ``task_type``, ``n_classes``, ``class_counts``,
            ``imbalance_ratio``, ``target_dtype``.
        """
        target = df[target_col]
        task_type = cls._infer_task_type(target, config, log)
        result: dict[str, Any] = {
            "task_type": task_type,
            "target_col": target_col,
            "target_dtype": str(target.dtype),
            "n_unique": int(target.nunique()),
            "n_classes": None,
            "class_counts": None,
            "imbalance_ratio": None,
        }

        if task_type == "classification":
            cls._analyze_class_balance(target, target_col, result, log)

        return result

    @classmethod
    def _infer_task_type(
        cls,
        target: pd.Series,  # type: ignore[type-arg]
        config: MentorConfig,
        log: DecisionLog,
    ) -> str:
        # Explicit config override
        if config.task_type is not None:
            log.append(DecisionRecord(
                component="TargetAnalyzer",
                action="use_explicit_task_type",
                rationale=(
                    f"Task type '{config.task_type}' was explicitly set in MentorConfig. "
                    "Skipping automatic inference."
                ),
                data={"task_type": config.task_type, "source": "config"},
                severity=Severity.INFO,
            ))
            return config.task_type

        n_unique = target.nunique()
        dtype = target.dtype
        unique_vals = set(target.dropna().unique())

        # Boolean or binary integer → classification
        if pd.api.types.is_bool_dtype(dtype) or unique_vals <= {0, 1, 0.0, 1.0}:
            task_type = "classification"
            rationale = (
                f"Target column has dtype '{dtype}' with unique values "
                f"{sorted(unique_vals)}. Binary target → classification."
            )
        elif pd.api.types.is_numeric_dtype(dtype) and n_unique > cls._CLASSIFICATION_UNIQUE_THRESHOLD:
            task_type = "regression"
            rationale = (
                f"Target column is numeric with {n_unique} unique values "
                f"(> threshold of {cls._CLASSIFICATION_UNIQUE_THRESHOLD}). "
                "Treated as a continuous regression target."
            )
        elif not pd.api.types.is_numeric_dtype(dtype):
            task_type = "classification"
            rationale = (
                f"Target column is non-numeric (dtype='{dtype}') with "
                f"{n_unique} unique classes. Treated as classification."
            )
        else:
            task_type = "classification"
            rationale = (
                f"Target column is numeric with only {n_unique} unique values "
                f"(<= threshold {cls._CLASSIFICATION_UNIQUE_THRESHOLD}). "
                "Treated as a discrete classification target."
            )

        log.append(DecisionRecord(
            component="TargetAnalyzer",
            action="infer_task_type",
            rationale=rationale,
            data={"task_type": task_type, "n_unique": n_unique, "dtype": str(dtype)},
            severity=Severity.INFO,
        ))
        return task_type

    @staticmethod
    def _analyze_class_balance(
        target: pd.Series,  # type: ignore[type-arg]
        target_col: str,
        result: dict[str, Any],
        log: DecisionLog,
    ) -> None:
        """Detect class imbalance and populate result dict."""
        counts = target.value_counts()
        n_total = len(target.dropna())
        n_classes = len(counts)
        minority_pct = float(counts.min() / n_total)
        majority_pct = float(counts.max() / n_total)
        imbalance_ratio = float(counts.max() / counts.min()) if counts.min() > 0 else float("inf")

        result["n_classes"] = n_classes
        result["class_counts"] = {str(k): int(v) for k, v in counts.items()}
        result["imbalance_ratio"] = round(imbalance_ratio, 2)

        if minority_pct < 0.05:
            severity = Severity.CRITICAL
            rationale = (
                f"Target '{target_col}' has severe class imbalance: "
                f"minority class is only {minority_pct:.1%} of samples "
                f"(imbalance ratio: {imbalance_ratio:.1f}x). "
                "A naïve model will ignore the minority class entirely. "
                "Strongly recommend SMOTE oversampling, class_weight='balanced', "
                "or a threshold-moving strategy. Use F1-score or AUC-PR, not accuracy."
            )
        elif minority_pct < 0.20:
            severity = Severity.WARNING
            rationale = (
                f"Target '{target_col}' has moderate class imbalance: "
                f"minority class is {minority_pct:.1%} of samples "
                f"(imbalance ratio: {imbalance_ratio:.1f}x). "
                "Consider class_weight='balanced' and using F1-score for evaluation."
            )
        else:
            severity = Severity.INFO
            rationale = (
                f"Target '{target_col}' has {n_classes} classes. "
                f"Majority class: {majority_pct:.1%}, minority class: {minority_pct:.1%}. "
                "Class distribution is reasonably balanced."
            )

        log.append(DecisionRecord(
            component="TargetAnalyzer",
            action="analyze_class_balance",
            rationale=rationale,
            data={
                "n_classes": n_classes,
                "minority_pct": round(minority_pct, 4),
                "imbalance_ratio": round(imbalance_ratio, 2),
            },
            severity=severity,
        ))


# ---------------------------------------------------------------------------
# CorrelationAnalyzer — internal
# ---------------------------------------------------------------------------


class CorrelationAnalyzer:
    """
    Computes pairwise Pearson correlations between numeric features and
    flags near-duplicate pairs that may indicate multicollinearity.
    """

    @classmethod
    def analyze(
        cls,
        df: pd.DataFrame,
        config: MentorConfig,
        log: DecisionLog,
        target_col: str | None = None,
    ) -> dict[str, Any]:
        """
        Compute correlation matrix and flag high-correlation feature pairs.

        Parameters
        ----------
        df : pd.DataFrame
            Feature DataFrame (target column excluded before calling).
        config : MentorConfig
            ``correlation_threshold`` controls flagging sensitivity.
        log : DecisionLog
            Decision log.
        target_col : str | None
            If provided, target column is excluded from the correlation matrix.

        Returns
        -------
        dict[str, Any]
            Keys: ``high_correlation_pairs``, ``correlation_matrix``.
        """
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if target_col and target_col in numeric_cols:
            numeric_cols = [c for c in numeric_cols if c != target_col]

        if len(numeric_cols) < 2:
            log.append(DecisionRecord(
                component="CorrelationAnalyzer",
                action="skip_correlation",
                rationale=(
                    f"Only {len(numeric_cols)} numeric feature(s) found. "
                    "Pairwise correlation analysis requires at least 2 numeric columns."
                ),
                data={"n_numeric_cols": len(numeric_cols)},
                severity=Severity.INFO,
            ))
            return {"high_correlation_pairs": [], "correlation_matrix": {}}

        corr_matrix = df[numeric_cols].corr(method="pearson")
        high_pairs: list[dict[str, Any]] = []

        # Iterate upper triangle only (avoid duplicate pairs)
        for i, col_a in enumerate(numeric_cols):
            for col_b in numeric_cols[i + 1 :]:
                corr_val = corr_matrix.loc[col_a, col_b]
                if abs(corr_val) >= config.correlation_threshold:
                    high_pairs.append({
                        "col_a": col_a,
                        "col_b": col_b,
                        "correlation": round(float(corr_val), 4),
                    })
                    log.append(DecisionRecord(
                        component="CorrelationAnalyzer",
                        action="flag_high_correlation",
                        rationale=(
                            f"Features '{col_a}' and '{col_b}' have Pearson correlation "
                            f"{corr_val:.3f} (threshold: {config.correlation_threshold}). "
                            "These columns carry nearly identical information — keeping both "
                            "inflates the feature space without adding signal. "
                            "Recommend dropping or combining one of them in preprocessing."
                        ),
                        data={"col_a": col_a, "col_b": col_b, "correlation": round(float(corr_val), 4)},
                        severity=Severity.WARNING,
                    ))

        if not high_pairs:
            log.append(DecisionRecord(
                component="CorrelationAnalyzer",
                action="no_high_correlation",
                rationale=(
                    f"No feature pair exceeds the correlation threshold "
                    f"({config.correlation_threshold}). "
                    "Multicollinearity does not appear to be a concern."
                ),
                data={"threshold": config.correlation_threshold, "n_pairs_checked": len(numeric_cols) * (len(numeric_cols) - 1) // 2},
                severity=Severity.INFO,
            ))

        # Convert matrix to a serialisable nested dict
        corr_dict = {
            col: {c: round(float(v), 4) for c, v in row.items()}
            for col, row in corr_matrix.to_dict().items()
        }

        return {"high_correlation_pairs": high_pairs, "correlation_matrix": corr_dict}


# ---------------------------------------------------------------------------
# DatasetAnalyzer — public facade
# ---------------------------------------------------------------------------


class DatasetAnalyzer:
    """
    Intelligent dataset analyzer — the first stage of every mentorml pipeline.

    Implements the ``Analyzable`` protocol.

    This class orchestrates four internal analyzers:

    1. **Structural analysis** — rows, columns, duplicates, memory.
    2. **Column profiling** — per-column dtype, missing values, outliers,
       distribution shape.
    3. **Target analysis** — task type inference, class imbalance.
    4. **Correlation analysis** — multicollinearity detection.

    Every finding is emitted as a ``DecisionRecord`` to the provided
    ``DecisionLog``.

    Parameters
    ----------
    config : MentorConfig
        Global configuration controlling thresholds and behaviour.

    Examples
    --------
    ::

        import pandas as pd
        from mentorml import MentorConfig
        from mentorml.core import DecisionLog
        from mentorml.analysis import DatasetAnalyzer

        df = pd.read_csv("titanic.csv")
        config = MentorConfig(target_column="Survived")
        log = DecisionLog()

        analyzer = DatasetAnalyzer(config)
        results = analyzer.analyze(df, log)

        print(results["target"]["task_type"])       # "classification"
        print(results["n_columns_flagged"])          # int
        for record in log.filter(log.Severity.WARNING):
            print(record)
    """

    def __init__(self, config: MentorConfig) -> None:
        self.config = config

    def analyze(
        self,
        df: pd.DataFrame,
        log: DecisionLog,
    ) -> dict[str, Any]:
        """
        Run the full analysis pipeline on ``df``.

        Parameters
        ----------
        df : pd.DataFrame
            Raw input dataset.  Must not be empty.
        log : DecisionLog
            Decision log.  All findings are appended here.

        Returns
        -------
        dict[str, Any]
            Structured analysis result with keys:

            - ``n_rows`` / ``n_cols`` — dataset dimensions
            - ``n_duplicates`` / ``duplicate_pct`` — duplicate row statistics
            - ``memory_mb`` — approximate in-memory size
            - ``column_profiles`` — ``dict[str, ColumnProfile]``
            - ``target`` — target analysis dict (or ``None``)
            - ``correlation`` — correlation analysis dict
            - ``flagged_columns`` — categorised column flags
            - ``n_columns_flagged`` — total flagged column count

        Raises
        ------
        DataValidationError
            If ``df`` is empty or has no columns.
        InsufficientDataError
            If ``df`` has fewer than ``_MIN_ROWS`` rows.
        """
        self._validate_input(df, log)

        log.append(DecisionRecord(
            component="DatasetAnalyzer",
            action="start_analysis",
            rationale=(
                f"Beginning dataset analysis: {df.shape[0]:,} rows × "
                f"{df.shape[1]} columns. "
                f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB."
            ),
            data={
                "n_rows": df.shape[0],
                "n_cols": df.shape[1],
                "columns": list(df.columns),
                "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
            },
            severity=Severity.INFO,
        ))

        # 1. Structural analysis
        n_duplicates, duplicate_pct = self._analyze_duplicates(df, log)

        # 2. Column profiling
        column_profiles = self._profile_all_columns(df, log)

        # 3. Flag structural issues
        flagged = self._flag_column_issues(column_profiles, log)

        # 4. Target analysis (if target_column configured)
        target_result: dict[str, Any] | None = None
        if self.config.target_column:
            if self.config.target_column in df.columns:
                target_result = TargetAnalyzer.analyze(
                    df, self.config.target_column, self.config, log
                )
            else:
                log.append(DecisionRecord(
                    component="DatasetAnalyzer",
                    action="target_column_missing",
                    rationale=(
                        f"Configured target column '{self.config.target_column}' "
                        "was not found in the DataFrame. "
                        "Target analysis will be skipped."
                    ),
                    data={
                        "configured_target": self.config.target_column,
                        "available_columns": list(df.columns),
                    },
                    severity=Severity.WARNING,
                ))

        # 5. Correlation analysis (exclude target)
        df_features = df.drop(columns=[self.config.target_column]) if (
            self.config.target_column and self.config.target_column in df.columns
        ) else df
        correlation_result = CorrelationAnalyzer.analyze(
            df_features, self.config, log, target_col=self.config.target_column
        )

        # 6. Summary record
        n_flagged = len(flagged["constant"]) + len(flagged["id_like"]) + len(flagged["high_missing"])
        log.append(DecisionRecord(
            component="DatasetAnalyzer",
            action="analysis_complete",
            rationale=(
                f"Dataset analysis complete. "
                f"{n_flagged} column(s) flagged for attention: "
                f"{len(flagged['constant'])} constant, "
                f"{len(flagged['id_like'])} ID-like, "
                f"{len(flagged['high_missing'])} high-missing. "
                f"{len(correlation_result['high_correlation_pairs'])} high-correlation pair(s) detected."
            ),
            data={
                "n_flagged_columns": n_flagged,
                "n_high_correlation_pairs": len(correlation_result["high_correlation_pairs"]),
                "flagged": flagged,
            },
            severity=Severity.INFO if n_flagged == 0 else Severity.WARNING,
        ))

        return {
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
            "n_duplicates": n_duplicates,
            "duplicate_pct": round(duplicate_pct, 4),
            "column_profiles": column_profiles,
            "target": target_result,
            "correlation": correlation_result,
            "flagged_columns": flagged,
            "n_columns_flagged": n_flagged,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_input(self, df: pd.DataFrame, log: DecisionLog) -> None:
        """Raise early if the DataFrame cannot be analysed."""
        if df.empty or len(df.columns) == 0:
            raise DataValidationError(
                "Input DataFrame is empty — cannot analyse.",
                context={"shape": df.shape},
            )
        if len(df) < _MIN_ROWS:
            raise InsufficientDataError(
                f"Dataset has only {len(df)} rows. "
                f"At least {_MIN_ROWS} rows are required for reliable analysis.",
                context={"n_rows": len(df), "minimum": _MIN_ROWS},
            )

    def _analyze_duplicates(
        self, df: pd.DataFrame, log: DecisionLog
    ) -> tuple[int, float]:
        """Detect and report duplicate rows."""
        n_duplicates = int(df.duplicated().sum())
        duplicate_pct = n_duplicates / len(df)

        if n_duplicates > 0:
            severity = Severity.WARNING if duplicate_pct > 0.05 else Severity.INFO
            log.append(DecisionRecord(
                component="DatasetAnalyzer",
                action="detect_duplicates",
                rationale=(
                    f"Found {n_duplicates:,} duplicate rows ({duplicate_pct:.1%} of dataset). "
                    + (
                        "This level of duplication is significant and likely indicates a "
                        "data pipeline issue (e.g., double-joins, repeated ingestion). "
                        "Duplicates will artificially inflate model confidence."
                        if duplicate_pct > 0.05
                        else "A small number of exact duplicates is common in real datasets."
                    )
                ),
                data={"n_duplicates": n_duplicates, "duplicate_pct": round(duplicate_pct, 4)},
                severity=severity,
            ))
        else:
            log.append(DecisionRecord(
                component="DatasetAnalyzer",
                action="detect_duplicates",
                rationale="No duplicate rows detected. Dataset appears structurally clean.",
                data={"n_duplicates": 0},
                severity=Severity.INFO,
            ))

        return n_duplicates, duplicate_pct

    def _profile_all_columns(
        self, df: pd.DataFrame, log: DecisionLog
    ) -> dict[str, ColumnProfile]:
        """Profile every column and emit per-column findings."""
        profiles: dict[str, ColumnProfile] = {}
        n_rows = len(df)

        for col in df.columns:
            profile = ColumnProfiler.profile(df[col], n_rows, self.config)
            profiles[col] = profile

            # Emit finding for missing values
            if profile.missing_pct > 0:
                if profile.missing_pct >= self.config.missing_threshold:
                    severity = Severity.WARNING
                    action = "flag_high_missing"
                    rationale = (
                        f"Column '{col}' has {profile.missing_pct:.1%} missing values "
                        f"(>= drop threshold {self.config.missing_threshold:.0%}). "
                        "This column will be a candidate for removal in preprocessing."
                    )
                else:
                    severity = Severity.INFO
                    action = "note_missing_values"
                    rationale = (
                        f"Column '{col}' has {profile.missing_pct:.1%} missing values "
                        f"(< drop threshold {self.config.missing_threshold:.0%}). "
                        "Will be imputed in preprocessing."
                    )
                log.append(DecisionRecord(
                    component="DatasetAnalyzer",
                    action=action,
                    rationale=rationale,
                    data={"column": col, "missing_pct": round(profile.missing_pct, 4), "n_missing": profile.n_missing},
                    severity=severity,
                ))

            # Emit finding for skewed numeric columns
            if profile.skewness is not None and abs(profile.skewness) > 1.0:
                direction = "right (positive)" if profile.skewness > 0 else "left (negative)"
                log.append(DecisionRecord(
                    component="DatasetAnalyzer",
                    action="flag_skewed_distribution",
                    rationale=(
                        f"Column '{col}' is {direction}-skewed (skewness={profile.skewness:.2f}). "
                        "Skewed distributions can degrade linear model performance. "
                        "A log or Box-Cox transform will be considered in preprocessing."
                    ),
                    data={"column": col, "skewness": profile.skewness, "kurtosis": profile.kurtosis},
                    severity=Severity.INFO,
                ))

            # Emit finding for outliers
            if profile.outlier_pct is not None and profile.outlier_pct > 0.05:
                log.append(DecisionRecord(
                    component="DatasetAnalyzer",
                    action="flag_outliers",
                    rationale=(
                        f"Column '{col}' has {profile.n_outliers} outlier(s) "
                        f"({profile.outlier_pct:.1%} of non-missing values, IQR method). "
                        "High outlier density may distort mean-based statistics and "
                        "degrade tree-based model splits."
                    ),
                    data={"column": col, "n_outliers": profile.n_outliers, "outlier_pct": round(profile.outlier_pct, 4)},
                    severity=Severity.WARNING,
                ))

        return profiles

    def _flag_column_issues(
        self,
        profiles: dict[str, ColumnProfile],
        log: DecisionLog,
    ) -> dict[str, list[str]]:
        """
        Identify constant, ID-like, and high-missing columns.

        Returns a dict of ``{issue_type: [column_names]}``.
        """
        constant_cols = [name for name, p in profiles.items() if p.is_constant]
        id_like_cols = [name for name, p in profiles.items() if p.is_id_like]
        high_missing_cols = [
            name for name, p in profiles.items()
            if p.missing_pct >= self.config.missing_threshold
        ]

        if constant_cols:
            log.append(DecisionRecord(
                component="DatasetAnalyzer",
                action="flag_constant_columns",
                rationale=(
                    f"Columns {constant_cols} have zero variance — every row has the same value. "
                    "Constant features carry no predictive signal and must be dropped before training. "
                    "Their presence often indicates a data pipeline or filtering bug."
                ),
                data={"columns": constant_cols, "n_constant": len(constant_cols)},
                severity=Severity.WARNING,
            ))

        if id_like_cols:
            log.append(DecisionRecord(
                component="DatasetAnalyzer",
                action="flag_id_like_columns",
                rationale=(
                    f"Columns {id_like_cols} have very high cardinality (>95% unique values). "
                    "These look like identifier columns (e.g., user_id, order_id). "
                    "Identifiers should never be used as model features — they cause overfitting "
                    "and don't generalise to unseen data."
                ),
                data={"columns": id_like_cols, "n_id_like": len(id_like_cols)},
                severity=Severity.WARNING,
            ))

        return {
            "constant": constant_cols,
            "id_like": id_like_cols,
            "high_missing": high_missing_cols,
        }