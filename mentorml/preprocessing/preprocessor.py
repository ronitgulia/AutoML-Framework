"""
mentorml.preprocessing.preprocessor
--------------------------------------
Phase 3: Explainable Preprocessor

``ExplainablePreprocessor`` learns transformation strategies from the
``AnalysisResult`` (Phase 1) and ``QualityReport`` (Phase 2), then applies
them to produce a clean, model-ready DataFrame.

Every decision — which imputation method, which scaler, which encoder — is
narrated via a ``DecisionRecord`` so the pipeline audit trail answers *why*
each transformation was chosen.

Design
------
- Follows sklearn's ``fit / transform`` pattern (implements ``Fittable``).
- ``fit()`` learns parameters (medians, modes, encoder mappings) and records
  every decision.
- ``transform()`` applies the learned parameters deterministically.
- Raises ``ComponentNotFittedError`` if ``transform`` is called before ``fit``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from mentorml.config import MentorConfig
from mentorml.core.decision import DecisionLog, DecisionRecord, Severity
from mentorml.core.exceptions import ComponentNotFittedError

logger = logging.getLogger(__name__)

# Skewness threshold above which log-transform is applied
_SKEW_THRESHOLD = 1.0


def _get_profile_attr(profile, key, default=None):
    """Get attribute from ColumnProfile dataclass or plain dict."""
    if isinstance(profile, dict):
        return profile.get(key, default)
    return getattr(profile, key, default)

class ExplainablePreprocessor:
    """
    Phase 3: Explainable data preprocessing.

    Learns and applies:

    1. **Column drops** — drops columns flagged by QualityReport as candidates
       for removal (missing > threshold, ID-like, constant).
    2. **Imputation** — median for numeric, mode for categorical.
    3. **Scaling** — StandardScaler by default; MinMaxScaler for non-negative
       heavy-tailed distributions.
    4. **Encoding** — One-hot encoding for low-cardinality categoricals;
       ordinal (label) encoding for high-cardinality.
    5. **Datetime extraction** — extracts year, month, day, dayofweek from
       datetime columns.

    Every strategy is justified with a ``DecisionRecord``.

    Parameters
    ----------
    config : MentorConfig
        Global configuration (uses ``cardinality_threshold``).

    Examples
    --------
    ::

        prep = ExplainablePreprocessor(config)
        prep.fit(df, log, analysis_result, quality_report)
        df_clean = prep.transform(df, log)
    """

    def __init__(self, config: MentorConfig) -> None:
        self.config = config
        self._fitted = False

        # Learned parameters (set during fit)
        self._cols_to_drop: list[str] = []
        self._numeric_cols: list[str] = []
        self._categorical_cols: list[str] = []
        self._datetime_cols: list[str] = []
        self._impute_values: dict[str, Any] = {}
        self._scaler: StandardScaler | MinMaxScaler | None = None
        self._scaler_cols: list[str] = []
        self._ohe_cols: list[str] = []
        self._ohe_categories: dict[str, list[str]] = {}
        self._label_enc_cols: list[str] = []
        self._label_encoders: dict[str, LabelEncoder] = {}
        self._output_columns: list[str] = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _decide_drops(
        self,
        df: pd.DataFrame,
        log: DecisionLog,
        analysis_result: dict[str, Any] | None,
        quality_report: dict[str, Any] | None,
    ) -> list[str]:
        """Decide which columns to drop and record each decision."""
        to_drop: set[str] = set()

        # From quality report (high missing)
        if quality_report:
            for col in quality_report.get("columns_to_drop", []):
                if col in df.columns:
                    to_drop.add(col)
                    log.append(
                        DecisionRecord(
                            component="ExplainablePreprocessor",
                            action=f"drop_column:{col}",
                            rationale=(
                                f"Dropping '{col}': missing values exceed "
                                f"configured threshold "
                                f"({self.config.missing_threshold:.0%})."
                            ),
                            severity=Severity.WARNING,
                            data={"column": col, "reason": "missing_critical"},
                        )
                    )

        # From analysis result (id-like, constant)
        if analysis_result:
            for col, profile in analysis_result.get("column_profiles", {}).items():
                if col not in df.columns:
                    continue
                if _get_profile_attr(profile, "is_constant", None):
                    to_drop.add(col)
                    log.append(
                        DecisionRecord(
                            component="ExplainablePreprocessor",
                            action=f"drop_column:{col}",
                            rationale=(
                                f"Dropping '{col}': constant column (zero variance) "
                                "carries no predictive information."
                            ),
                            severity=Severity.INFO,
                            data={"column": col, "reason": "constant"},
                        )
                    )
                elif _get_profile_attr(profile, "is_id_like", None):
                    to_drop.add(col)
                    log.append(
                        DecisionRecord(
                            component="ExplainablePreprocessor",
                            action=f"drop_column:{col}",
                            rationale=(
                                f"Dropping '{col}': ID-like column "
                                "(near-unique cardinality, no predictive value)."
                            ),
                            severity=Severity.INFO,
                            data={"column": col, "reason": "id_like"},
                        )
                    )

        return list(to_drop)

    def _classify_columns(
        self,
        df: pd.DataFrame,
        log: DecisionLog,
        analysis_result: dict[str, Any] | None,
    ) -> None:
        """Classify columns into numeric, categorical, datetime buckets."""
        profiles = (analysis_result or {}).get("column_profiles", {})

        for col in df.columns:
            profile = profiles.get(col, {})
            cat = _get_profile_attr(profile, "dtype_category", "")

            if cat == "datetime" or pd.api.types.is_datetime64_any_dtype(df[col]):
                self._datetime_cols.append(col)
            elif cat == "numeric" or pd.api.types.is_numeric_dtype(df[col]):
                self._numeric_cols.append(col)
            elif cat in ("categorical", "boolean", "text") or df[col].dtype == object:
                self._categorical_cols.append(col)

    def _learn_imputation(self, df: pd.DataFrame, log: DecisionLog) -> None:
        """Learn median (numeric) or mode (categorical) imputation values."""
        for col in self._numeric_cols:
            if df[col].isna().any():
                val = float(df[col].median())
                self._impute_values[col] = val
                log.append(
                    DecisionRecord(
                        component="ExplainablePreprocessor",
                        action=f"impute_strategy:{col}",
                        rationale=(
                            f"Imputing '{col}' with median={val:.4g}. "
                            "Median is robust to outliers for numeric columns."
                        ),
                        severity=Severity.INFO,
                        data={"column": col, "strategy": "median", "value": val},
                    )
                )

        for col in self._categorical_cols:
            if df[col].isna().any():
                mode_vals = df[col].mode()
                val = str(mode_vals.iloc[0]) if len(mode_vals) > 0 else "UNKNOWN"
                self._impute_values[col] = val
                log.append(
                    DecisionRecord(
                        component="ExplainablePreprocessor",
                        action=f"impute_strategy:{col}",
                        rationale=(
                            f"Imputing '{col}' with mode='{val}'. "
                            "Mode preserves the most common category."
                        ),
                        severity=Severity.INFO,
                        data={"column": col, "strategy": "mode", "value": val},
                    )
                )

    def _learn_scaler(
        self,
        df: pd.DataFrame,
        log: DecisionLog,
        analysis_result: dict[str, Any] | None,
    ) -> None:
        """Learn StandardScaler or MinMaxScaler for numeric columns."""
        if not self._numeric_cols:
            return

        self._scaler_cols = self._numeric_cols.copy()

        # Use MinMaxScaler if any column is non-negative with high skew
        profiles = (analysis_result or {}).get("column_profiles", {})
        use_minmax = False
        for col in self._scaler_cols:
            profile = profiles.get(col, {})
            skew = _get_profile_attr(profile, "skewness", None) or 0.0
            min_val = _get_profile_attr(profile, "min_val", None) or 0.0
            if abs(skew) > _SKEW_THRESHOLD and min_val >= 0:
                use_minmax = True
                break

        if use_minmax:
            self._scaler = MinMaxScaler()
            scaler_name = "MinMaxScaler"
            rationale = (
                "Applying MinMaxScaler: detected non-negative highly-skewed "
                "numeric columns. MinMaxScaler preserves the zero lower bound."
            )
        else:
            self._scaler = StandardScaler()
            scaler_name = "StandardScaler"
            rationale = (
                "Applying StandardScaler (zero mean, unit variance). "
                "Default choice for normally-distributed numeric features."
            )

        # Fit on non-null values (after imputation fill for fitting)
        df_numeric = df[self._scaler_cols].copy()
        for col in self._scaler_cols:
            if df_numeric[col].isna().any():
                df_numeric[col] = df_numeric[col].fillna(
                    self._impute_values.get(col, df_numeric[col].median())
                )
        self._scaler.fit(df_numeric)

        log.append(
            DecisionRecord(
                component="ExplainablePreprocessor",
                action=f"scaler_selected:{scaler_name}",
                rationale=rationale,
                severity=Severity.INFO,
                data={"scaler": scaler_name, "columns": self._scaler_cols},
            )
        )

