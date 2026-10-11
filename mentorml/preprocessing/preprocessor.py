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
    # Fittable interface
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        log: DecisionLog,
        analysis_result: dict[str, Any] | None = None,
        quality_report: dict[str, Any] | None = None,
    ) -> "ExplainablePreprocessor":
        """
        Learn transformation parameters from ``df``.

        Parameters
        ----------
        df : pd.DataFrame
            Training data (must include target column if present).
        log : DecisionLog
            Decision log to append fitting decisions to.
        analysis_result : dict | None
            Output of ``DatasetAnalyzer.analyze()``.  Used to read column
            profiles (dtype categories, skewness, id-like flags).
        quality_report : dict | None
            Output of ``DataQualityAssessor.analyze()``.  Used to identify
            columns to drop and impute.

        Returns
        -------
        ExplainablePreprocessor
            Returns ``self`` to allow method chaining.
        """
        log.append(
            DecisionRecord(
                component="ExplainablePreprocessor",
                action="fit_start",
                rationale=(
                    f"Fitting preprocessor on {len(df)} rows × "
                    f"{len(df.columns)} columns."
                ),
                data={},

                severity=Severity.INFO,
            )
        )

        # --- 1. Determine columns to drop ---
        self._cols_to_drop = self._decide_drops(df, log, analysis_result, quality_report)
        df_work = df.drop(columns=self._cols_to_drop, errors="ignore").copy()

        # --- 2. Classify remaining columns ---
        self._classify_columns(df_work, log, analysis_result)

        # --- 3. Learn imputation values ---
        self._learn_imputation(df_work, log)

        # --- 4. Learn scaler ---
        self._learn_scaler(df_work, log, analysis_result)

        # --- 5. Learn encoders ---
        self._learn_encoders(df_work, log)

        self._fitted = True

        log.append(
            DecisionRecord(
                component="ExplainablePreprocessor",
                action="fit_complete",
                rationale=(
                    f"Preprocessing fit complete. "
                    f"Dropping {len(self._cols_to_drop)} columns, "
                    f"imputing {len(self._impute_values)}, "
                    f"OHE on {self._ohe_cols}, "
                    f"label-encoding on {self._label_enc_cols}."
                ),
                severity=Severity.INFO,
                data={
                    "cols_to_drop": self._cols_to_drop,
                    "cols_to_impute": list(self._impute_values.keys()),
                    "ohe_cols": self._ohe_cols,
                    "label_enc_cols": self._label_enc_cols,
                    "scaler": type(self._scaler).__name__ if self._scaler else "none",
                },
            )
        )
        return self

    def transform(
        self,
        df: pd.DataFrame,
        log: DecisionLog,
    ) -> pd.DataFrame:
        """
        Apply the learned transformations to ``df``.

        Parameters
        ----------
        df : pd.DataFrame
            Data to transform (training or held-out).
        log : DecisionLog
            Decision log to append transform records to.

        Returns
        -------
        pd.DataFrame
            Fully preprocessed, model-ready DataFrame.

        Raises
        ------
        ComponentNotFittedError
            If called before ``fit()``.
        """
        if not self._fitted:
            raise ComponentNotFittedError(
                "ExplainablePreprocessor",
                "Call fit() before transform().",
            )

        df_out = df.drop(columns=self._cols_to_drop, errors="ignore").copy()

        # Imputation
        for col, value in self._impute_values.items():
            if col in df_out.columns:
                df_out[col] = df_out[col].fillna(value)

        # Datetime extraction
        for col in self._datetime_cols:
            if col in df_out.columns:
                dt = pd.to_datetime(df_out[col], errors="coerce")
                df_out[f"{col}_year"] = dt.dt.year
                df_out[f"{col}_month"] = dt.dt.month
                df_out[f"{col}_day"] = dt.dt.day
                df_out[f"{col}_dayofweek"] = dt.dt.dayofweek
                df_out.drop(columns=[col], inplace=True)

        # Scaling
        if self._scaler is not None and self._scaler_cols:
            valid = [c for c in self._scaler_cols if c in df_out.columns]
            if valid:
                df_out[valid] = self._scaler.transform(df_out[valid])

        # OHE
        for col in self._ohe_cols:
            if col not in df_out.columns:
                continue
            categories = self._ohe_categories[col]
            for cat in categories:
                df_out[f"{col}__{cat}"] = (df_out[col] == cat).astype(int)
            df_out.drop(columns=[col], inplace=True)

        # Label encoding
        for col, enc in self._label_encoders.items():
            if col not in df_out.columns:
                continue
            df_out[col] = df_out[col].astype(str).map(
                lambda x, e=enc: (
                    e.transform([x])[0]
                    if x in e.classes_
                    else -1
                )
            )

        log.append(
            DecisionRecord(
                component="ExplainablePreprocessor",
                action="transform_complete",
                rationale=(
                    f"Transform produced {len(df_out)} rows × "
                    f"{len(df_out.columns)} columns."
                ),
                severity=Severity.INFO,
                data={"output_shape": [len(df_out), len(df_out.columns)]},
            )
        )
        return df_out

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

    def _learn_encoders(self, df: pd.DataFrame, log: DecisionLog) -> None:
        """Learn OHE (low cardinality) or label encoding (high cardinality)."""
        for col in self._categorical_cols:
            n_unique = df[col].nunique(dropna=True)
            if n_unique <= self.config.cardinality_threshold:
                cats = sorted(df[col].dropna().unique().tolist())
                self._ohe_cols.append(col)
                self._ohe_categories[col] = [str(c) for c in cats]
                log.append(
                    DecisionRecord(
                        component="ExplainablePreprocessor",
                        action=f"encoding_strategy:{col}",
                        rationale=(
                            f"One-hot encoding '{col}' ({n_unique} unique values ≤ "
                            f"cardinality_threshold={self.config.cardinality_threshold}). "
                            "OHE avoids implicit ordinal relationships."
                        ),
                        severity=Severity.INFO,
                        data={
                            "column": col,
                            "strategy": "one_hot",
                            "n_categories": n_unique,
                        },
                    )
                )
            else:
                enc = LabelEncoder()
                enc.fit(df[col].fillna("UNKNOWN").astype(str))
                self._label_enc_cols.append(col)
                self._label_encoders[col] = enc
                log.append(
                    DecisionRecord(
                        component="ExplainablePreprocessor",
                        action=f"encoding_strategy:{col}",
                        rationale=(
                            f"Label-encoding '{col}' ({n_unique} unique values > "
                            f"cardinality_threshold={self.config.cardinality_threshold}). "
                            "OHE would create too many columns; ordinal encoding used."
                        ),
                        severity=Severity.INFO,
                        data={
                            "column": col,
                            "strategy": "label_encode",
                            "n_categories": n_unique,
                        },
                    )
                )