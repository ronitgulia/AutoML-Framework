"""
mentorml.features.feature_engineer
-------------------------------------
Phase 4: Feature Engineering

Applies feature engineering transformations guided by the ``AnalysisResult``
from Phase 1.  Every transformation is narrated via ``DecisionRecord``.

Transformations applied
-----------------------
1. **Log transform** for heavily right-skewed numeric columns (skewness > 1).
2. **Drop near-duplicate features** (Pearson r > correlation_threshold from config).
3. **Polynomial interaction** for the top-2 correlated numeric feature pairs.
4. **Boolean flags** for columns with notable outlier rates.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from mentorml.config import MentorConfig
from mentorml.core.decision import DecisionLog, DecisionRecord, Severity
from mentorml.core.exceptions import ComponentNotFittedError

logger = logging.getLogger(__name__)

_SKEW_THRESHOLD = 1.0
_LOG_SHIFT = 1.0  # added before log to handle zeros


def _get_profile_attr(profile, key, default=None):
    """Get attribute from ColumnProfile dataclass or plain dict."""
    if isinstance(profile, dict):
        return profile.get(key, default)
    return getattr(profile, key, default)

class FeatureEngineer:
    """
    Phase 4: Explainable feature engineering.

    Learns from the dataset which transformations to apply, then
    deterministically applies them during ``transform()``.

    Parameters
    ----------
    config : MentorConfig
        Global configuration.

    Examples
    --------
    ::

        fe = FeatureEngineer(config)
        fe.fit(df, log, analysis_result)
        df_features = fe.transform(df, log)
    """

    def __init__(self, config: MentorConfig) -> None:
        self.config = config
        self._fitted = False
        self._log_transform_cols: list[str] = []
        self._drop_correlated: list[str] = []
        self._interaction_pairs: list[tuple[str, str]] = []
        self._outlier_flag_cols: list[str] = []
        self._outlier_fences: dict[str, tuple[float, float]] = {}

    # ------------------------------------------------------------------
    # Fittable interface
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        log: DecisionLog,
        analysis_result: dict[str, Any] | None = None,
    ) -> "FeatureEngineer":
        """
        Learn feature engineering parameters from ``df``.

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed training data (numeric only expected).
        log : DecisionLog
            Decision log to record engineering choices.
        analysis_result : dict | None
            Output of ``DatasetAnalyzer.analyze()``.

        Returns
        -------
        FeatureEngineer
            ``self`` for method chaining.
        """
        log.append(
            DecisionRecord(
                component="FeatureEngineer",
                action="fit_start",
                rationale=(
                    f"Starting feature engineering fit on "
                    f"{len(df)} rows × {len(df.columns)} columns."
                ),
                data={},

                severity=Severity.INFO,
            )
        )

        numeric_df = df.select_dtypes(include="number")
        profiles = (analysis_result or {}).get("column_profiles", {})

        # 1. Log transform for right-skewed columns
        self._learn_log_transforms(numeric_df, profiles, log)

        # 2. Drop near-duplicate correlated columns
        self._learn_correlation_drops(numeric_df, log)

        # 3. Polynomial interactions for top correlated pairs
        self._learn_interactions(numeric_df, log)

        # 4. Outlier flag columns
        self._learn_outlier_flags(numeric_df, profiles, log)

        self._fitted = True

        log.append(
            DecisionRecord(
                component="FeatureEngineer",
                action="fit_complete",
                rationale=(
                    f"Feature engineering fit complete. "
                    f"Log-transforms: {self._log_transform_cols}. "
                    f"Drop correlated: {self._drop_correlated}. "
                    f"Interactions: {self._interaction_pairs}. "
                    f"Outlier flags: {self._outlier_flag_cols}."
                ),
                severity=Severity.INFO,
                data={
                    "log_transform_cols": self._log_transform_cols,
                    "drop_correlated": self._drop_correlated,
                    "interaction_pairs": [list(p) for p in self._interaction_pairs],
                    "outlier_flag_cols": self._outlier_flag_cols,
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
        Apply learned feature engineering transformations to ``df``.

        Parameters
        ----------
        df : pd.DataFrame
            Data to transform.
        log : DecisionLog
            Decision log to record transform actions.

        Returns
        -------
        pd.DataFrame
            Feature-engineered DataFrame.

        Raises
        ------
        ComponentNotFittedError
            If called before ``fit()``.
        """
        if not self._fitted:
            raise ComponentNotFittedError(
                "FeatureEngineer",
                "Call fit() before transform().",
            )

        df_out = df.copy()

        # Log transforms
        for col in self._log_transform_cols:
            if col in df_out.columns:
                df_out[f"{col}_log"] = np.log1p(df_out[col].clip(lower=0))

        # Drop correlated
        to_drop = [c for c in self._drop_correlated if c in df_out.columns]
        if to_drop:
            df_out.drop(columns=to_drop, inplace=True)

        # Interaction features
        for col_a, col_b in self._interaction_pairs:
            if col_a in df_out.columns and col_b in df_out.columns:
                df_out[f"{col_a}_x_{col_b}"] = df_out[col_a] * df_out[col_b]

        # Outlier flag features
        for col in self._outlier_flag_cols:
            if col in df_out.columns and col in self._outlier_fences:
                lower, upper = self._outlier_fences[col]
                df_out[f"{col}_is_outlier"] = (
                    (df_out[col] < lower) | (df_out[col] > upper)
                ).astype(int)

        log.append(
            DecisionRecord(
                component="FeatureEngineer",
                action="transform_complete",
                rationale=(
                    f"Feature engineering transform produced "
                    f"{len(df_out)} rows × {len(df_out.columns)} columns."
                ),
                severity=Severity.INFO,
                data={"output_shape": [len(df_out), len(df_out.columns)]},
            )
        )
        return df_out

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _learn_log_transforms(
        self,
        numeric_df: pd.DataFrame,
        profiles: dict[str, Any],
        log: DecisionLog,
    ) -> None:
        """Flag right-skewed columns for log1p transform."""
        for col in numeric_df.columns:
            series = numeric_df[col].dropna()
            if len(series) < 4:
                continue
            profile = profiles.get(col, {})
            skew = _get_profile_attr(profile, "skewness", None) or float(series.skew())
            if skew > _SKEW_THRESHOLD:
                self._log_transform_cols.append(col)
                log.append(
                    DecisionRecord(
                        component="FeatureEngineer",
                        action=f"log_transform:{col}",
                        rationale=(
                            f"Creating log1p({col}) feature. "
                            f"Skewness={skew:.2f} > {_SKEW_THRESHOLD} indicates "
                            "a right-skewed distribution; log-transform compresses "
                            "the tail and improves linear model performance."
                        ),
                        severity=Severity.INFO,
                        data={"column": col, "skewness": round(skew, 4)},
                    )
                )

    def _learn_correlation_drops(
        self,
        numeric_df: pd.DataFrame,
        log: DecisionLog,
    ) -> None:
        """Drop one column from each highly-correlated pair."""
        if len(numeric_df.columns) < 2:
            return

        corr = numeric_df.corr().abs()
        dropped: set[str] = set()
        cols = list(numeric_df.columns)

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a, b = cols[i], cols[j]
                if a in dropped or b in dropped:
                    continue
                r = corr.loc[a, b]
                if r >= self.config.correlation_threshold:
                    # Drop the one with more missing
                    drop_col = b
                    keep_col = a
                    self._drop_correlated.append(drop_col)
                    dropped.add(drop_col)
                    log.append(
                        DecisionRecord(
                            component="FeatureEngineer",
                            action=f"drop_correlated:{drop_col}",
                            rationale=(
                                f"Dropping '{drop_col}' (r={r:.3f} with '{keep_col}', "
                                f"≥ correlation_threshold={self.config.correlation_threshold}). "
                                "Near-duplicate features add noise without new information."
                            ),
                            severity=Severity.WARNING,
                            data={
                                "drop_col": drop_col,
                                "keep_col": keep_col,
                                "pearson_r": round(float(r), 4),
                            },
                        )
                    )

    def _learn_interactions(
        self,
        numeric_df: pd.DataFrame,
        log: DecisionLog,
    ) -> None:
        """Create interaction features for the top-2 correlated pairs."""
        if len(numeric_df.columns) < 2:
            return

        remaining = [
            c for c in numeric_df.columns if c not in self._drop_correlated
        ]
        if len(remaining) < 2:
            return

        corr = numeric_df[remaining].corr().abs()
        pairs: list[tuple[float, str, str]] = []
        for i in range(len(remaining)):
            for j in range(i + 1, len(remaining)):
                a, b = remaining[i], remaining[j]
                r = float(corr.loc[a, b])
                if r < self.config.correlation_threshold:
                    pairs.append((r, a, b))

        pairs.sort(key=lambda x: x[0], reverse=True)
        top_pairs = pairs[:2]

        for r_val, a, b in top_pairs:
            self._interaction_pairs.append((a, b))
            log.append(
                DecisionRecord(
                    component="FeatureEngineer",
                    action=f"interaction_feature:{a}_x_{b}",
                    rationale=(
                        f"Creating interaction feature '{a}_x_{b}' "
                        f"(Pearson r={r_val:.3f}). "
                        "Multiplicative interactions can capture non-linear "
                        "relationships that linear models miss."
                    ),
                    severity=Severity.INFO,
                    data={"col_a": a, "col_b": b, "pearson_r": round(r_val, 4)},
                )
            )

    def _learn_outlier_flags(
        self,
        numeric_df: pd.DataFrame,
        profiles: dict[str, Any],
        log: DecisionLog,
    ) -> None:
        """Create binary outlier-flag features for columns with high outlier rates."""
        for col in numeric_df.columns:
            series = numeric_df[col].dropna()
            if len(series) < 4:
                continue
            profile = profiles.get(col, {})
            outlier_pct = _get_profile_attr(profile, "outlier_pct", None) or 0.0
            if outlier_pct < 0.05:
                continue

            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            self._outlier_flag_cols.append(col)
            self._outlier_fences[col] = (lower, upper)

            log.append(
                DecisionRecord(
                    component="FeatureEngineer",
                    action=f"outlier_flag:{col}",
                    rationale=(
                        f"Creating binary outlier flag '{col}_is_outlier' "
                        f"({outlier_pct:.1%} outlier rate). "
                        "The flag lets the model learn a separate pattern for "
                        "extreme values without distorting the main distribution."
                    ),
                    severity=Severity.INFO,
                    data={
                        "column": col,
                        "outlier_pct": round(outlier_pct, 4),
                        "iqr_lower": round(lower, 4),
                        "iqr_upper": round(upper, 4),
                    },
                )
            )