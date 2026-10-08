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
