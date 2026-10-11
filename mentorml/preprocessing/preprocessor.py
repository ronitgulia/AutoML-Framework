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

