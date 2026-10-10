"""
mentorml.modeling.model_selector
----------------------------------
Phase 5: Model Selection & Tuning

``ModelSelector`` evaluates a curated set of candidate estimators via
cross-validated scoring, selects the best one, and narrates the decision.

Candidate models
----------------
Classification:
  - LogisticRegression (fast baseline, interpretable)
  - RandomForestClassifier (robust, handles non-linearity)
  - GradientBoostingClassifier (typically highest accuracy)

Regression:
  - Ridge (fast baseline, regularised linear)
  - RandomForestRegressor (robust, non-linear)
  - GradientBoostingRegressor (typically highest accuracy)

Selection criterion
-------------------
``roc_auc`` (classification) or ``neg_root_mean_squared_error`` (regression),
averaged over ``config.cv_folds`` stratified folds.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score

from mentorml.config import MentorConfig
from mentorml.core.decision import DecisionLog, DecisionRecord, Severity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Candidate model registry
# ---------------------------------------------------------------------------

_CLASSIFICATION_CANDIDATES: dict[str, Any] = {
    "LogisticRegression": lambda cfg: LogisticRegression(
        max_iter=500,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    ),
    "RandomForestClassifier": lambda cfg: RandomForestClassifier(
        n_estimators=100,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    ),
    "GradientBoostingClassifier": lambda cfg: GradientBoostingClassifier(
        n_estimators=100,
        random_state=cfg.random_state,
    ),
}

_REGRESSION_CANDIDATES: dict[str, Any] = {
    "Ridge": lambda cfg: Ridge(random_state=cfg.random_state),
    "RandomForestRegressor": lambda cfg: RandomForestRegressor(
        n_estimators=100,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    ),
    "GradientBoostingRegressor": lambda cfg: GradientBoostingRegressor(
        n_estimators=100,
        random_state=cfg.random_state,
    ),
}


# ---------------------------------------------------------------------------
# ModelSelector
# ---------------------------------------------------------------------------


class ModelSelector:
    """
    Phase 5: Cross-validated model selection.

    Evaluates candidate models for the inferred task type and returns the
    best fitted estimator together with a full ranking ``DecisionRecord``.

    Parameters
    ----------
    config : MentorConfig
        Global configuration (``cv_folds``, ``random_state``, ``n_jobs``).

    Examples
    --------
    ::

        selector = ModelSelector(config)
        result = selector.select(X_train, y_train, log, task_type="classification")
        model = result["best_model"]
    """

    def __init__(self, config: MentorConfig) -> None:
        self.config = config
        self._best_model: Any = None
        self._best_model_name: str = ""
        self._scores: dict[str, float] = {}

