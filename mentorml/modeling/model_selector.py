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


