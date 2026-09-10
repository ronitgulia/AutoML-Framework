"""
AutoML Framework - Model Selector
Searches over a catalogue of ML models and returns the best one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    name: str
    model: Any
    mean_score: float
    std_score: float
    params: Dict[str, Any] = field(default_factory=dict)


CLASSIFICATION_CATALOGUE = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
}

REGRESSION_CATALOGUE = {
    "ridge": Ridge(),
    "decision_tree": DecisionTreeRegressor(random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
}


class ModelSelector:
    """
    Evaluates multiple model families via cross-validation and returns
    a ranked list of results.

    Parameters
    ----------
    task : str
        'classification' or 'regression'.
    metric : str
        Scikit-learn scorer string, e.g. 'roc_auc', 'f1', 'r2'.
    cv : int
        Number of cross-validation folds.
    n_jobs : int
        Parallel jobs for cross-validation.
    """

    def __init__(
        self,
        task: str = "classification",
        metric: str = "roc_auc",
        cv: int = 5,
        n_jobs: int = -1,
    ):
        self.task = task
        self.metric = metric
        self.cv = cv
        self.n_jobs = n_jobs
        self._catalogue = (
            CLASSIFICATION_CATALOGUE
            if task == "classification"
            else REGRESSION_CATALOGUE
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> List[ModelResult]:
        """
        Run cross-validation for every model in the catalogue.

        Returns a list of ModelResult sorted by mean_score descending.
        """
        results: List[ModelResult] = []

        for name, model in self._catalogue.items():
            logger.info("Evaluating %s ...", name)
            try:
                scores = cross_val_score(
                    model,
                    X,
                    y,
                    cv=self.cv,
                    scoring=self.metric,
                    n_jobs=self.n_jobs,
                )
                result = ModelResult(
                    name=name,
                    model=model,
                    mean_score=float(scores.mean()),
                    std_score=float(scores.std()),
                )
                results.append(result)
                logger.info(
                    "  %s: %.4f ± %.4f", name, result.mean_score, result.std_score
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("  %s failed: %s", name, exc)

        results.sort(key=lambda r: r.mean_score, reverse=True)
        self.results_ = results
        self.best_ = results[0] if results else None
        return results

    @property
    def best_model(self) -> Optional[Any]:
        """Returns the estimator with the highest cross-val score."""
        return self.best_.model if self.best_ else None
