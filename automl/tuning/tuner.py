"""
AutoML Framework - Hyperparameter Tuner
Uses Optuna (TPE sampler) to optimise model hyperparameters.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

import numpy as np
import optuna
from sklearn.model_selection import cross_val_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Search spaces per model family                                               #
# --------------------------------------------------------------------------- #

def _rf_space(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }


def _gbm_space(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
    }


def _lr_space(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "C": trial.suggest_float("C", 1e-4, 1e2, log=True),
        "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
    }


SEARCH_SPACES: Dict[str, Callable] = {
    "random_forest": _rf_space,
    "gradient_boosting": _gbm_space,
    "logistic_regression": _lr_space,
}


class HyperparameterTuner:
    """
    Wraps an Optuna study to tune a scikit-learn compatible estimator.

    Parameters
    ----------
    estimator_cls : type
        The un-instantiated sklearn estimator class.
    model_name : str
        Key in SEARCH_SPACES; determines which hyperparameter space to use.
    metric : str
        Scikit-learn scorer string.
    cv : int
        Cross-validation folds.
    n_trials : int
        Maximum number of Optuna trials.
    timeout : float or None
        Wall-clock budget in seconds (optional).
    direction : str
        'maximize' or 'minimize'.
    """

    def __init__(
        self,
        estimator_cls,
        model_name: str,
        metric: str = "roc_auc",
        cv: int = 5,
        n_trials: int = 50,
        timeout: Optional[float] = None,
        direction: str = "maximize",
    ):
        self.estimator_cls = estimator_cls
        self.model_name = model_name
        self.metric = metric
        self.cv = cv
        self.n_trials = n_trials
        self.timeout = timeout
        self.direction = direction

    def _objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        space_fn = SEARCH_SPACES.get(self.model_name)
        if space_fn is None:
            raise ValueError(f"No search space defined for '{self.model_name}'")

        params = space_fn(trial)
        model = self.estimator_cls(**params, random_state=42)
        scores = cross_val_score(model, X, y, cv=self.cv, scoring=self.metric)
        return float(scores.mean())

    def tune(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Run the Optuna study and return the best hyperparameters.
        """
        study = optuna.create_study(
            direction=self.direction,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        )
        study.optimize(
            lambda trial: self._objective(trial, X, y),
            n_trials=self.n_trials,
            timeout=self.timeout,
        )

        self.study_ = study
        self.best_params_ = study.best_params
        self.best_score_ = study.best_value

        logger.info(
            "Best %s after %d trials: %.4f | params=%s",
            self.metric,
            len(study.trials),
            self.best_score_,
            self.best_params_,
        )
        return self.best_params_

    def best_estimator(self) -> Any:
        """Returns an estimator instantiated with the best found params."""
        return self.estimator_cls(**self.best_params_, random_state=42)
