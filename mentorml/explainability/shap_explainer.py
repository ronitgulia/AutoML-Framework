"""
mentorml.explainability.shap_explainer
-----------------------------------------
Phase 6: SHAP / Feature Importance Explainer

Provides model-agnostic feature importance explanations.  Uses SHAP if
available, otherwise falls back to sklearn's permutation importance.

This graceful degradation means the core pipeline works without the
``mentorml[shap]`` extra.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from mentorml.config import MentorConfig
from mentorml.core.decision import DecisionLog, DecisionRecord, Severity

logger = logging.getLogger(__name__)

# Optional SHAP import — graceful fallback if not installed
try:
    import shap as _shap

    _SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SHAP_AVAILABLE = False


class SHAPExplainer:
    """
    Phase 6: Feature importance explainer.

    Uses SHAP TreeExplainer (if shap is installed) or sklearn permutation
    importance (fallback) to rank features by their contribution to the model.

    Parameters
    ----------
    config : MentorConfig
        Global configuration.

    Examples
    --------
    ::

        explainer = SHAPExplainer(config)
        result = explainer.explain(model, X_test, log)
        # result["feature_importances"] → sorted list of (feature, importance)
    """

    def __init__(self, config: MentorConfig) -> None:
        self.config = config

    def explain(
        self,
        model: Any,
        X: pd.DataFrame,
        log: DecisionLog,
        y: pd.Series | None = None,
    ) -> dict[str, Any]:
        """
        Generate feature importance explanations.

        Parameters
        ----------
        model : Any
            A fitted sklearn-compatible estimator.
        X : pd.DataFrame
            Feature matrix (typically the held-out test set).
        log : DecisionLog
            Decision log to append explanation metadata to.
        y : pd.Series | None
            Target values — required for permutation importance fallback.

        Returns
        -------
        dict[str, Any]
            Keys:

            - ``"method"`` – ``"shap"`` or ``"permutation"``
            - ``"feature_importances"`` – list of ``{"feature", "importance"}``
              dicts, sorted descending
            - ``"top_features"`` – top-5 feature names
            - ``"shap_values"`` – raw SHAP values array (if SHAP used)
        """
        log.append(
            DecisionRecord(
                component="SHAPExplainer",
                action="explanation_start",
                rationale=(
                    f"Generating feature importance for model "
                    f"'{type(model).__name__}' on "
                    f"{len(X)} rows × {len(X.columns)} features."
                ),
                data={},

                severity=Severity.INFO,
            )
        )

        if _SHAP_AVAILABLE:
            result = self._shap_explain(model, X, log)
        else:
            result = self._permutation_explain(model, X, y, log)

        top_features = [
            r["feature"] for r in result["feature_importances"][:5]
        ]
        result["top_features"] = top_features

        log.append(
            DecisionRecord(
                component="SHAPExplainer",
                action="explanation_complete",
                rationale=(
                    f"Explanation complete (method={result['method']}). "
                    f"Top features: {top_features}."
                ),
                severity=Severity.INFO,
                data={
                    "method": result["method"],
                    "top_features": top_features,
                    "n_features": len(X.columns),
                },
            )
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _shap_explain(
        model: Any,
        X: pd.DataFrame,
        log: DecisionLog,
    ) -> dict[str, Any]:
        """Use SHAP TreeExplainer or KernelExplainer."""
        try:
            explainer = _shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            method = "shap_tree"
        except Exception:  # noqa: BLE001
            # Fallback to KernelExplainer on small sample
            sample = X.iloc[: min(50, len(X))]
            explainer = _shap.KernelExplainer(model.predict, sample)
            shap_values = explainer.shap_values(sample)
            X = sample
            method = "shap_kernel"

        # For multi-class SHAP (list of arrays OR 3D array), take mean absolute
        if isinstance(shap_values, list):
            # Old SHAP: list of 2D arrays, one per class
            arr = np.abs(np.array(shap_values)).mean(axis=0)
        else:
            arr = np.abs(shap_values)

        # arr may be 3D (samples, features, classes) or 2D (samples, features)
        if arr.ndim == 3:
            mean_abs = arr.mean(axis=(0, 2))  # mean over samples and classes
        elif arr.ndim == 2:
            mean_abs = arr.mean(axis=0)       # mean over samples
        else:
            mean_abs = arr

        # Ensure 1D
        mean_abs = np.asarray(mean_abs).flatten()

        importances = [
            {"feature": col, "importance": round(float(imp), 6)}
            for col, imp in zip(X.columns, mean_abs)
        ]
        importances.sort(key=lambda d: d["importance"], reverse=True)

        log.append(
            DecisionRecord(
                component="SHAPExplainer",
                action=f"shap_computed:{method}",
                rationale=(
                    f"SHAP values computed using {method}. "
                    "Mean absolute SHAP value represents average feature impact."
                ),
                severity=Severity.INFO,
                data={"method": method},
            )
        )
        return {
            "method": method,
            "feature_importances": importances,
            "shap_values": shap_values,
        }

    @staticmethod
    def _permutation_explain(
        model: Any,
        X: pd.DataFrame,
        y: pd.Series | None,
        log: DecisionLog,
    ) -> dict[str, Any]:
        """Fallback: sklearn permutation importance."""
        from sklearn.inspection import permutation_importance

        log.append(
            DecisionRecord(
                component="SHAPExplainer",
                action="fallback_permutation_importance",
                rationale=(
                    "SHAP library not installed. "
                    "Falling back to sklearn permutation importance. "
                    "Install mentorml[shap] for SHAP-based explanations."
                ),
                data={},

                severity=Severity.WARNING,
            )
        )

        if y is None:
            # Cannot run permutation without y — return model's built-in importances
            if hasattr(model, "feature_importances_"):
                raw = model.feature_importances_
            elif hasattr(model, "coef_"):
                raw = np.abs(model.coef_).flatten()[: len(X.columns)]
            else:
                raw = np.ones(len(X.columns))

            importances = [
                {"feature": col, "importance": round(float(imp), 6)}
                for col, imp in zip(X.columns, raw)
            ]
            importances.sort(key=lambda d: d["importance"], reverse=True)
            return {"method": "builtin_importance", "feature_importances": importances}

        perm = permutation_importance(
            model,
            X,
            y,
            n_repeats=10,
            random_state=42,
        )
        importances = [
            {
                "feature": col,
                "importance": round(float(imp), 6),
            }
            for col, imp in zip(X.columns, perm.importances_mean)
        ]
        importances.sort(key=lambda d: d["importance"], reverse=True)
        return {"method": "permutation", "feature_importances": importances}