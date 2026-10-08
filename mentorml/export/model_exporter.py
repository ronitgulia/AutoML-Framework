"""
mentorml.export.model_exporter
---------------------------------
Phase 9: Deployment-ready Model Export

Saves all artefacts needed to reproduce predictions in a new environment:

- ``model.joblib``          — the fitted sklearn estimator
- ``preprocessor.joblib``   — the fitted ExplainablePreprocessor
- ``feature_names.json``    — list of feature names expected at inference time
- ``decisions.json``        — full decision log as structured JSON
- ``report.html``           — the interactive HTML report
- ``predict.py``            — a standalone inference script

Usage of the exported package
------------------------------
::

    # In a new environment with only scikit-learn + joblib installed:
    python predict.py --input new_data.csv --output predictions.csv
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import joblib

from mentorml.core.decision import DecisionLog, DecisionRecord, Severity

logger = logging.getLogger(__name__)

_PREDICT_SCRIPT_TEMPLATE = '''"""
Auto-generated inference script by mentorml.
Usage: python predict.py --input new_data.csv --output predictions.csv
"""
import argparse
import json
import joblib
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="mentorml inference")
    parser.add_argument("--input", required=True, help="CSV path for new data")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path")
    args = parser.parse_args()

    # Load artefacts
    model = joblib.load("model.joblib")
    preprocessor = joblib.load("preprocessor.joblib")
    with open("feature_names.json") as f:
        feature_names = json.load(f)

    # Load and preprocess
    df = pd.read_csv(args.input)
    from mentorml.core.decision import DecisionLog
    log = DecisionLog()
    df_proc = preprocessor.transform(df, log)

    # Align columns
    for col in feature_names:
        if col not in df_proc.columns:
            df_proc[col] = 0
    df_proc = df_proc[[c for c in feature_names if c in df_proc.columns]]

    # Predict
    predictions = model.predict(df_proc)
    out = pd.DataFrame({"prediction": predictions})
    out.to_csv(args.output, index=False)
    print(f"Predictions written to {args.output}")

if __name__ == "__main__":
    main()
'''


class ModelExporter:
    """
    Phase 9: Export trained pipeline artefacts.

    Writes all files needed to reproduce predictions to a single directory,
    including a self-contained inference script.

    Parameters
    ----------
    project_name : str
        Used to label the export directory.

    Examples
    --------
    ::

        exporter = ModelExporter(project_name="churn_model")
        export_path = exporter.export(
            model=fitted_model,
            preprocessor=fitted_preprocessor,
            feature_names=list(X_train.columns),
            log=log,
            report_path="reports/churn_20240101.html",
            output_dir="exports/",
        )
        print(f"Exported to {export_path}")
    """

    def __init__(self, project_name: str = "ml_project") -> None:
        self.project_name = project_name.lower().replace(" ", "_")

    def export(
        self,
        model: Any,
        preprocessor: Any,
        feature_names: list[str],
        log: DecisionLog,
        output_dir: str = ".",
        report_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Export all artefacts to ``output_dir/<project_name>/``.

        Parameters
        ----------
        model : Any
            Fitted sklearn-compatible estimator.
        preprocessor : Any
            Fitted ``ExplainablePreprocessor`` instance.
        feature_names : list[str]
            Column names of the feature matrix used for training.
        log : DecisionLog
            Full pipeline decision log.
        output_dir : str
            Parent directory for the export folder.
        report_path : str | None
            Path to the generated HTML report (will be copied).
        metadata : dict | None
            Extra metadata to embed in the export manifest.

        Returns
        -------
        str
            Absolute path to the export directory.
        """
