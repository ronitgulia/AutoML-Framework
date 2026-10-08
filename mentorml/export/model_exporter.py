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

