"""
examples/quickstart.py
-----------------------
Minimal demonstration of the Phase 0 core abstractions.

Run with:  python examples/quickstart.py
"""

import logging

import pandas as pd

import mentorml
from mentorml import MentorConfig
from mentorml.core import DecisionLog, DecisionRecord, Severity

# Configure logging so mentorml output is visible
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

print(f"mentorml version: {mentorml.__version__}\n")

# ---------------------------------------------------------------------------
# 1. Create a config
# ---------------------------------------------------------------------------
config = MentorConfig(
    target_column="churn",
    task_type="classification",
    missing_threshold=0.3,
    verbose=True,
)
print(config.summary())
print()

# ---------------------------------------------------------------------------
# 2. Create a decision log
# ---------------------------------------------------------------------------
log = DecisionLog()

# Simulate decisions that a real component would emit
log.append(
    DecisionRecord(
        component="DatasetAnalyzer",
        action="infer_task_type",
        rationale=(
            "Target column 'churn' has 2 unique values (0, 1) and dtype int64. "
            "Inferred task type: binary classification."
        ),
        data={"column": "churn", "n_unique": 2, "inferred_type": "classification"},
        severity=Severity.INFO,
    )
)

log.append(
    DecisionRecord(
        component="DataQualityAssessor",
        action="flag_missing_values",
        rationale=(
            "Column 'tenure_months' has 14.2% missing values. "
            "This is below the drop threshold (30%). Will impute with median."
        ),
        data={"column": "tenure_months", "missing_pct": 0.142, "strategy": "median"},
        severity=Severity.WARNING,
    )
)

log.append(
    DecisionRecord(
        component="DataQualityAssessor",
        action="flag_class_imbalance",
        rationale=(
            "Target 'churn' has a 92/8 class split. This severe imbalance will "
            "cause naive models to predict the majority class almost exclusively. "
            "Recommend SMOTE oversampling or class_weight='balanced'."
        ),
        data={"majority_class": 0, "majority_pct": 0.92, "minority_pct": 0.08},
        severity=Severity.CRITICAL,
    )
)

print(f"Decision log: {log!r}\n")

# ---------------------------------------------------------------------------
# 3. Inspect the log
# ---------------------------------------------------------------------------
print("=== All decisions ===")
for record in log:
    print(f"  {record}")

print("\n=== CRITICAL decisions only ===")
for record in log.filter(Severity.CRITICAL):
    print(f"  {record}")

# ---------------------------------------------------------------------------
# 4. Serialise to JSON (audit trail)
# ---------------------------------------------------------------------------
print("\n=== JSON audit trail (first record) ===")
import json

first = log[0].to_dict()
print(json.dumps(first, indent=2))