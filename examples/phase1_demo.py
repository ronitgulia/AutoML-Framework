"""Phase 1 live demo script."""

import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

from mentorml import DatasetAnalyzer, MentorConfig
from mentorml.core import DecisionLog, Severity

rng = np.random.default_rng(42)
n = 300
df = pd.DataFrame(
    {
        "user_id": range(1, n + 1),
        "age": np.where(rng.random(n) < 0.15, np.nan, rng.normal(35, 12, n)),
        "income": rng.exponential(50000, n),
        "income_log": np.log1p(rng.exponential(50000, n)),
        "city": rng.choice(["NYC", "LA", "Chicago", "Houston", None], n),
        "constant": ["same"] * n,
        "churned": rng.choice([0, 1], n, p=[0.88, 0.12]),
    }
)

config = MentorConfig(target_column="churned", correlation_threshold=0.7)
log = DecisionLog()
analyzer = DatasetAnalyzer(config)
results = analyzer.analyze(df, log)

print("=== ANALYSIS SUMMARY ===")
print(f"Shape: {results['n_rows']} rows x {results['n_cols']} cols")
print(f"Duplicates: {results['n_duplicates']}")
print(f"Flagged columns: {results['n_columns_flagged']}")
print(f"  Constant:     {results['flagged_columns']['constant']}")
print(f"  ID-like:      {results['flagged_columns']['id_like']}")
print(f"  High-missing: {results['flagged_columns']['high_missing']}")
print(f"Task type: {results['target']['task_type']}")
print(f"Imbalance ratio: {results['target']['imbalance_ratio']}x")

print("\n=== DECISION LOG SUMMARY ===")
print(f"Total records: {len(log)}")
n_warn = len([r for r in log if r.severity == Severity.WARNING])
n_crit = len([r for r in log if r.severity == Severity.CRITICAL])
print(f"  INFO:     {len(log) - n_warn - n_crit}")
print(f"  WARNING:  {n_warn}")
print(f"  CRITICAL: {n_crit}")

print("\n=== CRITICAL FINDINGS ===")
for r in log.filter(Severity.CRITICAL):
    print(f"  [{r.component}] {r.rationale[:130]}")

print("\n=== INCOME COLUMN PROFILE ===")
p = results["column_profiles"]["income"]
print(f"  dtype_category: {p.dtype_category}")
print(f"  skewness:       {p.skewness:.3f}")
print(f"  outlier_pct:    {p.outlier_pct:.1%}")
print(f"  mean:           {p.mean:.1f}")
print(f"  median:         {p.median:.1f}")