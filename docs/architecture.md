# MentorML — Architecture Overview

This document describes the internal design of the MentorML framework for contributors and advanced users.

---

## High-Level Design

MentorML follows a **linear, phase-based pipeline** architecture. Each phase is a self-contained component that reads the current dataset state, applies transformations or analysis, and appends structured decisions to a shared `DecisionLog`.

```
Raw Data
  → DatasetAnalyzer
  → DataQualityAssessor
  → ExplainablePreprocessor
  → FeatureEngineer
  → ModelSelector
  → SHAPExplainer
  → BusinessNarrator
  → HTMLReportGenerator
  → ModelExporter
```

All phases share a single `DecisionLog` that accumulates an immutable, timestamped audit trail of every decision made.

---

## Core Abstractions (`mentorml/core/`)

### `DecisionLog` & `DecisionRecord` (`decision.py`)

The heart of MentorML's explainability system.

- `DecisionRecord` — a single logged decision, containing:
  - `component` — which phase emitted it
  - `action` — a short machine-readable action name
  - `rationale` — plain-English explanation
  - `data` — arbitrary structured metadata dict
  - `severity` — `Severity.INFO`, `Severity.WARNING`, or `Severity.CRITICAL`
  - `timestamp` — UTC datetime of when the decision was made

- `DecisionLog` — an ordered, filterable collection of `DecisionRecord` objects.
  - Supports iteration, indexing, and `log.filter(severity)` queries.
  - Each record can be serialized to a dict via `record.to_dict()` for JSON export.

### `MentorConfig` (`config.py`)

A Python `dataclass` that acts as the single source of truth for all pipeline parameters.
Validated at construction time via `__post_init__`. No magic numbers exist anywhere in the codebase — all thresholds are config fields.

### Protocols (`protocols.py`)

Structural interfaces that define the contract every pipeline component must fulfill.
Components are designed to be plug-and-play — swapping implementations does not require changes to the orchestrator.

### Exceptions (`exceptions.py`)

A typed exception hierarchy rooted at `MentorMLError`, with sub-classes like `ConfigurationError` for precise, catchable error handling.

---

## Phase Components

### Phase 1 — `DatasetAnalyzer` (`mentorml/analysis/`)

- Profiles every column: dtype, cardinality, null count, skewness
- Infers task type (`classification` vs `regression`) from the target column
- Computes Pearson correlation matrix
- Logs all findings as `DecisionRecord` entries

### Phase 2 — `DataQualityAssessor` (`mentorml/quality/`)

- Detects duplicate rows
- Flags columns with excessive missing values against `missing_threshold`
- Identifies outliers via IQR boundaries
- Detects class imbalance in the target column
- Emits `CRITICAL`-severity decisions for actionable issues

### Phase 3 — `ExplainablePreprocessor` (`mentorml/preprocessing/`)

- Imputes missing numerical values (median strategy)
- Encodes categorical features adaptively:
  - Low cardinality → one-hot encoding
  - High cardinality (>`cardinality_threshold`) → ordinal or target encoding
- Applies skewness-aware scaling (log transform + StandardScaler for skewed features)
- Drops columns exceeding `missing_threshold`
- Every column-level decision is logged with its rationale

### Phase 4 — `FeatureEngineer` (`mentorml/features/`)

- Generates pairwise interaction terms
- Creates polynomial features
- Prunes features exceeding `correlation_threshold` (keeps one of each near-duplicate pair)
- Removes near-zero-variance features below `variance_threshold`

### Phase 5 — `ModelSelector` (`mentorml/modeling/`)

- Runs stratified k-fold cross-validation (`cv_folds`) across all candidate models:
  - **Boosting**: LightGBM, XGBoost, CatBoost
  - **Classical**: RandomForest, LogisticRegression, Ridge
- Builds a leaderboard ranked by primary metric (ROC-AUC for classification, RMSE for regression)
- Logs the winner and runner-up decisions with metric deltas

### Phase 6 — `SHAPExplainer` (`mentorml/explainability/`)

- Computes TreeSHAP values for the winning model
- Produces global feature importance ranking
- Supports local explanation for individual predictions

### Phase 7 — `BusinessNarrator` (`mentorml/narrative/`)

- Translates the `DecisionLog` and model results into a plain-English narrative
- Structures output as: executive summary → data quality findings → model results → recommendations

### Phase 8 — `HTMLReportGenerator` (`mentorml/reporting/`)

- Produces a single self-contained `.html` file — no external assets required
- Contains: model leaderboard, SHAP charts, data quality summary, full decision audit log
- Written to `report_output_dir` from `MentorConfig`

### Phase 9 — `ModelExporter` (`mentorml/export/`)

- Serializes the winning model and preprocessing pipeline to `.joblib`
- Writes a `metadata.json` with run configuration and metric results
- Generates a standalone `infer.py` inference script with no MentorML dependency

---

## Data Flow

```
┌─────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│  Input  │───▶│   DatasetAnalyzer    │───▶│  DataQualityAssessor │
│  Data   │    └──────────────────────┘    └──────────┬───────────┘
└─────────┘                                           │
                                                      ▼
                                         ┌──────────────────────────┐
                                         │  ExplainablePreprocessor │
                                         └──────────┬───────────────┘
                                                    │
                                                    ▼
                                         ┌──────────────────────┐
                                         │   FeatureEngineer    │
                                         └──────────┬───────────┘
                                                    │
                                                    ▼
                                         ┌──────────────────────┐
                                         │    ModelSelector     │
                                         │  (CV Tournament)     │
                                         └──────────┬───────────┘
                                                    │
                              ┌─────────────────────┼────────────────────┐
                              ▼                     ▼                    ▼
                    ┌──────────────┐     ┌─────────────────┐   ┌──────────────┐
                    │ SHAPExplainer│     │ BusinessNarrator│   │ ModelExporter│
                    └──────────────┘     └────────┬────────┘   └──────────────┘
                                                  │
                                                  ▼
                                        ┌──────────────────────┐
                                        │  HTMLReportGenerator │
                                        └──────────────────────┘
```

---

## Configuration System

All pipeline behaviour is controlled via `MentorConfig`. See [config.py](../mentorml/config.py) for the full field reference.

| Parameter | Default | Description |
|---|---|---|
| `target_column` | `None` | Prediction target column |
| `task_type` | `None` | `"classification"`, `"regression"`, or auto-inferred |
| `missing_threshold` | `0.40` | Drop columns above this missing fraction |
| `cardinality_threshold` | `50` | High-cardinality threshold for encoding strategy |
| `correlation_threshold` | `0.95` | Collinearity pruning threshold |
| `variance_threshold` | `0.01` | Near-zero-variance pruning threshold |
| `test_size` | `0.20` | Holdout fraction |
| `cv_folds` | `5` | Cross-validation folds |
| `random_state` | `42` | Global reproducibility seed |
| `n_jobs` | `-1` | Parallelism (-1 = all cores) |
| `verbose` | `True` | Console logging |
| `report_output_dir` | `"."` | HTML report destination |
| `export_output_dir` | `"."` | Model artifact destination |

---

## Testing Strategy

| Layer | Location | Coverage |
|---|---|---|
| Unit | `tests/unit/` | Config validation, DecisionLog, individual components |
| Integration | `tests/integration/` | Full pipeline — classification and regression end-to-end |

Run with:
```bash
pytest --cov=mentorml tests/
```
