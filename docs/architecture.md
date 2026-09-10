# Architecture Overview

## High-Level Design

The AutoML Framework follows a modular pipeline architecture:

```
Raw Data → Preprocessing → Feature Engineering → Model Search → Tuning → Export
```

## Core Components

### 1. Data Ingestion
- Supports CSV, Parquet, JSON, and SQL sources
- Schema inference and data type casting
- Missing value detection and reporting

### 2. Preprocessing Module (`automl/preprocessing/`)
- Numerical imputation (mean, median, KNN)
- Categorical encoding (one-hot, ordinal, target encoding)
- Outlier detection via IQR and Z-score
- Scaling: StandardScaler, MinMaxScaler, RobustScaler

### 3. Feature Engineering (`automl/features/`)
- Polynomial feature generation
- Interaction terms
- Date/time feature extraction
- Recursive Feature Elimination (RFE)
- SHAP-based feature importance ranking

### 4. Model Search (`automl/models/`)
Supported model families:
| Category | Algorithms |
|----------|-----------|
| Linear   | LogisticRegression, Ridge, Lasso |
| Tree     | DecisionTree, RandomForest, GBM |
| Boosting | XGBoost, LightGBM, CatBoost |
| Neural   | MLP, TabNet |

### 5. Hyperparameter Tuning (`automl/tuning/`)
- **Strategy**: Bayesian Optimization (Optuna backend)
- **Budget**: configurable `n_trials` and `timeout`
- **Pruning**: Median pruner for early stopping of bad trials
- **Parallelism**: supports multi-process and distributed tuning

### 6. Experiment Tracking
- MLflow integration for metric and artifact logging
- Auto-generated HTML report per run

## Data Flow Diagram

```
┌─────────┐    ┌──────────────┐    ┌───────────────────┐
│  Input  │───▶│ Preprocessor │───▶│ Feature Engineer  │
│  Data   │    └──────────────┘    └─────────┬─────────┘
└─────────┘                                  │
                                             ▼
                                   ┌──────────────────┐
                                   │   Model Selector  │
                                   └────────┬─────────┘
                                            │
                                            ▼
                                   ┌──────────────────┐
                                   │  HPO (Optuna)    │
                                   └────────┬─────────┘
                                            │
                                            ▼
                                   ┌──────────────────┐
                                   │  Best Pipeline   │
                                   │  + MLflow Log    │
                                   └──────────────────┘
```

## Configuration

All pipeline behaviour is controlled via YAML configs under `configs/`. Example:

```yaml
task: classification
metric: roc_auc
time_budget: 600  # seconds

preprocessing:
  impute_strategy: median
  scale: true

tuning:
  n_trials: 100
  sampler: TPE
```
