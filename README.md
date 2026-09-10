# AutoML Framework

An end-to-end Automated Machine Learning framework that handles data preprocessing, feature engineering, model selection, hyperparameter tuning, and deployment — all with minimal human intervention.

## Features

- 🔍 **Auto Feature Engineering** — automatic feature selection and transformation
- 🤖 **Model Selection** — searches over classical ML and neural architectures
- ⚙️ **Hyperparameter Tuning** — Bayesian optimization and random search
- 📊 **Experiment Tracking** — logs metrics, params, and artifacts per run
- 🚀 **One-Click Export** — export trained pipelines as ONNX or pickle

## Quick Start

```bash
pip install -r requirements.txt
python train.py --config configs/default.yaml
```

## Project Structure

```
AutoML Framework/
├── automl/
│   ├── preprocessing/
│   ├── features/
│   ├── models/
│   └── tuning/
├── configs/
├── docs/
├── workflows/
└── tests/
```

## License

MIT
