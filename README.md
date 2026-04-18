# NIDS — NSL-KDD Network Intrusion Detection

Python pipeline for classifying malicious vs benign **NSL-KDD** connection records: ingestion and cleaning, **20+ engineered features**, scaled numerics with one-hot categoricals, and stubs for Random Forest / XGBoost training, evaluation, and a Matplotlib/Seaborn SOC-style dashboard.

**Repository:** [github.com/shivvv3824/nids](https://github.com/shivvv3824/nids)

## Requirements

- Python **3.10+**
- NSL-KDD files **`KDDTrain+.txt`** and **`KDDTest+.txt`** (comma-separated, no header row)

Download the NSL-KDD Train+ and Test+ archives from a trusted mirror (search for “NSL-KDD” and `KDDTrain+.txt` / `KDDTest+.txt`), then place both files under `data/`.

## Quick start

```bash
git clone https://github.com/shivvv3824/nids.git
cd nids
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Add KDDTrain+.txt and KDDTest+.txt to data/
python main.py --phase 1
```

Phase 1 writes `artifacts/phase1_feature_bundle.joblib` (dense feature matrices + fitted `ColumnTransformer`).

### CLI options

```text
python main.py --help
```

- `--phase` — `1` (default), `2`, `3`, `4`, or `all`. Phases 2–4 are placeholders until implemented.
- `--data-dir` — folder containing NSL files (default: `./data`).
- `--train-file` / `--test-file` — filenames inside `data-dir` (defaults: `KDDTrain+.txt`, `KDDTest+.txt`).

## Project layout

```text
data/                    # NSL-KDD text files (not committed)
src/
  data_loader.py         # Load, clean, targets (binary + multi-class label)
  feature_engine.py      # Feature engineering + StandardScaler + OneHotEncoder
  train_models.py          # Phase 2: RF / XGBoost (stub)
  evaluate.py            # Phase 3: metrics / FPR / importance (stub)
dashboard/
  threat_visualizer.py   # Phase 4: SOC dashboard (stub)
models/                  # Saved models (gitignored artifacts)
main.py                  # Orchestrator
requirements.txt
```

## Roadmap

| Phase | Status | Description |
|------|--------|-------------|
| 1 | Implemented | Load NSL-KDD, clean, engineer features, fit preprocessing |
| 2 | Planned | Baseline + Random Forest + XGBoost, tuning, `joblib` to `models/` |
| 3 | Planned | Classification report, ROC-AUC, FPR vs baseline, feature importance |
| 4 | Planned | Multi-panel SOC triage figures (PCA/t-SNE, IoC bars, confusion heatmap, KDE/violin) |

## License

See [LICENSE](LICENSE).
