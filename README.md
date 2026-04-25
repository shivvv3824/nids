# NIDS — NSL-KDD Network Intrusion Detection

Python pipeline for classifying malicious vs benign **NSL-KDD** connection records: ingestion and cleaning, **20+ engineered features**, scaled numerics with one-hot categoricals, **tuned Random Forest + XGBoost** (with a logistic baseline), evaluation (accuracy, ROC-AUC, **FPR vs baseline**), and a **Matplotlib/Seaborn SOC-style dashboard** (PCA embedding, IoC importances, confusion matrix, attack-family byte-ratio distribution plot).

**Repository:** [github.com/shivvv3824/nids](https://github.com/shivvv3824/nids)

## Requirements

- Python **3.10+** (3.9 often works; use 3.10+ when possible)
- NSL-KDD files **`KDDTrain+.txt`** and **`KDDTest+.txt`** (comma-separated, no header row)
- **macOS + XGBoost:** if import fails with `libomp.dylib`, install OpenMP: `brew install libomp`. If XGBoost still cannot load, Phase 2 automatically falls back to **`HistGradientBoostingClassifier`** (saved as `models/xgboost.joblib` for a consistent downstream path).

Download the NSL-KDD Train+ and Test+ archives from a trusted mirror (search for “NSL-KDD” and `KDDTrain+.txt` / `KDDTest+.txt`), then place both files under `data/`.

## Quick start

```bash
git clone https://github.com/shivvv3824/nids.git
cd nids
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Add KDDTrain+.txt and KDDTest+.txt to data/
python main.py --phase all          # full pipeline (1→4)
# or stepwise: 1 preprocess, 2 train, 3 evaluate, 4 dashboard
python main.py --phase all --fast   # smaller search; good for laptops / smoke tests
```

Outputs:

- `artifacts/phase1_feature_bundle.joblib` — dense matrices + fitted preprocessor  
- `models/*.joblib` — baseline logistic regression, tuned RF, tuned XGBoost (or HGB fallback)  
- `artifacts/evaluation_report.json` — metrics, FPR deltas vs baseline, importances  
- `artifacts/soc_dashboard.png` — SOC triage figure

### CLI options

```text
python main.py --help
```

- `--phase` — `1` (default), `2`, `3`, `4`, or `all` (runs preprocessing → training → evaluation → dashboard).
- `--fast` — fewer CV folds / search iterations for quicker runs.
- `--data-dir` — folder containing NSL files (default: `./data`).
- `--artifacts-dir` / `--models-dir` — override output locations.
- `--train-file` / `--test-file` — filenames inside `data-dir` (defaults: `KDDTrain+.txt`, `KDDTest+.txt`).

## Project layout

```text
data/                    # NSL-KDD text files (not committed)
src/
  data_loader.py         # Load, clean, targets (binary + multi-class label)
  feature_engine.py      # Feature engineering + StandardScaler + OneHotEncoder
  train_models.py        # Phase 2: baseline + RandomizedSearchCV RF + XGB (HGB fallback)
  evaluate.py            # Phase 3: reports, ROC-AUC, FPR vs baseline, importances
dashboard/
  threat_visualizer.py   # Phase 4: SOC dashboard PNG
models/                  # Saved models (gitignored artifacts)
main.py                  # Orchestrator
requirements.txt
```

## Roadmap

| Phase | Status | Description |
|------|--------|-------------|
| 1 | Implemented | Load NSL-KDD, clean, engineer features, fit preprocessing |
| 2 | Implemented | Logistic baseline + tuned RF + tuned XGBoost (HGB fallback), `joblib` to `models/` |
| 3 | Implemented | Classification report, ROC-AUC, FPR vs baseline, RF/XGB importances → JSON |
| 4 | Implemented | SOC dashboard: PCA scatter, top-10 IoC bars, confusion heatmap, family byte-ratio distribution |

## Proof of work

Validated in this environment with a synthetic NSL-KDD-shaped dataset (smoke test for code-path execution), using:

```bash
python main.py --phase all --fast \
  --data-dir /path/to/tmp_synth_data \
  --artifacts-dir /path/to/tmp_artifacts \
  --models-dir /path/to/tmp_models
```

Observed results:

- All four phases completed successfully (1 → 4) end-to-end.
- `Phase 3` produced `evaluation_report.json`.
- `Phase 4` produced `soc_dashboard.png`.
- If XGBoost cannot load (`libomp.dylib` missing on macOS), pipeline falls back to `HistGradientBoostingClassifier` and still writes `models/xgboost.joblib`.

## License

See [LICENSE](LICENSE).
