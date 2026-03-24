# Assignment 5 - Minimal MLOps Pipeline

This repo uses a tiny `scikit-learn` Iris classifier (CPU-friendly) and a 2-job GitHub Actions pipeline.

## Local quick run

```bash
python -m pip install -r requirements.txt
python train.py
python check_threshold.py
```

Force a failing run locally:

```bash
FORCE_LOW_ACCURACY=true python train.py
python check_threshold.py
```

## GitHub Actions evidence runs

1. Add repository secret: `MLFLOW_TRACKING_URI` (your MLflow tracking server URI).
2. Run workflow `mlops-assignment-5` manually (`workflow_dispatch`) with:
   - `force_low_accuracy = true` for failed evidence (< 0.85)
   - `force_low_accuracy = false` for successful evidence (> 0.85)
3. Capture screenshots from the Actions run page:
   - failed run showing threshold check failure
   - successful run showing `deploy` job completed
