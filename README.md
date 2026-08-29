# Diabetes Risk Prediction Model

Predicts diabetes risk from health survey indicators, served through a FastAPI REST API.

The project benchmarks five classifiers across 50 randomized runs, trains the winner with a properly tuned decision threshold, and exposes it via an API with single-record and batch CSV endpoints.

---

## Dataset

[CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) (UCI ML Repository, ID 891), downloaded automatically at runtime — no manual setup needed.

The data is heavily imbalanced, which the pipeline handles by undersampling the majority class. Five composite features are engineered from the raw columns (`BMI_Category`, `TotalUnhealthyDays`, `CardioRiskScore`, `HealthyLifestyleScore`, `MobilityHealthBurden`), and seven raw columns are dropped afterward, leaving 19 features. Full schema in `feature_schema.json`.

---

## Project Structure

| File | Purpose |
|---|---|
| `diabetes_model_comparison.py` | Benchmarks 5 models over 50 balanced runs and ranks them |
| `train_final_model.py` | Trains the winning model and saves the deployed artifacts |
| `main.py` | FastAPI service |
| `best_diabetes_model_final.pkl` | Trained model (loaded by the API) |
| `best_threshold_final.pkl` | Tuned decision threshold (0.40) |
| `feature_schema.json` | Feature dtypes, allowed values, ranges |
| `undersampling_50_runs_*.csv` | Benchmark results |

---

## How It Works

**Model comparison.** Logistic Regression, CatBoost, XGBoost, Random Forest, and Decision Tree are each wrapped in a scikit-learn pipeline with scaling on the continuous features. Each of 50 runs draws a fresh balanced sample with a different seed and splits it 70/15/15 into train/validation/test — fit on train, threshold tuned on validation, scored on test. Repeating 50 times gives mean ± SD per metric, so the ranking reflects stable performance rather than one lucky split.

**Threshold tuning.** Rather than the default 0.5 cutoff, thresholds from 0.10 to 0.90 are scanned and the one maximizing F1 among those reaching at least 0.72 recall is chosen. Missing a diabetic case is costlier than a false alarm, so recall is a constraint rather than something to trade away.

**Ranking.** `Overall Score = 0.40 × ROC-AUC + 0.30 × F1 + 0.20 × Recall + 0.10 × Accuracy`. ROC-AUC weighs most as a threshold-independent measure; accuracy weighs least, since it misleads on imbalanced data.

**Final training.** `train_final_model.py` reads the winner from the ranking CSV, fits on a train split only, tunes the threshold on a held-out validation set the model never saw, then refits on train+validation for the deployed artifact while keeping that already-locked threshold. This keeps the threshold honest — tuning it on rows the model was fit on would inflate it.

---

## Results

Averaged over 50 balanced runs:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Overall Score |
|---|---|---|---|---|---|---|
| **CatBoost** | 0.7267 | 0.6731 | 0.8824 | 0.7635 | 0.8145 | **0.8040** |
| XGBoost | 0.7260 | 0.6723 | 0.8829 | 0.7631 | 0.8144 | 0.8039 |
| Logistic Regression | 0.7210 | 0.6669 | 0.8839 | 0.7601 | 0.8090 | 0.8005 |
| Decision Tree | 0.7106 | 0.6577 | 0.8797 | 0.7525 | 0.7984 | 0.7921 |
| Random Forest | 0.7024 | 0.6495 | 0.8805 | 0.7474 | 0.7866 | 0.7852 |

CatBoost ranked first and is the deployed model, though its margin over XGBoost is well inside the run-to-run standard deviation — the two are effectively tied. Logistic regression trailing by only ~0.005 suggests the signal here is largely linear.

The tuned threshold of **0.40** deliberately favors recall: it catches roughly 88% of diabetic cases, at the cost of precision around 0.67. For a screening tool, a false positive that prompts a doctor's visit beats a missed case.

---

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/mxstafa690/diabetes-prediction-model.git
cd diabetes-prediction-model

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Usage

Run the API:

```bash
uvicorn main:app --reload
```

Serves at `http://127.0.0.1:8000`, with interactive docs at `/docs`.

Retrain from scratch:

```bash
python diabetes_model_comparison.py   # benchmark (long-running: 250 model fits)
python train_final_model.py           # train + save the final model
```

`train_final_model.py` accepts `--model`, `--seed`, `--val-size`, `--min-recall`, `--no-refit`, `--model-out`, and `--threshold-out`. Run it with `--help` for details.

---

## API Reference

### `GET /`

Health check. Returns `{"status": "backend is running"}`.

### `POST /predict`

Single prediction. Rate limit: 20/minute per IP.

```json
{
  "HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 32.0,
  "Stroke": 0, "HeartDiseaseorAttack": 0, "Smoker": 1,
  "PhysActivity": 0, "Fruits": 0, "Veggies": 1,
  "HvyAlcoholConsump": 0, "MentHlth": 5, "PhysHlth": 10,
  "GenHlth": 4, "DiffWalk": 1, "Sex": 1,
  "Age": 9, "Education": 4, "Income": 5
}
```

`Age` here is the 1–13 BRFSS bracket code, not age in years.

Response:

```json
{
  "prediction": 1,
  "probability": 0.7412,
  "threshold": 0.4,
  "result": "Higher diabetes risk",
  "advice": "The result suggests higher diabetes risk. Please consider consulting a doctor."
}
```

### `POST /predict-batch`

Scores a CSV upload. Rate limit: 5/minute per IP, max 5,000 rows.

```bash
curl -X POST http://127.0.0.1:8000/predict-batch -F "file=@patients.csv"
```

Required columns (any order):

```
HighBP, HighChol, CholCheck, BMI, Stroke, HeartDiseaseorAttack, Smoker,
PhysActivity, Fruits, Veggies, HvyAlcoholConsump, MentHlth, PhysHlth,
GenHlth, DiffWalk, Sex, AgeYears, Education, Income
```

This endpoint takes **`AgeYears`** — real age in years — and converts it to the bracket code internally. Rows are validated individually, so invalid rows come back with per-field error messages while the valid ones are still scored.

```json
{
  "total_rows": 3,
  "valid_rows": 2,
  "invalid_rows": 1,
  "threshold": 0.4,
  "results": [
    { "row": 1, "prediction": 0, "probability": 0.2103, "result": "Lower diabetes risk", "advice": "..." },
    { "row": 2, "errors": ["'GenHlth' must be between 1 and 5, got 7"] },
    { "row": 3, "prediction": 1, "probability": 0.8021, "result": "Higher diabetes risk", "advice": "..." }
  ]
}
```

### Field reference

| Field | Valid values |
|---|---|
| `HighBP`, `HighChol`, `CholCheck`, `Stroke`, `HeartDiseaseorAttack`, `Smoker`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`, `DiffWalk` | 0 or 1 |
| `Sex` | 0 = female, 1 = male |
| `BMI` | 12–98 |
| `MentHlth`, `PhysHlth` | 0–30 (days in past month) |
| `GenHlth` | 1 (excellent) – 5 (poor) |
| `Age` | 1–13 bracket code (`/predict` only) |
| `AgeYears` | 18–150 (`/predict-batch` only) |
| `Education` | 1–6 |
| `Income` | 1–8 |

---

## Deployment

The included `Procfile` works on any PaaS that supports it (Railway, Render, Heroku):

```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Port to bind |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins — set to your real frontend domain(s) in production |

Both `.pkl` artifacts are committed, so the API starts without a training step.

---

## Disclaimer

This is an educational project, **not a medical device**. It must not be used for diagnosis or clinical decisions. Predictions come from a statistical model trained on self-reported survey data and carry substantial error — at the chosen threshold, roughly a third of positive predictions are false positives. Anyone concerned about their diabetes risk should consult a healthcare professional.

---

## License

<!-- TODO: add a LICENSE file and name it here. MIT is a common choice. -->
