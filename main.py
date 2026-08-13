import io
import os

import joblib
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# =========================================================
# LOAD MODEL
# =========================================================
print("🔄 Loading model...")
model = joblib.load("best_diabetes_model_final.pkl")
print("✅ Model loaded successfully")

print("🔄 Loading threshold...")
threshold = joblib.load("best_threshold_final.pkl")  # --- CHANGED ---
print(f"✅ Threshold loaded: {threshold}")

# Exact feature order/list your training script produced
# (original columns minus drop_cols, in original order, plus engineered features)
feature_columns = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Stroke",
    "HvyAlcoholConsump", "GenHlth", "MentHlth", "PhysHlth",
    "DiffWalk", "Sex", "Age", "Education", "Income",
    "BMI_Category", "TotalUnhealthyDays", "CardioRiskScore",
    "HealthyLifestyleScore", "MobilityHealthBurden"
]

# =========================================================
# CREATE API APP
# =========================================================
app = FastAPI()

# =========================================================
# RATE LIMITING
# Limits requests per client IP so the public API can't be
# scraped or abused for cost/DoS. Adjust the limits below as needed.
# =========================================================
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # set ALLOWED_ORIGINS env var to your real frontend domain(s) in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    print("✅ Health check hit")
    return {"status": "backend is running"}

# =========================================================
# WEBSITE INPUT SCHEMA (single prediction)
# (includes raw fields needed to COMPUTE engineered features,
#  even though some are dropped before the final predict step)
# =========================================================
class DiabetesInput(BaseModel):
    HighBP: int
    HighChol: int
    CholCheck: int
    BMI: float
    Stroke: int
    HeartDiseaseorAttack: int
    Smoker: int
    PhysActivity: int
    Fruits: int
    Veggies: int
    HvyAlcoholConsump: int
    MentHlth: int
    PhysHlth: int
    GenHlth: int
    DiffWalk: int
    Sex: int
    Age: int
    Education: int
    Income: int

# =========================================================
# AGE CONVERSION (real age in years -> 1-13 BRFSS bracket code)
# Used by the batch/CSV endpoint, which takes real age rather than
# the raw bracket code, for a better end-user experience.
# =========================================================
AGE_BRACKETS = [
    (18, 24, 1),
    (25, 29, 2),
    (30, 34, 3),
    (35, 39, 4),
    (40, 44, 5),
    (45, 49, 6),
    (50, 54, 7),
    (55, 59, 8),
    (60, 64, 9),
    (65, 69, 10),
    (70, 74, 11),
    (75, 79, 12),
    (80, 150, 13),
]

def age_years_to_bracket(age_years):
    """Converts a real age in years to the 1-13 BRFSS age bracket code
    the model was trained on. Raises ValueError if out of supported range."""
    for lo, hi, code in AGE_BRACKETS:
        if lo <= age_years <= hi:
            return code
    raise ValueError(
        f"AgeYears must be between 18 and 150 (survey only covers adults), got {age_years}"
    )

# =========================================================
# FEATURE ENGINEERING (must match training script exactly)
# =========================================================
def bmi_to_category(bmi):
    if bmi < 18.5:
        return 0
    elif bmi < 25:
        return 1
    elif bmi < 30:
        return 2
    else:
        return 3


def compute_engineered_features(row: dict) -> dict:
    """Adds the 5 engineered features to a row dict that already contains
    all raw fields (including 'Age' as the 1-13 bracket code). Shared by
    both the single-prediction and batch/CSV code paths."""
    row = dict(row)

    row["BMI_Category"] = bmi_to_category(row["BMI"])

    row["TotalUnhealthyDays"] = row["MentHlth"] + row["PhysHlth"]

    row["CardioRiskScore"] = (
        row["HighBP"] +
        row["HighChol"] +
        row["HeartDiseaseorAttack"] +
        row["Stroke"]
    )

    row["HealthyLifestyleScore"] = (
        row["PhysActivity"] +
        row["Fruits"] +
        row["Veggies"] +
        (1 - row["Smoker"]) +
        (1 - row["HvyAlcoholConsump"])
    )

    row["MobilityHealthBurden"] = (
        row["DiffWalk"] +
        row["GenHlth"] +
        row["PhysHlth"]
    )

    return row


def prepare_features(data: DiabetesInput) -> pd.DataFrame:
    row = data.dict()
    row = compute_engineered_features(row)

    input_df = pd.DataFrame([row])

    # Keep only the exact columns the model was trained on, in the exact order
    input_df = input_df[feature_columns]

    return input_df

# =========================================================
# PREDICTION ENDPOINT (single record)
# =========================================================
@app.post("/predict")
@limiter.limit("20/minute")
def predict_diabetes(request: Request, data: DiabetesInput):
    print("📩 Incoming request:", data.dict())

    input_df = prepare_features(data)
    print("🧮 Feature vector sent to model:")
    print(input_df)

    probability = model.predict_proba(input_df)[0][1]
    prediction = int(probability >= threshold)

    if prediction == 1:
        result = "Higher diabetes risk"
        advice = "The result suggests higher diabetes risk. Please consider consulting a doctor."
    else:
        result = "Lower diabetes risk"
        advice = "The result suggests lower diabetes risk. Continue maintaining a healthy lifestyle."

    response = {
        "prediction": prediction,
        "probability": round(float(probability), 4),
        "threshold": round(float(threshold), 4),
        "result": result,
        "advice": advice
    }

    print("📤 Sending response:", response)
    return response

# =========================================================
# BATCH / CSV PREDICTION ENDPOINT
# =========================================================

# CSV must contain exactly these columns (any order). Note "AgeYears" is
# the person's real age -- NOT the 1-13 bracket code the model actually
# trains on. It gets converted server-side in validate_and_build_row().
REQUIRED_CSV_COLUMNS = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Stroke",
    "HeartDiseaseorAttack", "Smoker", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "MentHlth", "PhysHlth", "GenHlth", "DiffWalk",
    "Sex", "AgeYears", "Education", "Income",
]

BINARY_FIELDS = [
    "HighBP", "HighChol", "CholCheck", "Stroke", "HeartDiseaseorAttack",
    "Smoker", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump",
    "DiffWalk", "Sex",
]

SCALE_FIELDS = {
    "GenHlth": (1, 5),
    "Education": (1, 6),
    "Income": (1, 8),
}

RANGE_FIELDS = {
    "BMI": (12, 98),
    "MentHlth": (0, 30),
    "PhysHlth": (0, 30),
}

MAX_ROWS = 5000  # adjust as needed


def validate_and_build_row(raw_row: dict, row_number: int):
    """Validates one CSV row and, if valid, returns the fully engineered
    feature row (dict) ready for the model. Returns (feature_row, errors)
    where feature_row is None if validation failed."""
    errors = []

    for col in REQUIRED_CSV_COLUMNS:
        if col not in raw_row or pd.isna(raw_row[col]):
            errors.append(f"missing value for '{col}'")

    if errors:
        return None, errors

    try:
        row = {col: raw_row[col] for col in REQUIRED_CSV_COLUMNS}

        for f in BINARY_FIELDS:
            row[f] = int(row[f])
            if row[f] not in (0, 1):
                errors.append(f"'{f}' must be 0 or 1, got {row[f]}")

        for f, (lo, hi) in SCALE_FIELDS.items():
            row[f] = int(row[f])
            if not (lo <= row[f] <= hi):
                errors.append(f"'{f}' must be between {lo} and {hi}, got {row[f]}")

        for f, (lo, hi) in RANGE_FIELDS.items():
            row[f] = int(row[f])
            if not (lo <= row[f] <= hi):
                errors.append(f"'{f}' must be between {lo} and {hi}, got {row[f]}")

        age_years = float(row["AgeYears"])
        try:
            row["Age"] = age_years_to_bracket(age_years)
        except ValueError as e:
            errors.append(str(e))

    except (TypeError, ValueError) as e:
        errors.append(f"could not parse row: {e}")
        return None, errors

    if errors:
        return None, errors

    del row["AgeYears"]
    feature_row = compute_engineered_features(row)

    return feature_row, []


@app.post("/predict-batch")
@limiter.limit("5/minute")
async def predict_batch(request: Request, file: UploadFile = File(...)):
    print(f"📩 Incoming batch file: {file.filename}")

    raw_bytes = await file.read()

    try:
        csv_df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as e:
        return {"error": f"Could not parse CSV: {e}"}

    if len(csv_df) == 0:
        return {"error": "CSV file contains no data rows."}

    if len(csv_df) > MAX_ROWS:
        return {"error": f"CSV contains {len(csv_df)} rows, which exceeds the {MAX_ROWS}-row limit per upload."}

    missing_columns = [c for c in REQUIRED_CSV_COLUMNS if c not in csv_df.columns]
    if missing_columns:
        return {"error": f"CSV is missing required column(s): {missing_columns}"}

    feature_rows = []      # engineered feature dicts, valid rows only
    feature_row_numbers = []  # original CSV row numbers matching feature_rows
    results = []            # final per-row output, valid rows filled in after inference
    errors_by_row = {}      # row_number -> list[str]

    for i, raw_row in enumerate(csv_df.to_dict(orient="records")):
        row_number = i + 1  # 1-indexed, matches row order in the file (excluding header)
        feature_row, errors = validate_and_build_row(raw_row, row_number)

        if errors:
            errors_by_row[row_number] = errors
        else:
            feature_rows.append(feature_row)
            feature_row_numbers.append(row_number)

    if feature_rows:
        batch_input_df = pd.DataFrame(feature_rows)[feature_columns]
        print(f"🧮 Scoring {len(batch_input_df)} valid row(s) out of {len(csv_df)} total")

        probabilities = model.predict_proba(batch_input_df)[:, 1]

        for row_number, probability in zip(feature_row_numbers, probabilities):
            prediction = int(probability >= threshold)
            result = "Higher diabetes risk" if prediction == 1 else "Lower diabetes risk"
            advice = (
                "The result suggests higher diabetes risk. Please consider consulting a doctor."
                if prediction == 1 else
                "The result suggests lower diabetes risk. Continue maintaining a healthy lifestyle."
            )
            results.append({
                "row": row_number,
                "prediction": prediction,
                "probability": round(float(probability), 4),
                "result": result,
                "advice": advice,
            })

    for row_number, errors in errors_by_row.items():
        results.append({
            "row": row_number,
            "errors": errors,
        })

    results.sort(key=lambda r: r["row"])

    response = {
        "total_rows": len(csv_df),
        "valid_rows": len(feature_rows),
        "invalid_rows": len(errors_by_row),
        "threshold": round(float(threshold), 4),
        "results": results,
    }

    print(f"📤 Sending batch response: {len(feature_rows)} scored, {len(errors_by_row)} invalid")
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)