from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================
# Load pipeline (preprocessor + model are one object)
# =============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

try:
    pipeline = joblib.load(MODEL_PATH)
    logger.info("Pipeline loaded successfully from %s", MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(f"Model file not found at {MODEL_PATH}")

app = FastAPI()

# =============================
# INPUT SCHEMA
# =============================
class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# =============================
# PREPROCESSING (mirrors training cleaning)
# =============================
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]
NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

def preprocess_input(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    # Juste nettoyage léger (comme au training)
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(str).str.strip()

    return df
    
def preprocess_inputs(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(str).str.strip()

    # Validate columns match training exactly
    trained_cat = pipeline.named_steps["preprocessor"].transformers[0][2]
    trained_num = pipeline.named_steps["preprocessor"].transformers[1][2]
    expected_cols = list(trained_cat) + list(trained_num)
    missing = set(expected_cols) - set(df.columns)
    extra = set(df.columns) - set(expected_cols)

    if missing:
        raise ValueError(f"Missing columns vs training: {missing}")
    if extra:
        raise ValueError(f"Extra columns not seen in training: {extra}")

    # Enforce column order to match training
    return df[expected_cols]
# =============================
# PREDICT ENDPOINT
# =============================
@app.post("/predict")
def predict(input_data: CustomerInput):
    try:
        data = input_data.dict()
        df = preprocess_input(data)

        proba = pipeline.predict_proba(df)[0][1]
        prediction = (proba > 0.35)

        logger.info(f"Input: {data} | Prediction: {prediction} | Proba: {proba:.4f}")

        return {
            "churn_prediction": int(prediction),
            "churn_probability": round(float(proba), 4)
        }
    except Exception as e:
        logger.error("Prediction failed: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/debug/pipeline")
def debug_pipeline():
    ct = pipeline.named_steps["preprocessor"]
    return {
        "expected_cat_cols": ct.transformers[0][2],
        "expected_num_cols": ct.transformers[1][2],
        "feature_names_out": list(ct.get_feature_names_out())
    }

# =============================
# HEALTH CHECK
# =============================
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}