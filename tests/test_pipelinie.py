import joblib
import os
import pandas as pd

MODEL_PATH = os.path.join("models", "model.pkl")

def test_model_load():
    model = joblib.load(MODEL_PATH)
    assert model is not None


def test_model_prediction():
    model = joblib.load(MODEL_PATH)

    sample = pd.DataFrame([{
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 10,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 50,
        "TotalCharges": 500
    }])

    pred = model.predict(sample)
    assert pred[0] in [0, 1]