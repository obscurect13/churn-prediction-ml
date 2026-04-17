import pandas as pd
import joblib
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ======================
# 1. LOAD & CLEAN DATA
# ======================
df = pd.read_csv("../data/telco.csv")
df.columns = df.columns.str.strip()

# Replace blank strings with NaN
df = df.replace(r'^\s*$', np.nan, regex=True)

# Fix TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")


# ======================
# 2. DEFINE COLUMNS
# ======================
num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())
    
categorical_cols = [
    "gender", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

for col in categorical_cols:
    df[col] = df[col].astype(str).fillna("Unknown").str.strip()

# ======================
# 3. SPLIT FEATURES / TARGET
# ======================
target = "Churn"
X = df.drop(columns=[target, "customerID"])
y = df[target].map({"Yes": 1, "No": 0})

# ======================
# 4. TRAIN / TEST SPLIT
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ======================
# 5. PREPROCESSOR
# ======================
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ("num", "passthrough", num_cols)
    ],
    remainder="drop",
    verbose_feature_names_out=False
    
)
# ======================
# 6. PIPELINE
# ======================
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", XGBClassifier(eval_metric="logloss", random_state=42))
])

# ======================
# 7. GRID SEARCH
# ======================
param_grid = {
    "smote__k_neighbors": [3, 5, 7],
    "model__n_estimators": [100, 200, 500],
    "model__max_depth": [3, 4, 5],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__scale_pos_weight": [1, 3, 5]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    scoring='f1',
    cv=cv,
    n_jobs=-1,
)

grid_search.fit(X_train, y_train)

print("Best params:", grid_search.best_params_)
print(f"Best f1-score: {grid_search.best_score_:.2%}")

# ======================
# 8. MODEL
# ======================
model = grid_search.best_estimator_

model.fit(X_train, y_train)

# ======================
# 9. EVALUATION
# ======================
#preds = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]
y_pred = (proba > 0.35)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Churn', 'Churn'])
disp.plot(cmap=plt.cm.Blues)

plt.title("Confusion Matrix")
plt.show()
# ======================
# 10. SAVE MODEL & SAMPLE
# ======================
os.makedirs("../models", exist_ok=True)

X_sample = X_test.sample(200, random_state=42)
joblib.dump(X_sample, "../models/X_sample.pkl")

joblib.dump(model, "../models/model.pkl")
print("Model saved successfully")