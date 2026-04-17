import joblib
import os
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

clf = joblib.load(MODEL_PATH)
print("Loaded:", type(clf))

if isinstance(clf, Pipeline):
    print("Pipeline steps:", list(clf.named_steps.keys()))
    xgb_model = clf.named_steps["model"]
else:
    xgb_model = clf

tmp_path = os.path.join(os.path.expanduser("~"), "xgb_temp.json")

booster = xgb_model.get_booster()
booster.save_model(tmp_path)
booster.load_model(tmp_path)
print("XGBoost booster re-serialized OK")

os.remove(tmp_path)

# Save full pipeline back
joblib.dump(clf, MODEL_PATH)
print("Saved pipeline back to", MODEL_PATH)

# Sanity check
clf2 = joblib.load(MODEL_PATH)
if isinstance(clf2, Pipeline):
    steps = list(clf2.named_steps.keys())
    assert "preprocessor" in steps
    assert "model" in steps
    print("All good —", steps)