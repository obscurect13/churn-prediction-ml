import streamlit as st
import requests
import shap
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Churn Prediction", layout="centered")

API_URL = "http://api:8000/predict"

@st.cache_resource
def load_model_and_explainer():
    model = joblib.load("models/model.pkl")
    X_sample = joblib.load("models/X_sample.pkl")

    from sklearn.pipeline import Pipeline
    if isinstance(model, Pipeline):
        preprocessor = model.named_steps["preprocessor"]
        xgb_model = model.named_steps["model"]
        X_transformed = preprocessor.transform(X_sample)
        feature_names = list(preprocessor.get_feature_names_out())
    else:
        xgb_model = model
        X_transformed = X_sample
        feature_names = list(X_sample.columns)

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_transformed)

    shap_values.feature_names = feature_names

    return explainer, shap_values, X_transformed, feature_names

explainer, shap_values, X_transformed, feature_names = load_model_and_explainer()

tabs = st.tabs(["🔮 Churn Prediction", "📊 SHAP Analysis"])

# =========================
# TAB 1 — PREDICTION
# =========================
with tabs[0]:
    st.title("🚀 Customer Churn Prediction")
    st.write("Fill customer information below to predict churn risk.")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

    with col2:
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
        total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)

    if st.button("Predict Churn", type="primary"):
        payload = {
            "gender": gender, "SeniorCitizen": senior,
            "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone,
            "MultipleLines": multiple_lines, "InternetService": internet,
            "OnlineSecurity": online_security, "OnlineBackup": online_backup,
            "DeviceProtection": device_protection, "TechSupport": tech_support,
            "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment, "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }

        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                result = response.json()
                churn_label = "Yes ⚠️" if result["churn_prediction"] else "No ✅"
                proba = result["churn_probability"]

                st.success("Prediction completed!")
                m1, m2 = st.columns(2)
                m1.metric("Churn Prediction", churn_label)
                m2.metric("Churn Probability", f"{proba:.2%}")

                color = "red" if proba > 0.35 else "orange" if proba > 0.2 else "green"
                st.markdown(f"""
                    <div style='background:{color};padding:8px;border-radius:6px;
                                text-align:center;color:white;font-weight:bold'>
                        Risk level: {"High" if proba > 0.35 else "Medium" if proba > 0.2 else "Low"}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"API Error {response.status_code}: {response.text}")

        except Exception as e:
            st.error(f"Request failed: {str(e)}")

# =========================
# TAB 2 — SHAP
# =========================
with tabs[1]:
    st.title("📊 SHAP Explainability")
    st.write("Analyse des features qui influencent le churn")

    st.subheader("📌 Feature Importance globale")
    fig1, ax1 = plt.subplots()
    shap.plots.bar(shap_values, max_display=10, show=False)
    st.pyplot(fig1)
    plt.close(fig1)  # ✅ FIX 6: close figures to prevent memory leak

    st.subheader("👤 Explication individuelle")
    index = st.slider("Choisir un client", 0, len(X_transformed) - 1, 0)

    fig2, ax2 = plt.subplots()
    shap.plots.waterfall(shap_values[index], show=False)
    st.pyplot(fig2)
    plt.close(fig2)