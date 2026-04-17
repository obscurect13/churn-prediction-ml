# 📊 Churn Prediction Project

This project is an end-to-end **Machine Learning system for customer churn prediction**, including training, API serving, and a Streamlit dashboard.

---

# 🚀 Project Overview

The goal is to predict customer churn using historical telecom data. The project includes:

* Data preprocessing
* Model training with XGBoost
* REST API with FastAPI
* Interactive dashboard with Streamlit
* Production-ready Docker deployment

---

# 🧱 Project Architecture

```
User → Streamlit UI → FastAPI Model API → ML Model (XGBoost)
```

In production, the system is containerized:

```
Docker Compose
│
├── streamlit service (UI)
├── fastapi service (API)
```

# ⚖️ Handling Class Imbalance (SMOTE)

The dataset is imbalanced, which can bias the model toward the majority class.

To address this, SMOTE (Synthetic Minority Over-sampling Technique) is used to generate synthetic samples for the minority class.

SMOTE is integrated into a scikit-learn Pipeline and applied only on training folds during cross-validation to avoid data leakage.

# 🐳 Docker Setup

This project is fully containerized using Docker.

## Build the containers

```bash
docker-compose build
```

## Run the application

```bash
docker-compose up
```

---

# 🌐 Services

| Service   | Description           | Port |
| --------- | --------------------- | ---- |
| Streamlit | User Dashboard        | 8501 |
| FastAPI   | Prediction API        | 8000 |

---

# 📦 Model

* Algorithm: XGBoost Classifier
* Output: Probability of churn
* Preprocessing: Encoding + Scaling + Pipeline

---

# 📊 Features

* Customer demographics
* Contract type
* Services subscribed
* Billing information

---

# 🔍 Explainability

SHAP values are used to interpret model predictions and understand feature importance.

---

# 🛠️ Tech Stack

* Python
* Scikit-learn
* XGBoost
* FastAPI
* Streamlit
* imbalanced-learn
* Docker

---

# 📌 Author

Omar SAID
