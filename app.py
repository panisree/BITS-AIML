import streamlit as st
from model.logistic_regression import run_model as lr
from model.decision_tree import run_model as dt
from model.knn import run_model as knn
from model.naive_bayes import run_model as nb
from model.random_forest import run_model as rf
from model.xgboost_model import run_model as xgb

st.title("Breast Cancer Classification App")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is None:
    st.warning("Upload dataset to start.")
    st.stop()

data_path = pd.read_csv(uploaded_file)

if "diagnosis" not in data_path.columns:
    st.error("Dataset must contain 'diagnosis' column.")
    st.stop()

if "id" in data_path.columns:
    data_path = data_path.drop(columns=["id"])

if data_path["diagnosis"].dtype == "object":
    data_path["diagnosis"] = data_path["diagnosis"].map({"M": 1, "B": 0})

X = data_path.drop("diagnosis", axis=1)
y = data_path["diagnosis"]

model_choice = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "KNN",
     "Naive Bayes", "Random Forest", "XGBoost"]
)

if st.button("Run Model"):

    if model_choice == "Logistic Regression":
        results = lr(data_path)
    elif model_choice == "Decision Tree":
        results = dt(data_path)
    elif model_choice == "KNN":
        results = knn(data_path)
    elif model_choice == "Naive Bayes":
        results = nb(data_path)
    elif model_choice == "Random Forest":
        results = rf(data_path)
    else:
        results = xgb(data_path)

    st.subheader("Evaluation Metrics")

    for key, value in results.items():
        if key != "Confusion Matrix":
            st.write(f"{key}: {value}")

    st.subheader("Confusion Matrix")
    st.write(results["Confusion Matrix"])
