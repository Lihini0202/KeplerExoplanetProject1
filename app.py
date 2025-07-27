import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from tabulate import tabulate
import shap
import matplotlib.pyplot as plt
import xgboost as xgb
import pickle
import os

# Streamlit app title
st.title("Kepler Exoplanet Classification Dashboard")

# Function to load datasets
@st.cache_data
def load_data():
    try:
        # Replace with actual paths to your dataset files
        summary_data = pd.read_csv("cumulative.csv")
        raw_train_data = pd.read_csv("exoTrain.csv")
        raw_test_data = pd.read_csv("exoTest.csv")
        return summary_data, raw_train_data, raw_test_data
    except FileNotFoundError:
        st.error("Dataset files not found. Please ensure 'cumulative.csv', 'exoTrain.csv', and 'exoTest.csv' are in the app directory.")
        return None, None, None

# Load datasets
summary_data, raw_train_data, raw_test_data = load_data()

# Display dataset information
st.header("Dataset Information")
if summary_data is not None:
    st.write(f"**Summary Dataset Shape**: {summary_data.shape}")
    st.write(f"**Train Light Curves Shape**: {raw_train_data.shape}")
    st.write(f"**Test Light Curves Shape**: {raw_test_data.shape}")
    
    st.subheader("First 5 Rows of Summary Data")
    st.dataframe(summary_data.head())
    
    st.subheader("First 5 Rows of Raw Training Data")
    st.dataframe(raw_train_data.head())

# Load pre-trained XGBoost model (assumes model is saved as 'xgb_model.pkl')
try:
    xgb_model = pickle.load(open("xgb_model.pkl", "rb"))
except FileNotFoundError:
    st.error("XGBoost model file 'xgb_model.pkl' not found. Please ensure the model is saved in the app directory.")
    xgb_model = None

# Placeholder for CNN metrics (since CNN model training is complex and not included here)
# These values are taken from the notebook output for demonstration
cnn_accuracies = [0.9551]  # Example value from notebook
cnn_aucs = [0.7501]
cnn_precisions = [0.0222]
cnn_recalls = [0.0922]
cnn_f1s = [0.0356]

# Function to compute metrics
def compute_metrics():
    if xgb_model is None or raw_test_data is None:
        return None, None, None, None, None
    
    # Assuming X_test and y_test are prepared from raw_test_data
    # For simplicity, assume labels are in 'LABEL' column (1-based indexing adjusted to 0-based)
    X_test = raw_test_data.drop(columns=['LABEL']).values
    y_test = raw_test_data['LABEL'].values - 1  # Adjust labels to 0 and 1
    y_pred_xgb = xgb_model.predict(X_test)
    
    xgb_accuracy = round(accuracy_score(y_test, y_pred_xgb) * 100, 2)
    xgb_auc = round(roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1]), 4)
    xgb_precision = round(precision_score(y_test, y_pred_xgb, zero_division=0), 4)
    xgb_recall = round(recall_score(y_test, y_pred_xgb, zero_division=0), 4)
    xgb_f1 = round(f1_score(y_test, y_pred_xgb, zero_division=0), 4)
    
    return xgb_accuracy, xgb_auc, xgb_precision, xgb_recall, xgb_f1

# Performance Comparison
st.header("Model Performance Comparison")
if xgb_model is not None and raw_test_data is not None:
    xgb_accuracy, xgb_auc, xgb_precision, xgb_recall, xgb_f1 = compute_metrics()
    
    table = [
        ["XGBoost", xgb_accuracy, xgb_auc, xgb_precision, xgb_recall, xgb_f1],
        ["1D-CNN", round(np.mean(cnn_accuracies) * 100, 2), round(np.mean(cnn_aucs), 4),
         round(np.mean(cnn_precisions), 4), round(np.mean(cnn_recalls), 4), round(np.mean(cnn_f1s), 4)]
    ]
    headers = ["Model", "Accuracy (%)", "AUC", "CONFIRMED Precision", "CONFIRMED Recall", "CONFIRMED F1"]
    
    st.subheader("Summary Table")
    st.text(tabulate(table, headers=headers, tablefmt="grid"))

# SHAP Analysis
st.header("SHAP Feature Importance")
if xgb_model is not None and raw_test_data is not None:
    st.subheader("Select SHAP Plot Type")
    plot_type = st.selectbox("Choose plot type:", ["Bar", "Dot"])
    
    # Compute SHAP values
    X_test = raw_test_data.drop(columns=['LABEL'])
    explainer = shap.Explainer(xgb_model)
    shap_values = explainer(X_test)
    
    # Generate and display SHAP plot
    fig, ax = plt.subplots()
    if plot_type == "Bar":
        shap.summary_plot(shap_values, X_test, feature_names=X_test.columns, plot_type='bar', show=False)
        plt.title("SHAP Feature Importance - Bar Plot")
    else:
        shap.summary_plot(shap_values, X_test, feature_names=X_test.columns, show=False)
        plt.title("SHAP Feature Importance - Dot Plot")
    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()

st.write("Note: The CNN model is not included in this app due to its complexity and training requirements. Metrics are displayed based on the notebook output.")
