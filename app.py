import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from PIL import Image
import os
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, precision_score, recall_score
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import tensorflow as tf

# --- Custom Loss Function ---
def focal_loss(gamma=1.0, alpha=0.75):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
        bce = -y_true * tf.math.log(y_pred) - (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        focal_factor = tf.pow(1.0 - p_t, gamma)
        alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        return tf.reduce_mean(alpha_t * focal_factor * bce)
    return focal_loss_fixed

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="Kepler Exoplanet Explorer",
    page_icon="🔭",
    layout="wide"
)

# --- Load Data ---
@st.cache_data
def load_data():
    """Load Kepler datasets from CSV files."""
    try:
        summary = pd.read_csv("datasets/cumulative.csv")
        train = pd.read_csv("datasets/exoTrain.csv")
        test = pd.read_csv("datasets/exoTest.csv")
        return summary, train, test
    except FileNotFoundError as e:
        st.error(f"Error loading data files: {e}")
        return None, None, None

# --- Load Models ---
@st.cache_resource
def load_xgb_model():
    """Load XGBoost model from pickle file."""
    try:
        return joblib.load("models/xgb_model.pkl")
    except Exception as e:
        st.error(f"Error loading XGBoost model: {e}")
        return None

@st.cache_resource
def load_cnn_model(fold=1):
    """Load CNN model from file."""
    try:
        model_path = f"models/exoplanet_cnn_fold{fold}.keras"
        return load_model(model_path, custom_objects={'focal_loss_fixed': focal_loss()})
    except Exception as e:
        st.error(f"Error loading CNN model: {e}")
        return None

@st.cache_resource
def load_scaler():
    """Load StandardScaler from pickle file."""
    try:
        return joblib.load("models/scaler.pkl")
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        return None

@st.cache_resource
def load_label_encoder():
    """Load LabelEncoder from pickle file."""
    try:
        return joblib.load("models/label_encoder.pkl")
    except Exception as e:
        st.error(f"Error loading label encoder: {e}")
        return None

# --- Page: Home ---
def home():
    st.title("🔭 Kepler Exoplanet Explorer")
    st.markdown("---")
    st.markdown("""
        ### Detect exoplanets from Kepler light curve data using machine learning
        
        This app allows you to:
        - Explore the Kepler exoplanet dataset
        - Visualize light curves and data distributions
        - Evaluate pre-trained machine learning models
        - Make predictions on test data
    """)
    
    st.subheader("🎥 NASA's Kepler Mission")
    st.video("https://www.youtube.com/watch?v=3yij1rJOefM")

# --- Page: Data Explorer ---
def data_explorer(summary, train, test):
    st.header("📊 Data Explorer")
    
    if summary is not None:
        st.subheader("KOI Summary Data")
        st.dataframe(summary.head())

        with st.expander("Show Summary Statistics"):
            st.dataframe(summary.describe())

    if train is not None:
        st.subheader("Training Data Sample")
        st.dataframe(train.head())

    if test is not None:
        st.subheader("Test Data Sample")
        st.dataframe(test.head())

# --- Page: Visual Analytics ---
def visual_analytics(summary):
    st.header("📈 Visual Analytics")
    
    if summary is not None:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            sns.countplot(x="koi_disposition", data=summary, ax=ax)
            ax.set_title("KOI Disposition Distribution")
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots()
            summary["koi_disposition"].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
            ax.set_ylabel("")
            ax.set_title("Disposition Proportions")
            st.pyplot(fig)

# --- Page: SHAP Explainability ---
def shap_explainer():
    st.header("🧠 SHAP Explainability")

    st.subheader("Feature Importance")
    try:
        bar_img = Image.open("shap_summary_bar.png")
        st.image(bar_img, use_column_width=True)
    except FileNotFoundError:
        st.error("SHAP bar plot image not found.")

    st.subheader("Detailed Feature Impact")
    try:
        dot_img = Image.open("shap_summary_dot.png")
        st.image(dot_img, use_column_width=True)
    except FileNotFoundError:
        st.error("SHAP dot plot image not found.")

# --- Page: CNN Fold Evaluation ---
def cnn_fold_evaluator(test_df):
    st.header("📊 CNN Fold Evaluation")
    
    if test_df is None:
        st.warning("Test data not loaded")
        return
    
    X_test = test_df.drop("LABEL", axis=1).values
    y_test = test_df["LABEL"].values - 1  # Convert labels to 0 and 1

    scaler = load_scaler()
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
        X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    else:
        st.error("Scaler not loaded, using raw data.")
        X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    folds = [1, 2, 3, 4, 5]
    selected_fold = st.selectbox("Select Fold", folds)

    model = load_cnn_model(selected_fold)
    if model is None:
        st.error(f"CNN model for fold {selected_fold} not found")
        return
    
    st.write(f"Evaluating CNN model (Fold {selected_fold})...")
    
    y_pred_proba = model.predict(X_test_cnn)
    y_pred = (y_pred_proba > 0.3).astype("int").ravel()
    acc = accuracy_score(y_test, y_pred)

    st.metric(f"Fold {selected_fold} Accuracy", f"{acc:.2%}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Not Exoplanet", "Exoplanet"],
                yticklabels=["Not Exoplanet", "Exoplanet"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("AUC", f"{auc:.4f}")
    col2.metric("Precision", f"{precision:.4f}")
    col3.metric("Recall", f"{recall:.4f}")

# --- Page: Prediction Playground ---
def prediction_playground(test_df, model_type="XGBoost"):
    st.header("🚀 Prediction Playground")
    
    if test_df is None:
        st.warning("Test data not loaded")
        return
    
    if model_type == "XGBoost":
        model = load_xgb_model()
    else:
        model = load_cnn_model(1)  # Default to fold 1
    
    if model is None:
        st.error(f"{model_type} model not found")
        return
    
    scaler = load_scaler()
    if scaler is None:
        st.error("Scaler not found")
        return
    
    y_test = test_df['LABEL'].values - 1
    X_test = test_df.drop("LABEL", axis=1).values

    idx = st.slider("Select a Star Index", 0, len(test_df)-1, 0)
    star = X_test[idx]
    actual = "Exoplanet" if y_test[idx] == 1 else "Not Exoplanet"

    st.subheader(f"Light Curve for Star #{idx}")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(star, color="royalblue")
    ax.set_title(f"Actual: {actual}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Flux")
    st.pyplot(fig)

    if st.button("Run Prediction"):
        star_scaled = scaler.transform([star])
        
        if model_type == "XGBoost":
            prob = model.predict_proba(star_scaled)[0][1]
        else:
            star_cnn = star_scaled.reshape(star_scaled.shape[0], star_scaled.shape[1], 1)
            prob = model.predict(star_cnn)[0][0]
        
        pred = "Exoplanet" if prob > 0.5 else "Not Exoplanet"

        st.subheader("Prediction Result")
        col1, col2, col3 = st.columns(3)
        col1.metric("Prediction", pred)
        col2.metric("Confidence", f"{prob:.2%}")
        col3.metric("Actual", actual)
        
        st.progress(prob if pred == "Exoplanet" else 1-prob)

# --- Main App ---
def main():
    # Load data
    summary_df, train_df, test_df = load_data()
    
    # Sidebar navigation
    st.sidebar.title("🌌 Navigation")
    page = st.sidebar.radio("Go to", [
        "🏠 Home",
        "📊 Data Explorer",
        "📈 Visual Analytics",
        "🧠 SHAP Explainability",
        "📊 CNN Fold Evaluation",
        "🚀 Prediction Playground"
    ])

    if page == "🏠 Home":
        home()
    elif page == "📊 Data Explorer":
        data_explorer(summary_df, train_df, test_df)
    elif page == "📈 Visual Analytics":
        visual_analytics(summary_df)
    elif page == "🧠 SHAP Explainability":
        shap_explainer()
    elif page == "📊 CNN Fold Evaluation":
        cnn_fold_evaluator(test_df)
    elif page == "🚀 Prediction Playground":
        prediction_playground(test_df)

if __name__ == "__main__":
    main()
