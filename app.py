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

# --- Load XGBoost Model ---
@st.cache_resource
def load_xgb_model():
    """Load XGBoost model from pickle file."""
    try:
        return joblib.load("models/xgb_model.pkl")
    except Exception as e:
        st.error(f"Error loading XGBoost model: {e}")
        return None

# --- Load Label Encoder ---
@st.cache_resource
def load_label_encoder():
    """Load LabelEncoder from pickle file."""
    try:
        return joblib.load("models/label_encoder.pkl")
    except Exception as e:
        st.error(f"Error loading label encoder: {e}")
        return None

# --- Load Scaler ---
@st.cache_resource
def load_scaler():
    """Load StandardScaler from pickle file."""
    try:
        return joblib.load("models/scaler.pkl")
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        return None

# --- Page: Home ---
def home():
    st.title("🔭 Kepler Exoplanet Explorer")
    st.markdown("---")
    st.markdown("""
        Explore NASA's Kepler dataset and see how ML models detect exoplanets.
    """)
    st.subheader("🎥 NASA’s Kepler Mission")
    st.video("https://www.youtube.com/watch?v=3yij1rJOefM")

# --- Page: Data Explorer ---
def data_explorer(summary, train, test):
    st.header("📊 Data Explorer")
    
    if summary is not None:
        st.subheader("KOI Summary Data")
        st.dataframe(summary)

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

    st.subheader("Bar Plot")
    try:
        bar_img = Image.open("shap_summary_bar.png")
        st.image(bar_img, use_column_width=True)
    except FileNotFoundError:
        st.error("SHAP bar plot image not found.")

    st.subheader("Dot Plot")
    try:
        dot_img = Image.open("shap_summary_dot.png")
        st.image(dot_img, use_column_width=True)
    except FileNotFoundError:
        st.error("SHAP dot plot image not found.")

# --- Page: CNN Fold Evaluation ---
def cnn_fold_evaluator(test_df):
    st.header("📊 CNN Fold Evaluation")
    
    if test_df is not None:
        X_test = test_df.drop("LABEL", axis=1).values
        y_test = test_df["LABEL"].values - 1  # assuming labels are 1 & 2

        scaler = load_scaler()
        if scaler is not None:
            X_test_scaled = scaler.transform(X_test)
            X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
        else:
            st.error("Scaler not loaded, using raw data.")
            X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        folds = [1, 2, 3, 4, 5]
        selected_fold = st.selectbox("Select Fold", folds)

        model_path = f"models/exoplanet_cnn_fold{selected_fold}.keras"
        if os.path.exists(model_path):
            try:
                model = load_model(model_path, custom_objects={'focal_loss_fixed': focal_loss()})
                st.write(f"X_test_cnn shape: {X_test_cnn.shape}")
                y_pred_proba = model.predict(X_test_cnn)
                y_pred = (y_pred_proba > 0.3).astype("int").ravel()
                acc = accuracy_score(y_test, y_pred)

                st.metric(f"Fold {selected_fold} Accuracy", f"{acc:.2%}")

                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=["Not Exoplanet", "Exoplanet"],
                            yticklabels=["Not Exoplanet", "Exoplanet"])
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

                # Additional metrics
                auc = roc_auc_score(y_test, y_pred_proba)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                st.write(f"AUC: {auc:.4f}")
                st.write(f"Precision: {precision:.4f}")
                st.write(f"Recall: {recall:.4f}")
            except Exception as e:
                st.error(f"Error loading or predicting with model: {e}")
        else:
            st.error("Model file not found.")

# --- Page: Prediction Playground (XGBoost only) ---
def prediction_playground(test_df, model):
    st.header("🚀 Prediction Playground")
    
    if test_df is not None and model is not None:
        y_test = test_df['LABEL'].values - 1
        X_test = test_df.drop("LABEL", axis=1).values

        idx = st.slider("Select a Star Index", 0, len(test_df)-1, 0)
        star = X_test[idx]
        actual = "Exoplanet" if y_test[idx] == 1 else "Not Exoplanet"

        st.subheader(f"Light Curve for Star #{idx}")
        fig, ax = plt.subplots()
        ax.plot(star, color="royalblue")
        ax.set_title(f"Actual: {actual}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Flux")
        st.pyplot(fig)

        if st.button("Run Prediction"):
            prob = model.predict_proba([star])[0][1]
            pred = "Exoplanet" if prob > 0.5 else "Not Exoplanet"

            st.subheader("Prediction Result")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Prediction", pred)
                st.write(f"Confidence: {prob:.2%}")
            with col2:
                st.metric("Actual", actual)
            st.progress(prob)

# --- Main Driver ---
summary_df, train_df, test_df = load_data()
xgb_model = load_xgb_model()
label_encoder = load_label_encoder()  # Load label encoder if needed

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
    prediction_playground(test_df, xgb_model)
