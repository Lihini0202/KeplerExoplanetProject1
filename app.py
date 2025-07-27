import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils import class_weight, resample
import xgboost as xgb
from tabulate import tabulate
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score

# Set page config
st.set_page_config(page_title="Exoplanet Detection", layout="wide")

# Title
st.title("Exoplanet Detection from Kepler Light Curves")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Exploration", "Model Training", "Predictions"])

# Load data function
@st.cache_data
def load_data():
    try:
        summary_data = pd.read_csv("datasets/cumulative.csv")
        raw_train_data = pd.read_csv("datasets/exoTrain.csv")
        raw_test_data = pd.read_csv("datasets/exoTest.csv")
        return summary_data, raw_train_data, raw_test_data
    except FileNotFoundError:
        st.error("ERROR: Dataset files not found.")
        return None, None, None

# Focal Loss Function
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

# Data augmentation function
def augment_data(X, y, sigma=0.04, max_shift=200):
    X_aug = X.copy()
    noise = np.random.normal(0, sigma, X.shape)
    X_aug += noise
    shifts = np.random.randint(-max_shift, max_shift, size=X.shape[0])
    for i in range(X.shape[0]):
        X_aug[i] = np.roll(X_aug[i], shifts[i], axis=0)
    return X_aug, y

# Load data
summary_data, raw_train_data, raw_test_data = load_data()

if page == "Data Exploration":
    st.header("Data Exploration")
    
    if summary_data is not None and raw_train_data is not None and raw_test_data is not None:
        st.success("Datasets Loaded Successfully!")
        st.write(f"Summary Dataset Shape: {summary_data.shape}")
        st.write(f"Train Light Curves Shape: {raw_train_data.shape}")
        st.write(f"Test Light Curves Shape: {raw_test_data.shape}")
        
        st.subheader("First 5 Rows from Summary Data:")
        st.dataframe(summary_data.head())
        
        st.subheader("First 5 Rows from Raw Training Data:")
        st.dataframe(raw_train_data.head())
        
        # Sample light curves visualization
        st.subheader("Sample Light Curves")
        num_samples = st.slider("Number of samples to display", 1, 5, 3)
        sample_indices = np.random.choice(raw_train_data.shape[0], num_samples, replace=False)
        
        for i, idx in enumerate(sample_indices):
            label = raw_train_data.iloc[idx]['LABEL']
            flux_values = raw_train_data.iloc[idx, 1:].values  # Skip LABEL column
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(flux_values, label=f"Sample #{idx} | Label: {'CONFIRMED' if label == 2 else 'FALSE POSITIVE'}")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Flux (Brightness)")
            ax.set_title(f"Light Curve for Sample #{idx}")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

elif page == "Model Training":
    st.header("Model Training")
    
    if raw_train_data is not None and raw_test_data is not None:
        st.subheader("Data Preparation")
        
        # Preprocessing steps would go here
        st.write("Preprocessing steps would be implemented here...")
        
        st.subheader("Model Selection")
        model_type = st.selectbox("Select model type", ["CNN", "XGBoost"])
        
        if model_type == "CNN":
            st.write("CNN model architecture and training would be implemented here...")
            
            # Example CNN architecture
            input_shape = (3197, 1)
            inputs = Input(shape=input_shape)
            x = Conv1D(32, 3, activation='relu', padding='same', kernel_regularizer=l2(0.01))(inputs)
            x = MaxPooling1D(2)(x)
            x = BatchNormalization()(x)
            x = Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
            x = MaxPooling1D(2)(x)
            x = BatchNormalization()(x)
            x = Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
            x = GlobalAveragePooling1D()(x)
            x = Dropout(0.5)(x)
            outputs = Dense(1, activation='sigmoid')(x)
            
            model = Model(inputs, outputs)
            
            st.code("""
            input_shape = (3197, 1)
            inputs = Input(shape=input_shape)
            x = Conv1D(32, 3, activation='relu', padding='same', kernel_regularizer=l2(0.01))(inputs)
            x = MaxPooling1D(2)(x)
            x = BatchNormalization()(x)
            x = Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
            x = MaxPooling1D(2)(x)
            x = BatchNormalization()(x)
            x = Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
            x = GlobalAveragePooling1D()(x)
            x = Dropout(0.5)(x)
            outputs = Dense(1, activation='sigmoid')(x)
            
            model = Model(inputs, outputs)
            """, language='python')
            
            if st.button("Train CNN Model"):
                with st.spinner("Training model..."):
                    # Here you would add the actual training code
                    st.success("Training complete!")
                    
        elif model_type == "XGBoost":
            st.write("XGBoost model training would be implemented here...")
            
            if st.button("Train XGBoost Model"):
                with st.spinner("Training model..."):
                    # Here you would add the actual training code
                    st.success("Training complete!")

elif page == "Predictions":
    st.header("Make Predictions")
    
    st.write("This section would allow users to upload new light curves for prediction.")
    
    uploaded_file = st.file_uploader("Upload a light curve file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("Uploaded data:")
            st.dataframe(data.head())
            
            # Here you would add code to make predictions on the uploaded data
            if st.button("Predict"):
                with st.spinner("Making prediction..."):
                    # Simulate prediction
                    prediction = np.random.choice([0, 1])
                    confidence = np.random.uniform(0.8, 0.99)
                    
                    if prediction == 1:
                        st.success(f"Prediction: EXOPLANET DETECTED (confidence: {confidence:.2%})")
                    else:
                        st.error(f"Prediction: NO EXOPLANET DETECTED (confidence: {confidence:.2%})")
                        
        except Exception as e:
            st.error(f"Error reading file: {e}")
