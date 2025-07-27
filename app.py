import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
from tabulate import tabulate
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib as mpl
import requests
import io

# Define the space-themed color palette
COLOR_PALETTE = {
    'dark_slate': '#1C2526',  # Near-black background
    'starry_white': '#F4F4F9',  # Light gray-white for text
    'nebula_purple': '#5E4B8C'  # Vibrant purple for accents
}

# Set Matplotlib color cycle
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[COLOR_PALETTE['nebula_purple']])

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    session_vars = {
        'cnn_model': None,
        'xgb_model': None,
        'scaler_cnn': None,
        'scaler_xgb': None,
        'label_encoder': None,
        'features_to_use': None,
        'data_loaded': False,
        'num_classes': None,
        'prediction_results': None,
        'cnn_config': {'conv_layers': 2, 'filters': [8, 16], 'kernel_size': 5, 'dense_units': 32, 'dropout_rate': 0.3, 'learning_rate': 1e-3},
        'xgb_config': {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'subsample': 0.8}
    }
    for var, default in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default

def download_gdrive_file(url):
    """Downloads a large file from Google Drive that has a virus scan warning page."""
    with requests.Session() as session:
        # Initial request to get the confirmation token
        response = session.get(url, stream=True)
        token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
        
        # Second request with the token to start the download
        if token:
            params = {'confirm': token}
            response = session.get(url, params=params, stream=True)
        
        # Read the content
        file_content = response.content
        return file_content

# Load datasets
@st.cache_data(show_spinner="Loading datasets...")
def load_data():
    """Load and cache datasets with error handling"""
    try:
        # Load exoTrain.csv from Google Drive
        with st.spinner("Downloading exoTrain.csv from Google Drive..."):
            exotrain_url = "https://drive.google.com/uc?export=download&id=1MxDUIqQ6S9cmi068I62xkbCwZHkpAeJI"
            file_content = download_gdrive_file(exotrain_url)
            raw_data = pd.read_csv(io.BytesIO(file_content))
        
        # Try to load cumulative.csv locally (optional)
        summary_data = None
        try:
            summary_data = pd.read_csv("datasets/cumulative.csv")
        except FileNotFoundError:
            st.sidebar.warning("`cumulative.csv` not found. Summary data features will be disabled.")
        
        st.session_state.data_loaded = True
        return None, raw_data, summary_data
        
    except Exception as e:
        st.error(f"Error loading datasets: {e}")
        st.session_state.data_loaded = False
        return None, None, None

# [Rest of your existing code remains the same...]

def main():
    """Main application function"""
    setup_page_config()
    initialize_session_state()
    
    os.makedirs("datasets", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    _, raw_data, summary_data = load_data()
    
    st.sidebar.title("🌌 Exoplanet Detection")
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigation", 
                          ["Home", "Data Exploration", "Model Training", 
                           "Model Comparison", "Predictions", 
                           "Prediction Analysis", "Feedback"],
                          index=0)
    
    if page == "Home":
        landing_page()
    elif page == "Data Exploration":
        data_exploration_page(raw_data, summary_data)
    elif page == "Model Training":
        model_training_page(raw_data, summary_data)
    elif page == "Model Comparison":
        model_comparison_interface()
    elif page == "Predictions":
        predictions_page()
    elif page == "Prediction Analysis":
        prediction_analysis()
    elif page == "Feedback":
        user_feedback()

if __name__ == "__main__":
    main()
