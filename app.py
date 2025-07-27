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

# ===== PAGE CONFIGURATION =====
st.set_page_config(
    page_title="Exoplanet Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🌌"
)

# Define the space-themed color palette
COLOR_PALETTE = {
    'dark_slate': '#1C2526',
    'starry_white': '#F4F4F9',
    'nebula_purple': '#5E4B8C'
}

# Set Matplotlib color cycle
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[COLOR_PALETTE['nebula_purple']])

# Custom CSS
st.markdown(f"""
    <style>
    :root {{ background-color: {COLOR_PALETTE['dark_slate']} !important; }}
    .main {{ background-color: {COLOR_PALETTE['dark_slate']} !important; color: {COLOR_PALETTE['starry_white']} !important; }}
    .stSidebar {{ background-color: {COLOR_PALETTE['dark_slate']} !important; color: {COLOR_PALETTE['starry_white']} !important; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 10px; background-color: {COLOR_PALETTE['dark_slate']} !important; }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px; padding: 0 25px; background-color: {COLOR_PALETTE['nebula_purple']} !important;
        color: {COLOR_PALETTE['starry_white']} !important; border-radius: 8px 8px 0px 0px; font-weight: bold;
    }}
    .stButton>button {{
        border-radius: 8px; height: 40px; width: 100%;
        background-color: {COLOR_PALETTE['nebula_purple']} !important;
        color: {COLOR_PALETTE['starry_white']} !important; font-weight: bold; border: none;
    }}
    </style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if 'data_loaded' not in st.session_state:
        st.session_state.update({
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
        })

def clear_memory():
    tf.keras.backend.clear_session()
    gc.collect()
    if 'cnn_model' in st.session_state:
        del st.session_state.cnn_model
    if 'xgb_model' in st.session_state:
        del st.session_state.xgb_model

def focal_loss(gamma=2.0, alpha=0.5):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        bce = -y_true * tf.math.log(y_pred) - (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        focal_factor = tf.pow(1.0 - p_t, gamma)
        alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        return tf.reduce_mean(alpha_t * focal_factor * bce)
    return focal_loss_fixed

def download_gdrive_file(url):
    with requests.Session() as session:
        response = session.get(url, stream=True)
        token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
        if token:
            response = session.get(url, params={'confirm': token}, stream=True)
        return response.content

@st.cache_data(show_spinner="Loading datasets...")
def load_data():
    try:
        with st.spinner("Downloading exoTrain.csv..."):
            url = "https://drive.google.com/uc?export=download&id=1MxDUIqQ6S9cmi068I62xkbCwZHkpAeJI"
            raw_data = pd.read_csv(io.BytesIO(download_gdrive_file(url)))
        
        summary_data = None
        try:
            summary_data = pd.read_csv("datasets/cumulative.csv")
        except FileNotFoundError:
            st.sidebar.warning("cumulative.csv not found")
        
        st.session_state.data_loaded = True
        return None, raw_data, summary_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

def plot_light_curves(data, n_samples=5):
    try:
        if 'LABEL' not in data.columns:
            samples = data.sample(n=min(n_samples, len(data)))
            fig = make_subplots(rows=1, cols=1, subplot_titles=("Light Curves",))
            for idx, row in samples.iterrows():
                fig.add_trace(go.Scatter(y=row, mode='lines', name=f'Star {idx}',
                                      line=dict(color=COLOR_PALETTE['nebula_purple'])))
        else:
            exo_samples = data[data['LABEL'] == 2].sample(n=min(n_samples, len(data[data['LABEL'] == 2])))
            non_samples = data[data['LABEL'] == 1].sample(n=min(n_samples, len(data[data['LABEL'] == 1])))
            
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Exoplanet", "Non-Exoplanet"))
            for idx, row in exo_samples.iterrows():
                fig.add_trace(go.Scatter(y=row[1:], mode='lines', name=f'Exo {idx}',
                                      line=dict(color=COLOR_PALETTE['nebula_purple'])), row=1, col=1)
            for idx, row in non_samples.iterrows():
                fig.add_trace(go.Scatter(y=row[1:], mode='lines', name=f'Non {idx}',
                                      line=dict(color=COLOR_PALETTE['starry_white'])), row=1, col=2)

        fig.update_layout(
            height=400, showlegend=False,
            paper_bgcolor=COLOR_PALETTE['dark_slate'],
            plot_bgcolor=COLOR_PALETTE['dark_slate'],
            font=dict(color=COLOR_PALETTE['starry_white'])
        )
        return fig
    except Exception as e:
        st.error(f"Plotting error: {e}")
        return None

def load_cnn_model():
    try:
        clear_memory()
        st.session_state.cnn_model = load_model(
            'models/cnn_exoplanet_model.keras',
            custom_objects={'focal_loss_fixed': focal_loss()}
        )
        st.session_state.scaler_cnn = joblib.load('models/scaler_cnn.pkl')
        st.success("CNN model loaded!")
        return True
    except Exception as e:
        st.error(f"CNN load error: {e}")
        return False

def load_xgb_model():
    try:
        clear_memory()
        st.session_state.xgb_model = joblib.load('models/xgb_model.pkl')
        st.session_state.scaler_xgb = joblib.load('models/scaler_xgb.pkl')
        st.session_state.label_encoder = joblib.load('models/label_encoder.pkl')
        st.session_state.features_to_use = joblib.load('models/features_to_use.pkl')
        st.success("XGBoost model loaded!")
        return True
    except Exception as e:
        st.error(f"XGB load error: {e}")
        return False

def landing_page():
    st.title("🌌 Exoplanet Detection Dashboard")
    st.markdown("""
    ### Discover Exoplanets Through Data
    - **Data Exploration**: View light curves
    - **Model Training**: Build classification models
    - **Predictions**: Classify new observations
    """)
    
    _, raw_data, _ = load_data()
    if raw_data is not None:
        fig = plot_light_curves(raw_data, 3)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

def data_exploration_page(raw_data, summary_data):
    st.title("📊 Data Exploration")
    tab1, tab2 = st.tabs(["Light Curves", "Summary Data"])
    
    with tab1:
        st.header("Light Curve Data")
        if raw_data is not None:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Stars", len(raw_data))
            if 'LABEL' in raw_data.columns:
                col2.metric("Exoplanets", len(raw_data[raw_data['LABEL'] == 2]))
                col3.metric("Non-Exoplanets", len(raw_data[raw_data['LABEL'] == 1]))
            
            n_samples = st.slider("Samples to show", 1, 10, 3)
            fig = plot_light_curves(raw_data, n_samples)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

def predict_with_cnn(data):
    try:
        X = data.drop(columns=['LABEL'], errors='ignore').values
        X_scaled = st.session_state.scaler_cnn.transform(X)
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        
        predictions = st.session_state.cnn_model.predict(X_reshaped).flatten()
        
        results = pd.DataFrame({
            'Star ID': data.index,
            'Prediction': ['Exoplanet' if p > 0.5 else 'Non-Exoplanet' for p in predictions],
            'Confidence': [f"{p*100:.2f}%" if p > 0.5 else f"{(1-p)*100:.2f}%" for p in predictions]
        })
        return results
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def predictions_page():
    st.title("🔮 Predictions")
    model_choice = st.selectbox("Select Model", ["CNN", "XGBoost"])
    
    if st.button(f"🔄 Load {model_choice} Model"):
        if model_choice == "CNN":
            load_cnn_model()
        else:
            load_xgb_model()
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(data)} records")
        
        if model_choice == "CNN" and st.session_state.cnn_model:
            results = predict_with_cnn(data)
            if results is not None:
                st.dataframe(results)
                fig = px.pie(results, names='Prediction', 
                           color_discrete_map={'Exoplanet': COLOR_PALETTE['nebula_purple'], 
                                             'Non-Exoplanet': COLOR_PALETTE['starry_white']})
                st.plotly_chart(fig, use_container_width=True)

def main():
    initialize_session_state()
    os.makedirs("datasets", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    _, raw_data, summary_data = load_data()
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", 
                          ["Home", "Data Exploration", "Predictions"])
    
    if page == "Home":
        landing_page()
    elif page == "Data Exploration":
        data_exploration_page(raw_data, summary_data)
    elif page == "Predictions":
        predictions_page()

if __name__ == "__main__":
    main()
