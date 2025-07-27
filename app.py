import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib as mpl
import os
import requests
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import xgboost as xgb

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
    """Initialize session state variables"""
    session_vars = {
        'data_loaded': False,
        'raw_data': None,
        'summary_data': None,
        'cnn_model': None,
        'xgb_model': None,
        'scaler_cnn': None,
        'scaler_xgb': None,
        'label_encoder': None,
        'features_to_use': None,
        'num_classes': None,
        'prediction_results': None
    }
    for var, default in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default

# Custom CSS for space-themed styling
def setup_page_config():
    """Configure page settings and styles with space-themed palette"""
    st.set_page_config(
        page_title="Exoplanet Detection Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="🌌"
    )
    
    st.markdown(f"""
        <style>
        .main {{
            background-color: {COLOR_PALETTE['dark_slate']} !important;
            color: {COLOR_PALETTE['starry_white']} !important;
        }}
        .stSidebar {{
            background-color: {COLOR_PALETTE['dark_slate']} !important;
            color: {COLOR_PALETTE['starry_white']} !important;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 10px;
            background-color: {COLOR_PALETTE['dark_slate']} !important;
        }}
        .stTabs [data-baseweb="tab"] {{
            height: 50px;
            padding: 0 25px;
            background-color: {COLOR_PALETTE['nebula_purple']} !important;
            color: {COLOR_PALETTE['starry_white']} !important;
            border-radius: 8px 8px 0px 0px;
            font-weight: bold;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {COLOR_PALETTE['starry_white']} !important;
            color: {COLOR_PALETTE['dark_slate']} !important;
        }}
        .stTabs [data-baseweb="tab-highlight"] {{
            background-color: {COLOR_PALETTE['nebula_purple']} !important;
        }}
        .stButton>button {{
            border-radius: 8px;
            height: 40px;
            width: 100%;
            background-color: {COLOR_PALETTE['nebula_purple']} !important;
            color: {COLOR_PALETTE['starry_white']} !important;
            font-weight: bold;
            border: none;
        }}
        .stButton>button:hover {{
            background-color: {COLOR_PALETTE['starry_white']} !important;
            color: {COLOR_PALETTE['dark_slate']} !important;
        }}
        .stDataFrame {{
            border-radius: 8px;
            background-color: {COLOR_PALETTE['dark_slate']} !important;
            color: {COLOR_PALETTE['starry_white']} !important;
        }}
        .plot-container {{
            border-radius: 8px;
            background-color: {COLOR_PALETTE['dark_slate']} !important;
            color: {COLOR_PALETTE['starry_white']} !important;
            padding: 20px;
        }}
        .metric-card {{
            background-color: {COLOR_PALETTE['dark_slate']} !important;
            color: {COLOR_PALETTE['starry_white']} !important;
            border-radius: 8px;
            padding: 15px;
        }}
        h1, h2, h3, h4, h5, h6, p, div, span {{
            color: {COLOR_PALETTE['starry_white']} !important;
        }}
        </style>
    """, unsafe_allow_html=True)

# Focal Loss Function for CNN
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

# Load datasets
@st.cache_data(show_spinner="Loading datasets...")
def load_data():
    """Load and cache datasets with error handling"""
    try:
        # Load exoTrain.csv from Google Drive
        primary_url = "https://drive.google.com/uc?export=download&id=1MxDUIqQ6S9cmi068I62xkbCwZHkpAeJI"
        st.write("Checking accessibility of exoTrain.csv...")
        response = requests.head(primary_url, allow_redirects=True)
        if response.status_code != 200:
            raise Exception(f"Failed to access Google Drive URL (status code: {response.status_code}). Ensure the file is set to 'Anyone with the link'.")

        st.write("Loading exoTrain.csv from Google Drive...")
        raw_data = pd.read_csv(primary_url)
        
        # Debug: Display column names
        st.write(f"exoTrain.csv columns: {list(raw_data.columns)}")
        
        st.session_state.raw_data = raw_data
        st.session_state.data_loaded = True
        st.success("exoTrain.csv loaded successfully!")

        # Attempt to load cumulative.csv (optional)
        summary_data = None
        if os.path.exists("datasets/cumulative.csv"):
            st.write("Loading cumulative.csv from repository...")
            summary_data = pd.read_csv("datasets/cumulative.csv")
            st.write(f"cumulative.csv columns: {list(summary_data.columns)}")
            st.session_state.summary_data = summary_data
            st.success("cumulative.csv loaded successfully!")
        else:
            st.warning("cumulative.csv not found in repository. Summary data features disabled.")

        return raw_data, summary_data
    except Exception as e:
        st.error(f"Failed to load datasets: {e}")
        st.write("Please ensure the Google Drive URL for exoTrain.csv is publicly accessible ('Anyone with the link').")
        st.session_state.data_loaded = False
        return None, None

# Load CNN model
def load_cnn_model():
    """Safely load CNN model with memory cleanup"""
    try:
        tf.keras.backend.clear_session()
        if 'cnn_model' in st.session_state:
            del st.session_state.cnn_model
        # Placeholder: Replace with actual Google Drive URL for the model
        model_url = "YOUR_CNN_MODEL_GOOGLE_DRIVE_URL"
        model_path = "cnn_exoplanet_model.keras"
        if model_url:
            response = requests.get(model_url)
            with open(model_path, 'wb') as f:
                f.write(response.content)
        st.session_state.cnn_model = load_model(
            model_path,
            custom_objects={'focal_loss_fixed': focal_loss(gamma=2.0, alpha=0.5)}
        )
        # Placeholder: Replace with actual Google Drive URL for the scaler
        scaler_url = "YOUR_SCALER_CNN_GOOGLE_DRIVE_URL"
        scaler_path = "scaler_cnn.pkl"
        if scaler_url:
            response = requests.get(scaler_url)
            with open(scaler_path, 'wb') as f:
                f.write(response.content)
        st.session_state.scaler_cnn = joblib.load(scaler_path)
        st.success("CNN model and scaler loaded successfully!")
        return True
    except Exception as e:
        st.error(f"Error loading CNN model: {e}")
        return False

# Load XGBoost model
def load_xgb_model():
    """Safely load XGBoost model with components"""
    try:
        if 'xgb_model' in st.session_state:
            del st.session_state.xgb_model
        # Placeholder: Replace with actual Google Drive URL for the model
        model_url = "YOUR_XGB_MODEL_GOOGLE_DRIVE_URL"
        model_path = "xgb_model.pkl"
        if model_url:
            response = requests.get(model_url)
            with open(model_path, 'wb') as f:
                f.write(response.content)
        st.session_state.xgb_model = joblib.load(model_path)
        # Placeholder: Replace with actual Google Drive URL for the scaler
        scaler_url = "YOUR_SCALER_XGB_GOOGLE_DRIVE_URL"
        scaler_path = "scaler_xgb.pkl"
        if scaler_url:
            response = requests.get(scaler_url)
            with open(scaler_path, 'wb') as f:
                f.write(response.content)
        st.session_state.scaler_xgb = joblib.load(scaler_path)
        # Placeholder: Replace with actual Google Drive URL for the encoder
        encoder_url = "YOUR_LABEL_ENCODER_GOOGLE_DRIVE_URL"
        encoder_path = "label_encoder.pkl"
        if encoder_url:
            response = requests.get(encoder_url)
            with open(encoder_path, 'wb') as f:
                f.write(response.content)
        st.session_state.label_encoder = joblib.load(encoder_path)
        # Placeholder: Replace with actual Google Drive URL for features
        features_url = "YOUR_FEATURES_TO_USE_GOOGLE_DRIVE_URL"
        features_path = "features_to_use.pkl"
        if features_url:
            response = requests.get(features_url)
            with open(features_path, 'wb') as f:
                f.write(response.content)
        st.session_state.features_to_use = joblib.load(features_path)
        st.session_state.num_classes = len(st.session_state.label_encoder.classes_)
        st.success("XGBoost model and components loaded successfully!")
        return True
    except Exception as e:
        st.error(f"Error loading XGBoost model: {e}")
        return False

# Light curve visualization
def plot_light_curves(data, n_samples=5, label_col='LABEL'):
    """Visualize sample light curves with space-themed palette"""
    try:
        if label_col not in data.columns:
            st.warning(f"Column '{label_col}' not found. Skipping exoplanet/non-exoplanet differentiation.")
            # Plot all samples without differentiation
            samples = data.sample(n=min(n_samples, len(data)))
            fig_plotly = make_subplots(rows=1, cols=1, subplot_titles=("Sample Light Curves",))
            
            for idx, row in samples.iterrows():
                fig_plotly.add_trace(
                    go.Scatter(
                        x=list(range(len(row))),
                        y=row,
                        mode='lines',
                        name=f'Star {idx}',
                        line=dict(color=COLOR_PALETTE['nebula_purple'])
                    ),
                    row=1, col=1
                )
        else:
            exoplanet_samples = data[data[label_col] == 2].sample(n=min(n_samples, len(data[data[label_col] == 2])))
            non_exoplanet_samples = data[data[label_col] == 1].sample(n=min(n_samples, len(data[data[label_col] == 1])))
            
            fig_plotly = make_subplots(rows=1, cols=2, subplot_titles=("Exoplanet Stars", "Non-Exoplanet Stars"))
            
            for idx, row in exoplanet_samples.iterrows():
                fig_plotly.add_trace(
                    go.Scatter(
                        x=list(range(len(row[1:]))),
                        y=row[1:],
                        mode='lines',
                        name=f'Exoplanet Star {idx}',
                        line=dict(color=COLOR_PALETTE['nebula_purple'])
                    ),
                    row=1, col=1
                )
            
            for idx, row in non_exoplanet_samples.iterrows():
                fig_plotly.add_trace(
                    go.Scatter(
                        x=list(range(len(row[1:]))),
                        y=row[1:],
                        mode='lines',
                        name=f'Non-Exoplanet Star {idx}',
                        line=dict(color=COLOR_PALETTE['starry_white'])
                    ),
                    row=1, col=2
                )
        
        fig_plotly.update_layout(
            height=400,
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor=COLOR_PALETTE['dark_slate'],
            plot_bgcolor=COLOR_PALETTE['dark_slate'],
            font=dict(color=COLOR_PALETTE['starry_white'])
        )
        
        if label_col in data.columns:
            fig_plotly.update_xaxes(title_text="Time", title_font=dict(color=COLOR_PALETTE['starry_white']), row=1, col=1)
            fig_plotly.update_xaxes(title_text="Time", title_font=dict(color=COLOR_PALETTE['starry_white']), row=1, col=2)
            fig_plotly.update_yaxes(title_text="Flux", title_font=dict(color=COLOR_PALETTE['starry_white']), row=1, col=2)
        else:
            fig_plotly.update_xaxes(title_text="Time", title_font=dict(color=COLOR_PALETTE['starry_white']), row=1, col=1)
            fig_plotly.update_yaxes(title_text="Flux", title_font=dict(color=COLOR_PALETTE['starry_white']), row=1, col=1)
        
        return fig_plotly
    except Exception as e:
        st.error(f"Error generating light curves: {e}")
        return None

# Landing Page
def landing_page(raw_data):
    """Render the landing page"""
    st.title("🌌 Exoplanet Detection Dashboard")
    st.markdown(f"""
    <div style="color: {COLOR_PALETTE['starry_white']}">
    ## Discover Exoplanets Through Data
    
    This interactive dashboard allows you to:
    - Explore exoplanet light curve data
    - Visualize light curves and dataset statistics
    - Make predictions with pre-trained models (if available)
    
    ### Key Features:
    🔭 **Data Exploration**: View light curves and exoplanet characteristics  
    🔮 **Predictions**: Classify new observations (requires pre-trained models)  
    💬 **Feedback**: Share your suggestions for improvement
    
    *Navigate using the sidebar to get started!*
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Quick Start Guide"):
        with st.expander("Getting Started"):
            st.markdown(f"""
            <div style="color: {COLOR_PALETTE['starry_white']}">
            1. **Data Exploration**: View light curves and dataset statistics
            2. **Predictions**: Upload data for classification (requires pre-trained models)
            3. **Note**: Model training is disabled due to resource constraints
            
            Pro Tip: Start with the Data Exploration tab to understand the dataset!
            </div>
            """, unsafe_allow_html=True)
    
    if st.session_state.data_loaded and raw_data is not None:
        fig_plotly = plot_light_curves(raw_data, 2)
        if fig_plotly:
            st.plotly_chart(fig_plotly, use_container_width=True)
        else:
            st.warning("Failed to generate sample light curves.")
    else:
        st.error("Dataset not loaded. Please check the data source.")

# Data Exploration Page
def data_exploration_page(raw_data, summary_data):
    """Render the data exploration page"""
    st.title("📊 Data Exploration")
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["Light Curve Data", "Summary Data"])
    
    with tab1:
        st.header("Light Curve Data Overview")
        
        if raw_data is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Stars", len(raw_data))
            with col2:
                if 'LABEL' in raw_data.columns:
                    st.metric("Exoplanet Stars", len(raw_data[raw_data['LABEL'] == 2]))
                else:
                    st.metric("Exoplanet Stars", "N/A (LABEL column missing)")
            with col3:
                if 'LABEL' in raw_data.columns:
                    st.metric("Non-Exoplanet Stars", len(raw_data[raw_data['LABEL'] == 1]))
                else:
                    st.metric("Non-Exoplanet Stars", "N/A (LABEL column missing)")
            
            st.markdown("---")
            
            st.subheader("Sample Light Curves")
            n_samples = st.slider("Number of samples to display", 1, 10, 3)
            fig_plotly = plot_light_curves(raw_data, n_samples)
            if fig_plotly:
                st.plotly_chart(fig_plotly, use_container_width=True)
            
            st.markdown("---")
            
            st.subheader("Flux Distribution Comparison")
            if 'LABEL' in raw_data.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    try:
                        fig_plotly = px.histogram(
                            raw_data[raw_data['LABEL'] == 2].iloc[:, 1:].mean(axis=1),
                            nbins=50,
                            title="Exoplanet Stars Flux Distribution",
                            color_discrete_sequence=[COLOR_PALETTE['nebula_purple']],
                            labels={'value': 'Average Flux'}
                        )
                        fig_plotly.update_layout(
                            paper_bgcolor=COLOR_PALETTE['dark_slate'],
                            plot_bgcolor=COLOR_PALETTE['dark_slate'],
                            font=dict(color=COLOR_PALETTE['starry_white'])
                        )
                        st.plotly_chart(fig_plotly, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Failed to plot exoplanet flux distribution: {e}")
                
                with col2:
                    try:
                        fig_plotly = px.histogram(
                            raw_data[raw_data['LABEL'] == 1].iloc[:, 1:].mean(axis=1),
                            nbins=50,
                            title="Non-Exoplanet Stars Flux Distribution",
                            color_discrete_sequence=[COLOR_PALETTE['starry_white']],
                            labels={'value': 'Average Flux'}
                        )
                        fig_plotly.update_layout(
                            paper_bgcolor=COLOR_PALETTE['dark_slate'],
                            plot_bgcolor=COLOR_PALETTE['dark_slate'],
                            font=dict(color=COLOR_PALETTE['starry_white'])
                        )
                        st.plotly_chart(fig_plotly, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Failed to plot non-exoplanet flux distribution: {e}")
            else:
                st.warning("Flux distribution plots skipped due to missing 'LABEL' column.")
        else:
            st.error("Light curve data not available.")
    
    with tab2:
        st.header("Summary Data Overview")
        
        if summary_data is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Observations", len(summary_data))
            with col2:
                if 'koi_disposition' in summary_data.columns:
                    dispositions = summary_data['koi_disposition'].value_counts()
                    st.metric("Confirmed Exoplanets", dispositions.get('CONFIRMED', 0))
                else:
                    st.metric("Confirmed Exoplanets", "N/A (koi_disposition column missing)")
            
            st.markdown("---")
            
            st.subheader("Exoplanet Dispositions")
            if 'koi_disposition' in summary_data.columns:
                try:
                    fig_plotly = px.pie(
                        summary_data,
                        names='koi_disposition',
                        hole=0.3,
                        color_discrete_sequence=[COLOR_PALETTE['nebula_purple'], COLOR_PALETTE['starry_white'], COLOR_PALETTE['dark_slate']]
                    )
                    fig_plotly.update_layout(
                        paper_bgcolor=COLOR_PALETTE['dark_slate'],
                        plot_bgcolor=COLOR_PALETTE['dark_slate'],
                        font=dict(color=COLOR_PALETTE['starry_white'])
                    )
                    st.plotly_chart(fig_plotly, use_container_width=True)
                except Exception as e:
                    st.warning(f"Failed to plot dispositions: {e}")
            else:
                st.warning("Exoplanet dispositions plot skipped due to missing 'koi_disposition' column.")
            
            st.markdown("---")
            
            st.subheader("Feature Distributions")
            numeric_cols = summary_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                feature = st.selectbox(
                    "Select feature to visualize",
                    numeric_cols,
                    index=0
                )
                
                try:
                    if 'koi_disposition' in summary_data.columns:
                        fig_plotly = px.histogram(
                            summary_data,
                            x=feature,
                            color='koi_disposition',
                            marginal="box",
                            nbins=50,
                            barmode='overlay',
                            color_discrete_sequence=[COLOR_PALETTE['nebula_purple'], COLOR_PALETTE['starry_white'], COLOR_PALETTE['dark_slate']]
                        )
                    else:
                        fig_plotly = px.histogram(
                            summary_data,
                            x=feature,
                            nbins=50,
                            title=f"Distribution of {feature}",
                            color_discrete_sequence=[COLOR_PALETTE['nebula_purple']]
                        )
                    fig_plotly.update_layout(
                        paper_bgcolor=COLOR_PALETTE['dark_slate'],
                        plot_bgcolor=COLOR_PALETTE['dark_slate'],
                        font=dict(color=COLOR_PALETTE['starry_white'])
                    )
                    st.plotly_chart(fig_plotly, use_container_width=True)
                except Exception as e:
                    st.warning(f"Failed to plot feature distribution: {e}")
            else:
                st.warning("No numeric columns available for feature distribution.")
        else:
            st.warning("Summary data not available. Please provide cumulative.csv.")

# Predict with CNN
def predict_with_cnn(data):
    """Make predictions with CNN model in batches"""
    try:
        if st.session_state.scaler_cnn is None:
            raise ValueError("CNN scaler not loaded.")
        
        tf.keras.backend.clear_session()
        
        processed_data = data.copy()
        flux_cols = [col for col in data.columns if col != 'LABEL']
        
        X = processed_data.drop(columns=['LABEL'] if 'LABEL' in processed_data.columns else [], errors='ignore').values
        X_scaled = st.session_state.scaler_cnn.transform(X)
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        
        batch_size = min(1000, len(X_reshaped))
        predictions = []
        for i in range(0, len(X_reshaped), batch_size):
            batch = X_reshaped[i:i + batch_size]
            preds = st.session_state.cnn_model.predict(batch, verbose=0)
            predictions.extend(preds.flatten())
        
        predictions = (np.array(predictions) > 0.5).astype(int)
        
        results = pd.DataFrame({
            'Star ID': processed_data.index,
            'Prediction': ['Exoplanet' if p == 1 else 'Non-Exoplanet' for p in predictions],
            'Confidence': [f"{max(pred, 1-pred)*100:.2f}%" for pred in predictions]
        })
        
        st.session_state.prediction_results = results
        return results
    except Exception as e:
        st.error(f"CNN prediction error: {e}")
        return None

# Predict with XGBoost
def predict_with_xgb(data):
    """Make predictions with XGBoost model"""
    try:
        if st.session_state.scaler_xgb is None or st.session_state.label_encoder is None:
            raise ValueError("XGBoost scaler or label encoder not loaded.")
        
        missing_features = set(st.session_state.features_to_use) - set(data.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
            
        processed_data = data.copy()
        X = processed_data[st.session_state.features_to_use]
        X = X.dropna(axis=1, how='all').replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean(numeric_only=True))
        X_scaled = st.session_state.scaler_xgb.transform(X)
        
        predictions = st.session_state.xgb_model.predict(X_scaled)
        prediction_probs = st.session_state.xgb_model.predict_proba(X_scaled)
        predictions = st.session_state.label_encoder.inverse_transform(predictions)
        
        results = pd.DataFrame({
            'Star ID': processed_data.index,
            'Prediction': predictions,
            'Confidence': [f"{max(probs)*100:.2f}%" for probs in prediction_probs]
        })
        
        st.session_state.prediction_results = results
        return results
    except Exception as e:
        st.error(f"XGBoost prediction error: {e}")
        return None

# Predictions Page
def predictions_page():
    """Render the predictions page"""
    st.title("🔮 Predictions")
    st.markdown("---")
    
    model_choice = st.selectbox("Select Model for Prediction", ["CNN", "XGBoost"])
    
    col1, col2 = st.columns(2)
    with col1:
        if model_choice == "CNN":
            if st.button("🔄 Verify CNN Model"):
                if load_cnn_model():
                    st.success("CNN model ready for predictions!")
        else:
            if st.button("🔄 Verify XGBoost Model"):
                if load_xgb_model():
                    st.success("XGBoost model ready for predictions!")
    
    uploaded_file = st.file_uploader("Upload CSV file for prediction", type=["csv"])
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.success(f"Successfully uploaded {uploaded_file.name} with {len(data)} records")
            
            if model_choice == "CNN":
                if st.session_state.cnn_model is None or st.session_state.scaler_cnn is None:
                    st.error("Please load the CNN model first")
                else:
                    with st.spinner("Making predictions..."):
                        results = predict_with_cnn(data)
                        if results is not None:
                            st.subheader("Prediction Results")
                            st.dataframe(results.style.apply(
                                lambda x: ['background-color: ' + COLOR_PALETTE['nebula_purple'] if val == 'Exoplanet' else 'background-color: ' + COLOR_PALETTE['dark_slate'] for val in x],
                                subset=['Prediction']
                            ))
                            
                            fig_plotly = px.pie(
                                results,
                                names='Prediction',
                                title='Prediction Distribution',
                                color='Prediction',
                                color_discrete_map={'Exoplanet': COLOR_PALETTE['nebula_purple'], 'Non-Exoplanet': COLOR_PALETTE['starry_white']}
                            )
                            fig_plotly.update_layout(
                                paper_bgcolor=COLOR_PALETTE['dark_slate'],
                                plot_bgcolor=COLOR_PALETTE['dark_slate'],
                                font=dict(color=COLOR_PALETTE['starry_white'])
                            )
                            st.plotly_chart(fig_plotly, use_container_width=True)
            
            else:  # XGBoost
                if (st.session_state.xgb_model is None or 
                    st.session_state.scaler_xgb is None or 
                    st.session_state.label_encoder is None):
                    st.error("Please load the XGBoost model first")
                else:
                    with st.spinner("Making predictions..."):
                        results = predict_with_xgb(data)
                        if results is not None:
                            st.subheader("Prediction Results")
                            st.dataframe(results.style.apply(
                                lambda x: ['background-color: ' + COLOR_PALETTE['nebula_purple'] if val == 'CONFIRMED' else 
                                         ('background-color: ' + COLOR_PALETTE['starry_white'] if val == 'CANDIDATE' else 'background-color: ' + COLOR_PALETTE['dark_slate']) for val in x],
                                subset=['Prediction']
                            ))
                            
                            fig_plotly = px.pie(
                                results,
                                names='Prediction',
                                title='Prediction Distribution',
                                color='Prediction',
                                color_discrete_map={
                                    'CONFIRMED': COLOR_PALETTE['nebula_purple'],
                                    'CANDIDATE': COLOR_PALETTE['starry_white'],
                                    'FALSE POSITIVE': COLOR_PALETTE['dark_slate']
                                }
                            )
                            fig_plotly.update_layout(
                                paper_bgcolor=COLOR_PALETTE['dark_slate'],
                                plot_bgcolor=COLOR_PALETTE['dark_slate'],
                                font=dict(color=COLOR_PALETTE['starry_white'])
                            )
                            st.plotly_chart(fig_plotly, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error processing file: {e}")

# User Feedback System
def user_feedback():
    """Render the user feedback interface"""
    st.header("💬 Provide Feedback")
    
    with st.form("feedback_form"):
        name = st.text_input("Your Name (Optional)")
        rating = st.slider("Rate this dashboard", 1, 5, 3)
        comments = st.text_area("Suggestions or Comments")
        
        submitted = st.form_submit_button("Submit Feedback")
        if submitted:
            st.success("Thank you for your feedback! 🎉")
            st.balloons()

# Main App
def main():
    """Main application function"""
    setup_page_config()
    initialize_session_state()
    
    raw_data, summary_data = load_data()
    
    st.sidebar.title("🌌 Exoplanet Detection")
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigation", 
                          ["Home", "Data Exploration", "Predictions", "Feedback"],
                          index=0)
    
    if page == "Home":
        landing_page(raw_data)
    elif page == "Data Exploration":
        data_exploration_page(raw_data, summary_data)
    elif page == "Predictions":
        predictions_page()
    elif page == "Feedback":
        user_feedback()

if __name__ == "__main__":
    main()
