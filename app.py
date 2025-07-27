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
        st.session_state.raw_data = raw_data
        st.session_state.data_loaded = True
        st.success("exoTrain.csv loaded successfully!")

        # Attempt to load cumulative.csv (optional)
        summary_data = None
        # This path assumes the 'datasets' folder is in the same directory as the script.
        # Update path if your directory structure is different.
        summary_path = "cumulative.csv" 
        if os.path.exists(summary_path):
            st.write("Loading cumulative.csv from repository...")
            summary_data = pd.read_csv(summary_path)
            st.session_state.summary_data = summary_data
            st.success("cumulative.csv loaded successfully!")
        else:
            st.warning("cumulative.csv not found in the root directory. Summary data features will be disabled.")

        return raw_data, summary_data
    except Exception as e:
        st.error(f"Failed to load datasets: {e}")
        st.session_state.data_loaded = False
        return None, None

# Load CNN model
def load_cnn_model():
    """Safely load CNN model with memory cleanup"""
    try:
        tf.keras.backend.clear_session()
        # Placeholder: Replace with actual Google Drive URL for the model
        model_url = "YOUR_CNN_MODEL_GOOGLE_DRIVE_URL"
        model_path = "cnn_exoplanet_model.keras"
        # The following block is a placeholder for downloading the model from a URL
        # if model_url != "YOUR_CNN_MODEL_GOOGLE_DRIVE_URL":
        #     response = requests.get(model_url)
        #     with open(model_path, 'wb') as f:
        #         f.write(response.content)

        st.session_state.cnn_model = load_model(
            model_path,
            custom_objects={'focal_loss_fixed': focal_loss(gamma=2.0, alpha=0.5)}
        )
        
        # Placeholder: Replace with actual Google Drive URL for the scaler
        scaler_url = "YOUR_SCALER_CNN_GOOGLE_DRIVE_URL"
        scaler_path = "scaler_cnn.pkl"
        # if scaler_url != "YOUR_SCALER_CNN_GOOGLE_DRIVE_URL":
        #     response = requests.get(scaler_url)
        #     with open(scaler_path, 'wb') as f:
        #         f.write(response.content)
        
        st.session_state.scaler_cnn = joblib.load(scaler_path)
        st.success("CNN model and scaler loaded successfully!")
        return True
    except Exception as e:
        st.error(f"Error loading CNN model: {e}. Ensure 'cnn_exoplanet_model.keras' and 'scaler_cnn.pkl' are in the root directory.")
        return False

# Load XGBoost model
def load_xgb_model():
    """Safely load XGBoost model with components"""
    try:
        # Placeholder: Replace with actual Google Drive URL for the model
        model_url = "YOUR_XGB_MODEL_GOOGLE_DRIVE_URL"
        model_path = "xgb_model.pkl"
        st.session_state.xgb_model = joblib.load(model_path)
        
        # Placeholder: Replace with actual Google Drive URL for the scaler
        scaler_url = "YOUR_SCALER_XGB_GOOGLE_DRIVE_URL"
        scaler_path = "scaler_xgb.pkl"
        st.session_state.scaler_xgb = joblib.load(scaler_path)

        # Placeholder: Replace with actual Google Drive URL for the encoder
        encoder_url = "YOUR_LABEL_ENCODER_GOOGLE_DRIVE_URL"
        encoder_path = "label_encoder.pkl"
        st.session_state.label_encoder = joblib.load(encoder_path)

        # Placeholder: Replace with actual Google Drive URL for features
        features_url = "YOUR_FEATURES_TO_USE_GOOGLE_DRIVE_URL"
        features_path = "features_to_use.pkl"
        st.session_state.features_to_use = joblib.load(features_path)
        
        st.session_state.num_classes = len(st.session_state.label_encoder.classes_)
        st.success("XGBoost model and components loaded successfully!")
        return True
    except Exception as e:
        st.error(f"Error loading XGBoost model: {e}. Ensure model, scaler, encoder, and features files are in the root directory.")
        return False

# Light curve visualization
def plot_light_curves(data, n_samples=5, label_col='LABEL'):
    """Visualize sample light curves with space-themed palette"""
    try:
        if label_col not in data.columns:
            st.warning(f"Column '{label_col}' not found. Skipping exoplanet/non-exoplanet differentiation.")
            samples = data.sample(n=min(n_samples, len(data)))
            fig_plotly = make_subplots(rows=1, cols=1, subplot_titles=("Sample Light Curves",))
            
            for idx, row in samples.iterrows():
                fig_plotly.add_trace(
                    go.Scatter(
                        x=list(range(len(row))), y=row, mode='lines', name=f'Star {idx}',
                        line=dict(color=COLOR_PALETTE['nebula_purple'])
                    ), row=1, col=1
                )
        else:
            exoplanet_data = data[data[label_col] == 2]
            non_exoplanet_data = data[data[label_col] == 1]

            exoplanet_samples = exoplanet_data.sample(n=min(n_samples, len(exoplanet_data)))
            non_exoplanet_samples = non_exoplanet_data.sample(n=min(n_samples, len(non_exoplanet_data)))
            
            fig_plotly = make_subplots(rows=1, cols=2, subplot_titles=("Exoplanet Stars", "Non-Exoplanet Stars"))
            
            for idx, row in exoplanet_samples.iterrows():
                fig_plotly.add_trace(
                    go.Scatter(
                        x=list(range(len(row[1:]))), y=row[1:], mode='lines', name=f'Exoplanet Star {idx}',
                        line=dict(color=COLOR_PALETTE['nebula_purple'])
                    ), row=1, col=1
                )
            
            for idx, row in non_exoplanet_samples.iterrows():
                fig_plotly.add_trace(
                    go.Scatter(
                        x=list(range(len(row[1:]))), y=row[1:], mode='lines', name=f'Non-Exoplanet Star {idx}',
                        line=dict(color=COLOR_PALETTE['starry_white'])
                    ), row=1, col=2
                )
        
        fig_plotly.update_layout(
            height=400, showlegend=False, margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor=COLOR_PALETTE['dark_slate'], plot_bgcolor=COLOR_PALETTE['dark_slate'],
            font=dict(color=COLOR_PALETTE['starry_white'])
        )
        
        fig_plotly.update_xaxes(title_text="Time", title_font=dict(color=COLOR_PALETTE['starry_white']))
        fig_plotly.update_yaxes(title_text="Flux", title_font=dict(color=COLOR_PALETTE['starry_white']))
        
        return fig_plotly
    except Exception as e:
        st.error(f"Error generating light curves: {e}")
        return None

# Landing Page
def landing_page(raw_data):
    """Render the landing page"""
    st.title("🌌 Exoplanet Detection Dashboard")
    st.markdown("""
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
    """)
    
    if raw_data is not None:
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
                st.metric("Total Stars", f"{len(raw_data):,}")
            with col2:
                metric_val = f"{len(raw_data[raw_data['LABEL'] == 2]):,}" if 'LABEL' in raw_data.columns else "N/A"
                st.metric("Exoplanet Stars", metric_val)
            with col3:
                metric_val = f"{len(raw_data[raw_data['LABEL'] == 1]):,}" if 'LABEL' in raw_data.columns else "N/A"
                st.metric("Non-Exoplanet Stars", metric_val)
            
            st.markdown("---")
            st.subheader("Sample Light Curves")
            n_samples = st.slider("Number of samples to display", 1, 10, 3)
            fig_light_curves = plot_light_curves(raw_data, n_samples)
            if fig_light_curves:
                st.plotly_chart(fig_light_curves, use_container_width=True)
        else:
            st.error("Light curve data not available.")
    
    with tab2:
        st.header("Summary Data Overview")
        if summary_data is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Observations", f"{len(summary_data):,}")
            with col2:
                if 'koi_disposition' in summary_data.columns:
                    dispositions = summary_data['koi_disposition'].value_counts()
                    st.metric("Confirmed Exoplanets", f"{dispositions.get('CONFIRMED', 0):,}")
                else:
                    st.metric("Confirmed Exoplanets", "N/A")

            st.markdown("---")
            st.subheader("Exoplanet Dispositions")
            if 'koi_disposition' in summary_data.columns:
                fig_disposition = px.pie(
                    summary_data, names='koi_disposition', hole=0.3,
                    color_discrete_sequence=[COLOR_PALETTE['nebula_purple'], COLOR_PALETTE['starry_white'], '#868e96']
                )
                fig_disposition.update_layout(
                    paper_bgcolor=COLOR_PALETTE['dark_slate'], plot_bgcolor=COLOR_PALETTE['dark_slate'],
                    font=dict(color=COLOR_PALETTE['starry_white'])
                )
                st.plotly_chart(fig_disposition, use_container_width=True)
        else:
            st.warning("Summary data (cumulative.csv) not available.")

# Predict with CNN
def predict_with_cnn(data):
    """Make predictions with CNN model in batches"""
    try:
        if st.session_state.scaler_cnn is None:
            raise ValueError("CNN scaler not loaded.")
        
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
        st.error(f"CNN prediction error: {e}")
        return None

# Predict with XGBoost
def predict_with_xgb(data):
    """Make predictions with XGBoost model"""
    try:
        if any(comp is None for comp in [st.session_state.scaler_xgb, st.session_state.label_encoder, st.session_state.features_to_use]):
            raise ValueError("XGBoost components (scaler, encoder, features) not loaded.")
        
        missing_features = set(st.session_state.features_to_use) - set(data.columns)
        if missing_features:
            raise ValueError(f"Missing required features for XGBoost: {missing_features}")
            
        X = data[st.session_state.features_to_use].copy()
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(X.mean(numeric_only=True), inplace=True)
        X_scaled = st.session_state.scaler_xgb.transform(X)
        
        predictions_encoded = st.session_state.xgb_model.predict(X_scaled)
        prediction_probs = st.session_state.xgb_model.predict_proba(X_scaled)
        predictions = st.session_state.label_encoder.inverse_transform(predictions_encoded)
        
        results = pd.DataFrame({
            'Star ID': data.index,
            'Prediction': predictions,
            'Confidence': [f"{max(probs)*100:.2f}%" for probs in prediction_probs]
        })
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
    
    # Button to load/verify model existence
    button_label = f"🔄 Verify {model_choice} Model"
    if st.button(button_label):
        if model_choice == "CNN":
            load_cnn_model()
        else:
            load_xgb_model()

    uploaded_file = st.file_uploader("Upload CSV file for prediction", type=["csv"])
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.success(f"Successfully uploaded {uploaded_file.name} with {len(data)} records")
            
            with st.spinner("Making predictions..."):
                results = None
                if model_choice == "CNN":
                    if st.session_state.cnn_model:
                        results = predict_with_cnn(data)
                    else:
                        st.error("Please verify the CNN model before making predictions.")
                else: # XGBoost
                    if st.session_state.xgb_model:
                        results = predict_with_xgb(data)
                    else:
                        st.error("Please verify the XGBoost model before making predictions.")
                
                if results is not None:
                    st.subheader("Prediction Results")
                    st.dataframe(results)
                    
                    st.subheader("Prediction Distribution")
                    fig_pie = px.pie(
                        results, names='Prediction', title='Prediction Distribution',
                        color='Prediction',
                        color_discrete_map={
                            'Exoplanet': COLOR_PALETTE['nebula_purple'], 
                            'Non-Exoplanet': COLOR_PALETTE['starry_white'],
                            'CONFIRMED': COLOR_PALETTE['nebula_purple'],
                            'CANDIDATE': COLOR_PALETTE['starry_white'],
                            'FALSE POSITIVE': '#868e96'
                        }
                    )
                    fig_pie.update_layout(
                        paper_bgcolor=COLOR_PALETTE['dark_slate'],
                        plot_bgcolor=COLOR_PALETTE['dark_slate'],
                        font=dict(color=COLOR_PALETTE['starry_white'])
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")

# Feedback Page
def feedback_page():
    """Render the feedback page"""
    st.title("💬 Feedback & Suggestions")
    st.markdown("---")
    st.write("We value your input! Please use the form below to share your thoughts.")
    
    with st.form("feedback_form", clear_on_submit=True):
        feedback_text = st.text_area("Your Feedback", placeholder="What do you like? What could be improved?")
        submitted = st.form_submit_button("Submit Feedback")
        if submitted:
            st.success("Thank you for your feedback! We appreciate you helping us improve.")
            st.balloons()

# Main application logic
def main():
    """Main function to run the Streamlit app"""
    initialize_session_state()
    setup_page_config()
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Data Exploration", "Predictions", "Feedback"],
        key="navigation_radio"
    )
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This dashboard is for educational and demonstration purposes. "
        "Model training is disabled in this live environment."
    )
    
    # Load data only once
    if not st.session_state.data_loaded:
        load_data()
    
    raw_data = st.session_state.get('raw_data')
    summary_data = st.session_state.get('summary_data')
    
    # Page routing
    if page == "Home":
        landing_page(raw_data)
    elif page == "Data Exploration":
        data_exploration_page(raw_data, summary_data)
    elif page == "Predictions":
        predictions_page()
    elif page == "Feedback":
        feedback_page()

if __name__ == "__main__":
    main()
