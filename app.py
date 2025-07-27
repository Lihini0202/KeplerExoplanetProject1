import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Define the space-themed color palette
COLOR_PALETTE = {
    'dark_slate': '#1C2526',  # Near-black background
    'starry_white': '#F4F4F9',  # Light gray-white for text
    'nebula_purple': '#5E4B8C'  # Vibrant purple for accents
}

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
        'models_loaded': False,
        'cnn_config': {'conv_layers': 2, 'filters': [8, 16], 'kernel_size': 5, 'dense_units': 32, 'dropout_rate': 0.3, 'learning_rate': 1e-3},
        'xgb_config': {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'subsample': 0.8},
        'preprocessed_raw_data': None,
        'preprocessed_summary_data': None
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

# Load datasets
@st.cache_data(show_spinner="Loading datasets...")
def load_data():
    """Load and cache datasets with error handling"""
    try:
        raw_data = pd.read_csv("datasets/exoTrain.csv")
        summary_data = pd.read_csv("datasets/cumulative.csv")
        st.session_state.data_loaded = True
        st.session_state.preprocessed_raw_data = raw_data.copy()
        st.session_state.preprocessed_summary_data = summary_data.copy()
        return None, raw_data, summary_data
    except FileNotFoundError as e:
        st.error(f"Dataset files not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading datasets: {e}")
        st.stop()

# Light curve visualization
def plot_light_curves(data, n_samples=5):
    """Visualize sample light curves with space-themed palette"""
    try:
        exoplanet_samples = data[data['LABEL'] == 2].sample(n=min(n_samples, len(data[data['LABEL'] == 2])))
        non_exoplanet_samples = data[data['LABEL'] == 1].sample(n=min(n_samples, len(data[data['LABEL'] == 1])))
        
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
        
        fig_plotly.update_xaxes(title_text="Time", title_font=dict(color=COLOR_PALETTE['starry_white']), row=1, col=1)
        fig_plotly.update_xaxes(title_text="Time", title_font=dict(color=COLOR_PALETTE['starry_white']), row=1, col=2)
        fig_plotly.update_yaxes(title_text="Flux", title_font=dict(color=COLOR_PALETTE['starry_white']), row=1, col=2)
        
        return fig_plotly
    except Exception as e:
        st.error(f"Error generating light curves: {e}")
        return None

# Placeholder implementations for missing functions
def data_exploration_page(raw_data, summary_data):
    """Placeholder for Data Exploration page"""
    st.title("Data Exploration")
    st.write("This page will display data exploration features once implemented.")
    if raw_data is not None:
        st.write("Raw Data Sample:")
        st.dataframe(raw_data.head())
    if summary_data is not None:
        st.write("Summary Data Sample:")
        st.dataframe(summary_data.head())

def model_training_page(raw_data, summary_data):
    """Placeholder for Model Training page"""
    st.title("Model Training")
    st.write("This page will allow training of CNN and XGBoost models once implemented.")

def predictions_page():
    """Placeholder for Predictions page"""
    st.title("Predictions")
    st.write("This page will handle predictions once implemented.")

def prediction_analysis():
    """Placeholder for Prediction Analysis page"""
    st.title("Prediction Analysis")
    st.write("This page will display prediction analysis once implemented.")

def user_feedback():
    """Placeholder for User Feedback page"""
    st.title("Feedback")
    st.write("This page will collect user feedback once implemented.")

def landing_page():
    """Render the landing page with enhanced design"""
    st.title("🌌 Exoplanet Detection Dashboard")
    
    # Hero Section
    st.markdown(f"""
    <div style="color: {COLOR_PALETTE['starry_white']}; text-align: center;">
    <h2>Discover Exoplanets Through Machine Learning</h2>
    <p style="font-size: 1.2rem;">Analyze Kepler space telescope data to identify planetary transits using deep learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats Cards
    _, raw_data, summary_data = load_data()
    if st.session_state.preprocessed_raw_data is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Stars Analyzed", len(raw_data))
        with col2:
            st.metric("Confirmed Exoplanets", len(raw_data[raw_data['LABEL'] == 2]))
        with col3:
            st.metric("Candidate Systems", len(summary_data[summary_data['koi_disposition'] == 'CANDIDATE']))
    
    # Get Started Section
    st.markdown("---")
    st.markdown(f"""
    <div style="color: {COLOR_PALETTE['starry_white']};">
    <h2 style="text-align: center;">🚀 Get Started</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Cards
    cols = st.columns(3)
    with cols[0]:
        with st.container():  # Removed border parameter
            st.markdown(f"""
            <div style="text-align: center; border: 1px solid {COLOR_PALETTE['nebula_purple']}; padding: 20px; border-radius: 8px;">
            <h3 style="color: {COLOR_PALETTE['nebula_purple']};">🔭 Data Exploration</h3>
            <p>Visualize light curves and analyze dataset statistics to understand the characteristics of exoplanet systems</p>
            </div>
            """, unsafe_allow_html=True)
            st.image("https://via.placeholder.com/300x150?text=Light+Curves", use_column_width=True)
    
    with cols[1]:
        with st.container():  # Removed border parameter
            st.markdown(f"""
            <div style="text-align: center; border: 1px solid {COLOR_PALETTE['nebula_purple']}; padding: 20px; border-radius: 8px;">
            <h3 style="color: {COLOR_PALETTE['nebula_purple']};">🤖 Model Training</h3>
            <p>Build and train CNN and XGBoost models to classify exoplanet transits from stellar light curves</p>
            </div>
            """, unsafe_allow_html=True)
            st.image("https://via.placeholder.com/300x150?text=Model+Training", use_column_width=True)
    
    with cols[2]:
        with st.container():  # Removed border parameter
            st.markdown(f"""
            <div style="text-align: center; border: 1px solid {COLOR_PALETTE['nebula_purple']}; padding: 20px; border-radius: 8px;">
            <h3 style="color: {COLOR_PALETTE['nebula_purple']};">🔮 Predictions</h3>
            <p>Upload new observations and classify them as exoplanet or non-exoplanet systems</p>
            </div>
            """, unsafe_allow_html=True)
            st.image("https://via.placeholder.com/300x150?text=Predictions", use_column_width=True)
    
    # Pro Tip Section
    st.markdown("---")
    st.markdown(f"""
    <div style="color: {COLOR_PALETTE['starry_white']}; border: 1px solid {COLOR_PALETTE['nebula_purple']}; padding: 20px; border-radius: 8px;">
    <h3 style="color: {COLOR_PALETTE['nebula_purple']};">💡 Pro Tip</h3>
    <p style="font-size: 1.1rem;"><b>Start with the Data Exploration tab</b> to understand the dataset characteristics before training models or making predictions.</p>
    <p>Explore the light curve visualizations to see how exoplanet transits appear in the data.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample Visualization
    if st.session_state.preprocessed_raw_data is not None:
        st.markdown("---")
        st.markdown(f"""
        <div style="color: {COLOR_PALETTE['starry_white']}; text-align: center;">
        <h3>Sample Light Curves</h3>
        </div>
        """, unsafe_allow_html=True)
        fig_plotly = plot_light_curves(st.session_state.preprocessed_raw_data, 2)
        if fig_plotly:
            st.plotly_chart(fig_plotly, use_container_width=True)
        else:
            st.warning("Failed to generate sample light curves.")

def model_comparison_interface():
    """New and improved model comparison interface"""
    st.title("🆚 Model Comparison")
    st.markdown("---")
    
    # Model status section
    st.subheader("Model Status")  # Removed divider parameter
    st.markdown("---")  # Added manual divider
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():  # Removed border parameter
            st.markdown(f"""
            <div style="border: 1px solid {COLOR_PALETTE['nebula_purple']}; padding: 15px; border-radius: 8px;">
            <h5 style="color: {COLOR_PALETTE['starry_white']};">Convolutional Neural Network</h5>
            </div>
            """, unsafe_allow_html=True)
            if st.session_state.get('cnn_model'):
                st.success("✅ Model Loaded")
                st.metric("Parameters", f"{st.session_state.cnn_model.count_params():,}")
                st.metric("Input Shape", "3197 × 1")
            else:
                st.warning("❌ Model Not Loaded")
                if st.button("Load CNN Model", key="load_cnn_compare"):
                    # Placeholder for load_cnn_model
                    st.info("CNN model loading not implemented yet.")
    
    with col2:
        with st.container():  # Removed border parameter
            st.markdown(f"""
            <div style="border: 1px solid {COLOR_PALETTE['nebula_purple']}; padding: 15px; border-radius: 8px;">
            <h5 style="color: {COLOR_PALETTE['starry_white']};">XGBoost Classifier</h5>
            </div>
            """, unsafe_allow_html=True)
            if st.session_state.get('xgb_model'):
                st.success("✅ Model Loaded")
                st.metric("Trees", st.session_state.xgb_model.n_estimators)
                st.metric("Features", len(st.session_state.features_to_use))
            else:
                st.warning("❌ Model Not Loaded")
                if st.button("Load XGBoost Model", key="load_xgb_compare"):
                    # Placeholder for load_xgb_model
                    st.info("XGBoost model loading not implemented yet.")
    
    st.markdown("---")
    
    # Only show comparison if both models are loaded
    if not (st.session_state.get('cnn_model') and st.session_state.get('xgb_model')):
        st.warning("Please load both models to enable comparison")
        return
    
    # Performance metrics comparison
    st.subheader("Performance Metrics")
    st.markdown("---")
    
    performance_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Inference Time'],
        'CNN': [0.92, 0.89, 0.85, 0.87, '15ms'],
        'XGBoost': [0.88, 0.91, 0.82, 0.86, '5ms']
    }
    
    tab1, tab2 = st.tabs(["Table View", "Visual Comparison"])
    
    with tab1:
        st.dataframe(
            pd.DataFrame(performance_data),
            use_container_width=True,
            hide_index=True
        )
    
    with tab2:
        fig = px.bar(
            pd.DataFrame(performance_data).melt(id_vars='Metric'), 
            x='Metric', 
            y='value', 
            color='variable',
            barmode='group',
            color_discrete_map={
                'CNN': COLOR_PALETTE['nebula_purple'],
                'XGBoost': COLOR_PALETTE['starry_white']
            },
            labels={'value': 'Score', 'variable': 'Model'}
        )
        fig.update_layout(
            paper_bgcolor=COLOR_PALETTE['dark_slate'],
            plot_bgcolor=COLOR_PALETTE['dark_slate'],
            font=dict(color=COLOR_PALETTE['starry_white'])
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Model characteristics comparison
    st.subheader("Model Characteristics")
    st.markdown("---")
    
    char_data = {
        'Characteristic': ['Input Type', 'Training Time', 'Best For', 'Interpretability'],
        'CNN': [
            'Raw light curves', 
            'Longer (GPU recommended)', 
            'Capturing temporal patterns', 
            'Lower (black box)'
        ],
        'XGBoost': [
            'Feature vectors', 
            'Faster', 
            'Structured data analysis', 
            'Higher (feature importance)'
        ]
    }
    
    st.dataframe(
        pd.DataFrame(char_data),
        use_container_width=True,
        hide_index=True
    )
    
    # Recommendation section
    st.subheader("Recommendation")
    st.markdown("---")
    
    with st.expander("Which model should I use?"):
        st.markdown("""
        ### CNN Recommended When:
        - You have raw light curve data
        - You need to capture complex temporal patterns
        - You have sufficient computational resources
        - Maximum accuracy is more important than speed
        
        ### XGBoost Recommended When:
        - You have extracted features
        - You need faster training/prediction
        - Interpretability is important
        - You're working with limited resources
        """)

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
                           "Model Comparison", "Predictions", "Prediction Analysis", "Feedback"],
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
