import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib as mpl
import os
import requests

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

# Load datasets
@st.cache_data(show_spinner="Loading datasets...")
def load_data():
    """Load and cache datasets with error handling"""
    try:
        # Verify Google Drive URL accessibility
        url = "https://drive.google.com/uc?export=download&id=1MxDUIqQ6S9cmi068I62xkbCwZHkpAeJI"
        st.write("Checking accessibility of exoTrain.csv...")
        response = requests.head(url, allow_redirects=True)
        if response.status_code != 200:
            raise Exception(f"Failed to access Google Drive URL (status code: {response.status_code}). Ensure the file is set to 'Anyone with the link'.")

        # Load exoTrain.csv from Google Drive
        st.write("Loading exoTrain.csv from Google Drive...")
        raw_data = pd.read_csv(url)
        st.session_state.raw_data = raw_data
        st.session_state.data_loaded = True
        st.success("exoTrain.csv loaded successfully!")

        # Attempt to load cumulative.csv (assumed not present; update if available)
        summary_data = None
        if os.path.exists("datasets/cumulative.csv"):
            st.write("Loading cumulative.csv from repository...")
            summary_data = pd.read_csv("datasets/cumulative.csv")
            st.session_state.summary_data = summary_data
            st.success("cumulative.csv loaded successfully!")
        else:
            st.warning("cumulative.csv not found in repository. Summary data features disabled.")

        return raw_data, summary_data
    except Exception as e:
        st.error(f"Failed to load datasets: {e}")
        st.write("Please ensure the Google Drive URL for exoTrain.csv is publicly accessible ('Anyone with the link') and try again.")
        st.session_state.data_loaded = False
        return None, None

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
    
    ### Key Features:
    🔭 **Data Exploration**: View light curves and exoplanet characteristics  
    *Navigate using the sidebar to get started!*
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Quick Start Guide"):
        with st.expander("Getting Started"):
            st.markdown(f"""
            <div style="color: {COLOR_PALETTE['starry_white']}">
            1. **Data Exploration**: View light curves and dataset statistics
            2. **Note**: Model training and predictions are disabled due to missing model files.
            
            Pro Tip: Start with the Data Exploration tab to understand the dataset!
            </div>
            """, unsafe_allow_html=True)
    
    if st.session_state.data_loaded:
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
                st.metric("Exoplanet Stars", len(raw_data[raw_data['LABEL'] == 2]))
            with col3:
                st.metric("Non-Exoplanet Stars", len(raw_data[raw_data['LABEL'] == 1]))
            
            st.markdown("---")
            
            st.subheader("Sample Light Curves")
            n_samples = st.slider("Number of samples to display", 1, 10, 3)
            fig_plotly = plot_light_curves(raw_data, n_samples)
            if fig_plotly:
                st.plotly_chart(fig_plotly, use_container_width=True)
            
            st.markdown("---")
            
            st.subheader("Flux Distribution Comparison")
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
            st.error("Light curve data not available.")
    
    with tab2:
        st.header("Summary Data Overview")
        
        if summary_data is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Observations", len(summary_data))
            with col2:
                dispositions = summary_data['koi_disposition'].value_counts()
                st.metric("Confirmed Exoplanets", dispositions.get('CONFIRMED', 0))
            
            st.markdown("---")
            
            st.subheader("Exoplanet Dispositions")
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
            
            st.markdown("---")
            
            st.subheader("Feature Distributions")
            feature = st.selectbox(
                "Select feature to visualize",
                summary_data.select_dtypes(include=[np.number]).columns,
                index=0
            )
            
            try:
                fig_plotly = px.histogram(
                    summary_data,
                    x=feature,
                    color='koi_disposition',
                    marginal="box",
                    nbins=50,
                    barmode='overlay',
                    color_discrete_sequence=[COLOR_PALETTE['nebula_purple'], COLOR_PALETTE['starry_white'], COLOR_PALETTE['dark_slate']]
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
            st.warning("Summary data not available. Please provide cumulative.csv.")

# Main App
def main():
    """Main application function"""
    setup_page_config()
    initialize_session_state()
    
    raw_data, summary_data = load_data()
    
    st.sidebar.title("🌌 Exoplanet Detection")
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigation", 
                          ["Home", "Data Exploration"],
                          index=0)
    
    if page == "Home":
        landing_page(raw_data)
    elif page == "Data Exploration":
        data_exploration_page(raw_data, summary_data)

if __name__ == "__main__":
    main()
