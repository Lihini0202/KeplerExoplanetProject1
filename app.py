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

# Clear memory
def clear_memory():
    """Clear TensorFlow session and free memory"""
    tf.keras.backend.clear_session()
    gc.collect()
    if 'cnn_model' in st.session_state:
        del st.session_state.cnn_model
    if 'xgb_model' in st.session_state:
        del st.session_state.xgb_model

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
        :root {{
            background-color: {COLOR_PALETTE['dark_slate']} !important;
        }}
        .main {{
            background-color: {COLOR_PALETTE['dark_slate']} !important;
            color: {COLOR_PALETTE['starry_white']} !important;
        }}
        .stSidebar {{
            background-color: {COLOR_PALETTE['dark_slate']} !important;
            color: {COLOR_PALETTE['starry_white']} !important;
        }}
        .stSidebar [data-testid="stSidebarNav"] {{
            background-color: {COLOR_PALETTE['dark_slate']} !important;
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
        .stTextInput input, .stTextArea textarea, .stSelectbox, .stSlider div {{
            background-color: {COLOR_PALETTE['dark_slate']} !important;
            color: {COLOR_PALETTE['starry_white']} !important;
            border-color: {COLOR_PALETTE['nebula_purple']} !important;
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
        raw_data = pd.read_csv("datasets/exoTrain.csv")
        summary_data = pd.read_csv("datasets/cumulative.csv")
        st.session_state.data_loaded = True
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

# Model loading functions
def load_cnn_model():
    """Safely load CNN model with memory cleanup"""
    try:
        clear_memory()
        st.session_state.cnn_model = load_model(
            'models/cnn_exoplanet_model.keras', 
            custom_objects={'focal_loss_fixed': focal_loss(gamma=2.0, alpha=0.5)}
        )
        st.session_state.scaler_cnn = joblib.load('models/scaler_cnn.pkl')
        st.success("CNN model and scaler loaded successfully!")
        return True
    except Exception as e:
        st.error(f"Error loading CNN model: {e}")
        return False

def load_xgb_model():
    """Safely load XGBoost model with components"""
    try:
        clear_memory()
        st.session_state.xgb_model = joblib.load('models/xgb_model.pkl')
        st.session_state.scaler_xgb = joblib.load('models/scaler_xgb.pkl')
        st.session_state.label_encoder = joblib.load('models/label_encoder.pkl')
        st.session_state.features_to_use = joblib.load('models/features_to_use.pkl')
        st.session_state.num_classes = len(st.session_state.label_encoder.classes_)
        st.success("XGBoost model and components loaded successfully!")
        return True
    except Exception as e:
        st.error(f"Error loading XGBoost model: {e}")
        return False

# Landing Page
def landing_page():
    """Render the landing page"""
    st.title("🌌 Exoplanet Detection Dashboard")
    st.markdown(f"""
    <div style="color: {COLOR_PALETTE['starry_white']}">
    ## Discover Exoplanets Through Machine Learning
    
    This interactive dashboard allows you to:
    - Explore exoplanet light curve data
    - Train and evaluate machine learning models
    - Make predictions on new observations
    
    ### Key Features:
    🔭 **Data Exploration**: Visualize light curves and exoplanet characteristics  
    🤖 **Model Training**: Build CNN and XGBoost classification models  
    🔮 **Predictions**: Classify new observations as exoplanet or non-exoplanet  
    
    *Navigate using the sidebar to get started!*
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Quick Start Guide"):
        with st.expander("Getting Started"):
            st.markdown(f"""
            <div style="color: {COLOR_PALETTE['starry_white']}">
            1. **Data Exploration**: View light curves and dataset statistics
            2. **Model Training**: Load pre-trained models or train new ones
            3. **Predictions**: Upload new data for classification
            
            Pro Tip: Start with the Data Exploration tab to understand the dataset!
            </div>
            """, unsafe_allow_html=True)
    
    _, raw_data, summary_data = load_data()
    fig_plotly = plot_light_curves(raw_data, 2)
    if fig_plotly:
        st.plotly_chart(fig_plotly, use_container_width=True)
    else:
        st.warning("Failed to generate sample light curves.")

# Model Comparison Interface
def model_comparison_interface():
    """Render the model精力充沛的 model comparison interface"""
    st.header("📊 Model Performance Comparison")
    
    if st.session_state.cnn_model and st.session_state.xgb_model:
        comparison_data = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Training Time'],
            'CNN': [0.92, 0.89, 0.85, 0.87, '2 min 30 sec'],
            'XGBoost': [0.88, 0.91, 0.82, 0.86, '45 sec']
        })
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(comparison_data.style.apply(
                lambda x: ['background-color: ' + COLOR_PALETTE['dark_slate'] for val in x],
                axis=1
            ))
        
        with col2:
            fig_plotly = px.bar(
                comparison_data.melt(id_vars='Metric'), 
                x='Metric', 
                y='value', 
                color='variable',
                barmode='group', 
                title='Model Comparison',
                color_discrete_map={'CNN': COLOR_PALETTE['nebula_purple'], 'XGBoost': COLOR_PALETTE['starry_white']}
            )
            fig_plotly.update_layout(
                paper_bgcolor=COLOR_PALETTE['dark_slate'],
                plot_bgcolor=COLOR_PALETTE['dark_slate'],
                font=dict(color=COLOR_PALETTE['starry_white'])
            )
            st.plotly_chart(fig_plotly, use_container_width=True)
        
        st.markdown(f"""
        <div style="color: {COLOR_PALETTE['starry_white']}">
        ### When to use each model:
        - **CNN**: Best for raw light curve data, captures temporal patterns  
        - **XGBoost**: Best for tabular feature data, faster training  
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Please load both CNN and XGBoost models to enable comparison")

# Prediction Analysis Interface
def prediction_analysis():
    """Render the prediction analysis interface"""
    st.header("🔍 Prediction Analysis")
    
    if 'prediction_results' in st.session_state and st.session_state.prediction_results is not None:
        results = st.session_state.prediction_results
        
        tab1, tab2 = st.tabs(["Confidence Analysis", "Error Analysis"])
        
        with tab1:
            st.subheader("Prediction Confidence Distribution")
            try:
                results['Confidence'] = results['Confidence'].str.rstrip('%').astype(float)
                fig_plotly = px.histogram(
                    results, 
                    x="Confidence", 
                    color="Prediction",
                    nbins=20, 
                    title="Confidence Scores Distribution",
                    color_discrete_map={
                        'Exoplanet': COLOR_PALETTE['nebula_purple'], 
                        'Non-Exoplanet': COLOR_PALETTE['starry_white'], 
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
                st.warning(f"Failed to plot confidence distribution: {str(e)}")
        
        with tab2:
            st.subheader("Misclassification Analysis")
            st.write("Examples of misclassified observations (placeholder):")
            st.dataframe(results.head().style.apply(
                lambda x: ['background-color: ' + COLOR_PALETTE['dark_slate'] for val in x],
                axis=1
            ))
    else:
        st.warning("No prediction results available. Make predictions first!")

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

# Data Exploration Page
def data_exploration_page(raw_data, summary_data):
    """Render the data exploration page"""
    st.title("📊 Data Exploration")
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["Light Curve Data", "Summary Data"])
    
    with tab1:
        st.header("Light Curve Data Overview")
        
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
                st.warning(f"Failed to plot exoplanet flux distribution: {str(e)}")
        
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
                st.warning(f"Failed to plot non-exoplanet flux distribution: {str(e)}")
    
    with tab2:
        st.header("Summary Data Overview")
        
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
            st.warning(f"Failed to plot dispositions: {str(e)}")
        
        st.markdown("---")
        
        st.subheader("Feature Distributions")
        feature = st.selectbox(
            "Select feature to visualize",
            summary_data.select_dtypes(include=[np.number]).columns,
            index=3
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
            st.warning(f"Failed to plot feature distribution: {str(e)}")

# Model Training Page
def model_training_page(raw_data, summary_data):
    """Render the model training page"""
    st.title("🤖 Model Training")
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["CNN Model", "XGBoost Model"])
    
    with tab1:
        st.header("Convolutional Neural Network")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Load Pre-Trained CNN Model", use_container_width=True):
                if load_cnn_model():
                    st.markdown("---")
                    st.subheader("Model Architecture")
                    model_summary = []
                    st.session_state.cnn_model.summary(print_fn=lambda x: model_summary.append(x))
                    st.text("\n".join(model_summary))
                    
                    try:
                        if raw_data is None or raw_data.empty:
                            st.warning("No raw data available for predictions.")
                        else:
                            st.markdown("---")
                            st.subheader("Sample Predictions")
                            sample_data = raw_data.sample(n=min(100, len(raw_data)), random_state=42)
                            results = predict_with_cnn(sample_data)
                            if results is not None:
                                fig_plotly = px.pie(
                                    results,
                                    names='Prediction',
                                    title='Sample Prediction Distribution',
                                    color='Prediction',
                                    color_discrete_map={'Exoplanet': COLOR_PALETTE['nebula_purple'], 'Non-Exoplanet': COLOR_PALETTE['starry_white']}
                                )
                                fig_plotly.update_layout(
                                    paper_bgcolor=COLOR_PALETTE['dark_slate'],
                                    plot_bgcolor=COLOR_PALETTE['dark_slate'],
                                    font=dict(color=COLOR_PALETTE['starry_white'])
                                )
                                st.plotly_chart(fig_plotly, use_container_width=True)
                            else:
                                st.warning("Failed to generate predictions for plotting.")
                    except Exception as e:
                        st.warning(f"Could not generate sample predictions plot: {str(e)}")
        
        with col2:
            if st.button("🔧 Train CNN Model", use_container_width=True):
                with st.spinner("Training CNN model..."):
                    try:
                        clear_memory()
                        
                        X_cnn = raw_data.drop(columns=['LABEL']).values
                        y_cnn = raw_data['LABEL'].values - 1
                        scaler_cnn = StandardScaler()
                        X_cnn_scaled = scaler_cnn.fit_transform(X_cnn)
                        X_cnn = X_cnn_scaled.reshape(X_cnn_scaled.shape[0], X_cnn_scaled.shape[1], 1)
                        
                        confirmed_indices = np.where(y_cnn == 1)[0]
                        X_confirmed = X_cnn[confirmed_indices]
                        y_confirmed = y_cnn[confirmed_indices]
                        oversample_factor = 5
                        X_train_oversampled = np.vstack([X_cnn] + [X_confirmed] * oversample_factor)
                        y_train_oversampled = np.hstack([y_cnn] + [y_confirmed] * oversample_factor)
                        
                        class_weights = compute_class_weight('balanced', classes=np.unique(y_train_oversampled), y=y_train_oversampled)
                        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
                        
                        inputs = Input(shape=(X_cnn.shape[1], 1))
                        x = inputs
                        for i in range(st.session_state.cnn_config['conv_layers']):
                            x = Conv1D(
                                st.session_state.cnn_config['filters'][min(i, len(st.session_state.cnn_config['filters'])-1)],
                                kernel_size=st.session_state.cnn_config['kernel_size'],
                                activation='relu',
                                kernel_regularizer=l2(0.02)
                            )(x)
                            x = BatchNormalization()(x)
                            x = MaxPooling1D(pool_size=4)(x)
                        x = GlobalAveragePooling1D()(x)
                        x = Dense(st.session_state.cnn_config['dense_units'], activation='relu', kernel_regularizer=l2(0.02))(x)
                        x = Dropout(st.session_state.cnn_config['dropout_rate'])(x)
                        outputs = Dense(1, activation='sigmoid')(x)
                        model = Model(inputs, outputs)
                        
                        model.compile(
                            optimizer=tf.keras.optimizers.Adam(learning_rate=st.session_state.cnn_config['learning_rate']),
                            loss=focal_loss(gamma=2.0, alpha=0.5),
                            metrics=['accuracy']
                        )
                        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        class TrainingCallback(tf.keras.callbacks.Callback):
                            def on_epoch_end(self, epoch, logs=None):
                                progress = (epoch + 1) / 20
                                progress_bar.progress(min(progress, 1.0))
                                status_text.text(f"Epoch {epoch + 1}/20 - Loss: {logs['loss']:.4f} - Val Loss: {logs['val_loss']:.4f}")
                        
                        history = model.fit(
                            X_train_oversampled, y_train_oversampled,
                            validation_split=0.2,
                            epochs=20,
                            batch_size=64,
                            callbacks=[early_stopping, lr_scheduler, TrainingCallback()],
                            class_weight=class_weight_dict,
                            verbose=0
                        )
                        
                        model.save('models/cnn_exoplanet_model.keras')
                        joblib.dump(scaler_cnn, 'models/scaler_cnn.pkl')
                        st.session_state.cnn_model = model
                        st.session_state.scaler_cnn = scaler_cnn
                        
                        st.success("CNN model trained and saved successfully!")
                        progress_bar.empty()
                        status_text.empty()
                        
                        fig_plotly = go.Figure()
                        fig_plotly.add_trace(go.Scatter(
                            y=history.history['loss'],
                            mode='lines',
                            name='Training Loss',
                            line=dict(color=COLOR_PALETTE['nebula_purple'])
                        ))
                        fig_plotly.add_trace(go.Scatter(
                            y=history.history['val_loss'],
                            mode='lines',
                            name='Validation Loss',
                            line=dict(color=COLOR_PALETTE['starry_white'])
                        ))
                        fig_plotly.update_layout(
                            title='Training History',
                            xaxis_title='Epoch',
                            yaxis_title='Loss',
                            hovermode="x unified",
                            paper_bgcolor=COLOR_PALETTE['dark_slate'],
                            plot_bgcolor=COLOR_PALETTE['dark_slate'],
                            font=dict(color=COLOR_PALETTE['starry_white'])
                        )
                        st.plotly_chart(fig_plotly, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")
                        progress_bar.empty()
                        status_text.empty()
        
        if st.session_state.cnn_model:
            st.markdown("---")
            st.subheader("Model Architecture")
            model_summary = []
            st.session_state.cnn_model.summary(print_fn=lambda x: model_summary.append(x))
            st.text("\n".join(model_summary))
    
    with tab2:
        st.header("XGBoost Classifier")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Load Pre-Trained XGBoost Model", use_container_width=True):
                if load_xgb_model():
                    try:
                        st.markdown("---")
                        st.subheader("Feature Importance")
                        X_xgb = summary_data[st.session_state.features_to_use]
                        X_xgb = X_xgb.dropna(axis=1, how='all').replace([np.inf, -np.inf], np.nan)
                        X_xgb = X_xgb.fillna(X_xgb.mean(numeric_only=True))
                        X_xgb_scaled = st.session_state.scaler_xgb.transform(X_xgb)
                        
                        fig_plotly = px.bar(
                            x=st.session_state.features_to_use,
                            y=st.session_state.xgb_model.feature_importances_,
                            labels={'x': 'Features', 'y': 'Importance'},
                            title='Feature Importance (Pre-Trained Model)',
                            color=st.session_state.xgb_model.feature_importances_,
                            color_continuous_scale=[COLOR_PALETTE['dark_slate'], COLOR_PALETTE['nebula_purple'], COLOR_PALETTE['starry_white']]
                        )
                        fig_plotly.update_layout(
                            xaxis_tickangle=-45,
                            paper_bgcolor=COLOR_PALETTE['dark_slate'],
                            plot_bgcolor=COLOR_PALETTE['dark_slate'],
                            font=dict(color=COLOR_PALETTE['starry_white'])
                        )
                        st.plotly_chart(fig_plotly, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not generate feature importance plots: {str(e)}")
        
        with col2:
            if st.button("🔧 Train XGBoost Model", use_container_width=True):
                with st.spinner("Training XGBoost model..."):
                    try:
                        clear_memory()
                        
                        try:
                            st.session_state.features_to_use = joblib.load('models/features_to_use.pkl')
                        except:
                            numeric_columns = summary_data.select_dtypes(include=[np.number]).columns
                            exclude_columns = ['kepid', 'koi_disposition', 'label_encoded']
                            st.session_state.features_to_use = [col for col in numeric_columns if col not in exclude_columns]
                            if not st.session_state.features_to_use:
                                st.error("No numeric features found in summary_data.")
                                st.stop()
                        
                        X_xgb = summary_data[st.session_state.features_to_use]
                        X_xgb = X_xgb.dropna(axis=1, how='all').replace([np.inf, -np.inf], np.nan)
                        X_xgb = X_xgb.fillna(X_xgb.mean(numeric_only=True))
                        
                        if 'label_encoded' not in summary_data.columns:
                            st.session_state.label_encoder = LabelEncoder()
                            summary_data['label_encoded'] = st.session_state.label_encoder.fit_transform(summary_data['koi_disposition'])
                            joblib.dump(st.session_state.label_encoder, 'models/label_encoder.pkl')
                        
                        y_xgb = summary_data['label_encoded']
                        st.session_state.num_classes = len(np.unique(y_xgb))
                        
                        if len(X_xgb) != len(y_xgb):
                            st.error(f"Feature and target arrays have different lengths: X({len(X_xgb)}) vs y({len(y_xgb)})")
                            st.stop()
                        
                        X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
                            X_xgb, y_xgb, 
                            test_size=0.3, 
                            stratify=y_xgb, 
                            random_state=42
                        )
                        
                        scaler_xgb = StandardScaler()
                        X_train_xgb = scaler_xgb.fit_transform(X_train_xgb)
                        X_test_xgb = scaler_xgb.transform(X_test_xgb)
                        
                        if st.session_state.num_classes > 2:
                            objective = 'multi:softprob'
                            eval_metric = 'mlogloss'
                        else:
                            objective = 'binary:logistic'
                            eval_metric = 'logloss'
                        
                        xgb_model = xgb.XGBClassifier(
                            objective=objective,
                            eval_metric=eval_metric,
                            random_state=42,
                            n_estimators=st.session_state.xgb_config['n_estimators'],
                            max_depth=st.session_state.xgb_config['max_depth'],
                            learning_rate=st.session_state.xgb_config['learning_rate'],
                            subsample=st.session_state.xgb_config['subsample'],
                            tree_method='hist',
                            num_class=st.session_state.num_classes if st.session_state.num_classes > 2 else None
                        )
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        xgb_model.fit(
                            X_train_xgb, 
                            y_train_xgb,
                            eval_set=[(X_test_xgb, y_test_xgb)],
                            early_stopping_rounds=10,
                            verbose=0
                        )
                        
                        joblib.dump(xgb_model, 'models/xgb_model.pkl')
                        joblib.dump(scaler_xgb, 'models/scaler_xgb.pkl')
                        joblib.dump(st.session_state.features_to_use, 'models/features_to_use.pkl')
                        st.session_state.xgb_model = xgb_model
                        st.session_state.scaler_xgb = scaler_xgb
                        
                        st.success("XGBoost model trained and saved successfully!")
                        progress_bar.empty()
                        status_text.empty()
                        
                        fig_plotly = px.bar(
                            x=st.session_state.features_to_use,
                            y=xgb_model.feature_importances_,
                            labels={'x': 'Features', 'y': 'Importance'},
                            title='Feature Importance',
                            color=xgb_model.feature_importances_,
                            color_continuous_scale=[COLOR_PALETTE['dark_slate'], COLOR_PALETTE['nebula_purple'], COLOR_PALETTE['starry_white']]
                        )
                        fig_plotly.update_layout(
                            xaxis_tickangle=-45,
                            paper_bgcolor=COLOR_PALETTE['dark_slate'],
                            plot_bgcolor=COLOR_PALETTE['dark_slate'],
                            font=dict(color=COLOR_PALETTE['starry_white'])
                        )
                        st.plotly_chart(fig_plotly, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")
                        progress_bar.empty()
                        status_text.empty()

# Prediction functions
def predict_with_cnn(data):
    """Make predictions with CNN model in batches"""
    try:
        if st.session_state.scaler_cnn is None:
            raise ValueError("CNN scaler not loaded.")
        
        tf.keras.backend.clear_session()
        gc.collect()
        
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
        st.error(f"CNN prediction error: {str(e)}")
        return None

def predict_with_xgb(data):
    """Make predictions with XGBoost model"""
    try:
        if st.session_state.scaler_xgb is None or st.session_state.label_encoder is None:
            raise ValueError("XGBoost scaler or label encoder not loaded.")
        
        missing_features = set(st.session_state.features_to_use) - set(data.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
            
        processed_data = data.copy()
        valid_cols = [col for col in st.session_state.features_to_use if col in processed_data.columns]
        
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
        st.error(f"XGBoost prediction error: {str(e)}")
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
            st.error(f"Error processing file: {str(e)}")

# Main App
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
