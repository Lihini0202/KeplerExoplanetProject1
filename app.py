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
import shap
import pickle
import os
import gc
from tabulate import tabulate
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_extras.metric_cards import style_metric_cards

# Initialize session state and memory management
def initialize_session_state():
    """Initialize all session state variables"""
    if 'cnn_model' not in st.session_state:
        st.session_state.cnn_model = None
    if 'xgb_model' not in st.session_state:
        st.session_state.xgb_model = None
    if 'scaler_cnn' not in st.session_state:
        st.session_state.scaler_cnn = None
    if 'scaler_xgb' not in st.session_state:
        st.session_state.scaler_xgb = None
    if 'label_encoder' not in st.session_state:
        st.session_state.label_encoder = None
    if 'features_to_use' not in st.session_state:
        st.session_state.features_to_use = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False

# Clear memory and session state
def clear_memory():
    """Clear TensorFlow session and free memory"""
    tf.keras.backend.clear_session()
    gc.collect()
    if 'cnn_model' in st.session_state:
        del st.session_state.cnn_model
    if 'xgb_model' in st.session_state:
        del st.session_state.xgb_model

# Custom CSS for styling
def setup_page_config():
    """Configure page settings and styles"""
    st.set_page_config(
        page_title="Exoplanet Detection Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="🌌"
    )
    
    st.markdown("""
        <style>
        :root {
            background-color: #f0f2f6 !important;
        }
        .main {
            background-color: black !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding: 0 25px;
            background-color: #f8f9fa;
            border-radius: 8px 8px 0px 0px;
            gap: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ffffff;
        }
        .stTabs [data-baseweb="tab-highlight"] {
            background-color: #4CAF50;
        }
        .stButton>button {
            border-radius: 8px;
            height: 40px;
            width: 100%;
            background-color: #4CAF50 !important;
            color: white !important;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #45a049 !important;
        }
        .stDataFrame {
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .plot-container {
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 20px;
            background-color: white;
        }
        .metric-card {
            background-color: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
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

# Load datasets with caching and error handling
@st.cache_data(show_spinner="Loading datasets...")
def load_data():
    """Load and cache datasets with error handling"""
    try:
        raw_data = pd.read_csv("datasets/exoTrain.csv")
        summary_data = pd.read_csv("datasets/cumulative.csv")
        st.session_state.data_loaded = True
        return raw_data, summary_data
    except FileNotFoundError as e:
        st.error(f"Dataset files not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading datasets: {e}")
        st.stop()

# Light curve visualization with error handling
def plot_light_curves(data, n_samples=5):
    """Visualize sample light curves with error handling"""
    try:
        exoplanet_samples = data[data['LABEL'] == 2].sample(n=min(n_samples, len(data[data['LABEL'] == 2])))
        non_exoplanet_samples = data[data['LABEL'] == 1].sample(n=min(n_samples, len(data[data['LABEL'] == 1])))
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Exoplanet Stars", "Non-Exoplanet Stars"))
        
        for idx, row in exoplanet_samples.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(row[1:]))),
                    y=row[1:],
                    mode='lines',
                    name=f'Exoplanet Star {idx}',
                    line=dict(color='#FF6B6B')
                ),
                row=1, col=1
            )
        
        for idx, row in non_exoplanet_samples.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(row[1:]))),
                    y=row[1:],
                    mode='lines',
                    name=f'Non-Exoplanet Star {idx}',
                    line=dict(color='#4ECDC4')
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_yaxes(title_text="Flux", row=1, col=1)
        
        return fig
    except Exception as e:
        st.error(f"Error generating light curves: {e}")
        return None

# Model loading functions with memory management
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
        st.success("XGBoost model and components loaded successfully!")
        return True
    except Exception as e:
        st.error(f"Error loading XGBoost model: {e}")
        return False

# Data Exploration Page
def data_exploration_page(raw_data, summary_data):
    """Render the data exploration page"""
    st.title("📊 Data Exploration")
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["Light Curve Data", "Summary Data"])
    
    with tab1:
        st.header("Light Curve Data Overview")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Stars", len(raw_data))
        with col2:
            st.metric("Exoplanet Stars", len(raw_data[raw_data['LABEL'] == 2]))
        with col3:
            st.metric("Non-Exoplanet Stars", len(raw_data[raw_data['LABEL'] == 1]))
        
        st.markdown("---")
        
        # Automatic light curve display
        st.subheader("Sample Light Curves")
        n_samples = st.slider("Number of samples to display", 1, 10, 3)
        fig = plot_light_curves(raw_data, n_samples)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Flux distribution
        st.subheader("Flux Distribution Comparison")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                raw_data[raw_data['LABEL'] == 2].iloc[:, 1:].mean(axis=1),
                nbins=50,
                title="Exoplanet Stars Flux Distribution",
                color_discrete_sequence=['#FF6B6B'],
                labels={'value': 'Average Flux'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(
                raw_data[raw_data['LABEL'] == 1].iloc[:, 1:].mean(axis=1),
                nbins=50,
                title="Non-Exoplanet Stars Flux Distribution",
                color_discrete_sequence=['#4ECDC4'],
                labels={'value': 'Average Flux'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Summary Data Overview")
        
        # Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Observations", len(summary_data))
        with col2:
            dispositions = summary_data['koi_disposition'].value_counts()
            st.metric("Confirmed Exoplanets", dispositions.get('CONFIRMED', 0))
        
        st.markdown("---")
        
        # Disposition distribution
        st.subheader("Exoplanet Dispositions")
        fig = px.pie(
            summary_data,
            names='koi_disposition',
            hole=0.3,
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Feature distributions
        st.subheader("Feature Distributions")
        feature = st.selectbox(
            "Select feature to visualize",
            summary_data.select_dtypes(include=[np.number]).columns,
            index=3
        )
        
        fig = px.histogram(
            summary_data,
            x=feature,
            color='koi_disposition',
            marginal="box",
            nbins=50,
            barmode='overlay',
            color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
        )
        st.plotly_chart(fig, use_container_width=True)

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
                load_cnn_model()
        
        with col2:
            if st.button("🔧 Train CNN Model", use_container_width=True):
                with st.spinner("Training CNN model..."):
                    try:
                        clear_memory()
                        
                        # Data preparation
                        X_cnn = raw_data.drop(columns=['LABEL']).values
                        y_cnn = raw_data['LABEL'].values - 1
                        scaler_cnn = StandardScaler()
                        X_cnn_scaled = scaler_cnn.fit_transform(X_cnn)
                        X_cnn = X_cnn_scaled.reshape(X_cnn_scaled.shape[0], X_cnn_scaled.shape[1], 1)
                        
                        # Handle class imbalance
                        confirmed_indices = np.where(y_cnn == 1)[0]
                        X_confirmed = X_cnn[confirmed_indices]
                        y_confirmed = y_cnn[confirmed_indices]
                        oversample_factor = 5  # Reduced from 10 to save memory
                        X_train_oversampled = np.vstack([X_cnn] + [X_confirmed] * oversample_factor)
                        y_train_oversampled = np.hstack([y_cnn] + [y_confirmed] * oversample_factor)
                        
                        # Class weights
                        class_weights = compute_class_weight('balanced', classes=np.unique(y_train_oversampled), y=y_train_oversampled)
                        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
                        
                        # Simplified model architecture
                        inputs = Input(shape=(X_cnn.shape[1], 1))
                        x = Conv1D(8, kernel_size=5, activation='relu', kernel_regularizer=l2(0.02))(inputs)  # Reduced kernel size
                        x = BatchNormalization()(x)
                        x = MaxPooling1D(pool_size=4)(x)
                        x = Conv1D(16, kernel_size=5, activation='relu', kernel_regularizer=l2(0.02))(x)  # Reduced kernel size
                        x = BatchNormalization()(x)
                        x = MaxPooling1D(pool_size=4)(x)
                        x = GlobalAveragePooling1D()(x)
                        x = Dense(32, activation='relu', kernel_regularizer=l2(0.02))(x)
                        x = Dropout(0.3)(x)
                        outputs = Dense(1, activation='sigmoid')(x)
                        model = Model(inputs, outputs)
                        
                        # Training configuration
                        model.compile(optimizer='adam', loss=focal_loss(gamma=2.0, alpha=0.5), metrics=['accuracy'])
                        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)  # Reduced patience
                        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)  # Reduced patience
                        
                        # Training with progress
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        class TrainingCallback(tf.keras.callbacks.Callback):
                            def on_epoch_end(self, epoch, logs=None):
                                progress = (epoch + 1) / 20  # Reduced epochs
                                progress_bar.progress(min(progress, 1.0))
                                status_text.text(f"Epoch {epoch + 1}/20 - Loss: {logs['loss']:.4f} - Val Loss: {logs['val_loss']:.4f}")
                        
                        history = model.fit(
                            X_train_oversampled, y_train_oversampled,
                            validation_split=0.2,
                            epochs=20,  # Reduced epochs
                            batch_size=64,  # Reduced batch size
                            callbacks=[early_stopping, lr_scheduler, TrainingCallback()],
                            class_weight=class_weight_dict,
                            verbose=0
                        )
                        
                        # Save model and update session state
                        model.save('models/cnn_exoplanet_model.keras')
                        joblib.dump(scaler_cnn, 'models/scaler_cnn.pkl')
                        st.session_state.cnn_model = model
                        st.session_state.scaler_cnn = scaler_cnn
                        
                        # Display results
                        st.success("CNN model trained and saved successfully!")
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Plot training history
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=history.history['loss'],
                            mode='lines',
                            name='Training Loss',
                            line=dict(color='#FF6B6B')
                        ))
                        fig.add_trace(go.Scatter(
                            y=history.history['val_loss'],
                            mode='lines',
                            name='Validation Loss',
                            line=dict(color='#4ECDC4')
                        ))
                        fig.update_layout(
                            title='Training History',
                            xaxis_title='Epoch',
                            yaxis_title='Loss',
                            hovermode="x unified"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
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
                load_xgb_model()
        
        with col2:
            if st.button("🔧 Train XGBoost Model", use_container_width=True):
                with st.spinner("Training XGBoost model..."):
                    try:
                        clear_memory()
                        
                        # Feature selection
                        try:
                            st.session_state.features_to_use = joblib.load('models/features_to_use.pkl')
                        except:
                            numeric_columns = summary_data.select_dtypes(include=[np.number]).columns
                            exclude_columns = ['kepid', 'koi_disposition', 'label_encoded']
                            st.session_state.features_to_use = [col for col in numeric_columns if col not in exclude_columns]
                            if not st.session_state.features_to_use:
                                st.error("No numeric features found in summary_data.")
                                st.stop()
                        
                        # Data preparation
                        X_xgb = summary_data[st.session_state.features_to_use]
                        X_xgb = X_xgb.dropna(axis=1, how='all').replace([np.inf, -np.inf], np.nan)
                        X_xgb = X_xgb.fillna(X_xgb.mean(numeric_only=True))
                        
                        # Label encoding
                        if 'label_encoded' not in summary_data.columns:
                            st.session_state.label_encoder = LabelEncoder()
                            summary_data['label_encoded'] = st.session_state.label_encoder.fit_transform(summary_data['koi_disposition'])
                            joblib.dump(st.session_state.label_encoder, 'models/label_encoder.pkl')
                        
                        y_xgb = summary_data['label_encoded']
                        
                        # Train-test split
                        X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
                            X_xgb, y_xgb, test_size=0.3, stratify=y_xgb, random_state=42
                        )
                        
                        # Feature scaling
                        scaler_xgb = StandardScaler()
                        X_train_xgb = scaler_xgb.fit_transform(X_train_xgb)
                        X_test_xgb = scaler_xgb.transform(X_test_xgb)
                        
                        # Simplified model
                        xgb_model = xgb.XGBClassifier(
                            objective='binary:logistic',
                            eval_metric='logloss',
                            random_state=42,
                            n_estimators=50,  # Reduced from 100
                            max_depth=3,      # Reduced from 5
                            learning_rate=0.1,
                            tree_method='hist'  # More memory-efficient
                        )
                        
                        # Training with progress
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        xgb_model.fit(
                            X_train_xgb, 
                            y_train_xgb,
                            eval_set=[(X_test_xgb, y_test_xgb)],
                            early_stopping_rounds=10,
                            verbose=0
                        )
                        
                        # Save model and components
                        joblib.dump(xgb_model, 'models/xgb_model.pkl')
                        joblib.dump(scaler_xgb, 'models/scaler_xgb.pkl')
                        joblib.dump(st.session_state.features_to_use, 'models/features_to_use.pkl')
                        st.session_state.xgb_model = xgb_model
                        st.session_state.scaler_xgb = scaler_xgb
                        
                        # Display results
                        st.success("XGBoost model trained and saved successfully!")
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Feature importance
                        fig = px.bar(
                            x=st.session_state.features_to_use,
                            y=xgb_model.feature_importances_,
                            labels={'x': 'Features', 'y': 'Importance'},
                            title='Feature Importance',
                            color=xgb_model.feature_importances_,
                            color_continuous_scale='RdYlGn'
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # SHAP values (with sample limitation)
                        explainer = shap.TreeExplainer(xgb_model)
                        sample_idx = np.random.choice(X_test_xgb.shape[0], size=min(100, X_test_xgb.shape[0]), replace=False)
                        shap_values = explainer.shap_values(X_test_xgb[sample_idx])
                        
                        fig, ax = plt.subplots()
                        shap.summary_plot(
                            shap_values, 
                            X_test_xgb[sample_idx], 
                            feature_names=st.session_state.features_to_use, 
                            show=False,
                            plot_size=(10, 6)
                        )
                        st.pyplot(fig)
                        plt.close()
                        
                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")
                        progress_bar.empty()
                        status_text.empty()

# Prediction functions with memory management
def predict_with_cnn(data):
    """Make predictions with CNN model in batches"""
    try:
        # Clear memory and prepare data
        tf.keras.backend.clear_session()
        gc.collect()
        
        X = data.drop(columns=['LABEL'] if 'LABEL' in data.columns else [], errors='ignore').values
        X_scaled = st.session_state.scaler_cnn.transform(X)
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        
        # Predict in batches to avoid memory issues
        batch_size = min(1000, len(X_reshaped))
        predictions = []
        for i in range(0, len(X_reshaped), batch_size):
            batch = X_reshaped[i:i + batch_size]
            preds = st.session_state.cnn_model.predict(batch, verbose=0)
            predictions.extend(preds.flatten())
        
        predictions = (np.array(predictions) > 0.5).astype(int)
        
        # Format results
        results = pd.DataFrame({
            'Star ID': data.index,
            'Prediction': ['Exoplanet' if p == 1 else 'Non-Exoplanet' for p in predictions],
            'Confidence': [f"{max(pred, 1-pred)*100:.2f}%" for pred in predictions]
        })
        
        return results
    except Exception as e:
        st.error(f"CNN prediction error: {e}")
        return None

def predict_with_xgb(data):
    """Make predictions with XGBoost model"""
    try:
        # Verify features exist
        missing_features = set(st.session_state.features_to_use) - set(data.columns)
        if missing_features:
            st.error(f"Missing required features: {missing_features}")
            return None
            
        # Prepare data
        X = data[st.session_state.features_to_use]
        X = X.dropna(axis=1, how='all').replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean(numeric_only=True))
        X_scaled = st.session_state.scaler_xgb.transform(X)
        
        # Predict
        predictions = st.session_state.xgb_model.predict(X_scaled)
        prediction_probs = st.session_state.xgb_model.predict_proba(X_scaled)
        predictions = st.session_state.label_encoder.inverse_transform(predictions)
        
        # Format results
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
    
    # Model selection and verification
    model_choice = st.selectbox("Select Model for Prediction", ["CNN", "XGBoost"])
    
    col1, col2 = st.columns(2)
    with col1:
        if model_choice == "CNN":
            if st.button("🔄 Verify CNN Model", help="Check if CNN model is loaded properly"):
                if load_cnn_model():
                    st.success("CNN model ready for predictions!")
        else:
            if st.button("🔄 Verify XGBoost Model", help="Check if XGBoost model is loaded properly"):
                if load_xgb_model():
                    st.success("XGBoost model ready for predictions!")
    
    # File uploader
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
                            st.dataframe(results.style.applymap(
                                lambda x: 'background-color: #FF6B6B' if x == 'Exoplanet' else 'background-color: #4ECDC4',
                                subset=['Prediction']
                            ))
                            
                            # Visualization
                            fig = px.pie(
                                results,
                                names='Prediction',
                                title='Prediction Distribution',
                                color='Prediction',
                                color_discrete_map={'Exoplanet': '#FF6B6B', 'Non-Exoplanet': '#4ECDC4'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
            
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
                            st.dataframe(results.style.applymap(
                                lambda x: 'background-color: #FF6B6B' if x == 'CONFIRMED' else 
                                        ('background-color: #45B7D1' if x == 'CANDIDATE' else 'background-color: #4ECDC4'),
                                subset=['Prediction']
                            ))
                            
                            # Visualization
                            fig = px.pie(
                                results,
                                names='Prediction',
                                title='Prediction Distribution',
                                color='Prediction',
                                color_discrete_map={
                                    'CONFIRMED': '#FF6B6B',
                                    'CANDIDATE': '#45B7D1',
                                    'FALSE POSITIVE': '#4ECDC4'
                                }
                            )
                            st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error processing file: {e}")

# Main App
def main():
    """Main application function"""
    # Setup configuration
    setup_page_config()
    initialize_session_state()
    
    # Create necessary directories
    os.makedirs("datasets", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Load data
    raw_data, summary_data = load_data()
    
    # Navigation
    st.sidebar.title("🌌 Exoplanet Detection")
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigation", ["Data Exploration", "Model Training", "Predictions"], index=0)
    
    # Page routing
    if page == "Data Exploration":
        data_exploration_page(raw_data, summary_data)
    elif page == "Model Training":
        model_training_page(raw_data, summary_data)
    elif page == "Predictions":
        predictions_page()

if __name__ == "__main__":
    main()
