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
from tabulate import tabulate

# Custom CSS with Tailwind-inspired styling
st.set_page_config(
    page_title="Exoplanet Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🌌"
)
st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
    .main {
        background-color: #1a202c !important;
        color: #e2e8f0;
    }
    .stButton>button {
        @apply bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700;
    }
    .stTabs {
        @apply bg-gray-800 text-white p-4 rounded-lg;
    }
    .card {
        @apply bg-gray-700 p-4 rounded-lg shadow-lg;
    }
    .plot-container {
        @apply p-4 bg-gray-800 rounded-lg;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
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

# Sidebar for navigation
st.sidebar.title("Exoplanet Detection Dashboard")
page = st.sidebar.radio("Select Page", ["Data Exploration", "Model Training", "Predictions"])

# Create directories
os.makedirs("datasets", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Load datasets
@st.cache_data
def load_data():
    try:
        raw_data = pd.read_csv("datasets/exoTrain.csv")
        summary_data = pd.read_csv("datasets/cumulative.csv")
        return raw_data, summary_data
    except FileNotFoundError:
        st.error("Dataset files not found in 'datasets/' directory.")
        st.stop()
    return None, None

raw_data, summary_data = load_data()

if page == "Data Exploration":
    st.title("Exoplanet Data Exploration")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    dataset_choice = st.selectbox("Select Dataset", ["exoTrain.csv", "cumulative.csv"])
    if dataset_choice == "exoTrain.csv":
        st.write("Light curve data for CNN model")
        st.write(raw_data.head())
        st.subheader("Basic Statistics")
        st.write(raw_data.describe())
        tab1, tab2 = st.tabs(["Light Curve Plot", "Class Distribution"])
        with tab1:
            if st.checkbox("Show Light Curve Plot"):
                star_id = st.selectbox("Select Star ID", raw_data.index)
                plt.figure(figsize=(10, 4))
                plt.plot(raw_data.drop(columns=['LABEL']).iloc[star_id], label=f"Star {star_id}", color="#4CAF50")
                plt.xlabel("Time")
                plt.ylabel("Flux")
                plt.legend()
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.pyplot(plt)
                plt.close()
                st.markdown('</div>', unsafe_allow_html=True)
        with tab2:
            class_counts = raw_data['LABEL'].value_counts()
            plt.figure(figsize=(6, 4))
            plt.bar(class_counts.index, class_counts.values, color=["#ff4444", "#4CAF50"])
            plt.xlabel("Class (1=Exoplanet, 2=No Exoplanet)")
            plt.ylabel("Count")
            plt.title("Class Distribution")
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.pyplot(plt)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.write("Summary data for XGBoost model")
        st.write(summary_data.head())
        st.subheader("Basic Statistics")
        st.write(summary_data.describe())
        tab1, tab2 = st.tabs(["Feature Distribution", "Feature Importance Preview"])
        with tab1:
            feature = st.selectbox("Select Feature", summary_data.select_dtypes(include=[np.number]).columns)
            plt.figure(figsize=(10, 4))
            sns.histplot(summary_data[feature].dropna(), kde=True, color="#4CAF50")
            plt.xlabel(feature)
            plt.ylabel("Count")
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.pyplot(plt)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)
        with tab2:
            st.write("Feature importance will be available after training the XGBoost model.")
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Model Training":
    st.title("Model Training")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    model_choice = st.selectbox("Select Model", ["CNN", "XGBoost"])
    
    if model_choice == "CNN":
        st.subheader("CNN Model")
        if st.button("Load Pre-Trained CNN Model"):
            try:
                st.session_state.cnn_model = load_model('models/cnn_exoplanet_model.keras', custom_objects={'focal_loss_fixed': focal_loss(gamma=2.0, alpha=0.5)})
                st.session_state.scaler_cnn = joblib.load('models/scaler_cnn.pkl')
                st.success("Pre-trained CNN model and scaler loaded successfully!")
            except (FileNotFoundError, EOFError, pickle.PicklingError, pickle.UnpicklingError, AttributeError, ValueError) as e:
                st.error(f"Error loading CNN model or scaler: {e}")
        
        if st.button("Train CNN Model"):
            try:
                X_cnn = raw_data.drop(columns=['LABEL']).values
                y_cnn = raw_data['LABEL'].values - 1
                scaler_cnn = StandardScaler()
                X_cnn_scaled = scaler_cnn.fit_transform(X_cnn)
                X_cnn = X_cnn_scaled.reshape(X_cnn_scaled.shape[0], X_cnn_scaled.shape[1], 1)
                
                confirmed_indices = np.where(y_cnn == 1)[0]
                X_confirmed = X_cnn[confirmed_indices]
                y_confirmed = y_cnn[confirmed_indices]
                oversample_factor = 10
                X_train_oversampled = np.vstack([X_cnn] + [X_confirmed] * oversample_factor)
                y_train_oversampled = np.hstack([y_cnn] + [y_confirmed] * oversample_factor)
                
                class_weights = compute_class_weight('balanced', classes=np.unique(y_train_oversampled), y=y_train_oversampled)
                class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
                
                inputs = Input(shape=(X_cnn.shape[1], 1))
                x = Conv1D(8, kernel_size=7, activation='relu', kernel_regularizer=l2(0.02))(inputs)
                x = BatchNormalization()(x)
                x = MaxPooling1D(pool_size=4)(x)
                x = Conv1D(16, kernel_size=7, activation='relu', kernel_regularizer=l2(0.02))(x)
                x = BatchNormalization()(x)
                x = MaxPooling1D(pool_size=4)(x)
                x = GlobalAveragePooling1D()(x)
                x = Dense(32, activation='relu', kernel_regularizer=l2(0.02))(x)
                x = Dropout(0.3)(x)
                outputs = Dense(1, activation='sigmoid')(x)
                model = Model(inputs, outputs)
                
                model.compile(optimizer='adam', loss=focal_loss(gamma=2.0, alpha=0.5), metrics=['accuracy'])
                early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
                lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
                
                history = model.fit(
                    X_train_oversampled, y_train_oversampled,
                    validation_split=0.2,
                    epochs=30, batch_size=128,
                    callbacks=[early_stopping, lr_scheduler],
                    class_weight=class_weight_dict,
                    verbose=1
                )
                
                model.save('models/cnn_exoplanet_model.keras')
                joblib.dump(scaler_cnn, 'models/scaler_cnn.pkl')
                st.session_state.cnn_model = model
                st.session_state.scaler_cnn = scaler_cnn
                st.success("CNN model trained and saved successfully!")
                
                plt.figure(figsize=(10, 4))
                plt.plot(history.history['loss'], label='Training Loss', color="#4CAF50")
                plt.plot(history.history['val_loss'], label='Validation Loss', color="#ff4444")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.pyplot(plt)
                plt.close()
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Training failed: {str(e)}")
    
    else:
        st.subheader("XGBoost Model")
        if st.button("Load Pre-Trained XGBoost Model"):
            try:
                st.session_state.xgb_model = joblib.load('models/xgb_model.pkl')
                st.session_state.scaler_xgb = joblib.load('models/scaler_xgb.pkl')
                st.session_state.label_encoder = joblib.load('models/label_encoder.pkl')
                st.session_state.features_to_use = joblib.load('models/features_to_use.pkl')
                st.success("Pre-trained XGBoost model, scaler, label encoder, and features loaded successfully!")
            except (FileNotFoundError, EOFError, pickle.PicklingError, pickle.UnpicklingError, AttributeError) as e:
                st.error(f"Error loading XGBoost model, scaler, or features: {e}")
        
        if st.button("Train XGBoost Model"):
            try:
                st.session_state.features_to_use = joblib.load('models/features_to_use.pkl')
            except (FileNotFoundError, EOFError, pickle.PicklingError, pickle.UnpicklingError, AttributeError):
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
            X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
                X_xgb, y_xgb, test_size=0.3, stratify=y_xgb, random_state=42
            )
            
            scaler_xgb = StandardScaler()
            X_train_xgb = scaler_xgb.fit_transform(X_train_xgb)
            X_test_xgb = scaler_xgb.transform(X_test_xgb)
            
            xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42)
            xgb_model.fit(X_train_xgb, y_train_xgb)
            
            joblib.dump(xgb_model, 'models/xgb_model.pkl')
            joblib.dump(scaler_xgb, 'models/scaler_xgb.pkl')
            st.session_state.xgb_model = xgb_model
            st.session_state.scaler_xgb = scaler_xgb
            st.success("XGBoost model trained and saved successfully!")
            
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X_test_xgb)
            plt.figure(figsize=(10, 4))
            shap.summary_plot(shap_values, X_test_xgb, feature_names=st.session_state.features_to_use, show=False, color_bar=False)
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.pyplot(plt)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Feature Importance Bar Chart
            feature_importance = pd.DataFrame({
                'feature': st.session_state.features_to_use,
                'importance': xgb_model.feature_importances_
            }).sort_values('importance', ascending=False)
            plt.figure(figsize=(10, 4))
            plt.bar(feature_importance['feature'], feature_importance['importance'], color="#4CAF50")
            plt.xlabel("Features")
            plt.ylabel("Importance")
            plt.title("Feature Importance")
            plt.xticks(rotation=45, ha='right')
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.pyplot(plt)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Predictions":
    st.title("Exoplanet Predictions")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    model_choice = st.selectbox("Select Model for Prediction", ["CNN", "XGBoost"])
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        if model_choice == "CNN":
            if st.session_state.cnn_model is None or st.session_state.scaler_cnn is None:
                st.error("Please load or train the CNN model first.")
            else:
                try:
                    X = data.drop(columns=['LABEL'] if 'LABEL' in data.columns else [], errors='ignore').values
                    X_scaled = st.session_state.scaler_cnn.transform(X)
                    X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
                    predictions = st.session_state.cnn_model.predict(X_reshaped)
                    predictions = (predictions > 0.5).astype(int).flatten()
                    results = pd.DataFrame({
                        'Star ID': data.index,
                        'Prediction': ['Exoplanet' if p == 1 else 'Non-Exoplanet' for p in predictions]
                    })
                    st.write("CNN Predictions:")
                    st.write(tabulate(results, headers='keys', tablefmt='psql', showindex=False))
                except Exception as e:
                    st.error(f"Error making CNN predictions: {e}")
        
        else:
            if st.session_state.xgb_model is None or st.session_state.scaler_xgb is None or st.session_state.label_encoder is None:
                st.error("Please load or train the XGBoost model first.")
            else:
                try:
                    X = data[st.session_state.features_to_use]
                    X = X.dropna(axis=1, how='all').replace([np.inf, -np.inf], np.nan)
                    X = X.fillna(X.mean(numeric_only=True))
                    X_scaled = st.session_state.scaler_xgb.transform(X)
                    predictions = st.session_state.xgb_model.predict(X_scaled)
                    predictions = st.session_state.label_encoder.inverse_transform(predictions)
                    results = pd.DataFrame({
                        'Star ID': data.index,
                        'Prediction': predictions
                    })
                    st.write("XGBoost Predictions:")
                    st.write(tabulate(results, headers='keys', tablefmt='psql', showindex=False))
                except Exception as e:
                    st.error(f"Error making XGBoost predictions: {e}")
    st.markdown('</div>', unsafe_allow_html=True)
