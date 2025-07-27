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
import shap
import os
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Custom CSS and page config
st.set_page_config(page_title="Exoplanet Detection Dashboard", layout="wide", initial_sidebar_state="expanded", page_icon="🌌")
st.markdown("""
    <style>
    :root { background-color: #f0f2f6 !important; }
    .main { background-color: black !important; }
    .stButton>button { border-radius: 8px; height: 40px; width: 100%; background-color: #4CAF50 !important; color: white !important; font-weight: bold; }
    .stButton>button:hover { background-color: #45a049 !important; }
    </style>
""", unsafe_allow_html=True)

# Logging setup
logging.basicConfig(level=logging.INFO)

# Initialize session state
for key in ['cnn_model', 'xgb_model', 'scaler_cnn', 'scaler_xgb', 'label_encoder', 'features_to_use']:
    if key not in st.session_state:
        st.session_state[key] = None

# Focal Loss
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

# Data loading with fallback
@st.cache_data
def load_data():
    if os.path.exists("datasets/exoTrain.csv") and os.path.exists("datasets/cumulative.csv"):
        return pd.read_csv("datasets/exoTrain.csv"), pd.read_csv("datasets/cumulative.csv")
    uploaded_files = st.file_uploader("Upload exoTrain.csv and cumulative.csv", type=["csv"], accept_multiple_files=True)
    if uploaded_files and len(uploaded_files) == 2:
        raw_data = pd.read_csv(uploaded_files[0])
        summary_data = pd.read_csv(uploaded_files[1])
        os.makedirs("datasets", exist_ok=True)
        raw_data.to_csv("datasets/exoTrain.csv", index=False)
        summary_data.to_csv("datasets/cumulative.csv", index=False)
        return raw_data, summary_data
    st.error("Please upload both dataset files.")
    st.stop()

raw_data, summary_data = load_data()

# Light curve visualization
def plot_light_curves(data, n_samples=3):
    exoplanet_samples = data[data['LABEL'] == 2].sample(n=min(n_samples, len(data[data['LABEL'] == 2])))
    non_exoplanet_samples = data[data['LABEL'] == 1].sample(n=min(n_samples, len(data[data['LABEL'] == 1])))
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Exoplanet Stars", "Non-Exoplanet Stars"))
    for idx, row in exoplanet_samples.iterrows():
        fig.add_trace(go.Scatter(x=range(len(row[1:])), y=row[1:], mode='lines', line=dict(color='#FF6B6B')), row=1, col=1)
    for idx, row in non_exoplanet_samples.iterrows():
        fig.add_trace(go.Scatter(x=range(len(row[1:])), y=row[1:], mode='lines', line=dict(color='#4ECDC4')), row=1, col=2)
    fig.update_layout(height=400, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=1, col=2)
    fig.update_yaxes(title_text="Flux", row=1, col=1)
    return fig

# Page navigation
page = st.sidebar.radio("Navigation", ["Data Exploration", "Model Training", "Predictions"], index=0)

if page == "Data Exploration":
    st.title("📊 Data Exploration")
    st.subheader("Light Curve Data")
    n_samples = st.slider("Number of samples to display", 1, 10, 3)
    fig = plot_light_curves(raw_data, n_samples)
    st.plotly_chart(fig, use_container_width=True)

elif page == "Model Training":
    st.title("🤖 Model Training")
    tab1, tab2 = st.tabs(["CNN Model", "XGBoost Model"])

    with tab1:
        st.header("Convolutional Neural Network")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Load Pre-Trained CNN Model"):
                try:
                    st.session_state.cnn_model = load_model('models/cnn_exoplanet_model.keras', custom_objects={'focal_loss_fixed': focal_loss()})
                    st.session_state.scaler_cnn = joblib.load('models/scaler_cnn.pkl')
                    st.success("Pre-trained CNN model loaded!")
                except Exception as e:
                    st.error(f"Error: {e}")
        with col2:
            if st.button("🔧 Train CNN Model"):
                with st.spinner("Training CNN model..."):
                    try:
                        X_cnn = raw_data.drop(columns=['LABEL']).values
                        y_cnn = raw_data['LABEL'].values - 1
                        X_train, X_val, y_train, y_val = train_test_split(X_cnn, y_cnn, test_size=0.2, stratify=y_cnn, random_state=42)
                        scaler_cnn = StandardScaler()
                        X_train_scaled = scaler_cnn.fit_transform(X_train)
                        X_val_scaled = scaler_cnn.transform(X_val)
                        X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
                        X_val_reshaped = X_val_scaled.reshape(X_val_scaled.shape[0], X_val_scaled.shape[1], 1)

                        confirmed_indices = np.where(y_train == 1)[0]
                        X_confirmed = X_train_reshaped[confirmed_indices]
                        y_confirmed = y_train[confirmed_indices]
                        oversample_factor = 5
                        X_train_oversampled = np.vstack([X_train_reshaped] + [X_confirmed] * oversample_factor)
                        y_train_oversampled = np.hstack([y_train] + [y_confirmed] * oversample_factor)

                        class_weight_dict = {i: w for i, w in enumerate(compute_class_weight('balanced', classes=np.unique(y_train_oversampled), y=y_train_oversampled))}

                        inputs = Input(shape=(X_train_reshaped.shape[1], 1))
                        x = Conv1D(4, kernel_size=7, activation='relu', kernel_regularizer=l2(0.02))(inputs)
                        x = BatchNormalization()(x)
                        x = MaxPooling1D(pool_size=4)(x)
                        x = Conv1D(8, kernel_size=7, activation='relu', kernel_regularizer=l2(0.02))(x)
                        x = BatchNormalization()(x)
                        x = MaxPooling1D(pool_size=4)(x)
                        x = GlobalAveragePooling1D()(x)
                        x = Dense(16, activation='relu', kernel_regularizer=l2(0.02))(x)
                        x = Dropout(0.3)(x)
                        outputs = Dense(1, activation='sigmoid')(x)
                        model = Model(inputs, outputs)

                        model.compile(optimizer='adam', loss=focal_loss(), metrics=['accuracy'])
                        callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                                   ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)]

                        history = model.fit(X_train_oversampled, y_train_oversampled, validation_data=(X_val_reshaped, y_val),
                                          epochs=10, batch_size=256, callbacks=callbacks, class_weight=class_weight_dict, verbose=1)

                        os.makedirs("models", exist_ok=True)
                        model.save('models/cnn_exoplanet_model.keras')
                        joblib.dump(scaler_cnn, 'models/scaler_cnn.pkl')
                        st.session_state.cnn_model = model
                        st.session_state.scaler_cnn = scaler_cnn
                        st.success("CNN model trained and saved!")

                        # Training curves
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(y=history.history['loss'], mode='lines', name='Training Loss', line=dict(color='#FF6B6B')))
                        fig.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines', name='Validation Loss', line=dict(color='#4ECDC4')))
                        fig.add_trace(go.Scatter(y=history.history['accuracy'], mode='lines', name='Training Accuracy', line=dict(color='#FFB6C1')))
                        fig.add_trace(go.Scatter(y=history.history['val_accuracy'], mode='lines', name='Validation Accuracy', line=dict(color='#87CEEB')))
                        fig.update_layout(title='Training History', xaxis_title='Epoch', yaxis_title='Metrics')
                        st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")

        if st.session_state.cnn_model:
            st.subheader("Model Architecture")
            model_summary = []
            st.session_state.cnn_model.summary(print_fn=lambda x: model_summary.append(x))
            st.text("\n".join(model_summary))

    with tab2:
        st.header("XGBoost Classifier")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Load Pre-Trained XGBoost Model"):
                try:
                    st.session_state.xgb_model = joblib.load('models/xgb_model.pkl')
                    st.session_state.scaler_xgb = joblib.load('models/scaler_xgb.pkl')
                    st.session_state.label_encoder = joblib.load('models/label_encoder.pkl')
                    st.session_state.features_to_use = joblib.load('models/features_to_use.pkl')
                    st.success("XGBoost model loaded!")
                except Exception as e:
                    st.error(f"Error: {e}")
        with col2:
            if st.button("🔧 Train XGBoost Model"):
                with st.spinner("Training XGBoost model..."):
                    try:
                        if st.session_state.features_to_use is None:
                            numeric_columns = summary_data.select_dtypes(include=[np.number]).columns
                            exclude_columns = ['kepid', 'koi_disposition', 'label_encoded']
                            st.session_state.features_to_use = [col for col in numeric_columns if col not in exclude_columns]
                            if not st.session_state.features_to_use:
                                st.error("No numeric features found.")
                                st.stop()

                        X_xgb = summary_data[st.session_state.features_to_use].replace([np.inf, -np.inf], np.nan).fillna(summary_data[st.session_state.features_to_use].mean(numeric_only=True))
                        if 'label_encoded' not in summary_data.columns:
                            st.session_state.label_encoder = LabelEncoder()
                            summary_data['label_encoded'] = st.session_state.label_encoder.fit_transform(summary_data['koi_disposition'])
                            joblib.dump(st.session_state.label_encoder, 'models/label_encoder.pkl')

                        y_xgb = summary_data['label_encoded']
                        X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(X_xgb, y_xgb, test_size=0.3, stratify=y_xgb, random_state=42)
                        scaler_xgb = StandardScaler()
                        X_train_xgb = scaler_xgb.fit_transform(X_train_xgb)
                        X_test_xgb = scaler_xgb.transform(X_test_xgb)

                        xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42,
                                                    n_estimators=100, max_depth=5, learning_rate=0.1)
                        xgb_model.fit(X_train_xgb, y_train_xgb)

                        os.makedirs("models", exist_ok=True)
                        joblib.dump(xgb_model, 'models/xgb_model.pkl')
                        joblib.dump(scaler_xgb, 'models/scaler_xgb.pkl')
                        joblib.dump(st.session_state.features_to_use, 'models/features_to_use.pkl')
                        st.session_state.xgb_model = xgb_model
                        st.session_state.scaler_xgb = scaler_xgb
                        st.success("XGBoost model trained and saved!")

                        # Feature importance
                        fig = go.Figure(data=[go.Bar(x=st.session_state.features_to_use, y=xgb_model.feature_importances_)])
                        fig.update_layout(title='Feature Importance', xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)

                        # SHAP visualization
                        explainer = shap.TreeExplainer(xgb_model)
                        shap_values = explainer.shap_values(X_test_xgb)
                        fig, ax = plt.subplots()
                        shap.summary_plot(shap_values, X_test_xgb, feature_names=st.session_state.features_to_use, show=False)
                        st.pyplot(fig)
                        plt.close()

                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")

elif page == "Predictions":
    st.title("🔮 Predictions")
    model_choice = st.selectbox("Select Model for Prediction", ["CNN", "XGBoost"])
    uploaded_file = st.file_uploader("Upload CSV file for prediction", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.success(f"Uploaded {uploaded_file.name} with {len(data)} records")

        if model_choice == "CNN":
            if st.session_state.cnn_model is None or st.session_state.scaler_cnn is None:
                st.error("Load or train CNN model first.")
            else:
                try:
                    X = data.drop(columns=['LABEL'] if 'LABEL' in data.columns else [], errors='ignore').values
                    X_scaled = st.session_state.scaler_cnn.transform(X)
                    if X_scaled.shape[1] != st.session_state.cnn_model.input_shape[1]:
                        st.error(f"Input shape mismatch: {X_scaled.shape[1]} vs {st.session_state.cnn_model.input_shape[1]}")
                    else:
                        X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
                        predictions = (st.session_state.cnn_model.predict(X_reshaped) > 0.5).astype(int).flatten()
                        results = pd.DataFrame({
                            'Star ID': data.index,
                            'Prediction': ['Exoplanet' if p == 1 else 'Non-Exoplanet' for p in predictions],
                            'Confidence': [f"{max(pred, 1-pred)*100:.2f}%" for pred in st.session_state.cnn_model.predict(X_reshaped)]
                        })
                        st.dataframe(results)
                except Exception as e:
                    st.error(f"Prediction error: {e}")
        else:
            if st.session_state.xgb_model is None or st.session_state.scaler_xgb is None or st.session_state.label_encoder is None:
                st.error("Load or train XGBoost model first.")
            else:
                try:
                    X = data[st.session_state.features_to_use].replace([np.inf, -np.inf], np.nan).fillna(data[st.session_state.features_to_use].mean(numeric_only=True))
                    X_scaled = st.session_state.scaler_xgb.transform(X)
                    predictions = st.session_state.label_encoder.inverse_transform(st.session_state.xgb_model.predict(X_scaled))
                    prediction_probs = st.session_state.xgb_model.predict_proba(X_scaled)
                    results = pd.DataFrame({
                        'Star ID': data.index,
                        'Prediction': predictions,
                        'Confidence': [f"{max(probs)*100:.2f}%" for probs in prediction_probs]
                    })
                    st.dataframe(results)
                except Exception as e:
                    st.error(f"Prediction error: {e}")
