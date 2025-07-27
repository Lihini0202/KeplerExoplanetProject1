import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.utils import class_weight, resample
from sklearn.model_selection import train_test_split
import xgboost as xgb
from tabulate import tabulate
from sklearn.utils.class_weight import compute_class_weight
import shap
import joblib
import os

# Set page config
st.set_page_config(page_title="Exoplanet Detection", layout="wide")

# Title
st.title("Exoplanet Detection from Kepler Light Curves and Expert Features")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Exploration", "Model Training", "Predictions"])

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.model_type = None
    st.session_state.scaler_cnn = None
    st.session_state.scaler_xgb = None
    st.session_state.label_encoder = None
    st.session_state.model_metrics = []
    st.session_state.features_to_use = None

# Focal Loss Function
def focal_loss(gamma=1.0, alpha=0.75):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
        bce = -y_true * tf.math.log(y_pred) - (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        focal_factor = tf.pow(1.0 - p_t, gamma)
        alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        return tf.reduce_mean(alpha_t * focal_factor * bce)
    return focal_loss_fixed

# Data augmentation function
def augment_data(X, y, sigma=0.04, max_shift=200):
    X_aug = X.copy()
    noise = np.random.normal(0, sigma, X.shape)
    X_aug += noise
    shifts = np.random.randint(-max_shift, max_shift, size=X.shape[0])
    for i in range(X.shape[0]):
        X_aug[i] = np.roll(X_aug[i], shifts[i], axis=0)
    return X_aug, y

# Bootstrap evaluation function
def bootstrap_metrics(y_true, y_pred_proba, n_bootstraps=1000):
    aucs, recalls, precisions, f1s = [], [], [], []
    n_samples = len(y_true)
    for _ in range(n_bootstraps):
        idx = resample(np.arange(n_samples), stratify=y_true)
        y_true_boot = y_true[idx]
        y_pred_proba_boot = y_pred_proba[idx]
        y_pred_boot = (y_pred_proba_boot > 0.3).astype(int)
        aucs.append(roc_auc_score(y_true_boot, y_pred_proba_boot))
        report = classification_report(y_true_boot, y_pred_boot, output_dict=True, zero_division=0).get('1', {})
        recalls.append(report.get('recall', 0))
        precisions.append(report.get('precision', 0))
        f1s.append(report.get('f1-score', 0))
    auc_ci = np.percentile(aucs, [2.5, 97.5])
    recall_ci = np.percentile(recalls, [2.5, 97.5])
    precision_ci = np.percentile(precisions, [2.5, 97.5])
    f1_ci = np.percentile(f1s, [2.5, 97.5])
    return (
        np.mean(aucs), auc_ci,
        np.mean(recalls), recall_ci,
        np.mean(precisions), precision_ci,
        np.mean(f1s), f1_ci
    )

# Grad-CAM implementation
def apply_grad_cam(model, input_data, class_index=None, layer_name='conv1d_2'):
    input_data = np.expand_dims(input_data, axis=0)
    inputs = model.input
    conv_layer = model.get_layer(layer_name)
    grad_model = Model(inputs=inputs, outputs=[conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_data)
        if class_index is None:
            class_index = 1 if predictions[0] >= 0.3 else 0
        loss = predictions if class_index == 1 else 1.0 - predictions
    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=0)
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8
    heatmap_resized = np.interp(np.arange(3197), np.linspace(0, 3197, len(heatmap)), heatmap)
    return heatmap_resized, class_idx

# Load data
@st.cache_data
def load_data():
    try:
        summary_data = pd.read_csv("datasets/cumulative.csv")
        raw_train_data = pd.read_csv("datasets/exoTrain.csv")
        raw_test_data = pd.read_csv("datasets/exoTest.csv")
        return summary_data, raw_train_data, raw_test_data
    except FileNotFoundError:
        st.error("ERROR: Dataset files not found.")
        return None, None, None

summary_data, raw_train_data, raw_test_data = load_data()

if page == "Data Exploration":
    st.header("Data Exploration")
    
    if summary_data is not None and raw_train_data is not None and raw_test_data is not None:
        st.success("Datasets Loaded Successfully!")
        st.write(f"Summary Dataset Shape: {summary_data.shape}")
        st.write(f"Train Light Curves Shape: {raw_train_data.shape}")
        st.write(f"Test Light Curves Shape: {raw_test_data.shape}")
        
        st.subheader("First 5 Rows from Summary Data:")
        st.dataframe(summary_data.head())
        
        st.subheader("First 5 Rows from Raw Training Data:")
        st.dataframe(raw_train_data.head())
        
        st.subheader("Sample Light Curves")
        num_samples = st.slider("Number of samples to display", 1, 5, 3)
        sample_indices = np.random.choice(raw_train_data.shape[0], num_samples, replace=False)
        
        for i, idx in enumerate(sample_indices):
            label = raw_train_data.iloc[idx]['LABEL']
            flux_values = raw_train_data.iloc[idx, 1:].values
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(flux_values, label=f"Sample #{idx} | Label: {'CONFIRMED' if label == 2 else 'FALSE POSITIVE'}")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Flux (Brightness)")
            ax.set_title(f"Light Curve for Sample #{idx}")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

elif page == "Model Training":
    st.header("Model Training")
    
    if raw_train_data is not None and raw_test_data is not None and summary_data is not None:
        st.subheader("Data Preparation")
        
        # Load label encoder
        try:
            st.session_state.label_encoder = joblib.load('models/label_encoder.pkl')
        except FileNotFoundError:
            st.error("Label encoder not found. Please ensure 'models/label_encoder.pkl' is available.")
            st.stop()
        
        # Preprocessing for CNN
        X_train_cnn = raw_train_data.drop(columns=['LABEL']).values
        y_train_cnn = raw_train_data['LABEL'].values - 1
        X_test_cnn = raw_test_data.drop(columns=['LABEL']).values
        y_test_cnn = raw_test_data['LABEL'].values - 1
        try:
            st.session_state.scaler_cnn = joblib.load('models/scaler_cnn.pkl')
            X_train_cnn = st.session_state.scaler_cnn.transform(X_train_cnn)
            X_test_cnn = st.session_state.scaler_cnn.transform(X_test_cnn)
        except FileNotFoundError:
            st.warning("CNN scaler not found. Using new scaler (results may differ).")
            st.session_state.scaler_cnn = StandardScaler()
            X_train_cnn = st.session_state.scaler_cnn.fit_transform(X_train_cnn)
            X_test_cnn = st.session_state.scaler_cnn.transform(X_test_cnn)
        
        # Preprocessing for XGBoost
        numeric_columns = summary_data.select_dtypes(include=[np.number]).columns
        exclude_columns = ['kepid', 'koi_disposition', 'label_encoded']
        st.session_state.features_to_use = [col for col in numeric_columns if col not in exclude_columns]
        if not st.session_state.features_to_use:
            st.error("No numeric features found in summary_data.")
            st.stop()
        
        if 'label_encoded' not in summary_data.columns:
            summary_data['label_encoded'] = st.session_state.label_encoder.fit_transform(summary_data['koi_disposition'])
        X_xgb = summary_data[st.session_state.features_to_use].fillna(summary_data[st.session_state.features_to_use].mean(numeric_only=True))
        y_xgb = summary_data['label_encoded']
        X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
            X_xgb, y_xgb, test_size=0.3, stratify=y_xgb, random_state=42
        )
        try:
            st.session_state.scaler_xgb = joblib.load('models/scaler_xgb.pkl')
            X_train_xgb = st.session_state.scaler_xgb.transform(X_train_xgb)
            X_test_xgb = st.session_state.scaler_xgb.transform(X_test_xgb)
        except FileNotFoundError:
            st.warning("XGBoost scaler not found. Using new scaler (results may differ).")
            st.session_state.scaler_xgb = StandardScaler()
            X_train_xgb = st.session_state.scaler_xgb.fit_transform(X_train_xgb)
            X_test_xgb = st.session_state.scaler_xgb.transform(X_test_xgb)
        
        st.write(f"CNN Training set shape: {X_train_cnn.shape}, Test set shape: {X_test_cnn.shape}")
        st.write(f"XGBoost Training set shape: {X_train_xgb.shape}, Test set shape: {X_test_xgb.shape}")
        
        st.subheader("Model Selection")
        model_type = st.selectbox("Select model type", ["CNN", "XGBoost"])
        
        if model_type == "CNN":
            st.write("### CNN Model Architecture")
            st.code("""
            inputs = Input(shape=(3197, 1), name='input_layer')
            x = Conv1D(16, kernel_size=7, activation='relu', kernel_regularizer=l2(0.01))(inputs)
            x = BatchNormalization()(x)
            x = MaxPooling1D(pool_size=4)(x)
            x = Conv1D(32, kernel_size=7, activation='relu', kernel_regularizer=l2(0.01))(x)
            x = BatchNormalization()(x)
            x = MaxPooling1D(pool_size=4)(x)
            x = Conv1D(64, kernel_size=7, activation='relu', kernel_regularizer=l2(0.01))(x)
            x = BatchNormalization()(x)
            x = MaxPooling1D(pool_size=4)(x)
            x = GlobalAveragePooling1D()(x)
            x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
            x = Dropout(0.3)(x)
            outputs = Dense(1, activation='sigmoid')(x)
            model = Model(inputs, outputs)
            """, language='python')
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Train CNN Model (Lightweight)"):
                    with st.spinner("Training lightweight CNN model..."):
                        try:
                            subset_idx = np.random.choice(len(X_train_cnn), size=int(0.1 * len(X_train_cnn)), replace=False)
                            X_train_subset = X_train_cnn[subset_idx]
                            y_train_subset = y_train_cnn[subset_idx]
                            X_train_subset = X_train_subset.reshape(-1, 3197, 1)
                            X_test_cnn_reshaped = X_test_cnn.reshape(-1, 3197, 1)
                            
                            confirmed_indices = np.where(y_train_subset == 1)[0]
                            X_confirmed = X_train_subset[confirmed_indices]
                            y_confirmed = y_train_subset[confirmed_indices]
                            oversample_factor = 15
                            X_train_oversampled = np.vstack([X_train_subset] + [X_confirmed] * oversample_factor)
                            y_train_oversampled = np.hstack([y_train_subset] + [y_confirmed] * oversample_factor)
                            
                            if st.checkbox("Apply data augmentation (Warning: Increases memory usage)", key="cnn_aug"):
                                X_train_oversampled, y_train_oversampled = augment_data(X_train_oversampled, y_train_oversampled, sigma=0.04, max_shift=200)
                            
                            class_weights = compute_class_weight('balanced', classes=np.unique(y_train_oversampled), y=y_train_oversampled)
                            class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
                            
                            inputs = Input(shape=(3197, 1), name='input_layer')
                            x = Conv1D(16, kernel_size=7, activation='relu', kernel_regularizer=l2(0.01))(inputs)
                            x = BatchNormalization()(x)
                            x = MaxPooling1D(pool_size=4)(x)
                            x = Conv1D(32, kernel_size=7, activation='relu', kernel_regularizer=l2(0.01))(x)
                            x = BatchNormalization()(x)
                            x = MaxPooling1D(pool_size=4)(x)
                            x = Conv1D(64, kernel_size=7, activation='relu', kernel_regularizer=l2(0.01))(x)
                            x = BatchNormalization()(x)
                            x = MaxPooling1D(pool_size=4)(x)
                            x = GlobalAveragePooling1D()(x)
                            x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
                            x = Dropout(0.3)(x)
                            outputs = Dense(1, activation='sigmoid')(x)
                            model = Model(inputs, outputs)
                            
                            model.compile(optimizer='adam', loss=focal_loss(gamma=1.0, alpha=0.75), metrics=['accuracy'])
                            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                            lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
                            history = model.fit(
                                X_train_oversampled, y_train_oversampled,
                                validation_data=(X_test_cnn_reshaped, y_test_cnn),
                                epochs=10, batch_size=16,
                                callbacks=[early_stopping, lr_scheduler],
                                class_weight=class_weight_dict,
                                verbose=0
                            )
                            st.session_state.model = model
                            st.session_state.model_type = "CNN"
                            st.session_state.history = history
                            
                            y_pred_proba = model.predict(X_test_cnn_reshaped, verbose=0).flatten()
                            y_pred = (y_pred_proba > 0.3).astype(int)
                            accuracy = accuracy_score(y_test_cnn, y_pred)
                            auc = roc_auc_score(y_test_cnn, y_pred_proba)
                            report = classification_report(y_test_cnn, y_pred, output_dict=True, zero_division=0)
                            precision = report.get('1', {}).get('precision', 0)
                            recall = report.get('1', {}).get('recall', 0)
                            f1 = report.get('1', {}).get('f1-score', 0)
                            auc_mean, auc_ci, recall_mean, recall_ci, precision_mean, precision_ci, f1_mean, f1_ci = bootstrap_metrics(y_test_cnn, y_pred_proba)
                            
                            st.session_state.model_metrics.append(["CNN", accuracy * 100, auc, precision, recall, f1])
                            
                            st.success("CNN Training Complete!")
                            st.write("### CNN Performance Metrics")
                            st.write(f"Accuracy: {accuracy * 100:.2f}%")
                            st.write(f"AUC: {auc_mean:.4f} (95% CI: {auc_ci[0]:.4f} - {auc_ci[1]:.4f})")
                            st.write(f"CONFIRMED Precision: {precision_mean:.4f} (95% CI: {precision_ci[0]:.4f} - {precision_ci[1]:.4f})")
                            st.write(f"CONFIRMED Recall: {recall_mean:.4f} (95% CI: {recall_ci[0]:.4f} - {recall_ci[1]:.4f})")
                            st.write(f"CONFIRMED F1-Score: {f1_mean:.4f} (95% CI: {f1_ci[0]:.4f} - {f1_ci[1]:.4f})")
                            
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                            ax1.plot(history.history['loss'], label='Train Loss')
                            ax1.plot(history.history['val_loss'], label='Validation Loss')
                            ax1.set_title('Model Loss')
                            ax1.set_xlabel('Epoch')
                            ax1.set_ylabel('Loss')
                            ax1.legend()
                            ax2.plot(history.history['accuracy'], label='Train Accuracy')
                            ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
                            ax2.set_title('Model Accuracy')
                            ax2.set_xlabel('Epoch')
                            ax2.set_ylabel('Accuracy')
                            ax2.legend()
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            fpr, tpr, _ = roc_curve(y_test_cnn, y_pred_proba)
                            fig, ax = plt.subplots(figsize=(6, 4))
                            ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
                            ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
                            ax.set_xlabel('False Positive Rate')
                            ax.set_ylabel('True Positive Rate')
                            ax.set_title('ROC Curve')
                            ax.legend()
                            st.pyplot(fig)
                            
                            cm = confusion_matrix(y_test_cnn, y_pred)
                            fig, ax = plt.subplots(figsize=(6, 4))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['FALSE', 'CONFIRMED'], yticklabels=['FALSE', 'CONFIRMED'])
                            ax.set_title('CNN Confusion Matrix')
                            ax.set_xlabel('Predicted')
                            ax.set_ylabel('Actual')
                            st.pyplot(fig)
                            
                            st.write("### Grad-CAM Visualization")
                            sample_idx = st.selectbox("Select sample index for Grad-CAM", range(len(X_test_cnn)), key="cnn_gradcam")
                            heatmap, class_idx = apply_grad_cam(model, X_test_cnn_reshaped[sample_idx], layer_name='conv1d_2')
                            fig, ax = plt.subplots(figsize=(14, 4))
                            ax.plot(X_test_cnn[sample_idx], label='Light Curve')
                            ax.plot(heatmap * np.max(X_test_cnn[sample_idx]), label='Grad-CAM Heatmap', alpha=0.5)
                            ax.set_title(f"Grad-CAM for Sample #{sample_idx} (Predicted: {'CONFIRMED' if class_idx == 1 else 'FALSE POSITIVE'})")
                            ax.set_xlabel("Time Step")
                            ax.set_ylabel("Flux / Heatmap")
                            ax.legend()
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Error during training: {e}")
            
            with col2:
                if st.button("Load Pre-Trained CNN Model"):
                    with st.spinner("Loading pre-trained CNN model..."):
                        try:
                            model = tf.keras.models.load_model('models/cnn_exoplanet_model.keras', custom_objects={'focal_loss_fixed': focal_loss(gamma=1.0, alpha=0.75)})
                            st.session_state.model = model
                            st.session_state.model_type = "CNN"
                            X_test_cnn_reshaped = X_test_cnn.reshape(-1, 3197, 1)
                            y_pred_proba = model.predict(X_test_cnn_reshaped, verbose=0).flatten()
                            y_pred = (y_pred_proba > 0.3).astype(int)
                            accuracy = accuracy_score(y_test_cnn, y_pred)
                            auc = roc_auc_score(y_test_cnn, y_pred_proba)
                            report = classification_report(y_test_cnn, y_pred, output_dict=True, zero_division=0)
                            precision = report.get('1', {}).get('precision', 0)
                            recall = report.get('1', {}).get('recall', 0)
                            f1 = report.get('1', {}).get('f1-score', 0)
                            auc_mean, auc_ci, recall_mean, recall_ci, precision_mean, precision_ci, f1_mean, f1_ci = bootstrap_metrics(y_test_cnn, y_pred_proba)
                            st.session_state.model_metrics.append(["CNN", accuracy * 100, auc, precision, recall, f1])
                            st.success("CNN Model Loaded Successfully!")
                            st.write("### CNN Performance Metrics")
                            st.write(f"Accuracy: {accuracy * 100:.2f}%")
                            st.write(f"AUC: {auc_mean:.4f} (95% CI: {auc_ci[0]:.4f} - {auc_ci[1]:.4f})")
                            st.write(f"CONFIRMED Precision: {precision_mean:.4f} (95% CI: {precision_ci[0]:.4f} - {precision_ci[1]:.4f})")
                            st.write(f"CONFIRMED Recall: {recall_mean:.4f} (95% CI: {recall_ci[0]:.4f} - {recall_ci[1]:.4f})")
                            st.write(f"CONFIRMED F1-Score: {f1_mean:.4f} (95% CI: {f1_ci[0]:.4f} - {f1_ci[1]:.4f})")
                            fpr, tpr, _ = roc_curve(y_test_cnn, y_pred_proba)
                            fig, ax = plt.subplots(figsize=(6, 4))
                            ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
                            ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
                            ax.set_xlabel('False Positive Rate')
                            ax.set_ylabel('True Positive Rate')
                            ax.set_title('ROC Curve')
                            ax.legend()
                            st.pyplot(fig)
                            cm = confusion_matrix(y_test_cnn, y_pred)
                            fig, ax = plt.subplots(figsize=(6, 4))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['FALSE', 'CONFIRMED'], yticklabels=['FALSE', 'CONFIRMED'])
                            ax.set_title('CNN Confusion Matrix')
                            ax.set_xlabel('Predicted')
                            ax.set_ylabel('Actual')
                            st.pyplot(fig)
                            st.write("### Grad-CAM Visualization")
                            sample_idx = st.selectbox("Select sample index for Grad-CAM", range(len(X_test_cnn)), key="cnn_gradcam_pretrained")
                            heatmap, class_idx = apply_grad_cam(model, X_test_cnn_reshaped[sample_idx], layer_name='conv1d_2')
                            fig, ax = plt.subplots(figsize=(14, 4))
                            ax.plot(X_test_cnn[sample_idx], label='Light Curve')
                            ax.plot(heatmap * np.max(X_test_cnn[sample_idx]), label='Grad-CAM Heatmap', alpha=0.5)
                            ax.set_title(f"Grad-CAM for Sample #{sample_idx} (Predicted: {'CONFIRMED' if class_idx == 1 else 'FALSE POSITIVE'})")
                            ax.set_xlabel("Time Step")
                            ax.set_ylabel("Flux / Heatmap")
                            ax.legend()
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Error loading model: {e}")
        
        elif model_type == "XGBoost":
            st.write("### XGBoost Model")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Train XGBoost Model"):
                    with st.spinner("Training XGBoost model..."):
                        try:
                            xgb_model = xgb.XGBClassifier(
                                objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42
                            )
                            xgb_model.fit(X_train_xgb, y_train_xgb)
                            st.session_state.model = xgb_model
                            st.session_state.model_type = "XGBoost"
                            y_pred = xgb_model.predict(X_test_xgb)
                            y_pred_proba = xgb_model.predict_proba(X_test_xgb)[:, 1]
                            accuracy = accuracy_score(y_test_xgb, y_pred)
                            auc = roc_auc_score(y_test_xgb, y_pred_proba)
                            report = classification_report(y_test_xgb, y_pred, output_dict=True, zero_division=0)
                            precision = report.get('1', {}).get('precision', 0)
                            recall = report.get('1', {}).get('recall', 0)
                            f1 = report.get('1', {}).get('f1-score', 0)
                            auc_mean, auc_ci, recall_mean, recall_ci, precision_mean, precision_ci, f1_mean, f1_ci = bootstrap_metrics(y_test_xgb, y_pred_proba)
                            st.session_state.model_metrics.append(["XGBoost", accuracy * 100, auc, precision, recall, f1])
                            st.success("XGBoost Training Complete!")
                            st.write("### XGBoost Performance Metrics")
                            st.write(f"Accuracy: {accuracy * 100:.2f}%")
                            st.write(f"AUC: {auc_mean:.4f} (95% CI: {auc_ci[0]:.4f} - {auc_ci[1]:.4f})")
                            st.write(f"CONFIRMED Precision: {precision_mean:.4f} (95% CI: {precision_ci[0]:.4f} - {precision_ci[1]:.4f})")
                            st.write(f"CONFIRMED Recall: {recall_mean:.4f} (95% CI: {recall_ci[0]:.4f} - {recall_ci[1]:.4f})")
                            st.write(f"CONFIRMED F1-Score: {f1_mean:.4f} (95% CI: {f1_ci[0]:.4f} - {f1_ci[1]:.4f})")
                            cm = confusion_matrix(y_test_xgb, y_pred)
                            fig, ax = plt.subplots(figsize=(6, 4))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=st.session_state.label_encoder.classes_, yticklabels=st.session_state.label_encoder.classes_)
                            ax.set_title('XGBoost Confusion Matrix')
                            ax.set_xlabel('Predicted')
                            ax.set_ylabel('Actual')
                            st.pyplot(fig)
                            st.write("### SHAP Feature Importance")
                            explainer = shap.Explainer(xgb_model)
                            shap_values = explainer(X_test_xgb)
                            fig, ax = plt.subplots()
                            shap.summary_plot(shap_values, X_test_xgb, feature_names=st.session_state.features_to_use, plot_type='bar', show=False)
                            plt.title("SHAP Feature Importance - Bar Plot")
                            plt.tight_layout()
                            st.pyplot(fig)
                            fig, ax = plt.subplots()
                            shap.summary_plot(shap_values, X_test_xgb, feature_names=st.session_state.features_to_use, show=False)
                            plt.title("SHAP Feature Importance - Dot Plot")
                            plt.tight_layout()
                            st.pyplot(fig)
                            fpr, tpr, _ = roc_curve(y_test_xgb, y_pred_proba)
                            fig, ax = plt.subplots(figsize=(6, 4))
                            ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
                            ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
                            ax.set_xlabel('False Positive Rate')
                            ax.set_ylabel('True Positive Rate')
                            ax.set_title('ROC Curve')
                            ax.legend()
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Error during training: {e}")
            
            with col2:
                if st.button("Load Pre-Trained XGBoost Model"):
                    with st.spinner("Loading pre-trained XGBoost model..."):
                        try:
                            xgb_model = joblib.load('models/xgb_model.pkl')
                            st.session_state.model = xgb_model
                            st.session_state.model_type = "XGBoost"
                            y_pred = xgb_model.predict(X_test_xgb)
                            y_pred_proba = xgb_model.predict_proba(X_test_xgb)[:, 1]
                            accuracy = accuracy_score(y_test_xgb, y_pred)
                            auc = roc_auc_score(y_test_xgb, y_pred_proba)
                            report = classification_report(y_test_xgb, y_pred, output_dict=True, zero_division=0)
                            precision = report.get('1', {}).get('precision', 0)
                            recall = report.get('1', {}).get('recall', 0)
                            f1 = report.get('1', {}).get('f1-score', 0)
                            auc_mean, auc_ci, recall_mean, recall_ci, precision_mean, precision_ci, f1_mean, f1_ci = bootstrap_metrics(y_test_xgb, y_pred_proba)
                            st.session_state.model_metrics.append(["XGBoost", accuracy * 100, auc, precision, recall, f1])
                            st.success("XGBoost Model Loaded Successfully!")
                            st.write("### XGBoost Performance Metrics")
                            st.write(f"Accuracy: {accuracy * 100:.2f}%")
                            st.write(f"AUC: {auc_mean:.4f} (95% CI: {auc_ci[0]:.4f} - {auc_ci[1]:.4f})")
                            st.write(f"CONFIRMED Precision: {precision_mean:.4f} (95% CI: {precision_ci[0]:.4f} - {precision_ci[1]:.4f})")
                            st.write(f"CONFIRMED Recall: {recall_mean:.4f} (95% CI: {recall_ci[0]:.4f} - {recall_ci[1]:.4f})")
                            st.write(f"CONFIRMED F1-Score: {f1_mean:.4f} (95% CI: {f1_ci[0]:.4f} - {f1_ci[1]:.4f})")
                            cm = confusion_matrix(y_test_xgb, y_pred)
                            fig, ax = plt.subplots(figsize=(6, 4))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=st.session_state.label_encoder.classes_, yticklabels=st.session_state.label_encoder.classes_)
                            ax.set_title('XGBoost Confusion Matrix')
                            ax.set_xlabel('Predicted')
                            ax.set_ylabel('Actual')
                            st.pyplot(fig)
                            st.write("### SHAP Feature Importance")
                            explainer = shap.Explainer(xgb_model)
                            shap_values = explainer(X_test_xgb)
                            fig, ax = plt.subplots()
                            shap.summary_plot(shap_values, X_test_xgb, feature_names=st.session_state.features_to_use, plot_type='bar', show=False)
                            plt.title("SHAP Feature Importance - Bar Plot")
                            plt.tight_layout()
                            st.pyplot(fig)
                            fig, ax = plt.subplots()
                            shap.summary_plot(shap_values, X_test_xgb, feature_names=st.session_state.features_to_use, show=False)
                            plt.title("SHAP Feature Importance - Dot Plot")
                            plt.tight_layout()
                            st.pyplot(fig)
                            fpr, tpr, _ = roc_curve(y_test_xgb, y_pred_proba)
                            fig, ax = plt.subplots(figsize=(6, 4))
                            ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
                            ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
                            ax.set_xlabel('False Positive Rate')
                            ax.set_ylabel('True Positive Rate')
                            ax.set_title('ROC Curve')
                            ax.legend()
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Error loading model: {e}")
        
        if st.session_state.model_metrics:
            st.write("### Model Comparison")
            headers = ["Model", "Accuracy (%)", "AUC", "CONFIRMED Precision", "CONFIRMED Recall", "CONFIRMED F1"]
            st.table(tabulate(st.session_state.model_metrics, headers=headers, tablefmt="grid"))

elif page == "Predictions":
    st.header("Make Predictions")
    
    if st.session_state.model is None:
        st.warning("No model trained or loaded. Please go to 'Model Training' to train or load a model.")
    else:
        st.write(f"Using {st.session_state.model_type} model for predictions.")
        input_type = st.radio("Select input type", ["Raw Light Curve (CNN)", "Expert Features (XGBoost)"])
        
        if input_type == "Raw Light Curve (CNN)":
            uploaded_file = st.file_uploader("Upload a light curve CSV (3197 flux values)", type=["csv"])
            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                    if data.shape[1] == 3197:
                        X_new = st.session_state.scaler_cnn.transform(data.values)
                        X_new = X_new.reshape(-1, 3197, 1)
                        pred_proba = st.session_state.model.predict(X_new, verbose=0)[0][0]
                        prediction = 1 if pred_proba > 0.3 else 0
                        label = 'CONFIRMED' if prediction == 1 else 'FALSE POSITIVE'
                        st.success(f"Prediction: {label} (confidence: {pred_proba:.2%})")
                        
                        fig, ax = plt.subplots(figsize=(14, 4))
                        ax.plot(data.values[0], label='Uploaded Light Curve')
                        ax.set_xlabel("Time Step")
                        ax.set_ylabel("Flux (Brightness)")
                        ax.set_title("Uploaded Light Curve")
                        ax.legend()
                        ax.grid(True)
                        st.pyplot(fig)
                        
                        heatmap, class_idx = apply_grad_cam(st.session_state.model, X_new[0], layer_name='conv1d_2')
                        fig, ax = plt.subplots(figsize=(14, 4))
                        ax.plot(X_new[0, :, 0], label='Light Curve')
                        ax.plot(heatmap * np.max(X_new[0, :, 0]), label='Grad-CAM Heatmap', alpha=0.5)
                        ax.set_title(f"Grad-CAM (Predicted: {'CONFIRMED' if class_idx == 1 else 'FALSE POSITIVE'})")
                        ax.set_xlabel("Time Step")
                        ax.set_ylabel("Flux / Heatmap")
                        ax.legend()
                        st.pyplot(fig)
                    else:
                        st.error("Uploaded file must contain exactly 3197 flux values.")
                except Exception as e:
                    st.error(f"Error processing file: {e}")
        
        elif input_type == "Expert Features (XGBoost)":
            uploaded_file = st.file_uploader("Upload a CSV with expert features", type=["csv"])
            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                    if all(f in data.columns for f in st.session_state.features_to_use):
                        data = data[st.session_state.features_to_use].fillna(data[st.session_state.features_to_use].mean(numeric_only=True))
                        X_new = st.session_state.scaler_xgb.transform(data)
                        pred_proba = st.session_state.model.predict_proba(X_new)[:, 1][0]
                        prediction = 1 if pred_proba > 0.3 else 0
                        label = st.session_state.label_encoder.inverse_transform([prediction])[0]
                        st.success(f"Prediction: {label} (confidence: {pred_proba:.2%})")
                    else:
                        st.error(f"Uploaded file must contain features: {', '.join(st.session_state.features_to_use)}")
                except Exception as e:
                    st.error(f"Error processing file: {e}")

