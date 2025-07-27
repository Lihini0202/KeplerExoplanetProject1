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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils import class_weight, resample
import xgboost as xgb
from tabulate import tabulate
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score
import shap

# Set page config
st.set_page_config(page_title="Exoplanet Detection", layout="wide")

# Title
st.title("Exoplanet Detection from Kepler Light Curves")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Exploration", "Model Training", "Predictions"])

# Initialize session state for storing models and metrics
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.model_type = None
    st.session_state.scaler = None
    st.session_state.model_metrics = []

# Load data function
@st.cache_data
def load_data():
    try:
        summary_data = pd.read_csv("datasets/cumulative.csv")
        raw_train_data = pd.read_csv("datasets/exoTrain.csv")
        raw_test_data = pd.read_csv("datasets/exoTest.csv")
        return summary_data, raw_train_data, raw_test_data
    except FileNotFoundError:
        st.error("ERROR: Dataset files not found. Please ensure 'datasets/cumulative.csv', 'datasets/exoTrain.csv', and 'datasets/exoTest.csv' are available.")
        return None, None, None

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
        report = classification_report(y_true_boot, y_pred_boot, output_dict=True, zero_division=0)['1']
        recalls.append(report['recall'])
        precisions.append(report['precision'])
        f1s.append(report['f1-score'])
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
    return heatmap_resized, class_index

# Load data
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
        
        # Sample light curves visualization
        st.subheader("Sample Light Curves")
        num_samples = st.slider("Number of samples to display", 1, 5, 3)
        sample_indices = np.random.choice(raw_train_data.shape[0], num_samples, replace=False)
        
        for i, idx in enumerate(sample_indices):
            label = raw_train_data.iloc[idx]['LABEL']
            flux_values = raw_train_data.iloc[idx, 1:].values  # Skip LABEL column
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
    
    if raw_train_data is not None and raw_test_data is not None:
        st.subheader("Data Preparation")
        
        # Preprocessing
        st.write("Preprocessing data: Standardizing and encoding labels...")
        X_train = raw_train_data.drop(columns=['LABEL']).values
        y_train = raw_train_data['LABEL'].map({1: 0, 2: 1}).values  # Map 1 (FALSE) to 0, 2 (CONFIRMED) to 1
        X_test = raw_test_data.drop(columns=['LABEL']).values
        y_test = raw_test_data['LABEL'].map({1: 0, 2: 1}).values
        
        # Standardize features
        st.session_state.scaler = StandardScaler()
        X_train = st.session_state.scaler.fit_transform(X_train)
        X_test = st.session_state.scaler.transform(X_test)
        
        # Data augmentation for CNN
        if st.checkbox("Apply data augmentation for CNN"):
            st.write("Augmenting training data with Gaussian noise and random shifts...")
            X_train_aug, y_train_aug = augment_data(X_train, y_train, sigma=0.04, max_shift=200)
            X_train = np.concatenate([X_train, X_train_aug])
            y_train = np.concatenate([y_train, y_train_aug])
        
        st.write(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
        
        st.subheader("Model Selection")
        model_type = st.selectbox("Select model type", ["CNN", "XGBoost"])
        
        @st.cache_resource
        def train_cnn_model(X_train, y_train, X_test, y_test):
            X_train_cnn = X_train.reshape(-1, 3197, 1)
            X_test_cnn = X_test.reshape(-1, 3197, 1)
            input_shape = (3197, 1)
            inputs = Input(shape=input_shape)
            x = Conv1D(32, 3, activation='relu', padding='same', kernel_regularizer=l2(0.01))(inputs)
            x = MaxPooling1D(2)(x)
            x = BatchNormalization()(x)
            x = Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
            x = MaxPooling1D(2)(x)
            x = BatchNormalization()(x)
            x = Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
            x = GlobalAveragePooling1D()(x)
            x = Dropout(0.5)(x)
            outputs = Dense(1, activation='sigmoid')(x)
            model = Model(inputs, outputs)
            model.compile(optimizer='adam', loss=focal_loss(gamma=1.0, alpha=0.75), metrics=['accuracy'])
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weight_dict = dict(enumerate(class_weights))
            history = model.fit(
                X_train_cnn, y_train,
                validation_data=(X_test_cnn, y_test),
                epochs=20, batch_size=32,
                class_weight=class_weight_dict,
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            return model, history, X_test_cnn

        @st.cache_resource
        def train_xgb_model(X_train, y_train):
            xgb_model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False
            )
            xgb_model.fit(X_train, y_train)
            return xgb_model

        if model_type == "CNN":
            st.write("### CNN Model Architecture")
            st.code("""
            input_shape = (3197, 1)
            inputs = Input(shape=input_shape)
            x = Conv1D(32, 3, activation='relu', padding='same', kernel_regularizer=l2(0.01))(inputs)
            x = MaxPooling1D(2)(x)
            x = BatchNormalization()(x)
            x = Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
            x = MaxPooling1D(2)(x)
            x = BatchNormalization()(x)
            x = Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
            x = GlobalAveragePooling1D()(x)
            x = Dropout(0.5)(x)
            outputs = Dense(1, activation='sigmoid')(x)
            model = Model(inputs, outputs)
            """, language='python')
            
            if st.button("Train CNN Model"):
                with st.spinner("Training CNN model..."):
                    model, history, X_test_cnn = train_cnn_model(X_train, y_train, X_test, y_test)
                    st.session_state.model = model
                    st.session_state.model_type = "CNN"
                    
                    # Evaluate model
                    y_pred_proba = model.predict(X_test_cnn, verbose=0)
                    y_pred = (y_pred_proba > 0.3).astype(int)
                    accuracy = accuracy_score(y_test, y_pred)
                    auc = roc_auc_score(y_test, y_pred_proba)
                    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                    precision = report['1']['precision']
                    recall = report['1']['recall']
                    f1 = report['1']['f1-score']
                    auc_mean, auc_ci, recall_mean, recall_ci, precision_mean, precision_ci, f1_mean, f1_ci = bootstrap_metrics(y_test, y_pred_proba)
                    
                    # Store metrics
                    st.session_state.model_metrics.append(["CNN", accuracy * 100, auc, precision, recall, f1])
                    
                    # Display metrics
                    st.success("CNN Training Complete!")
                    st.write("### CNN Performance Metrics")
                    st.write(f"Accuracy: {accuracy * 100:.2f}%")
                    st.write(f"AUC: {auc_mean:.4f} (95% CI: {auc_ci[0]:.4f} - {auc_ci[1]:.4f})")
                    st.write(f"CONFIRMED Precision: {precision_mean:.4f} (95% CI: {precision_ci[0]:.4f} - {precision_ci[1]:.4f})")
                    st.write(f"CONFIRMED Recall: {recall_mean:.4f} (95% CI: {recall_ci[0]:.4f} - {recall_ci[1]:.4f})")
                    st.write(f"CONFIRMED F1-Score: {f1_mean:.4f} (95% CI: {f1_ci[0]:.4f} - {f1_ci[1]:.4f})")
                    
                    # Plot training history
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
                    
                    # ROC Curve
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
                    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('ROC Curve')
                    ax.legend()
                    st.pyplot(fig)
                    
                    # Grad-CAM Visualization
                    st.write("### Grad-CAM Visualization")
                    sample_idx = st.selectbox("Select sample index for Grad-CAM", range(len(X_test)))
                    heatmap, class_idx = apply_grad_cam(model, X_test_cnn[sample_idx], layer_name='conv1d_2')
                    fig, ax = plt.subplots(figsize=(14, 4))
                    ax.plot(X_test[sample_idx], label='Light Curve')
                    ax.plot(heatmap * np.max(X_test[sample_idx]), label='Grad-CAM Heatmap', alpha=0.5)
                    ax.set_title(f"Grad-CAM for Sample #{sample_idx} (Predicted: {'CONFIRMED' if class_idx == 1 else 'FALSE POSITIVE'})")
                    ax.set_xlabel("Time Step")
                    ax.set_ylabel("Flux / Heatmap")
                    ax.legend()
                    st.pyplot(fig)
        
        elif model_type == "XGBoost":
            st.write("### XGBoost Model Training")
            if st.button("Train XGBoost Model"):
                with st.spinner("Training XGBoost model..."):
                    xgb_model = train_xgb_model(X_train, y_train)
                    st.session_state.model = xgb_model
                    st.session_state.model_type = "XGBoost"
                    
                    # Evaluate model
                    y_pred = xgb_model.predict(X_test)
                    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
                    accuracy = accuracy_score(y_test, y_pred)
                    auc = roc_auc_score(y_test, y_pred_proba)
                    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                    precision = report['1']['precision']
                    recall = report['1']['recall']
                    f1 = report['1']['f1-score']
                    auc_mean, auc_ci, recall_mean, recall_ci, precision_mean, precision_ci, f1_mean, f1_ci = bootstrap_metrics(y_test, y_pred_proba)
                    
                    # Store metrics
                    st.session_state.model_metrics.append(["XGBoost", accuracy * 100, auc, precision, recall, f1])
                    
                    # Display metrics
                    st.success("XGBoost Training Complete!")
                    st.write("### XGBoost Performance Metrics")
                    st.write(f"Accuracy: {accuracy * 100:.2f}%")
                    st.write(f"AUC: {auc_mean:.4f} (95% CI: {auc_ci[0]:.4f} - {auc_ci[1]:.4f})")
                    st.write(f"CONFIRMED Precision: {precision_mean:.4f} (95% CI: {precision_ci[0]:.4f} - {precision_ci[1]:.4f})")
                    st.write(f"CONFIRMED Recall: {recall_mean:.4f} (95% CI: {recall_ci[0]:.4f} - {recall_ci[1]:.4f})")
                    st.write(f"CONFIRMED F1-Score: {f1_mean:.4f} (95% CI: {f1_ci[0]:.4f} - {f1_ci[1]:.4f})")
                    
                    # SHAP Analysis
                    st.write("### SHAP Feature Importance")
                    explainer = shap.Explainer(xgb_model)
                    shap_values = explainer(X_test)
                    
                    # SHAP summary plot (bar)
                    fig, ax = plt.subplots()
                    shap.summary_plot(shap_values, X_test, feature_names=raw_test_data.columns[1:], plot_type='bar', show=False)
                    plt.title("SHAP Feature Importance - Bar Plot")
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # SHAP summary plot (dot)
                    fig, ax = plt.subplots()
                    shap.summary_plot(shap_values, X_test, feature_names=raw_test_data.columns[1:], show=False)
                    plt.title("SHAP Feature Importance - Dot Plot")
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # ROC Curve
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
                    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('ROC Curve')
                    ax.legend()
                    st.pyplot(fig)
        
        # Display model comparison table
        if st.session_state.model_metrics:
            st.write("### Model Comparison")
            headers = ["Model", "Accuracy (%)", "AUC", "CONFIRMED Precision", "CONFIRMED Recall", "CONFIRMED F1"]
            st.table(tabulate(st.session_state.model_metrics, headers=headers, tablefmt="grid"))

elif page == "Predictions":
    st.header("Make Predictions")
    
    if st.session_state.model is None:
        st.warning("No model trained yet. Please train a model in the 'Model Training' section first.")
    else:
        st.write(f"Using trained {st.session_state.model_type} model for predictions.")
        uploaded_file = st.file_uploader("Upload a light curve file (CSV with 3197 flux values)", type=["csv"])
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.write("Uploaded data preview:")
                st.dataframe(data.head())
                
                if data.shape[1] == 3197:
                    X_new = st.session_state.scaler.transform(data.values)
                    if st.session_state.model_type == "CNN":
                        X_new = X_new.reshape(-1, 3197, 1)
                        pred_proba = st.session_state.model.predict(X_new, verbose=0)[0][0]
                    else:
                        pred_proba = st.session_state.model.predict_proba(X_new)[:, 1][0]
                    prediction = 1 if pred_proba > 0.3 else 0
                    
                    if prediction == 1:
                        st.success(f"Prediction: EXOPLANET DETECTED (confidence: {pred_proba:.2%})")
                    else:
                        st.error(f"Prediction: NO EXOPLANET DETECTED (confidence: {1 - pred_proba:.2%})")
                    
                    # Visualize uploaded light curve
                    st.write("### Uploaded Light Curve")
                    fig, ax = plt.subplots(figsize=(14, 4))
                    ax.plot(data.values[0], label='Uploaded Light Curve')
                    ax.set_xlabel("Time Step")
                    ax.set_ylabel("Flux (Brightness)")
                    ax.set_title("Uploaded Light Curve")
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)
                    
                    # Grad-CAM for CNN
                    if st.session_state.model_type == "CNN":
                        st.write("### Grad-CAM Visualization")
                        heatmap, class_idx = apply_grad_cam(st.session_state.model, X_new[0], layer_name='conv1d_2')
                        fig, ax = plt.subplots(figsize=(14, 4))
                        ax.plot(X_new[0, :, 0], label='Light Curve')
                        ax.plot(heatmap * np.max(X_new[0, :, 0]), label='Grad-CAM Heatmap', alpha=0.5)
                        ax.set_title(f"Grad-CAM for Uploaded Light Curve (Predicted: {'CONFIRMED' if class_idx == 1 else 'FALSE POSITIVE'})")
                        ax.set_xlabel("Time Step")
                        ax.set_ylabel("Flux / Heatmap")
                        ax.legend()
                        st.pyplot(fig)
                else:
                    st.error("Uploaded file must contain exactly 3197 flux values.")
            except Exception as e:
                st.error(f"Error processing file: {e}")
