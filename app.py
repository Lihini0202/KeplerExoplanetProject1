import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from PIL import Image
import os
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, precision_score, recall_score
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import tensorflow as tf

# --- Custom Loss Function ---
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

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="Kepler Exoplanet Detection",
    page_icon="🔭",
    layout="wide"
)

# --- Load Data ---
@st.cache_data
def load_data():
    """Load Kepler datasets from CSV files."""
    try:
        summary = pd.read_csv("datasets/cumulative.csv")
        train = pd.read_csv("datasets/exoTrain.csv")
        test = pd.read_csv("datasets/exoTest.csv")
        return summary, train, test
    except FileNotFoundError as e:
        st.error(f"Error loading data files: {e}")
        return None, None, None

# --- Load Models ---
@st.cache_resource
def load_xgb_model():
    """Load XGBoost model from pickle file."""
    try:
        return joblib.load("models/xgb_model.pkl")
    except Exception as e:
        st.error(f"Error loading XGBoost model: {e}")
        return None

@st.cache_resource
def load_cnn_model():
    """Load CNN model from file."""
    try:
        return load_model("models/exoplanet_cnn.keras", custom_objects={'focal_loss_fixed': focal_loss()})
    except Exception as e:
        st.error(f"Error loading CNN model: {e}")
        return None

# --- Page: Home ---
def home():
    st.title("🔭 Kepler Exoplanet Detection")
    st.markdown("---")
    st.markdown("""
        ### Detect exoplanets from Kepler light curve data using machine learning
        
        This app allows you to:
        - Explore the Kepler exoplanet dataset
        - Visualize light curves and data distributions
        - Train and evaluate machine learning models
        - Make predictions on new data
    """)
    
    if st.button("🚀 Get Started"):
        st.session_state.page = "Data Exploration"
        st.experimental_rerun()

# --- Page: Data Exploration ---
def data_exploration(summary, train, test):
    st.header("📊 Data Exploration")
    
    if summary is not None:
        st.subheader("Dataset Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Summary Records", len(summary))
        col2.metric("Training Samples", len(train))
        col3.metric("Test Samples", len(test))
        
        with st.expander("Show Summary Data"):
            st.dataframe(summary.head())
            
        with st.expander("Show Training Data"):
            st.dataframe(train.head())
            
        with st.expander("Show Test Data"):
            st.dataframe(test.head())
    
    st.subheader("Sample Light Curves")
    num_samples = st.slider("Number of samples to display", 1, 5, 3)
    
    if train is not None:
        sample_indices = np.random.choice(train.shape[0], num_samples, replace=False)
        
        for i, idx in enumerate(sample_indices):
            label = train.iloc[idx]['LABEL']
            flux_values = train.iloc[idx, 1:].values
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(flux_values, label=f"Sample #{idx} | Label: {'EXOPLANET' if label == 2 else 'NO EXOPLANET'}")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Flux (Brightness)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

# --- Page: Model Training ---
def model_training(train, test):
    st.header("🤖 Model Training")
    
    if train is None or test is None:
        st.warning("Please load data first")
        return
    
    model_type = st.selectbox("Select model type", ["CNN", "XGBoost"])
    
    if model_type == "CNN":
        st.subheader("CNN Model Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.slider("Epochs", 10, 100, 20)
            batch_size = st.slider("Batch size", 16, 128, 32)
        with col2:
            learning_rate = st.slider("Learning rate", 0.0001, 0.01, 0.001, 0.0001)
            validation_split = st.slider("Validation split", 0.1, 0.3, 0.2, 0.05)
        
        if st.button("Train CNN Model"):
            with st.spinner("Training CNN model..."):
                try:
                    # Preprocess data
                    X_train = train.drop("LABEL", axis=1).values
                    y_train = train["LABEL"].values - 1  # Convert labels to 0 and 1
                    X_test = test.drop("LABEL", axis=1).values
                    y_test = test["LABEL"].values - 1
                    
                    # Normalize data
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Reshape for CNN
                    X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
                    X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
                    
                    # Define CNN model
                    input_shape = (X_train_cnn.shape[1], 1)
                    inputs = Input(shape=input_shape)
                    x = Conv1D(32, 3, activation='relu', padding='same')(inputs)
                    x = MaxPooling1D(2)(x)
                    x = BatchNormalization()(x)
                    x = Conv1D(64, 3, activation='relu', padding='same')(x)
                    x = MaxPooling1D(2)(x)
                    x = BatchNormalization()(x)
                    x = Conv1D(128, 3, activation='relu', padding='same')(x)
                    x = GlobalAveragePooling1D()(x)
                    x = Dropout(0.5)(x)
                    outputs = Dense(1, activation='sigmoid')(x)
                    
                    model = tf.keras.Model(inputs, outputs)
                    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                                 loss=focal_loss(),
                                 metrics=['accuracy'])
                    
                    # Train model
                    history = model.fit(X_train_cnn, y_train,
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      validation_split=validation_split,
                                      verbose=1)
                    
                    # Evaluate model
                    y_pred_proba = model.predict(X_test_cnn)
                    y_pred = (y_pred_proba > 0.3).astype("int").ravel()
                    acc = accuracy_score(y_test, y_pred)
                    
                    # Show results
                    st.success("Training complete!")
                    st.metric("Test Accuracy", f"{acc:.2%}")
                    
                    # Plot training history
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    ax1.plot(history.history['accuracy'], label='Train Accuracy')
                    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
                    ax1.set_title('Model Accuracy')
                    ax1.set_ylabel('Accuracy')
                    ax1.set_xlabel('Epoch')
                    ax1.legend()
                    
                    ax2.plot(history.history['loss'], label='Train Loss')
                    ax2.plot(history.history['val_loss'], label='Validation Loss')
                    ax2.set_title('Model Loss')
                    ax2.set_ylabel('Loss')
                    ax2.set_xlabel('Epoch')
                    ax2.legend()
                    
                    st.pyplot(fig)
                    
                    # Save model
                    model.save("models/exoplanet_cnn.keras")
                    joblib.dump(scaler, "models/scaler.pkl")
                    st.success("Model saved successfully!")
                    
                except Exception as e:
                    st.error(f"Error during training: {e}")
    
    elif model_type == "XGBoost":
        st.subheader("XGBoost Model Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.slider("Number of trees", 50, 500, 100)
            max_depth = st.slider("Max depth", 3, 10, 6)
        with col2:
            learning_rate = st.slider("Learning rate", 0.01, 0.3, 0.1, 0.01)
            subsample = st.slider("Subsample ratio", 0.5, 1.0, 1.0, 0.05)
        
        if st.button("Train XGBoost Model"):
            with st.spinner("Training XGBoost model..."):
                try:
                    from xgboost import XGBClassifier
                    from sklearn.preprocessing import StandardScaler
                    
                    # Preprocess data
                    X_train = train.drop("LABEL", axis=1).values
                    y_train = train["LABEL"].values - 1  # Convert labels to 0 and 1
                    X_test = test.drop("LABEL", axis=1).values
                    y_test = test["LABEL"].values - 1
                    
                    # Normalize data
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Train model
                    model = XGBClassifier(n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        learning_rate=learning_rate,
                                        subsample=subsample,
                                        random_state=42)
                    
                    model.fit(X_train_scaled, y_train)
                    
                    # Evaluate model
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                    acc = accuracy_score(y_test, y_pred)
                    
                    # Show results
                    st.success("Training complete!")
                    st.metric("Test Accuracy", f"{acc:.2%}")
                    
                    # Confusion matrix
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                xticklabels=["No Exoplanet", "Exoplanet"],
                                yticklabels=["No Exoplanet", "Exoplanet"])
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    ax.set_title("Confusion Matrix")
                    st.pyplot(fig)
                    
                    # Save model
                    joblib.dump(model, "models/xgb_model.pkl")
                    joblib.dump(scaler, "models/scaler.pkl")
                    st.success("Model saved successfully!")
                    
                except Exception as e:
                    st.error(f"Error during training: {e}")

# --- Page: Model Evaluation ---
def model_evaluation(test):
    st.header("📊 Model Evaluation")
    
    if test is None:
        st.warning("Please load data first")
        return
    
    model_type = st.selectbox("Select model to evaluate", ["CNN", "XGBoost"])
    
    if model_type == "CNN":
        model = load_cnn_model()
        if model is None:
            st.warning("CNN model not found. Please train the model first.")
            return
        
        # Preprocess data
        X_test = test.drop("LABEL", axis=1).values
        y_test = test["LABEL"].values - 1
        
        # Load scaler
        try:
            scaler = joblib.load("models/scaler.pkl")
            X_test_scaled = scaler.transform(X_test)
            X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
        except Exception as e:
            st.error(f"Error loading scaler: {e}")
            return
        
        # Evaluate model
        y_pred_proba = model.predict(X_test_cnn)
        y_pred = (y_pred_proba > 0.3).astype("int").ravel()
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{acc:.2%}")
        col2.metric("AUC", f"{auc:.2f}")
        col3.metric("Precision", f"{precision:.2f}")
        col4.metric("Recall", f"{recall:.2f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["No Exoplanet", "Exoplanet"],
                    yticklabels=["No Exoplanet", "Exoplanet"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        
        # ROC curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        st.pyplot(fig)
    
    elif model_type == "XGBoost":
        model = load_xgb_model()
        if model is None:
            st.warning("XGBoost model not found. Please train the model first.")
            return
        
        # Preprocess data
        X_test = test.drop("LABEL", axis=1).values
        y_test = test["LABEL"].values - 1
        
        # Load scaler
        try:
            scaler = joblib.load("models/scaler.pkl")
            X_test_scaled = scaler.transform(X_test)
        except Exception as e:
            st.error(f"Error loading scaler: {e}")
            return
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{acc:.2%}")
        col2.metric("AUC", f"{auc:.2f}")
        col3.metric("Precision", f"{precision:.2f}")
        col4.metric("Recall", f"{recall:.2f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["No Exoplanet", "Exoplanet"],
                    yticklabels=["No Exoplanet", "Exoplanet"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        
        # Feature importance
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            from xgboost import plot_importance
            plot_importance(model, ax=ax)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not plot feature importance: {e}")

# --- Page: Prediction ---
def prediction(test):
    st.header("🔮 Make Predictions")
    
    if test is None:
        st.warning("Please load data first")
        return
    
    model_type = st.selectbox("Select model for prediction", ["CNN", "XGBoost"])
    
    if model_type == "CNN":
        model = load_cnn_model()
    else:
        model = load_xgb_model()
    
    if model is None:
        st.warning(f"{model_type} model not found. Please train the model first.")
        return
    
    # Load scaler
    try:
        scaler = joblib.load("models/scaler.pkl")
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        return
    
    # Select a sample from test data
    st.subheader("Select a Sample to Predict")
    sample_idx = st.slider("Sample index", 0, len(test)-1, 0)
    
    # Get the sample
    sample = test.iloc[sample_idx]
    actual_label = "Exoplanet" if sample["LABEL"] == 2 else "No Exoplanet"
    features = sample.drop("LABEL").values.reshape(1, -1)
    
    # Plot the light curve
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(features[0], color="royalblue")
    ax.set_title(f"Light Curve (Actual: {actual_label})")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Flux")
    st.pyplot(fig)
    
    if st.button("Predict"):
        with st.spinner("Making prediction..."):
            try:
                # Preprocess
                features_scaled = scaler.transform(features)
                
                if model_type == "CNN":
                    features_cnn = features_scaled.reshape(features_scaled.shape[0], features_scaled.shape[1], 1)
                    prob = model.predict(features_cnn)[0][0]
                else:
                    prob = model.predict_proba(features_scaled)[0][1]
                
                prediction = "Exoplanet" if prob > 0.5 else "No Exoplanet"
                confidence = prob if prediction == "Exoplanet" else 1 - prob
                
                # Display results
                col1, col2, col3 = st.columns(3)
                col1.metric("Prediction", prediction)
                col2.metric("Confidence", f"{confidence:.2%}")
                col3.metric("Actual", actual_label)
                
                # Show probability
                st.progress(prob)
                st.write(f"Exoplanet probability: {prob:.4f}")
                
            except Exception as e:
                st.error(f"Error during prediction: {e}")

# --- Main App ---
def main():
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "Home"
    
    # Load data
    summary, train, test = load_data()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    if st.sidebar.button("🏠 Home"):
        st.session_state.page = "Home"
    if st.sidebar.button("📊 Data Exploration"):
        st.session_state.page = "Data Exploration"
    if st.sidebar.button("🤖 Model Training"):
        st.session_state.page = "Model Training"
    if st.sidebar.button("📊 Model Evaluation"):
        st.session_state.page = "Model Evaluation"
    if st.sidebar.button("🔮 Prediction"):
        st.session_state.page = "Prediction"
    
    # Display the selected page
    if st.session_state.page == "Home":
        home()
    elif st.session_state.page == "Data Exploration":
        data_exploration(summary, train, test)
    elif st.session_state.page == "Model Training":
        model_training(train, test)
    elif st.session_state.page == "Model Evaluation":
        model_evaluation(test)
    elif st.session_state.page == "Prediction":
        prediction(test)

if __name__ == "__main__":
    main()
