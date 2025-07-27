import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import joblib
import os

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

# Create models directory
os.makedirs("models", exist_ok=True)

# Load data
try:
    raw_data = pd.read_csv("/content/drive/MyDrive/Keler/exoTrain.csv")  # For CNN
    baseline_df = pd.read_csv("/content/drive/MyDrive/Keler/cumulative.csv")  # For XGBoost
except FileNotFoundError:
    print("ERROR: Dataset files not found.")
    exit(1)

# CNN Training (Full Dataset)
print("Training CNN model...")
X_cnn = raw_data.drop(columns=['LABEL']).values
y_cnn = raw_data['LABEL'].values - 1  # Map 1 (FALSE) to 0, 2 (CONFIRMED) to 1
scaler_cnn = StandardScaler()
X_cnn_scaled = scaler_cnn.fit_transform(X_cnn)
X_cnn = X_cnn_scaled.reshape(X_cnn_scaled.shape[0], X_cnn_scaled.shape[1], 1)

# Oversampling
confirmed_indices = np.where(y_cnn == 1)[0]
X_confirmed = X_cnn[confirmed_indices]
y_confirmed = y_cnn[confirmed_indices]
oversample_factor = 15
X_train_oversampled = np.vstack([X_cnn] + [X_confirmed] * oversample_factor)
y_train_oversampled = np.hstack([y_cnn] + [y_confirmed] * oversample_factor)

# Data augmentation
X_train_aug, y_train_aug = augment_data(X_train_oversampled, y_train_oversampled, sigma=0.04, max_shift=200)

# Class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_aug), y=y_train_aug)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Define CNN model
inputs = Input(shape=(X_cnn.shape[1], 1), name='input_layer')
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

# Compile and train
model.compile(optimizer='adam', loss=focal_loss(gamma=1.0, alpha=0.75), metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
history = model.fit(
    X_train_aug, y_train_aug,
    validation_split=0.2,
    epochs=30, batch_size=64,
    callbacks=[early_stopping, lr_scheduler],
    class_weight=class_weight_dict,
    verbose=1
)

# Save CNN model and scaler
model.save('models/cnn_exoplanet_model.keras')
joblib.dump(scaler_cnn, 'models/scaler_cnn.pkl')
print("✅ Saved CNN model to 'models/cnn_exoplanet_model.keras' and scaler to 'models/scaler_cnn.pkl'")

# XGBoost Training
print("Training XGBoost model...")
# Define features_to_use based on your notebook
# Select only numeric columns for XGBoost
numeric_cols = baseline_df.select_dtypes(include=np.number).columns.tolist()
features_to_use_xgb = [col for col in numeric_cols if col not in ['kepid', 'label_encoded']]

if 'label_encoded' not in baseline_df.columns:
    label_encoder = LabelEncoder()
    baseline_df['label_encoded'] = label_encoder.fit_transform(baseline_df['koi_disposition'])
    joblib.dump(label_encoder, 'models/label_encoder.pkl')

X_xgb = baseline_df[features_to_use_xgb]
y_xgb = baseline_df['label_encoded']

# Handle missing values (if any, as per your notebook) - applied only to numeric features
X_xgb = X_xgb.fillna(X_xgb.mean())

# Standardize features
scaler_xgb = StandardScaler()
X_xgb_scaled = scaler_xgb.fit_transform(X_xgb)

# Train-test split
from sklearn.model_selection import train_test_split
X_train_xgb, _, y_train_xgb, _ = train_test_split(X_xgb_scaled, y_xgb, test_size=0.3, stratify=y_xgb, random_state=42)

# Train XGBoost
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42
)
xgb_model.fit(X_train_xgb, y_train_xgb)

# Save XGBoost model and scaler
joblib.dump(xgb_model, 'models/xgb_model.pkl')
joblib.dump(scaler_xgb, 'models/scaler_xgb.pkl')
print("✅ Saved XGBoost model to 'models/xgb_model.pkl' and scaler to 'models/scaler_xgb.pkl'")