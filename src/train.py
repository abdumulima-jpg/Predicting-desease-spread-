import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "processed_data.csv"
MODELS_DIR = BASE_DIR / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

# Target variable
TARGET = "Cases"

# Features
X = df.drop(columns=[TARGET])
y = df[TARGET]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Save feature column order for prediction
feature_columns = X.columns.tolist()

joblib.dump(
    feature_columns,
    MODELS_DIR / "feature_columns.pkl"
)


# ============================================================
# RANDOM FOREST REGRESSOR
# ============================================================

print("\nTraining Random Forest...")

random_forest = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

random_forest.fit(X_train, y_train)

print("Random Forest training completed.")


# ============================================================
# XGBOOST REGRESSOR
# ============================================================

print("\nTraining XGBoost...")

xgboost_model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    objective="reg:squarederror"
)

xgboost_model.fit(X_train, y_train)

print("XGBoost training completed.")


# ============================================================
# SCALE DATA FOR LSTM
# ============================================================

print("\nPreparing data for LSTM...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(
    scaler,
    MODELS_DIR / "scaler.pkl"
)


# ============================================================
# RESHAPE DATA FOR LSTM
# ============================================================

X_train_lstm = X_train_scaled.reshape(
    X_train_scaled.shape[0],
    1,
    X_train_scaled.shape[1]
)

X_test_lstm = X_test_scaled.reshape(
    X_test_scaled.shape[0],
    1,
    X_test_scaled.shape[1]
)


# ============================================================
# LSTM MODEL
# ============================================================

print("\nTraining LSTM...")

lstm_model = Sequential([
    LSTM(
        64,
        input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])
    ),
    Dense(32, activation="relu"),
    Dense(1)
])

lstm_model.compile(
    optimizer="adam",
    loss="mse"
)

lstm_model.fit(
    X_train_lstm,
    y_train,
    epochs=20,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

print("LSTM training completed.")


# ============================================================
# SAVE MODELS
# ============================================================

# Random Forest is saved as the main trained model
joblib.dump(
    random_forest,
    MODELS_DIR / "trained_model.pkl"
)

# Save XGBoost separately
joblib.dump(
    xgboost_model,
    MODELS_DIR / "xgboost_model.pkl"
)

# Save LSTM in Keras format
lstm_model.save(
    MODELS_DIR / "lstm_model.keras"
)


# ============================================================
# SAVE TEST DATA FOR EVALUATION
# ============================================================

test_data = X_test.copy()
test_data[TARGET] = y_test.values

test_data.to_csv(
    BASE_DIR / "outputs" / "test_data.csv",
    index=False
)


print("\n========================================")
print("ALL THREE MODELS TRAINED SUCCESSFULLY")
print("========================================")
print(f"Random Forest: {MODELS_DIR / 'trained_model.pkl'}")
print(f"XGBoost:       {MODELS_DIR / 'xgboost_model.pkl'}")
print(f"LSTM:          {MODELS_DIR / 'lstm_model.keras'}")
print(f"Scaler:        {MODELS_DIR / 'scaler.pkl'}")
print("========================================")