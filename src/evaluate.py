import json
import joblib
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_DATA = BASE_DIR / "outputs" / "test_data.csv"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD TEST DATA
# ============================================================

df = pd.read_csv(TEST_DATA)

TARGET = "Cases"

X_test = df.drop(columns=[TARGET])
y_test = df[TARGET]


# ============================================================
# LOAD FEATURE COLUMN ORDER
# ============================================================

feature_columns = joblib.load(
    MODELS_DIR / "feature_columns.pkl"
)

X_test = X_test[feature_columns]


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(name, y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    mse = mean_squared_error(y_true, y_pred)

    r2 = r2_score(y_true, y_pred)

    print(f"\n{name}")
    print("-" * 40)
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"R² Score: {r2}")

    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "R2": float(r2)
    }


# ============================================================
# RANDOM FOREST
# ============================================================

print("\nEvaluating Random Forest...")

random_forest = joblib.load(
    MODELS_DIR / "trained_model.pkl"
)

rf_predictions = random_forest.predict(X_test)

rf_results = evaluate_model(
    "Random Forest",
    y_test,
    rf_predictions
)


# ============================================================
# XGBOOST
# ============================================================

print("\nEvaluating XGBoost...")

xgboost_model = joblib.load(
    MODELS_DIR / "xgboost_model.pkl"
)

xgb_predictions = xgboost_model.predict(X_test)

xgb_results = evaluate_model(
    "XGBoost",
    y_test,
    xgb_predictions
)


# ============================================================
# LSTM
# ============================================================

print("\nEvaluating LSTM...")

scaler = joblib.load(
    MODELS_DIR / "scaler.pkl"
)

X_test_scaled = scaler.transform(X_test)

X_test_lstm = X_test_scaled.reshape(
    X_test_scaled.shape[0],
    1,
    X_test_scaled.shape[1]
)

lstm_model = load_model(
    MODELS_DIR / "lstm_model.keras"
)

lstm_predictions = lstm_model.predict(
    X_test_lstm,
    verbose=0
).flatten()

lstm_results = evaluate_model(
    "LSTM",
    y_test,
    lstm_predictions
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "Random Forest": rf_results,
    "XGBoost": xgb_results,
    "LSTM": lstm_results
}

metrics_path = OUTPUTS_DIR / "metrics.json"

with open(metrics_path, "w") as file:
    json.dump(results, file, indent=4)


print("\n========================================")
print("MODEL EVALUATION COMPLETED")
print("========================================")
print(f"Metrics saved to: {metrics_path}")
print("========================================")