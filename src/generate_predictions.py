import joblib
import pandas as pd
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_DATA = BASE_DIR / "outputs" / "test_data.csv"
MODEL_PATH = BASE_DIR / "models" / "trained_model.pkl"
FEATURE_COLUMNS_PATH = BASE_DIR / "models" / "feature_columns.pkl"
OUTPUT_PATH = BASE_DIR / "outputs" / "predictions.csv"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(TEST_DATA)

TARGET = "Cases"

X = df.drop(columns=[TARGET])


# ============================================================
# LOAD MODEL AND FEATURE COLUMNS
# ============================================================

model = joblib.load(MODEL_PATH)

feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

X = X[feature_columns]


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

predictions = model.predict(X)


# ============================================================
# CREATE OUTPUT FILE
# ============================================================

output = pd.DataFrame({
    "Actual_Cases": df[TARGET],
    "Predicted_Cases": predictions
})

output["Predicted_Cases"] = output["Predicted_Cases"].round(2)


output.to_csv(
    OUTPUT_PATH,
    index=False
)


print("========================================")
print("PREDICTIONS GENERATED SUCCESSFULLY")
print("========================================")
print(f"Predictions saved to: {OUTPUT_PATH}")
print(f"Number of predictions: {len(output)}")
print("========================================")