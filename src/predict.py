import joblib
import pandas as pd

from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "trained_model.pkl"
FEATURE_COLUMNS_PATH = BASE_DIR / "models" / "feature_columns.pkl"


# ============================================================
# LOAD MODEL AND FEATURE COLUMNS
# ============================================================

model = joblib.load(MODEL_PATH)

feature_columns = joblib.load(
    FEATURE_COLUMNS_PATH
)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_cases(features: dict):
    """
    Predict disease cases using the trained Random Forest model.

    Parameters:
        features (dict): Input feature values.

    Returns:
        float: Predicted number of cases.
    """

    # Convert input dictionary to DataFrame
    input_data = pd.DataFrame([features])

    # Make sure the features are in the same order
    # used during model training
    input_data = input_data[feature_columns]

    # Generate prediction
    prediction = model.predict(input_data)

    return float(prediction[0])


# ============================================================
# TEST THE MODULE
# ============================================================

if __name__ == "__main__":

    example_features = {
        "country": 178,
        "continent": 0,
        "population": 6115.0,
        "Recovered": 2.0,
        "Deaths": 2570.0,
        "Tests": 2226216.0,
        "year": 2024,
        "month": 6,
        "day_of_month": 30,
        "hour": 16
    }

    prediction = predict_cases(example_features)

    print(f"Predicted cases: {prediction:.2f}")