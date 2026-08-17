from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# DATA PATHS
# ============================================================

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "data.csv"

PROCESSED_DATA_PATH = (
    BASE_DIR / "data" / "processed" / "processed_data.csv"
)


# ============================================================
# MODEL PATHS
# ============================================================

MODELS_DIR = BASE_DIR / "models"

RANDOM_FOREST_MODEL_PATH = (
    MODELS_DIR / "trained_model.pkl"
)

XGBOOST_MODEL_PATH = (
    MODELS_DIR / "xgboost_model.pkl"
)

LSTM_MODEL_PATH = (
    MODELS_DIR / "lstm_model.keras"
)

SCALER_PATH = (
    MODELS_DIR / "scaler.pkl"
)

FEATURE_COLUMNS_PATH = (
    MODELS_DIR / "feature_columns.pkl"
)


# ============================================================
# OUTPUT PATHS
# ============================================================

OUTPUTS_DIR = BASE_DIR / "outputs"

PREDICTIONS_PATH = (
    OUTPUTS_DIR / "predictions.csv"
)

METRICS_PATH = (
    OUTPUTS_DIR / "metrics.json"
)

PLOTS_DIR = (
    OUTPUTS_DIR / "plots"
)


# ============================================================
# MODEL SETTINGS
# ============================================================

TARGET_COLUMN = "Cases"

TEST_SIZE = 0.20

RANDOM_STATE = 42