import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import joblib


BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DATA = BASE_DIR / "data" / "processed" / "processed_data.csv"
FEATURE_DATA = BASE_DIR / "data" / "processed" / "processed_data.csv"

ENCODER_DIR = BASE_DIR / "models"


def engineer_features():
    """Create machine-learning features from the processed dataset."""

    df = pd.read_csv(PROCESSED_DATA)

    # Convert date and time columns
    df["day"] = pd.to_datetime(df["day"], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    # Date-based features
    df["year"] = df["day"].dt.year
    df["month"] = df["day"].dt.month
    df["day_of_month"] = df["day"].dt.day

    # Time-based feature
    df["hour"] = df["time"].dt.hour

    # Encode categorical columns
    ENCODER_DIR.mkdir(parents=True, exist_ok=True)

    for column in ["country", "continent"]:
        if column in df.columns:
            encoder = LabelEncoder()
            df[column] = encoder.fit_transform(df[column].astype(str))

            joblib.dump(
                encoder,
                ENCODER_DIR / f"{column}_encoder.pkl"
            )

    # Remove original datetime columns
    df = df.drop(columns=["day", "time"])

    # Save engineered dataset
    df.to_csv(FEATURE_DATA, index=False)

    print(f"Feature-engineered data saved to: {FEATURE_DATA}")
    print(f"Feature-engineered dataset shape: {df.shape}")

    return df


if __name__ == "__main__":
    engineer_features()