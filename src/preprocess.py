import pandas as pd
from pathlib import Path


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA = BASE_DIR / "data" / "raw" / "data.csv"
PROCESSED_DATA = BASE_DIR / "data" / "processed" / "processed_data.csv"


def preprocess_data():
    """Load, clean, and save the processed disease-spread dataset."""

    # Load raw dataset
    df = pd.read_csv(RAW_DATA)

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Fill missing categorical values
    if "continent" in df.columns:
        df["continent"] = df["continent"].fillna(df["continent"].mode()[0])

    if "country" in df.columns:
        df["country"] = df["country"].fillna("Unknown")

    # Fill missing numerical values
    numerical_columns = [
        "population",
        "Cases",
        "Recovered",
        "Deaths",
        "Tests"
    ]

    for column in numerical_columns:
        if column in df.columns:
            df[column] = df[column].fillna(df[column].median())

    # Convert date column
    if "day" in df.columns:
        df["day"] = pd.to_datetime(df["day"], errors="coerce")

    # Convert time column
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

    # Remove rows where the main date is invalid
    if "day" in df.columns:
        df = df.dropna(subset=["day"])

    # Create processed-data directory if it doesn't exist
    PROCESSED_DATA.parent.mkdir(parents=True, exist_ok=True)

    # Save processed dataset
    df.to_csv(PROCESSED_DATA, index=False)

    print(f"Processed data saved to: {PROCESSED_DATA}")
    print(f"Processed dataset shape: {df.shape}")

    return df


if __name__ == "__main__":
    preprocess_data()