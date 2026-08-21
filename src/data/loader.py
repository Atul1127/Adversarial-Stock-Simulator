from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def load_stock_data(path: str) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path, parse_dates=["Date"])

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df = df.drop_duplicates("Date").sort_values("Date")

    numeric_columns = REQUIRED_COLUMNS[1:]
    df[numeric_columns] = df[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    df = df.dropna().set_index("Date")

    if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("Invalid non-positive price values found.")

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data["return"] = np.log(data["Close"]).diff()
    data["volume_change"] = np.log(data["Volume"]).diff()
    data["volatility_20"] = data["return"].rolling(20).std()
    data["price_range"] = (data["High"] - data["Low"]) / data["Close"]

    return data.replace([np.inf, -np.inf], np.nan).dropna()


def train_test_split_time_series(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    split = int(len(df) * train_ratio)

    if split <= 0 or split >= len(df):
        raise ValueError("Dataset is too small for this split.")

    return df.iloc[:split].copy(), df.iloc[split:].copy()
