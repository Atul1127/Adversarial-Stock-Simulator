import pandas as pd
import numpy as np


def load_stock_data(path):
    """
    Load historical stock data from a CSV file.
    Expected columns: Date, Open, High, Low, Close, Volume
    """
    df = pd.read_csv(path)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
        df = df.set_index("Date")

    required = ["Open", "High", "Low", "Close", "Volume"]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[required].dropna()

    return df


def create_features(df):
    """
    Create basic financial features for the simulator.
    """

    data = df.copy()

    data["Return"] = data["Close"].pct_change()

    data["Log_Return"] = np.log(
        data["Close"] / data["Close"].shift(1)
    )

    data["Volatility"] = (
        data["Return"]
        .rolling(window=20)
        .std()
    )

    data["Volume_Change"] = (
        data["Volume"].pct_change()
    )

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()

    return data


def train_test_split_time_series(df, train_ratio=0.8):
    """
    Time-aware train/test split.
    No random shuffling to avoid data leakage.
    """

    split_index = int(len(df) * train_ratio)

    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()

    return train, test
