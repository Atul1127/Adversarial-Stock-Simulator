import numpy as np
import pandas as pd


def create_sequences(
    df: pd.DataFrame,
    sequence_length: int = 30,
) -> np.ndarray:
    """Create rolling return sequences for generative modeling."""

    if "return" not in df.columns:
        raise ValueError("DataFrame must contain a 'return' column.")

    returns = df["return"].dropna().to_numpy(dtype=np.float32)

    if len(returns) <= sequence_length:
        raise ValueError("Not enough observations for requested sequence length.")

    sequences = []

    for i in range(len(returns) - sequence_length + 1):
        sequences.append(returns[i:i + sequence_length])

    return np.asarray(sequences, dtype=np.float32)


def normalize_sequences(
    sequences: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Normalize return sequences for stable model training."""

    mean = float(sequences.mean())
    std = float(sequences.std())

    if std == 0:
        raise ValueError("Cannot normalize zero-variance sequences.")

    normalized = (sequences - mean) / std

    return normalized.astype(np.float32), mean, std


def denormalize_sequences(
    sequences: np.ndarray,
    mean: float,
    std: float,
) -> np.ndarray:
    """Convert normalized sequences back to returns."""

    return sequences * std + mean
