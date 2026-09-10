import numpy as np
import pandas as pd

DEFAULT_FEATURES = ["return"]


def create_sequences(
    df: pd.DataFrame,
    sequence_length: int = 30,
    feature_columns: list[str] | None = None,
) -> np.ndarray:
    """Create rolling multivariate sequences for generative modeling."""
    feature_columns = feature_columns or DEFAULT_FEATURES
    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing sequence features: {missing}")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")

    values = df[feature_columns].dropna().to_numpy(dtype=np.float32)

    if len(values) <= sequence_length:
        raise ValueError("Not enough observations for requested sequence length.")

    return np.asarray(
        [
            values[i : i + sequence_length]
            for i in range(len(values) - sequence_length + 1)
        ],
        dtype=np.float32,
    )


def normalize_sequences(
    sequences: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize each generated feature independently.

    Constant features are valid and are represented by a unit scale so that
    normalization remains finite while denormalization reconstructs the
    original constant values exactly.
    """
    sequences = np.asarray(sequences, dtype=np.float32)
    if sequences.ndim != 3:
        raise ValueError("sequences must have shape (samples, timesteps, features).")
    if not np.isfinite(sequences).all():
        raise ValueError("sequences must contain only finite values.")

    mean = sequences.mean(axis=(0, 1)).astype(np.float32)
    std = sequences.std(axis=(0, 1)).astype(np.float32)

    # A zero-variance feature should normalize to zero rather than make the
    # entire dataset unusable. Unit scale preserves exact round-tripping.
    safe_std = np.where(std == 0, 1.0, std).astype(np.float32)
    normalized = (sequences - mean) / safe_std

    return normalized.astype(np.float32), mean, safe_std


def denormalize_sequences(
    sequences: np.ndarray,
    mean: np.ndarray | float,
    std: np.ndarray | float,
) -> np.ndarray:
    """Convert normalized multivariate sequences back to feature units."""
    return np.asarray(sequences) * np.asarray(std) + np.asarray(mean)
