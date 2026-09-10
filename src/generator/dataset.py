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
    if sequence_length <= 1:
        raise ValueError("sequence_length must be greater than one for autoregressive training.")

    values = df[feature_columns].dropna().to_numpy(dtype=np.float32)

    if len(values) <= sequence_length:
        raise ValueError("Not enough observations for requested sequence length.")

    return np.asarray(
        [values[i : i + sequence_length] for i in range(len(values) - sequence_length + 1)],
        dtype=np.float32,
    )


def transform_features(values: np.ndarray, feature_columns: list[str]) -> np.ndarray:
    """Transform skewed market features into stable model space."""
    values = np.asarray(values, dtype=np.float32).copy()
    columns = list(feature_columns)

    for idx, name in enumerate(columns):
        if name == "volume_change":
            values[..., idx] = np.sign(values[..., idx]) * np.log1p(np.abs(values[..., idx]))
        elif name == "price_range":
            if np.any(values[..., idx] < 0):
                raise ValueError("price_range must be non-negative before transformation.")
            values[..., idx] = np.log1p(values[..., idx])

    return values


def inverse_transform_features(values: np.ndarray, feature_columns: list[str]) -> np.ndarray:
    """Map transformed generator outputs back to market feature units."""
    values = np.asarray(values, dtype=np.float32).copy()
    columns = list(feature_columns)

    for idx, name in enumerate(columns):
        if name == "volume_change":
            values[..., idx] = np.sign(values[..., idx]) * np.expm1(np.abs(values[..., idx]))
        elif name == "price_range":
            values[..., idx] = np.maximum(np.expm1(values[..., idx]), 0.0)

    return values


def normalize_sequences(
    sequences: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize each generated feature independently."""
    sequences = np.asarray(sequences, dtype=np.float32)
    if sequences.ndim != 3:
        raise ValueError("sequences must have shape (samples, timesteps, features).")
    if not np.isfinite(sequences).all():
        raise ValueError("sequences must contain only finite values.")

    mean = sequences.mean(axis=(0, 1)).astype(np.float32)
    std = sequences.std(axis=(0, 1)).astype(np.float32)
    safe_std = np.where(std < 1e-8, 1.0, std).astype(np.float32)

    normalized = (sequences - mean) / safe_std
    return normalized.astype(np.float32), mean, safe_std


def denormalize_sequences(
    sequences: np.ndarray,
    mean: np.ndarray | float,
    std: np.ndarray | float,
) -> np.ndarray:
    """Convert normalized multivariate sequences back to transformed feature units."""
    return np.asarray(sequences) * np.asarray(std) + np.asarray(mean)
