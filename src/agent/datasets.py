import numpy as np
import pandas as pd
import torch

from src.generator.dataset import (
    denormalize_sequences,
    inverse_transform_features,
)
from src.generator.model import Generator

REQUIRED_SYNTHETIC_FEATURES = ["return", "volume_change", "price_range"]
EXPECTED_TRANSFORM = "signed_log1p_volume_log1p_range_v1"


def _load_generator(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    feature_columns = checkpoint.get("feature_columns")
    if feature_columns != REQUIRED_SYNTHETIC_FEATURES:
        raise ValueError(
            "Incompatible generator checkpoint. Retrain the generator with "
            "`python -m src.generator.train`."
        )
    if checkpoint.get("feature_transform") != EXPECTED_TRANSFORM:
        raise ValueError(
            "Incompatible generator feature transform. Retrain the generator."
        )

    feature_dim = checkpoint.get("output_dim", len(feature_columns))
    if feature_dim != len(feature_columns):
        raise ValueError("Generator checkpoint feature dimension is inconsistent.")

    model = Generator(
        checkpoint["noise_dim"],
        checkpoint["hidden_dim"],
        feature_dim,
        checkpoint.get("min_scale", 0.05),
        checkpoint.get("max_scale", 2.0),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return checkpoint, model


def _generate_episode(checkpoint, model, rng, length):
    feature_columns = checkpoint["feature_columns"]
    feature_dim = len(feature_columns)
    initial_states = torch.as_tensor(
        checkpoint["initial_states"], dtype=torch.float32
    )
    start_index = int(torch.randint(len(initial_states), (1,), generator=rng).item())
    current = initial_states[start_index].view(1, 1, feature_dim)
    hidden = None
    outputs = [current]

    with torch.no_grad():
        for _ in range(length - 1):
            noise = torch.randn(
                1,
                1,
                checkpoint["noise_dim"],
                generator=rng,
            )
            current, _, _, hidden = model.step(current, noise, hidden)
            outputs.append(current)

    values = torch.cat(outputs, dim=1).squeeze(0).numpy()
    values = denormalize_sequences(
        values,
        np.asarray(checkpoint["mean"]),
        np.asarray(checkpoint["std"]),
    )
    values = inverse_transform_features(values, feature_columns)

    frame = pd.DataFrame(values, columns=feature_columns)
    frame["volatility_20"] = (
        frame["return"]
        .rolling(20, min_periods=1)
        .std()
        .fillna(0.0)
    )
    return frame[["return", "volume_change", "volatility_20", "price_range"]]


def generate_synthetic_episodes(
    checkpoint_path="models/market_generator.pt",
    num_episodes=512,
    seed=42,
):
    """Generate independent synthetic market episodes matching training horizon."""
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive.")

    checkpoint, model = _load_generator(checkpoint_path)
    length = checkpoint["sequence_length"]
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    return [
        _generate_episode(checkpoint, model, rng, length)
        for _ in range(num_episodes)
    ]


def generate_synthetic_dataframe(
    checkpoint_path="models/market_generator.pt",
    num_sequences=1000,
    seed=42,
):
    """Generate synthetic data as a compatibility view of independent episodes.

    The returned frame contains an ``episode_id`` column. Training code should
    prefer ``generate_synthetic_episodes`` so episode boundaries are explicit.
    """
    episodes = generate_synthetic_episodes(
        checkpoint_path=checkpoint_path,
        num_episodes=num_sequences,
        seed=seed,
    )
    return pd.concat(
        [episode.assign(episode_id=index) for index, episode in enumerate(episodes)],
        ignore_index=True,
    )
