import numpy as np
import pandas as pd
import torch

from src.generator.dataset import (
    denormalize_sequences,
    inverse_transform_features,
)
from src.generator.model import Generator

REQUIRED_SYNTHETIC_FEATURES = ["return", "volume_change", "price_range"]


def generate_synthetic_dataframe(
    checkpoint_path="models/market_generator.pt",
    num_sequences=1000,
    seed=42,
):
    """Generate one reproducible continuous autoregressive market trajectory."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    feature_columns = checkpoint.get("feature_columns")
    if feature_columns != REQUIRED_SYNTHETIC_FEATURES:
        raise ValueError(
            "Incompatible generator checkpoint. Retrain the generator with "
            "`python -m src.generator.train` before training PPO."
        )

    if checkpoint.get("feature_transform") != "signed_log1p_volume_log1p_range_v1":
        raise ValueError(
            "Incompatible generator transform metadata. Retrain with "
            "`python -m src.generator.train`."
        )

    feature_dim = checkpoint.get("output_dim", len(feature_columns))
    if feature_dim != len(feature_columns):
        raise ValueError("Generator checkpoint feature dimension is inconsistent.")

    max_normalized_value = checkpoint.get("max_normalized_value", 3.0)
    model = Generator(
        checkpoint["noise_dim"],
        checkpoint["hidden_dim"],
        feature_dim,
        max_normalized_value,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total_steps = num_sequences * checkpoint["sequence_length"]
    initial_states = torch.as_tensor(checkpoint["initial_states"], dtype=torch.float32)
    if initial_states.ndim != 2 or initial_states.shape[1] != feature_dim:
        raise ValueError("Generator checkpoint initial states are invalid.")

    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    start_index = int(torch.randint(len(initial_states), (1,), generator=rng).item())
    current = initial_states[start_index].view(1, 1, feature_dim)
    hidden = None
    outputs = [current]

    with torch.no_grad():
        for _ in range(total_steps - 1):
            noise = torch.randn(
                1,
                1,
                checkpoint["noise_dim"],
                generator=rng,
            )
            current, hidden = model(current, noise, hidden)
            outputs.append(current)

    synthetic = torch.cat(outputs, dim=1).squeeze(0).numpy()
    synthetic = denormalize_sequences(
        synthetic,
        np.asarray(checkpoint["mean"]),
        np.asarray(checkpoint["std"]),
    )
    synthetic = inverse_transform_features(synthetic, feature_columns)

    synthetic_df = pd.DataFrame(synthetic, columns=feature_columns)
    synthetic_df["volatility_20"] = (
        synthetic_df["return"]
        .rolling(20, min_periods=1)
        .std()
        .fillna(0.0)
    )

    return synthetic_df[
        ["return", "volume_change", "volatility_20", "price_range"]
    ].reset_index(drop=True)
