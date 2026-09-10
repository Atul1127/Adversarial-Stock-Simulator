import numpy as np
import pandas as pd
import torch

from src.generator.dataset import denormalize_sequences
from src.generator.model import Generator


def generate_synthetic_dataframe(
    checkpoint_path="models/market_generator.pt",
    num_sequences=1000,
    seed=42,
):
    """Generate one continuous synthetic trajectory with return and activity features."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    feature_columns = checkpoint["feature_columns"]
    feature_dim = checkpoint.get("output_dim", len(feature_columns))
    model = Generator(
        checkpoint["noise_dim"],
        checkpoint["hidden_dim"],
        feature_dim,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sequence_length = checkpoint["sequence_length"]
    total_steps = num_sequences * sequence_length

    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    noise = torch.randn(
        1,
        total_steps,
        checkpoint["noise_dim"],
        generator=rng,
    )

    with torch.no_grad():
        synthetic = model(noise).squeeze(0).numpy()

    synthetic = denormalize_sequences(
        synthetic,
        np.asarray(checkpoint["mean"]),
        np.asarray(checkpoint["std"]),
    )

    synthetic_df = pd.DataFrame(
        synthetic,
        columns=feature_columns,
    )

    synthetic_df["volatility_20"] = (
        synthetic_df["return"]
        .rolling(20, min_periods=1)
        .std()
        .fillna(0.0)
    )

    # Preserve the environment's expected column order.
    return synthetic_df[
        ["return", "volume_change", "volatility_20", "price_range"]
    ].reset_index(drop=True)
