import numpy as np
import pandas as pd
import torch

from src.generator.dataset import denormalize_sequences
from src.generator.model import Generator

REQUIRED_SYNTHETIC_FEATURES = ["return", "volume_change", "price_range"]


def generate_synthetic_dataframe(
    checkpoint_path="models/market_generator.pt",
    num_sequences=1000,
    seed=42,
):
    """Generate one reproducible continuous multivariate market trajectory."""
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

    transforms = checkpoint.get("transforms", {})
    if transforms.get("price_range") != "log1p":
        raise ValueError(
            "Incompatible generator checkpoint transform metadata. Retrain "
            "the generator with `python -m src.generator.train`."
        )

    feature_dim = checkpoint.get("output_dim", len(feature_columns))
    if feature_dim != len(feature_columns):
        raise ValueError("Generator checkpoint feature dimension is inconsistent.")

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
    synthetic_df["price_range"] = np.expm1(synthetic_df["price_range"])
    synthetic_df["price_range"] = synthetic_df["price_range"].clip(lower=0.0)

    synthetic_df["volatility_20"] = (
        synthetic_df["return"]
        .rolling(20, min_periods=1)
        .std()
        .fillna(0.0)
    )

    return synthetic_df[
        ["return", "volume_change", "volatility_20", "price_range"]
    ].reset_index(drop=True)
