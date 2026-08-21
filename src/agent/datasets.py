import numpy as np
import pandas as pd
import torch

from src.generator.dataset import denormalize_sequences
from src.generator.model import Generator


def generate_synthetic_dataframe(
    checkpoint_path="models/market_generator.pt",
    num_sequences=1000,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model = Generator(
        checkpoint["noise_dim"],
        checkpoint["hidden_dim"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    noise = torch.randn(
        num_sequences,
        checkpoint["sequence_length"],
        checkpoint["noise_dim"],
    )

    with torch.no_grad():
        synthetic = model(noise).numpy()

    synthetic = denormalize_sequences(
        synthetic,
        checkpoint["mean"],
        checkpoint["std"],
    )

    returns = synthetic.reshape(-1)

    return pd.DataFrame({
        "return": returns,
        "volume_change": np.zeros_like(returns),
        "volatility_20": pd.Series(returns).rolling(20).std().bfill(),
        "price_range": np.abs(returns),
    })
