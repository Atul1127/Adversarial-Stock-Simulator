import numpy as np
import pandas as pd
import torch

from src.generator.dataset import denormalize_sequences
from src.generator.model import Generator


def generate_synthetic_dataframe(
    checkpoint_path="models/market_generator.pt",
    num_sequences=1000,
):
    """Generate one continuous synthetic market trajectory.

    ``num_sequences`` controls the approximate trajectory length using the
    generator's trained sequence length. A single LSTM rollout is used rather
    than flattening independently generated sequences, which would create
    artificial transitions between unrelated samples.
    """
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

    sequence_length = checkpoint["sequence_length"]
    total_steps = num_sequences * sequence_length

    noise = torch.randn(
        1,
        total_steps,
        checkpoint["noise_dim"],
    )

    with torch.no_grad():
        synthetic = model(noise).squeeze(0).numpy()

    returns = denormalize_sequences(
        synthetic,
        checkpoint["mean"],
        checkpoint["std"],
    )

    rolling_volatility = (
        pd.Series(returns)
        .rolling(20, min_periods=1)
        .std()
        .fillna(0.0)
        .to_numpy()
    )

    # There is no volume/OHLC output in the return-only generator. Use a
    # deterministic activity proxy derived from absolute returns rather than
    # pretending that synthetic volume is real observed volume.
    activity_proxy = np.log1p(np.abs(returns))
    price_range_proxy = np.abs(returns)

    return pd.DataFrame(
        {
            "return": returns,
            "volume_change": activity_proxy,
            "volatility_20": rolling_volatility,
            "price_range": price_range_proxy,
        }
    )
