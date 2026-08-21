import numpy as np
import torch

from src.data.loader import load_stock_data, create_features
from src.generator.dataset import create_sequences, denormalize_sequences
from src.generator.model import Generator


def autocorrelation(x, lag=1):
    x = np.asarray(x)
    return np.corrcoef(x[:-lag], x[lag:])[0, 1]


def main():
    checkpoint = torch.load(
        "models/market_generator.pt",
        map_location="cpu",
        weights_only=False,
    )

    model = Generator(
        checkpoint["noise_dim"],
        checkpoint["hidden_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    df = create_features(
        load_stock_data("data/raw/AAPL.csv")
    )

    real = create_sequences(
        df,
        checkpoint["sequence_length"],
    )

    real = real.flatten()

    noise = torch.randn(
        500,
        checkpoint["sequence_length"],
        checkpoint["noise_dim"],
    )

    with torch.no_grad():
        synthetic = model(noise).numpy()

    synthetic = denormalize_sequences(
        synthetic,
        checkpoint["mean"],
        checkpoint["std"],
    ).flatten()

    print("=" * 55)
    print("MARKET GENERATOR VALIDATION")
    print("=" * 55)

    print("\nReturn statistics")
    print(f"Real mean:       {real.mean():.6f}")
    print(f"Synthetic mean:  {synthetic.mean():.6f}")

    print(f"\nReal volatility:      {real.std():.6f}")
    print(f"Synthetic volatility: {synthetic.std():.6f}")

    print("\nTail behavior")
    print(f"Real 1% quantile:      {np.quantile(real, 0.01):.6f}")
    print(f"Synthetic 1% quantile: {np.quantile(synthetic, 0.01):.6f}")

    print(f"\nReal autocorrelation:      {autocorrelation(real):.6f}")
    print(f"Synthetic autocorrelation: {autocorrelation(synthetic):.6f}")

    print("\nValidation completed.")


if __name__ == "__main__":
    main()
