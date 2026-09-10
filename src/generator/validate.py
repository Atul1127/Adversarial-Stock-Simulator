import numpy as np
import torch

from src.data.loader import (
    create_features,
    load_stock_data,
    train_test_split_time_series,
)
from src.generator.dataset import denormalize_sequences
from src.generator.model import Generator


def autocorrelation(x: np.ndarray, lag: int = 1) -> float:
    x = np.asarray(x, dtype=np.float64)
    if len(x) <= lag or np.std(x[:-lag]) == 0 or np.std(x[lag:]) == 0:
        return 0.0
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


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

    full_df = create_features(load_stock_data("data/raw/AAPL.csv"))
    train_df, _ = train_test_split_time_series(
        full_df,
        train_ratio=checkpoint.get("train_ratio", 0.8),
    )

    # Validate against the same chronological training distribution used to
    # fit the generator. The held-out period is not used for tuning or checks.
    real = train_df["return"].to_numpy(dtype=np.float64)

    sequence_length = checkpoint["sequence_length"]
    total_steps = len(real)
    noise = torch.randn(
        1,
        total_steps,
        checkpoint["noise_dim"],
    )

    with torch.no_grad():
        synthetic = model(noise).squeeze(0).numpy()

    synthetic = denormalize_sequences(
        synthetic,
        checkpoint["mean"],
        checkpoint["std"],
    ).astype(np.float64)

    print("=" * 65)
    print("MARKET GENERATOR VALIDATION — TRAIN DISTRIBUTION")
    print("=" * 65)

    print("\nReturn statistics")
    print(f"Real mean:             {real.mean():.6f}")
    print(f"Synthetic mean:        {synthetic.mean():.6f}")
    print(f"Real volatility:       {real.std():.6f}")
    print(f"Synthetic volatility:  {synthetic.std():.6f}")

    print("\nTail behavior")
    for quantile in (0.01, 0.05, 0.95, 0.99):
        print(
            f"Q{quantile:.0%} real:       {np.quantile(real, quantile): .6f} | "
            f"synthetic: {np.quantile(synthetic, quantile): .6f}"
        )

    print("\nSerial dependence")
    print(f"Real lag-1 autocorrelation:       {autocorrelation(real):.6f}")
    print(f"Synthetic lag-1 autocorrelation:  {autocorrelation(synthetic):.6f}")
    print(f"Real lag-5 autocorrelation:       {autocorrelation(real, 5):.6f}")
    print(f"Synthetic lag-5 autocorrelation:  {autocorrelation(synthetic, 5):.6f}")

    print("\nExtreme-return frequency")
    real_threshold = np.quantile(np.abs(real), 0.95)
    synthetic_threshold = np.quantile(np.abs(synthetic), 0.95)
    print(
        f"Real |return| > real 95th percentile: "
        f"{np.mean(np.abs(real) > real_threshold):.2%}"
    )
    print(
        f"Synthetic |return| > real 95th percentile: "
        f"{np.mean(np.abs(synthetic) > real_threshold):.2%}"
    )
    print(
        f"Synthetic |return| > synthetic 95th percentile: "
        f"{np.mean(np.abs(synthetic) > synthetic_threshold):.2%}"
    )

    print(f"\nValidated continuous trajectory length: {len(synthetic):,}")
    print(f"Generator sequence length: {sequence_length}")
    print("\nValidation completed without using the held-out test period.")


if __name__ == "__main__":
    main()
