import numpy as np
import torch

from src.data.loader import (
    create_features,
    load_stock_data,
    train_test_split_time_series,
)
from src.generator.dataset import denormalize_sequences
from src.generator.model import Generator


FEATURE_COLUMNS = ["return", "volume_change", "price_range"]


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

    feature_columns = checkpoint.get("feature_columns")
    if feature_columns != FEATURE_COLUMNS:
        raise ValueError(
            "Incompatible generator checkpoint. Retrain with "
            "`python -m src.generator.train`."
        )

    feature_dim = checkpoint.get("output_dim")
    if feature_dim != len(feature_columns):
        raise ValueError("Generator checkpoint feature dimension is inconsistent.")

    model = Generator(
        checkpoint["noise_dim"],
        checkpoint["hidden_dim"],
        feature_dim,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    full_df = create_features(load_stock_data("data/raw/AAPL.csv"))
    train_df, _ = train_test_split_time_series(
        full_df,
        train_ratio=checkpoint.get("train_ratio", 0.8),
    )

    real = train_df[feature_columns].to_numpy(dtype=np.float64)
    total_steps = len(real)

    rng = torch.Generator(device="cpu")
    rng.manual_seed(checkpoint.get("seed", 42))
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
    ).astype(np.float64)

    print("=" * 70)
    print("MARKET GENERATOR VALIDATION — TRAIN DISTRIBUTION")
    print("=" * 70)

    for idx, feature in enumerate(feature_columns):
        real_feature = real[:, idx]
        synthetic_feature = synthetic[:, idx]
        print(f"\n{feature}")
        print(
            f"  Real mean/std:       {real_feature.mean(): .6f} / "
            f"{real_feature.std(): .6f}"
        )
        print(
            f"  Synthetic mean/std:  {synthetic_feature.mean(): .6f} / "
            f"{synthetic_feature.std(): .6f}"
        )
        for quantile in (0.01, 0.05, 0.95, 0.99):
            print(
                f"  Q{quantile:.0%}: real {np.quantile(real_feature, quantile): .6f} | "
                f"synthetic {np.quantile(synthetic_feature, quantile): .6f}"
            )

    real_returns = real[:, feature_columns.index("return")]
    synthetic_returns = synthetic[:, feature_columns.index("return")]
    print("\nReturn serial dependence")
    print(f"  Real lag-1:       {autocorrelation(real_returns): .6f}")
    print(f"  Synthetic lag-1:  {autocorrelation(synthetic_returns): .6f}")
    print(f"  Real lag-5:       {autocorrelation(real_returns, 5): .6f}")
    print(f"  Synthetic lag-5:  {autocorrelation(synthetic_returns, 5): .6f}")

    real_threshold = np.quantile(np.abs(real_returns), 0.95)
    print("\nExtreme-return frequency")
    print(
        "  Synthetic |return| above real 95th percentile: "
        f"{np.mean(np.abs(synthetic_returns) > real_threshold):.2%}"
    )

    print(f"\nValidated continuous trajectory length: {len(synthetic):,}")
    print(f"Generated features: {', '.join(feature_columns)}")
    print("Validation completed without using the held-out test period.")


if __name__ == "__main__":
    main()
