import numpy as np
import torch

from src.data.loader import (
    create_features,
    load_stock_data,
    train_test_split_time_series,
)
from src.generator.dataset import denormalize_sequences, inverse_transform_features, transform_features
from src.generator.model import Generator


FEATURE_COLUMNS = ["return", "volume_change", "price_range"]
EXPECTED_TRANSFORM = "signed_log1p_volume_log1p_range_v1"


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
    if checkpoint.get("feature_transform") != EXPECTED_TRANSFORM:
        raise ValueError(
            "Incompatible generator transform metadata. Retrain with "
            "`python -m src.generator.train`."
        )

    feature_dim = checkpoint.get("output_dim")
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

    full_df = create_features(load_stock_data("data/raw/AAPL.csv"))
    train_df, _ = train_test_split_time_series(
        full_df,
        train_ratio=checkpoint.get("train_ratio", 0.8),
    )

    real = train_df[feature_columns].to_numpy(dtype=np.float32)
    transformed_real = transform_features(real, feature_columns)
    normalized_real, _, _ = denormalize_sequences(
        transformed_real,
        np.asarray(checkpoint["mean"]),
        np.asarray(checkpoint["std"]),
    ), None, None
    del normalized_real  # transform_real is only used to document the training space.

    initial_states = torch.as_tensor(checkpoint["initial_states"], dtype=torch.float32)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(checkpoint.get("seed", 42))
    start_index = int(torch.randint(len(initial_states), (1,), generator=rng).item())
    current = initial_states[start_index].view(1, 1, feature_dim)
    hidden = None
    outputs = [current]

    with torch.no_grad():
        for _ in range(len(real) - 1):
            noise = torch.randn(
                1,
                1,
                checkpoint["noise_dim"],
                generator=rng,
            )
            current, hidden = model(current, noise, hidden)
            outputs.append(current)

    synthetic_model_space = torch.cat(outputs, dim=1).squeeze(0).numpy()
    synthetic = denormalize_sequences(
        synthetic_model_space,
        np.asarray(checkpoint["mean"]),
        np.asarray(checkpoint["std"]),
    )
    synthetic = inverse_transform_features(synthetic, feature_columns).astype(np.float64)

    real = real.astype(np.float64)
    print("=" * 70)
    print("MARKET GENERATOR VALIDATION — TRAIN DISTRIBUTION")
    print("=" * 70)

    for idx, feature in enumerate(feature_columns):
        real_feature = real[:, idx]
        synthetic_feature = synthetic[:, idx]
        print(f"\n{feature}")
        print(
            f"  Real mean/std:       {real_feature.mean(): .6f} / {real_feature.std(): .6f}"
        )
        print(
            f"  Synthetic mean/std:  {synthetic_feature.mean(): .6f} / {synthetic_feature.std(): .6f}"
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
    synthetic_extreme_rate = np.mean(np.abs(synthetic_returns) > real_threshold)
    print("\nExtreme-return frequency")
    print(
        "  Synthetic |return| above real 95th percentile: "
        f"{synthetic_extreme_rate:.2%}"
    )

    # A generator is considered usable only when it remains in a plausible
    # distributional range; these checks prevent training PPO on collapsed data.
    return_mean_ratio = abs(synthetic_returns.mean() - real_returns.mean()) / max(
        real_returns.std(), 1e-8
    )
    return_std_ratio = synthetic_returns.std() / max(real_returns.std(), 1e-8)
    if return_mean_ratio > 2.0 or not 0.25 <= return_std_ratio <= 2.0:
        raise RuntimeError(
            "Synthetic return distribution failed validation. "
            f"mean_offset_z={return_mean_ratio:.3f}, std_ratio={return_std_ratio:.3f}."
        )
    if synthetic_extreme_rate > 0.25:
        raise RuntimeError(
            "Synthetic tail frequency failed validation: "
            f"{synthetic_extreme_rate:.2%} above the real 95th-percentile threshold."
        )

    print(f"\nValidated trajectory length: {len(synthetic):,}")
    print(f"Generated features: {', '.join(feature_columns)}")
    print("Generator validation PASSED without using the held-out test period.")


if __name__ == "__main__":
    main()
