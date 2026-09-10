import numpy as np
import torch

from src.data.loader import (
    create_features,
    load_stock_data,
    train_test_split_time_series,
)
from src.generator.dataset import denormalize_sequences, inverse_transform_features
from src.generator.model import Generator


FEATURE_COLUMNS = ["return", "volume_change", "price_range"]
EXPECTED_TRANSFORM = "signed_log1p_volume_log1p_range_v1"
NUM_VALIDATION_EPISODES = 512


def autocorrelation(x: np.ndarray, lag: int = 1) -> float:
    x = np.asarray(x, dtype=np.float64)
    if len(x) <= lag:
        return 0.0
    x0 = x[:-lag] - x[:-lag].mean()
    x1 = x[lag:] - x[lag:].mean()
    denominator = np.sqrt((x0**2).mean() * (x1**2).mean())
    if denominator <= 1e-12:
        return 0.0
    return float((x0 * x1).mean() / denominator)


def _load_model(checkpoint):
    feature_dim = checkpoint["output_dim"]
    model = Generator(
        checkpoint["noise_dim"],
        checkpoint["hidden_dim"],
        feature_dim,
        checkpoint.get("min_scale", 0.05),
        checkpoint.get("max_scale", 2.0),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def generate_validation_episodes(checkpoint, num_episodes=NUM_VALIDATION_EPISODES):
    """Generate independent episodes using exactly the trained sequence horizon."""
    feature_columns = checkpoint["feature_columns"]
    feature_dim = checkpoint["output_dim"]
    length = checkpoint["sequence_length"]
    initial_states = torch.as_tensor(checkpoint["initial_states"], dtype=torch.float32)
    if initial_states.ndim != 2 or initial_states.shape[1] != feature_dim:
        raise ValueError("Generator checkpoint initial states are invalid.")

    model = _load_model(checkpoint)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(checkpoint.get("seed", 42))

    episodes = []
    with torch.no_grad():
        for _ in range(num_episodes):
            start_index = int(
                torch.randint(len(initial_states), (1,), generator=rng).item()
            )
            current = initial_states[start_index].view(1, 1, feature_dim)
            hidden = None
            outputs = [current]

            for _ in range(length - 1):
                noise = torch.randn(
                    1,
                    1,
                    checkpoint["noise_dim"],
                    generator=rng,
                )
                current, _, _, hidden = model.step(current, noise, hidden)
                outputs.append(current)

            model_space = torch.cat(outputs, dim=1).squeeze(0).numpy()
            transformed = denormalize_sequences(
                model_space,
                np.asarray(checkpoint["mean"]),
                np.asarray(checkpoint["std"]),
            )
            episodes.append(
                inverse_transform_features(
                    transformed,
                    feature_columns,
                ).astype(np.float64)
            )

    return episodes


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
            "Incompatible generator feature transform. Retrain with "
            "`python -m src.generator.train`."
        )

    full_df = create_features(load_stock_data("data/raw/AAPL.csv"))
    train_df, _ = train_test_split_time_series(
        full_df,
        train_ratio=checkpoint.get("train_ratio", 0.8),
    )
    real = train_df[feature_columns].to_numpy(dtype=np.float64)
    episodes = generate_validation_episodes(checkpoint)
    synthetic = np.concatenate(episodes, axis=0)

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

    return_idx = feature_columns.index("return")
    real_returns = real[:, return_idx]
    synthetic_returns = synthetic[:, return_idx]
    episode_lag1 = np.asarray(
        [autocorrelation(episode[:, return_idx]) for episode in episodes]
    )
    episode_lag5 = np.asarray(
        [autocorrelation(episode[:, return_idx], 5) for episode in episodes]
    )

    real_lag1 = autocorrelation(real_returns)
    real_lag5 = autocorrelation(real_returns, 5)
    synthetic_lag1 = float(episode_lag1.mean())
    synthetic_lag5 = float(episode_lag5.mean())

    print("\nReturn serial dependence")
    print(f"  Real lag-1:        {real_lag1: .6f}")
    print(f"  Synthetic lag-1:   {synthetic_lag1: .6f}")
    print(f"  Real lag-5:        {real_lag5: .6f}")
    print(f"  Synthetic lag-5:   {synthetic_lag5: .6f}")

    real_threshold = np.quantile(np.abs(real_returns), 0.95)
    synthetic_extreme_rate = float(
        np.mean(np.abs(synthetic_returns) > real_threshold)
    )

    return_mean_offset = abs(
        synthetic_returns.mean() - real_returns.mean()
    ) / max(real_returns.std(), 1e-8)
    return_std_ratio = synthetic_returns.std() / max(real_returns.std(), 1e-8)
    lag1_gap = abs(synthetic_lag1 - real_lag1)

    print("\nValidation gates")
    print(f"  Mean offset (real std units): {return_mean_offset:.3f}")
    print(f"  Volatility ratio:             {return_std_ratio:.3f}")
    print(f"  Extreme-return rate:          {synthetic_extreme_rate:.2%}")
    print(f"  Lag-1 correlation gap:        {lag1_gap:.3f}")

    failures = []
    if return_mean_offset > 2.0:
        failures.append(f"mean_offset_z={return_mean_offset:.3f}")
    if not 0.50 <= return_std_ratio <= 1.50:
        failures.append(f"std_ratio={return_std_ratio:.3f}")
    if not 0.01 <= synthetic_extreme_rate <= 0.20:
        failures.append(f"extreme_rate={synthetic_extreme_rate:.2%}")
    if lag1_gap > 0.20 or abs(synthetic_lag1) > 0.25:
        failures.append(
            f"lag1_real={real_lag1:.3f},lag1_synthetic={synthetic_lag1:.3f}"
        )

    if failures:
        raise RuntimeError(
            "Synthetic generator failed validation: " + "; ".join(failures)
        )

    print(
        f"\nValidated {len(episodes)} independent episodes x "
        f"{len(episodes[0])} steps."
    )
    print(f"Generated features: {', '.join(feature_columns)}")
    print("Generator validation PASSED without using the held-out test period.")


if __name__ == "__main__":
    main()
