from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.loader import (
    create_features,
    load_stock_data,
    train_test_split_time_series,
)
from src.generator.dataset import create_sequences, normalize_sequences, transform_features
from src.generator.model import Discriminator, Generator


SEED = 42
FEATURE_COLUMNS = ["return", "volume_change", "price_range"]
MAX_NORMALIZED_VALUE = 3.0
RECONSTRUCTION_WEIGHT = 5.0
MOMENT_WEIGHT = 0.5


def _rollout(generator, start_state, steps, noise_dim, device):
    """Free-run an autoregressive trajectory from a starting state."""
    current = start_state
    hidden = None
    outputs = []

    for _ in range(steps):
        noise = torch.randn(current.size(0), 1, noise_dim, device=device)
        current, hidden = generator(current, noise, hidden)
        outputs.append(current)

    return torch.cat(outputs, dim=1)


def train(
    data_path="data/raw/AAPL.csv",
    epochs=50,
    batch_size=64,
    noise_dim=16,
    hidden_dim=64,
    sequence_length=30,
    train_ratio=0.8,
    seed=SEED,
):
    """Train a bounded autoregressive adversarial market generator."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    df = create_features(load_stock_data(data_path))
    train_df, _ = train_test_split_time_series(df, train_ratio=train_ratio)

    raw_sequences = create_sequences(
        train_df,
        sequence_length=sequence_length,
        feature_columns=FEATURE_COLUMNS,
    )
    transformed = transform_features(raw_sequences, FEATURE_COLUMNS)
    sequences, mean, std = normalize_sequences(transformed)

    dataset = TensorDataset(torch.tensor(sequences))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_dim = len(FEATURE_COLUMNS)

    generator = Generator(
        noise_dim,
        hidden_dim,
        feature_dim,
        MAX_NORMALIZED_VALUE,
    ).to(device)
    discriminator = Discriminator(hidden_dim, feature_dim).to(device)

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    reconstruction = torch.nn.SmoothL1Loss()

    for epoch in range(epochs):
        for (real,) in loader:
            real = real.to(device)
            context = real[:, :1, :]
            target = real[:, 1:, :]
            steps = target.size(1)

            fake = _rollout(generator, context, steps, noise_dim, device)

            real_labels = torch.ones(real.size(0), 1, device=device)
            fake_labels = torch.zeros(real.size(0), 1, device=device)

            d_loss = (
                criterion(discriminator(target), real_labels)
                + criterion(discriminator(fake.detach()), fake_labels)
            ) / 2

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            fake = _rollout(generator, context, steps, noise_dim, device)
            g_adv = criterion(discriminator(fake), real_labels)
            g_recon = reconstruction(fake, target)

            fake_mean = fake.mean(dim=(0, 1))
            target_mean = target.mean(dim=(0, 1))
            fake_std = fake.std(dim=(0, 1))
            target_std = target.std(dim=(0, 1))
            g_moment = torch.mean((fake_mean - target_mean) ** 2) + torch.mean(
                (fake_std - target_std) ** 2
            )

            g_loss = (
                g_adv
                + RECONSTRUCTION_WEIGHT * g_recon
                + MOMENT_WEIGHT * g_moment
            )

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"D Loss: {d_loss.item():.4f} | "
                f"G Loss: {g_loss.item():.4f} | "
                f"Recon: {g_recon.item():.4f}"
            )

    initial_states = sequences[: min(128, len(sequences)), 0, :]

    Path("models").mkdir(exist_ok=True)
    torch.save(
        {
            "model_state_dict": generator.state_dict(),
            "noise_dim": noise_dim,
            "hidden_dim": hidden_dim,
            "sequence_length": sequence_length,
            "output_dim": feature_dim,
            "feature_columns": FEATURE_COLUMNS,
            "mean": mean,
            "std": std,
            "train_ratio": train_ratio,
            "seed": seed,
            "initial_states": initial_states,
            "max_normalized_value": MAX_NORMALIZED_VALUE,
            "feature_transform": "signed_log1p_volume_log1p_range_v1",
        },
        "models/market_generator.pt",
    )

    print("\nGenerator saved to models/market_generator.pt")


if __name__ == "__main__":
    train()
