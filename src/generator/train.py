from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.loader import (
    create_features,
    load_stock_data,
    train_test_split_time_series,
)
from src.generator.dataset import create_sequences, normalize_sequences
from src.generator.model import Discriminator, Generator


SEED = 42


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
    """Train the market generator using only the chronological training split."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    df = create_features(load_stock_data(data_path))
    train_df, _ = train_test_split_time_series(df, train_ratio=train_ratio)

    sequences = create_sequences(train_df, sequence_length)
    sequences, mean, std = normalize_sequences(sequences)

    dataset = TensorDataset(torch.tensor(sequences))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator(noise_dim, hidden_dim).to(device)
    discriminator = Discriminator(hidden_dim).to(device)

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        for (real,) in loader:
            real = real.to(device)
            batch_size_actual = real.size(0)

            noise = torch.randn(
                batch_size_actual,
                sequence_length,
                noise_dim,
                device=device,
            )
            fake = generator(noise)

            real_labels = torch.ones(batch_size_actual, 1, device=device)
            fake_labels = torch.zeros(batch_size_actual, 1, device=device)

            d_loss = (
                criterion(discriminator(real), real_labels)
                + criterion(discriminator(fake.detach()), fake_labels)
            ) / 2

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            noise = torch.randn(
                batch_size_actual,
                sequence_length,
                noise_dim,
                device=device,
            )
            fake = generator(noise)
            g_loss = criterion(discriminator(fake), real_labels)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"D Loss: {d_loss.item():.4f} | "
                f"G Loss: {g_loss.item():.4f}"
            )

    Path("models").mkdir(exist_ok=True)
    torch.save(
        {
            "model_state_dict": generator.state_dict(),
            "noise_dim": noise_dim,
            "hidden_dim": hidden_dim,
            "sequence_length": sequence_length,
            "mean": mean,
            "std": std,
            "train_ratio": train_ratio,
            "seed": seed,
        },
        "models/market_generator.pt",
    )

    print("\nGenerator saved to models/market_generator.pt")


if __name__ == "__main__":
    train()
