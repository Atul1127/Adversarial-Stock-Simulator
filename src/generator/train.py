from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.generator.dataset import create_sequences, normalize_sequences
from src.generator.model import Generator, Discriminator
from src.data.loader import load_stock_data, create_features


def train(
    data_path="data/raw/AAPL.csv",
    epochs=50,
    batch_size=64,
    noise_dim=16,
    hidden_dim=64,
    sequence_length=30,
):

    df = create_features(load_stock_data(data_path))

    sequences = create_sequences(df, sequence_length)
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

            # Train discriminator
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

            # Train generator
            noise = torch.randn(
                batch_size_actual,
                sequence_length,
                noise_dim,
                device=device,
            )

            fake = generator(noise)

            g_loss = criterion(
                discriminator(fake),
                real_labels,
            )

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
        },
        "models/market_generator.pt",
    )

    print("\nGenerator saved to models/market_generator.pt")


if __name__ == "__main__":
    train()
