import torch
import torch.nn as nn


class Generator(nn.Module):
    """LSTM generator for synthetic market return sequences."""

    def __init__(self, noise_dim: int = 16, hidden_dim: int = 64):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=noise_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, noise):
        x, _ = self.lstm(noise)
        return self.output(x).squeeze(-1)


class Discriminator(nn.Module):
    """LSTM discriminator for real vs synthetic return sequences."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, sequences):
        x = sequences.unsqueeze(-1)
        x, _ = self.lstm(x)
        return self.output(x[:, -1])
