import torch
import torch.nn as nn


class Generator(nn.Module):
    """Autoregressive LSTM generator for multivariate market features."""

    def __init__(
        self,
        noise_dim: int = 16,
        hidden_dim: int = 64,
        output_dim: int = 1,
        max_normalized_value: float = 3.0,
    ):
        super().__init__()
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        if max_normalized_value <= 0:
            raise ValueError("max_normalized_value must be positive.")

        self.output_dim = output_dim
        self.max_normalized_value = max_normalized_value
        self.lstm = nn.LSTM(
            input_size=output_dim + noise_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, current_state, noise, hidden=None):
        if current_state.ndim == 2:
            current_state = current_state.unsqueeze(1)
        if noise.ndim == 2:
            noise = noise.unsqueeze(1)

        x = torch.cat([current_state, noise], dim=-1)
        x, hidden = self.lstm(x, hidden)
        output = self.output(x)
        output = torch.tanh(output) * self.max_normalized_value
        return output, hidden


class Discriminator(nn.Module):
    """LSTM discriminator for real vs synthetic multivariate sequences."""

    def __init__(self, hidden_dim: int = 64, input_dim: int = 1):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, sequences):
        x, _ = self.lstm(sequences)
        return self.output(x[:, -1])
