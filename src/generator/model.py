import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """Probabilistic autoregressive LSTM for multivariate market features."""

    def __init__(
        self,
        noise_dim: int = 16,
        hidden_dim: int = 64,
        output_dim: int = 1,
        min_scale: float = 0.05,
        max_scale: float = 2.0,
    ):
        super().__init__()
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        if noise_dim <= 0 or hidden_dim <= 0:
            raise ValueError("noise_dim and hidden_dim must be positive.")
        if min_scale <= 0 or max_scale <= min_scale:
            raise ValueError("Invalid generator scale bounds.")

        self.output_dim = output_dim
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)

        self.lstm = nn.LSTM(
            input_size=output_dim + noise_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        self.scale_head = nn.Linear(hidden_dim, output_dim)

    def step(self, current_state, noise, hidden=None, sample_generator=None):
        """Predict and sample the next normalized market state."""
        if current_state.ndim == 2:
            current_state = current_state.unsqueeze(1)
        if noise.ndim == 2:
            noise = noise.unsqueeze(1)

        x = torch.cat([current_state, noise], dim=-1)
        hidden_output, hidden = self.lstm(x, hidden)

        mean = self.mean_head(hidden_output)
        scale = F.softplus(self.scale_head(hidden_output)) + self.min_scale
        scale = torch.clamp(scale, max=self.max_scale)

        if sample_generator is None:
            sample_noise = torch.randn_like(mean)
        else:
            sample_noise = torch.randn(
                mean.shape,
                dtype=mean.dtype,
                device=mean.device,
                generator=sample_generator,
            )

        sample = mean + scale * sample_noise
        sample = torch.clamp(sample, -4.0, 4.0)
        return sample, mean, scale, hidden

    def forward(self, current_state, noise, hidden=None, sample_generator=None):
        return self.step(current_state, noise, hidden, sample_generator)


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
