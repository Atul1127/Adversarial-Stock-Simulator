import numpy as np
import gymnasium as gym
import pandas as pd

from src.environment.trading_env import TradingEnv


def make_rolling_episodes(
    data: pd.DataFrame,
    episode_length: int = 30,
    stride: int = 1,
) -> list[pd.DataFrame]:
    """Create chronological rolling windows for episodic RL training."""
    if episode_length <= 1:
        raise ValueError("episode_length must be greater than one.")
    if stride <= 0:
        raise ValueError("stride must be positive.")

    data = data.reset_index(drop=True).copy()
    if len(data) < episode_length:
        raise ValueError("Data is shorter than the requested episode length.")

    return [
        data.iloc[start : start + episode_length].reset_index(drop=True).copy()
        for start in range(0, len(data) - episode_length + 1, stride)
    ]


class RandomEpisodeEnv(gym.Env):
    """Sample a complete market trajectory independently on every reset.

    Episodes are kept independent so PPO never learns a transition between
    unrelated real/synthetic trajectories. Optional weights control source
    sampling in the combined experiment.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        episodes: list[pd.DataFrame],
        weights: list[float] | None = None,
        initial_cash: float = 100_000.0,
        transaction_cost: float = 0.001,
    ):
        super().__init__()
        if not episodes:
            raise ValueError("episodes must contain at least one trajectory.")
        if any(len(episode) < 2 for episode in episodes):
            raise ValueError("Every episode must contain at least two rows.")

        if weights is None:
            weights_array = np.ones(len(episodes), dtype=np.float64)
        else:
            if len(weights) != len(episodes):
                raise ValueError("weights must match the number of episodes.")
            weights_array = np.asarray(weights, dtype=np.float64)
            if np.any(weights_array < 0) or weights_array.sum() <= 0:
                raise ValueError("weights must be non-negative and not all zero.")

        self.episodes = [episode.reset_index(drop=True).copy() for episode in episodes]
        self.weights = weights_array / weights_array.sum()
        self.initial_cash = float(initial_cash)
        self.transaction_cost = float(transaction_cost)
        self._episode_rng = np.random.default_rng()
        self._env: TradingEnv | None = None

        reference = TradingEnv(
            self.episodes[0],
            initial_cash=self.initial_cash,
            transaction_cost=self.transaction_cost,
        )
        self.action_space = reference.action_space
        self.observation_space = reference.observation_space

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._episode_rng = np.random.default_rng(seed)

        index = int(self._episode_rng.choice(len(self.episodes), p=self.weights))
        self._env = TradingEnv(
            self.episodes[index],
            initial_cash=self.initial_cash,
            transaction_cost=self.transaction_cost,
        )
        return self._env.reset(seed=seed)

    def step(self, action):
        if self._env is None:
            raise RuntimeError("reset() must be called before step().")
        return self._env.step(action)

    def render(self):
        return None

    def close(self):
        self._env = None
