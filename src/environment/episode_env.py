import numpy as np
import gymnasium as gym
import pandas as pd

from src.environment.trading_env import TradingEnv


class RandomEpisodeEnv(gym.Env):
    """Sample a complete market trajectory independently on every reset.

    This is used for synthetic training so PPO never learns from a transition
    between two unrelated generated trajectories. Optional weights allow the
    combined experiment to control how often real and synthetic episodes are
    sampled.
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

        lengths = [len(episode) for episode in episodes]
        if any(length < 2 for length in lengths):
            raise ValueError("Every episode must contain at least two rows.")

        if weights is None:
            weights_array = np.ones(len(episodes), dtype=np.float64)
        else:
            if len(weights) != len(episodes):
                raise ValueError("weights must match the number of episodes.")
            weights_array = np.asarray(weights, dtype=np.float64)
            if np.any(weights_array < 0) or weights_array.sum() <= 0:
                raise ValueError("weights must be non-negative and not all zero.")

        weights_array /= weights_array.sum()

        self.episodes = [episode.reset_index(drop=True).copy() for episode in episodes]
        self.weights = weights_array
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

        index = int(
            self._episode_rng.choice(
                len(self.episodes),
                p=self.weights,
            )
        )
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
