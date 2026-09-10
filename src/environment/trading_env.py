import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class TradingEnv(gym.Env):
    """Simple single-asset trading environment with mild risk awareness.

    The action chosen from observation t is applied to the return realized
    from t to t+1. Portfolio wealth uses simple returns.

    Action:
        -1.0 = fully short
         0.0 = neutral
        +1.0 = fully long

    Observation:
        four market features + current position + current drawdown

    Reward:
        log portfolio return minus a small current-drawdown penalty.
    """

    metadata = {"render_modes": []}
    DRAWDOWN_PENALTY = 0.02

    def __init__(
        self,
        data: pd.DataFrame,
        initial_cash: float = 100_000.0,
        transaction_cost: float = 0.001,
    ):
        super().__init__()

        required = [
            "return",
            "volume_change",
            "volatility_20",
            "price_range",
        ]
        missing = [column for column in required if column not in data.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        if len(data) < 2:
            raise ValueError("Trading environment requires at least two rows.")
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive.")
        if transaction_cost < 0:
            raise ValueError("transaction_cost cannot be negative.")

        self.data = data.reset_index(drop=True).copy()
        self.features = required
        self.initial_cash = float(initial_cash)
        self.transaction_cost = float(transaction_cost)

        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=np.full(len(self.features) + 2, -np.inf, dtype=np.float32),
            high=np.full(len(self.features) + 2, np.inf, dtype=np.float32),
            dtype=np.float32,
        )

    def _get_observation(self):
        row = self.data.iloc[self.current_step]
        values = row[self.features].to_numpy(dtype=np.float32)
        return np.concatenate(
            [
                values,
                np.array(
                    [self.position, self.current_drawdown],
                    dtype=np.float32,
                ),
            ]
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0.0
        self.portfolio_value = self.initial_cash
        self.peak_portfolio_value = self.initial_cash
        self.current_drawdown = 0.0
        return self._get_observation(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 1:
            raise ValueError("Trading action must contain exactly one value.")

        target_position = float(np.clip(action[0], -1.0, 1.0))
        previous_position = self.position
        self.position = target_position

        log_return = float(self.data.iloc[self.current_step + 1]["return"])
        market_return = float(np.expm1(log_return))
        turnover = abs(self.position - previous_position)
        trading_cost = turnover * self.transaction_cost

        portfolio_return = self.position * market_return - trading_cost
        portfolio_return = max(portfolio_return, -0.999)
        self.portfolio_value *= 1.0 + portfolio_return

        self.peak_portfolio_value = max(
            self.peak_portfolio_value,
            self.portfolio_value,
        )
        self.current_drawdown = max(
            0.0,
            1.0 - self.portfolio_value / self.peak_portfolio_value,
        )

        reward = float(
            np.log1p(portfolio_return)
            - self.DRAWDOWN_PENALTY * self.current_drawdown
        )

        self.current_step += 1
        terminated = self.current_step >= len(self.data) - 1
        truncated = False
        observation = (
            np.zeros(self.observation_space.shape, dtype=np.float32)
            if terminated
            else self._get_observation()
        )

        info = {
            "portfolio_value": self.portfolio_value,
            "position": self.position,
            "market_return": market_return,
            "portfolio_return": portfolio_return,
            "trading_cost": trading_cost,
            "drawdown": self.current_drawdown,
        }

        return observation, reward, terminated, truncated, info
