import numpy as np
import pandas as pd

from src.data.loader import create_features
from src.environment.trading_env import TradingEnv


def sample_data():
    n = 100

    return pd.DataFrame({
        "Open": np.ones(n) * 100,
        "High": np.ones(n) * 101,
        "Low": np.ones(n) * 99,
        "Close": np.linspace(100, 110, n),
        "Volume": np.ones(n) * 1_000_000,
    })


def test_feature_pipeline():
    df = create_features(sample_data())

    assert "return" in df.columns
    assert "volatility_20" in df.columns
    assert len(df) > 0


def test_environment():
    df = create_features(sample_data())
    env = TradingEnv(df)

    observation, _ = env.reset()

    assert observation.shape == env.observation_space.shape

    action = np.array([0.5], dtype=np.float32)

    observation, reward, terminated, truncated, info = env.step(action)

    assert np.isfinite(reward)
    assert observation.shape == env.observation_space.shape
