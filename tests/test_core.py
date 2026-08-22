import numpy as np
import pandas as pd

from src.data.loader import create_features
from src.environment.trading_env import TradingEnv


def sample_data():
    n = 100
    return pd.DataFrame(
        {
            "Open": np.ones(n) * 100,
            "High": np.ones(n) * 101,
            "Low": np.ones(n) * 99,
            "Close": np.linspace(100, 110, n),
            "Volume": np.ones(n) * 1_000_000,
        }
    )


def test_feature_pipeline():
    df = create_features(sample_data())

    assert {"return", "volatility_20", "volume_change", "price_range"}.issubset(df.columns)
    assert len(df) > 0
    assert np.isfinite(df[["return", "volatility_20", "volume_change", "price_range"]]).all().all()


def test_time_series_features_are_chronological():
    raw = sample_data()
    raw.index = pd.date_range("2020-01-01", periods=len(raw), freq="D")
    raw.index.name = "Date"

    df = create_features(raw)
    assert df.index.is_monotonic_increasing


def test_environment_reset_and_step():
    df = create_features(sample_data())
    env = TradingEnv(df)

    observation, _ = env.reset(seed=42)
    assert observation.shape == env.observation_space.shape
    assert env.action_space.contains(np.array([0.0], dtype=np.float32))
    assert env.action_space.contains(np.array([-1.0], dtype=np.float32))
    assert env.action_space.contains(np.array([1.0], dtype=np.float32))

    observation, reward, terminated, truncated, info = env.step(
        np.array([0.5], dtype=np.float32)
    )

    assert np.isfinite(reward)
    assert observation.shape == env.observation_space.shape
    assert info["position"] == 0.5
    assert info["portfolio_value"] > 0
    assert not truncated


def test_environment_rejects_invalid_data():
    df = create_features(sample_data())

    with np.testing.assert_raises(ValueError):
        TradingEnv(df.iloc[:1])

    with np.testing.assert_raises(ValueError):
        TradingEnv(df, initial_cash=0)

    with np.testing.assert_raises(ValueError):
        TradingEnv(df, transaction_cost=-0.01)
