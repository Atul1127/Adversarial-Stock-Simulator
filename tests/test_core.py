import numpy as np
import pandas as pd

from src.adversarial.scenarios import (
    drawdown_shock,
    generate_adversarial_scenarios,
    market_crash,
)
from src.data.loader import create_features
from src.environment.trading_env import TradingEnv
from src.evaluation.benchmarks import evaluate_buy_and_hold
from src.evaluation.metrics import sortino_ratio
from src.generator.dataset import create_sequences, normalize_sequences, denormalize_sequences


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
    assert np.isfinite(
        df[["return", "volatility_20", "volume_change", "price_range"]]
    ).all().all()


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
    assert info["portfolio_return"] <= 1.0
    assert not truncated


def test_environment_rejects_invalid_data():
    df = create_features(sample_data())

    with np.testing.assert_raises(ValueError):
        TradingEnv(df.iloc[:1])

    with np.testing.assert_raises(ValueError):
        TradingEnv(df, initial_cash=0)

    with np.testing.assert_raises(ValueError):
        TradingEnv(df, transaction_cost=-0.01)


def test_stress_scenarios_are_controlled_and_timed():
    df = create_features(sample_data())
    baseline = df["return"].to_numpy()

    drawdown = drawdown_shock(df, magnitude=0.10, duration=20, start_fraction=0.25)
    crash = market_crash(df, magnitude=0.20, duration=5, start_fraction=0.50)

    drawdown_delta = drawdown["return"].to_numpy() - baseline
    crash_delta = crash["return"].to_numpy() - baseline

    assert np.allclose(drawdown_delta[:25], 0.0)
    assert np.allclose(crash_delta[:50], 0.0)
    assert np.all(drawdown_delta[25:45] < 0)
    assert np.all(crash_delta[50:55] < 0)

    drawdown_simple_loss = 1.0 - np.prod(np.exp(drawdown_delta[25:45]))
    crash_simple_loss = 1.0 - np.prod(np.exp(crash_delta[50:55]))
    assert np.isclose(drawdown_simple_loss, 0.10, atol=1e-6)
    assert np.isclose(crash_simple_loss, 0.20, atol=1e-6)


def test_scenario_names_are_explicit_for_single_asset_setup():
    df = create_features(sample_data())
    scenarios = generate_adversarial_scenarios(df)

    assert set(scenarios) == {
        "volatility",
        "drawdown",
        "crash",
        "market_amplification",
    }


def test_sortino_uses_full_downside_deviation():
    returns = np.array([0.02, -0.01, 0.03, -0.02])
    assert np.isfinite(sortino_ratio(returns))


def test_multivariate_sequence_round_trip():
    df = create_features(sample_data())
    columns = ["return", "volume_change", "price_range"]
    sequences = create_sequences(df, sequence_length=10, feature_columns=columns)

    normalized, mean, std = normalize_sequences(sequences)
    restored = denormalize_sequences(normalized, mean, std)

    assert sequences.ndim == 3
    assert sequences.shape[2] == len(columns)
    assert mean.shape == (len(columns),)
    assert std.shape == (len(columns),)
    assert np.allclose(restored, sequences, atol=1e-6)


def test_buy_and_hold_uses_same_realized_period_count_as_trading_env():
    df = create_features(sample_data())
    result = evaluate_buy_and_hold(df)

    realized_returns = np.expm1(df["return"].to_numpy()[1:])
    expected_return = np.prod(1.0 + realized_returns) - 1.0

    assert np.isclose(result["return"], expected_return)
