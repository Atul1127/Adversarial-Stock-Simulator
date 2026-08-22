from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO

from src.data.loader import (
    load_stock_data,
    create_features,
    train_test_split_time_series,
)
from src.agent.datasets import generate_synthetic_dataframe
from src.environment.trading_env import TradingEnv


DATA_PATH = "data/raw/AAPL.csv"
MODEL_PATH = "models/ppo_combined"


def main():
    real = create_features(load_stock_data(DATA_PATH))
    train_real, _ = train_test_split_time_series(real, train_ratio=0.8)

    synthetic = generate_synthetic_dataframe()

    combined = pd.concat(
        [train_real, synthetic],
        ignore_index=True,
    )

    env = TradingEnv(combined)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=1,
    )

    model.learn(total_timesteps=50_000)

    Path("models").mkdir(exist_ok=True)
    model.save(MODEL_PATH)

    print(f"\nCombined PPO saved to {MODEL_PATH}.")


if __name__ == "__main__":
    main()
