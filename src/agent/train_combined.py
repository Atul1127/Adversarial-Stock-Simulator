from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO

from src.data.loader import load_stock_data, create_features
from src.agent.datasets import generate_synthetic_dataframe
from src.environment.trading_env import TradingEnv


def main():
    real = create_features(
        load_stock_data("data/raw/AAPL.csv")
    )

    synthetic = generate_synthetic_dataframe()

    combined = pd.concat(
        [real, synthetic],
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

    model.save("models/ppo_combined")

    print("\nCombined PPO saved.")


if __name__ == "__main__":
    main()
