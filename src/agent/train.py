from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from src.data.loader import load_stock_data, create_features, train_test_split_time_series
from src.environment.trading_env import TradingEnv


DATA_PATH = "data/raw/AAPL.csv"
MODEL_PATH = "models/ppo_real"


def main():
    df = create_features(load_stock_data(DATA_PATH))
    train_df, _ = train_test_split_time_series(df, train_ratio=0.8)

    env = TradingEnv(train_df)

    # Validate Gymnasium environment before training.
    check_env(env, warn=True)

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

    print(f"\nPPO model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
