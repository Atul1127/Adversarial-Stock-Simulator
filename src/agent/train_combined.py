from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from src.data.loader import (
    load_stock_data,
    create_features,
    train_test_split_time_series,
)
from src.agent.datasets import generate_synthetic_episodes
from src.environment.episode_env import RandomEpisodeEnv, make_rolling_episodes


DATA_PATH = "data/raw/AAPL.csv"
MODEL_PATH = "models/ppo_combined"
SEED = 42
EPISODE_LENGTH = 30
NUM_SYNTHETIC_EPISODES = 512


def main():
    real = create_features(load_stock_data(DATA_PATH))
    train_real, _ = train_test_split_time_series(real, train_ratio=0.8)
    real_episodes = make_rolling_episodes(
        train_real,
        episode_length=EPISODE_LENGTH,
    )
    synthetic_episodes = generate_synthetic_episodes(
        num_episodes=NUM_SYNTHETIC_EPISODES,
        seed=SEED,
    )

    # Sample 50% of episodes from real training windows and 50% from generated
    # windows. No transition exists between unrelated trajectories.
    episodes = real_episodes + synthetic_episodes
    real_weight = 0.5 / len(real_episodes)
    synthetic_weight = 0.5 / len(synthetic_episodes)
    weights = (
        [real_weight] * len(real_episodes)
        + [synthetic_weight] * len(synthetic_episodes)
    )

    env = RandomEpisodeEnv(episodes, weights=weights)
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
        seed=SEED,
        verbose=1,
    )

    model.learn(total_timesteps=50_000)

    Path("models").mkdir(exist_ok=True)
    model.save(MODEL_PATH)

    print(f"\nCombined PPO saved to {MODEL_PATH}.")


if __name__ == "__main__":
    main()
