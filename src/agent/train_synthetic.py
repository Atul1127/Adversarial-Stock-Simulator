from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from src.agent.datasets import generate_synthetic_episodes
from src.environment.episode_env import RandomEpisodeEnv


MODEL_PATH = "models/ppo_synthetic"
SEED = 42
NUM_EPISODES = 512


def main():
    episodes = generate_synthetic_episodes(
        num_episodes=NUM_EPISODES,
        seed=SEED,
    )
    env = RandomEpisodeEnv(episodes)
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

    print(f"\nSynthetic PPO saved to {MODEL_PATH}.")


if __name__ == "__main__":
    main()
