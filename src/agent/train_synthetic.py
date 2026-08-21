from pathlib import Path

from stable_baselines3 import PPO

from src.agent.datasets import generate_synthetic_dataframe
from src.environment.trading_env import TradingEnv


def main():
    data = generate_synthetic_dataframe()

    env = TradingEnv(data)

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

    model.save("models/ppo_synthetic")

    print("\nSynthetic PPO saved.")


if __name__ == "__main__":
    main()
