import numpy as np
from stable_baselines3 import PPO

from src.data.loader import load_stock_data, create_features
from src.environment.trading_env import TradingEnv


def main():
    df = create_features(
        load_stock_data("data/raw/AAPL.csv")
    )

    env = TradingEnv(df)
    model = PPO.load("models/ppo_real")

    obs, _ = env.reset()

    portfolio_values = [env.initial_cash]

    while True:
        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        obs, reward, terminated, truncated, info = env.step(action)

        portfolio_values.append(
            info["portfolio_value"]
        )

        if terminated or truncated:
            break

    values = np.asarray(portfolio_values)

    total_return = values[-1] / values[0] - 1

    running_max = np.maximum.accumulate(values)
    drawdown = values / running_max - 1
    max_drawdown = drawdown.min()

    print("=" * 50)
    print("PPO BASELINE")
    print("=" * 50)
    print(f"Initial portfolio: ${values[0]:,.2f}")
    print(f"Final portfolio:   ${values[-1]:,.2f}")
    print(f"Total return:      {total_return:.2%}")
    print(f"Max drawdown:      {max_drawdown:.2%}")


if __name__ == "__main__":
    main()
