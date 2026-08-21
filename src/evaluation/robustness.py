import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.data.loader import load_stock_data, create_features
from src.environment.trading_env import TradingEnv
from src.adversarial.scenarios import generate_adversarial_scenarios
from src.evaluation.metrics import (
    total_return,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    var,
    cvar,
)


def evaluate(model, data):
    env = TradingEnv(data)
    obs, _ = env.reset()

    portfolio_values = [env.initial_cash]
    rewards = []

    while True:
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        rewards.append(reward)
        portfolio_values.append(info["portfolio_value"])

        if terminated or truncated:
            break

    portfolio_values = np.asarray(portfolio_values)
    rewards = np.asarray(rewards)

    return {
        "return": total_return(portfolio_values),
        "sharpe": sharpe_ratio(rewards),
        "sortino": sortino_ratio(rewards),
        "max_drawdown": max_drawdown(portfolio_values),
        "var_95": var(rewards),
        "cvar_95": cvar(rewards),
    }


def main():
    df = create_features(
        load_stock_data("data/raw/AAPL.csv")
    )

    scenarios = {
        "real": df,
        **generate_adversarial_scenarios(df),
    }

    model = PPO.load("models/ppo_real")

    results = []

    for name, data in scenarios.items():
        metrics = evaluate(model, data)

        results.append({
            "scenario": name,
            **metrics,
        })

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 90)
    print("PPO ADVERSARIAL ROBUSTNESS REPORT")
    print("=" * 90)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    results_df.to_csv(
        "results/robustness_report.csv",
        index=False,
    )

    print("\nSaved: results/robustness_report.csv")


if __name__ == "__main__":
    main()
