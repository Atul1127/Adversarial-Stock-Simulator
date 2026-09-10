import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.data.loader import load_stock_data, create_features, train_test_split_time_series
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
    """Evaluate a model using actual portfolio-period returns."""
    env = TradingEnv(data)
    obs, _ = env.reset()

    portfolio_values = [env.initial_cash]
    portfolio_returns = []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)

        portfolio_values.append(info["portfolio_value"])
        portfolio_returns.append(info["portfolio_return"])

        if terminated or truncated:
            break

    portfolio_values = np.asarray(portfolio_values, dtype=float)
    portfolio_returns = np.asarray(portfolio_returns, dtype=float)

    return {
        "return": total_return(portfolio_values),
        "sharpe": sharpe_ratio(portfolio_returns),
        "sortino": sortino_ratio(portfolio_returns),
        "max_drawdown": max_drawdown(portfolio_values),
        "var_95": var(portfolio_returns),
        "cvar_95": cvar(portfolio_returns),
    }


def main():
    full_data = create_features(
        load_stock_data("data/raw/AAPL.csv")
    )
    _, test_data = train_test_split_time_series(full_data, train_ratio=0.8)

    # Final robustness evaluation is strictly out-of-sample.
    scenarios = {
        "real": test_data,
        **generate_adversarial_scenarios(test_data),
    }

    model = PPO.load("models/ppo_real")
    results = []

    for name, data in scenarios.items():
        metrics = evaluate(model, data)
        results.append({"scenario": name, **metrics})

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 90)
    print("PPO ADVERSARIAL ROBUSTNESS REPORT — OUT-OF-SAMPLE")
    print("=" * 90)
    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    output_path = "results/robustness_report.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
