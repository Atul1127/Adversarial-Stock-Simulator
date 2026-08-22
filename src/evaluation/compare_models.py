import pandas as pd
from stable_baselines3 import PPO

from src.data.loader import (
    load_stock_data,
    create_features,
    train_test_split_time_series,
)
from src.adversarial.scenarios import generate_adversarial_scenarios
from src.evaluation.robustness import evaluate


DATA_PATH = "data/raw/AAPL.csv"


def main():
    full_data = create_features(load_stock_data(DATA_PATH))
    _, test_data = train_test_split_time_series(full_data, train_ratio=0.8)

    # All model comparisons and stress tests use only the held-out period.
    scenarios = {
        "real": test_data,
        **generate_adversarial_scenarios(test_data),
    }

    models = {
        "real_ppo": "models/ppo_real",
        "synthetic_ppo": "models/ppo_synthetic",
        "combined_ppo": "models/ppo_combined",
    }

    results = []

    for model_name, model_path in models.items():
        print(f"\nEvaluating {model_name}...")
        model = PPO.load(model_path)

        for scenario_name, scenario_data in scenarios.items():
            metrics = evaluate(model, scenario_data)
            results.append({
                "model": model_name,
                "scenario": scenario_name,
                **metrics,
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv("results/model_comparison.csv", index=False)

    print("\n" + "=" * 100)
    print("FINAL MODEL COMPARISON — OUT-OF-SAMPLE")
    print("=" * 100)
    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    print("\nSaved: results/model_comparison.csv")


if __name__ == "__main__":
    main()
