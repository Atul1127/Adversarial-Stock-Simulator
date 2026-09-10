# Adversarial Stock Simulator

An experimental ML system combining **LSTM-based synthetic market generation, PPO reinforcement learning, and adversarial stress testing**.

The research question is simple: **does validated synthetic market experience improve trading-agent performance or robustness compared with real historical training alone?**

> Research / portfolio project. Not a live-trading system or investment advice.

## Pipeline

```text
Historical OHLCV
      ↓
Feature Engineering
      ↓
80% Train / 20% Test
      ↓
LSTM Generator + Discriminator
      ↓
Validated 30-step Synthetic Episodes
      ↓
┌────────────┬──────────────┬──────────────┐
│ Real PPO   │ Synthetic PPO│ Combined PPO │
└────────────┴──────────────┴──────────────┘
                    ↓
             Unseen Test Period
                    ↓
          Volatility / Drawdown / Crash
                    ↓
          Risk Metrics + Buy & Hold
```

## Data

Historical AAPL OHLCV data is transformed into:

- Log return
- Rolling volatility
- Log volume change
- Normalized intraday price range

The train/test split is chronological. The held-out 20% is never used to train or validate the generator.

## Synthetic Market Generator

The generator uses a recurrent probabilistic LSTM with an LSTM discriminator. It models three variables jointly:

- return
- volume change
- price range

The generator is trained on **30-step training windows** and produces independent 30-step synthetic episodes. We do not extrapolate a short-window model into an artificial multi-year sequence.

Before synthetic episodes are used for PPO, validation checks:

- mean and volatility
- tail quantiles
- lag-1 and lag-5 return autocorrelation
- extreme-return frequency

A failed generator validation stops the experiment.

## PPO Experiments

All variants use the same 30-step episodic training setup and seed:

**Real PPO** — random chronological windows from the 80% real training split.

**Synthetic PPO** — independently generated synthetic episodes.

**Combined PPO** — real and synthetic episodes sampled with equal source probability.

No real→synthetic or synthetic→real transition is introduced inside an episode.

The trading observation contains the four market features, current position, and current portfolio drawdown. The reward is portfolio log-return minus a small drawdown penalty, keeping the objective simple while discouraging excessive risk.

## Baseline

The final comparison includes an unlevered **buy-and-hold AAPL benchmark** so PPO performance is not interpreted in isolation.

## Stress Tests

The same unseen 20% test period is evaluated under three controlled scenarios:

| Scenario | Stress |
| --- | --- |
| Real | Unmodified held-out market |
| Volatility | 3× return-deviation dispersion |
| Drawdown | 10% cumulative negative shock over 20 steps |
| Crash | 20% cumulative negative shock over 5 steps |

A single asset cannot provide a genuine cross-asset correlation test, so no correlation metric is claimed.

## Metrics

Performance:

- Total return
- Sharpe ratio
- Sortino ratio

Risk:

- Maximum drawdown
- VaR 95%
- CVaR 95%

Metrics use actual portfolio-period returns rather than the PPO reward signal.

## Project Structure

```text
Adversarial-Stock-Simulator/
├── configs/
├── experiments/
├── src/
│   ├── agent/
│   ├── adversarial/
│   ├── data/
│   ├── environment/
│   ├── evaluation/
│   └── generator/
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/Atul1127/Adversarial-Stock-Simulator.git
cd Adversarial-Stock-Simulator
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python -m src.data.download
python -m src.generator.train
python -m src.generator.validate
pytest -q
python -m src.agent.train
python -m src.agent.train_synthetic
python -m src.agent.train_combined
python -m src.evaluation.compare_models
python -m src.evaluation.robustness
```

Run `src.generator.validate` before retraining PPO. The validator is the quality gate for synthetic data.

## Limitations

- Single asset: AAPL
- Simplified transaction costs and market microstructure
- Synthetic generation covers selected market features rather than full OHLCV
- Stress scenarios are controlled transformations, not learned adversarial policies
- One historical test split is used for the current experiment

## Future Work

- Multiple assets and true correlation testing
- Walk-forward evaluation
- Multiple seeds with confidence intervals
- Regime-conditioned generation
- Learned adversarial policies
- More realistic slippage and execution models
