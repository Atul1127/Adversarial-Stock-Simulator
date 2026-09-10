# Adversarial Stock Simulator

An experimental machine learning system combining **LSTM-based synthetic market generation, reinforcement learning, and adversarial stress testing** to study how synthetic market experience affects trading-agent performance and robustness.

The core research question is whether exposing a PPO trading agent to statistically validated synthetic market episodes improves risk-adjusted performance or robustness compared with training only on historical data.

> **Research / portfolio project:** this repository is designed for reproducible experimentation, not live trading or investment advice.

## Architecture

```text
Historical OHLCV Data
        │
        ▼
  Data Validation
        │
        ▼
 Feature Engineering
        │
        ├──────────────────┐
        ▼                  ▼
 Real Training Windows   LSTM Generator
        │                  │
        │                  ▼
        │            Synthetic 30-step
        │                Episodes
        │                  │
        └────────┬─────────┘
                 ▼
        PPO Episode Sampler
                 │
       ┌─────────┼─────────┐
       ▼         ▼         ▼
    Real PPO  Synthetic  Combined
                 PPO        PPO
       └─────────┬─────────┘
                 ▼
          Held-out Test Period
                 │
                 ▼
       Adversarial Stress Tests
                 │
                 ▼
       Risk & Robustness Analysis
```

## What the System Does

### 1. Market Data Pipeline

Historical OHLCV data is downloaded with Yahoo Finance and transformed into model-ready features:

- Data validation and schema checks
- Duplicate removal and chronological sorting
- Missing-value handling
- Log returns
- Rolling volatility
- Log volume change
- Normalized intraday price range
- Chronological train/test splitting

### 2. Synthetic Market Generation

A probabilistic autoregressive LSTM generator and LSTM discriminator are trained **only on the chronological training split**. The generator models three market variables jointly:

- Log return
- Log volume change
- Normalized price range

Skewed positive/negative features are transformed into stable model space before normalization. The generator predicts the next market state from the previous state and latent noise, with adversarial, likelihood, moment, and serial-dependence objectives.

Synthetic data is generated as **independent 30-step market episodes**, matching the generator's training horizon. The system does not extrapolate a 30-step generator into an artificial multi-year trajectory.

Generator validation compares synthetic and real training distributions using:

- Mean and volatility
- Tail quantiles
- Lag-1 and lag-5 return dependence
- Extreme-return frequency
- Explicit acceptance gates

The held-out test period is never used for generator training, validation, or tuning.

### 3. Reinforcement Learning

The trading environment uses **Gymnasium** and the agent uses **PPO** from Stable-Baselines3.

The continuous action represents portfolio exposure:

```text
-1.0  → Fully short
 0.0  → Neutral
+1.0  → Fully long
```

The environment models transaction costs, position turnover, portfolio value, and return-based rewards. Financial evaluation uses actual portfolio-period returns rather than the PPO reward signal.

To make the experiments comparable, all PPO variants use the same **30-step episodic training protocol**:

1. **Real PPO** samples random chronological 30-step windows from the real training split.
2. **Synthetic PPO** samples independently generated 30-step episodes.
3. **Combined PPO** samples real and synthetic episodes with equal source probability.

No artificial real→synthetic or synthetic→real transition is exposed to PPO.

### 4. Adversarial Stress Testing

The trained agents are evaluated under controlled market shocks:

| Scenario | Description |
| --- | --- |
| Real | Held-out historical market conditions |
| Volatility | 3× amplification of return deviations around the sample mean |
| Drawdown | Controlled 10% cumulative loss injected over a sustained mid-path window |
| Crash | Controlled 20% cumulative loss injected over a short mid-path window |
| Market amplification | 1.5× amplification of return deviations around the sample mean |

Because this repository currently uses a single asset, market amplification is explicitly **not** described as a cross-asset correlation test. Genuine correlation stress testing requires multiple assets.

All final comparisons and stress tests use the same unseen 20% historical test period.

### 5. Experimental Comparison

The experiment compares:

1. **Buy-and-hold benchmark** — passive unlevered exposure over the held-out period.
2. **Real-only PPO** — trained only on real training windows.
3. **Synthetic-only PPO** — trained only on generated episodes.
4. **Real + synthetic PPO** — trained on balanced real and synthetic episode sources.

All strategies are evaluated on the same unseen 20% real-data test period and its controlled stress scenarios.

## Evaluation Metrics

### Performance

- Total Return
- Sharpe Ratio
- Sortino Ratio

### Risk

- Maximum Drawdown
- Value at Risk (VaR)
- Conditional Value at Risk (CVaR)

Metrics are computed from actual portfolio-period returns. Sortino uses full downside deviation relative to a zero target. VaR and CVaR are reported as return quantiles, so negative values represent losses.

## Project Structure

```text
Adversarial-Stock-Simulator/
├── configs/
│   └── default.yaml
├── experiments/
│   ├── test_data_pipeline.py
│   └── test_real_data.py
├── src/
│   ├── agent/
│   │   ├── datasets.py
│   │   ├── evaluate.py
│   │   ├── train.py
│   │   ├── train_combined.py
│   │   └── train_synthetic.py
│   ├── adversarial/
│   │   └── scenarios.py
│   ├── data/
│   │   ├── download.py
│   │   └── loader.py
│   ├── environment/
│   │   ├── episode_env.py
│   │   └── trading_env.py
│   ├── evaluation/
│   │   ├── benchmarks.py
│   │   ├── compare_models.py
│   │   ├── metrics.py
│   │   └── robustness.py
│   └── generator/
│       ├── dataset.py
│       ├── model.py
│       ├── train.py
│       └── validate.py
├── tests/
│   └── test_core.py
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

Activate the environment:

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

**Windows CMD**

```cmd
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Pipeline

### 1. Download market data

```bash
python -m src.data.download
```

The default experiment downloads 10 years of AAPL data into `data/raw/`.

### 2. Train the synthetic market generator

```bash
python -m src.generator.train
```

The generator uses only the chronological training split.

### 3. Validate the generator

```bash
python -m src.generator.validate
```

Validation is a hard gate. Do not train PPO when the generator fails validation.

### 4. Train PPO variants

```bash
python -m src.agent.train
python -m src.agent.train_synthetic
python -m src.agent.train_combined
```

All training scripts use seed 42 and matched 30-step episode horizons.

### 5. Compare all strategies

```bash
python -m src.evaluation.compare_models
```

This produces `results/model_comparison.csv` and includes the buy-and-hold benchmark.

### 6. Evaluate robustness

```bash
python -m src.evaluation.robustness
```

This produces `results/robustness_report.csv` for the real PPO baseline.

Generated comparison outputs are written to `results/` and are intentionally ignored by Git.

## Testing

Run the automated tests with:

```bash
pytest -q
```

The suite covers feature generation, chronological ordering, environment behavior, episodic sampling, action validation, portfolio-return reporting, controlled stress scenarios, multivariate sequence normalization, and downside-risk metrics.

## Configuration

Core experiment settings are centralized in `configs/default.yaml`, including:

- Asset and historical period
- Sequence length and train split
- Generator architecture and optimization settings
- Trading capital and transaction costs
- PPO training budget
- Evaluation confidence level

The current default experiment uses **AAPL**, a 30-step episode/sequence length, an 80/20 chronological split, a 50,000-step PPO budget, and a 95% risk confidence level.

## Research Questions

1. Can a probabilistic recurrent adversarial generator produce statistically plausible multivariate market episodes?
2. How does PPO trained on synthetic episodes transfer to unseen real-market conditions?
3. Does combining validated synthetic episodes with real training windows improve robustness?
4. How sensitive are trading agents to volatility, drawdown, and crash scenarios?
5. Does synthetic-distribution quality predict downstream RL transfer?
6. How do learned policies compare with passive buy-and-hold?

## Limitations

This is an experimental research and portfolio project rather than a production trading platform.

- Single-asset experiments
- Simplified market microstructure
- Synthetic features are limited to returns, volume change, and price range
- Controlled stress scenarios rather than learned attacks
- Single-asset stress testing cannot measure true cross-asset correlation
- No live trading or execution infrastructure
- Limited historical universe

## Future Work

- Multivariate OHLCV/market generation with additional state variables
- Regime-conditioned generative models
- Learned adversarial policies
- Portfolio-level reinforcement learning
- Walk-forward evaluation
- Multiple random seeds and confidence intervals
- Hyperparameter optimization
- More realistic transaction and slippage models
- Genuine cross-asset correlation stress testing

## Disclaimer

This project is for **educational and research purposes only**. It is not financial advice and should not be used as the sole basis for real-world investment decisions.
