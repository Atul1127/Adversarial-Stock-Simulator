# Adversarial Stock Simulator

An experimental machine learning system that combines **synthetic market generation, reinforcement learning, and adversarial stress testing** to evaluate trading-agent robustness under normal and extreme market conditions.

The project investigates whether exposing a PPO trading agent to synthetic and adversarial market scenarios can improve its robustness compared with training only on historical data.

## Architecture

```text
                    Historical Market Data
                            |
                            v
                    Data Processing
                            |
                 +----------+----------+
                 |                     |
                 v                     v
          Real Market Data      LSTM Market Generator
                                       |
                                       v
                                Synthetic Returns
                                       |
                         +-------------+-------------+
                         |             |             |
                         v             v             v
                       Real       Synthetic      Combined
                         |             |             |
                         +-------------+-------------+
                                       |
                                       v
                              PPO Trading Agent
                                       |
                                       v
                            Adversarial Scenarios
                                       |
                                       v
                         Risk & Performance Analysis
```

## Key Features

### Market Data Pipeline

Historical OHLCV data is downloaded using Yahoo Finance and processed through a time-series pipeline.

The pipeline includes:

* Data validation
* Duplicate removal
* Chronological sorting
* Missing-value handling
* Log-return calculation
* Rolling volatility
* Volume-change features
* Price-range features
* Chronological train/test splitting

### Synthetic Market Generation

An **LSTM-based adversarial generator** learns patterns from historical market return sequences and generates synthetic return sequences from random noise.

The generative system contains:

* LSTM Generator
* LSTM Discriminator
* Sequence construction
* Normalization
* Synthetic sequence generation
* Statistical validation

Synthetic sequences are evaluated using:

* Return distribution
* Volatility
* Tail behavior
* Autocorrelation

### Reinforcement Learning

The trading environment is implemented using **Gymnasium** and the trading agent uses **Proximal Policy Optimization (PPO)** through Stable-Baselines3.

The agent controls portfolio exposure:

```text
-1.0  → Fully Short
 0.0  → Neutral
+1.0  → Fully Long
```

The environment includes:

* Continuous position sizing
* Transaction costs
* Portfolio tracking
* Market-return rewards
* Position turnover

### Adversarial Stress Testing

The trained agent is evaluated under controlled market shocks:

| Scenario    | Description                        |
| ----------- | ---------------------------------- |
| Real        | Historical market conditions       |
| Volatility  | Increased market volatility        |
| Drawdown    | Sustained negative movement        |
| Crash       | Concentrated severe market decline |
| Correlation | Amplified market movements         |

## Experiments

Three PPO training strategies are compared.

### 1. Real-Only PPO

```text
Historical Data
      ↓
     PPO
      ↓
Stress Testing
```

### 2. Synthetic-Only PPO

```text
Historical Data
      ↓
Synthetic Generator
      ↓
Synthetic Data
      ↓
     PPO
      ↓
Stress Testing
```

### 3. Real + Synthetic PPO

```text
Historical Data ──────┐
                      ├──→ PPO
Synthetic Data ───────┘
                      ↓
               Stress Testing
```

The main research question is whether synthetic market exposure improves robustness under previously unseen stress conditions.

## Evaluation Metrics

### Performance

* Total Return
* Sharpe Ratio
* Sortino Ratio

### Risk

* Maximum Drawdown
* Value at Risk (VaR)
* Conditional Value at Risk (CVaR)

The final experiment compares each PPO training strategy across normal and adversarial scenarios.

## Project Structure

```text
Adversarial-Stock-Simulator/
│
├── configs/
│   └── default.yaml
│
├── experiments/
│   ├── test_data_pipeline.py
│   └── test_real_data.py
│
├── notebooks/
│
├── src/
│   ├── agent/
│   │   ├── datasets.py
│   │   ├── evaluate.py
│   │   ├── train.py
│   │   ├── train_combined.py
│   │   └── train_synthetic.py
│   │
│   ├── adversarial/
│   │   └── scenarios.py
│   │
│   ├── data/
│   │   ├── download.py
│   │   └── loader.py
│   │
│   ├── environment/
│   │   └── trading_env.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── robustness.py
│   │   └── compare_models.py
│   │
│   └── generator/
│       ├── dataset.py
│       ├── model.py
│       ├── train.py
│       └── validate.py
│
├── tests/
│   └── test_core.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Atul1127/Adversarial-Stock-Simulator.git
cd Adversarial-Stock-Simulator
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For testing:

```bash
pip install pytest
```

## Usage

### 1. Download Market Data

```bash
python -m src.data.download
```

The default configuration downloads historical AAPL data.

### 2. Train the Market Generator

```bash
python -m src.generator.train
```

### 3. Validate Synthetic Data

```bash
python -m src.generator.validate
```

### 4. Train PPO on Real Data

```bash
python -m src.agent.train
```

### 5. Train PPO on Synthetic Data

```bash
python -m src.agent.train_synthetic
```

### 6. Train PPO on Real + Synthetic Data

```bash
python -m src.agent.train_combined
```

### 7. Run Robustness Evaluation

```bash
python -m src.evaluation.robustness
```

### 8. Compare Training Strategies

```bash
python -m src.evaluation.compare_models
```

The final comparison is saved to:

```text
results/model_comparison.csv
results/model_summary.csv
```

## Testing

Run the core tests with:

```bash
pytest -q
```

The test suite covers the core feature pipeline and Gymnasium trading environment.

## Research Questions

This project investigates:

1. Can an adversarial generative model produce useful synthetic market return sequences?
2. How does PPO trained on synthetic data perform on real market conditions?
3. Does combining real and synthetic data improve robustness?
4. How sensitive are trading agents to volatility and crash scenarios?
5. Do risk metrics reveal weaknesses that total return alone cannot capture?

## Limitations

This is an experimental research and portfolio project rather than a production trading platform.

Current limitations include:

* Single-asset experiments
* Simplified market microstructure
* Synthetic return generation rather than full OHLCV generation
* Controlled rather than learned adversarial attacks
* No live trading
* No execution infrastructure
* Limited historical universe

These limitations intentionally keep the project focused and the experiments interpretable.

## Future Work

Potential extensions include:

* Multi-asset market generation
* More advanced time-series generative models
* Learned adversarial policies
* Regime-conditioned generation
* Portfolio-level reinforcement learning
* Walk-forward evaluation
* Hyperparameter optimization
* More realistic transaction models

## Disclaimer

This project is for **educational and research purposes only**.

It is not financial advice and should not be used as the sole basis for real-world investment decisions.
