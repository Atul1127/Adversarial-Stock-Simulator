# Adversarial Stock Simulator

An experimental machine learning system combining **LSTM-based synthetic market generation, reinforcement learning, and adversarial stress testing** to evaluate trading-agent robustness under normal and extreme market conditions.

The core research question is whether exposing a PPO trading agent to synthetic and adversarial market scenarios improves robustness compared with training only on historical data.

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
        ├───────────────┐
        ▼               ▼
   Real Returns   LSTM Generator
                        │
                        ▼
                 Synthetic Returns
                        │
             ┌──────────┼──────────┐
             ▼          ▼          ▼
           Real     Synthetic   Combined
             └──────────┼──────────┘
                        ▼
                  PPO Trading Agent
                        │
                        ▼
             Adversarial Stress Tests
                        │
                        ▼
              Risk & Robustness Analysis
```

## What the System Does

### 1. Market Data Pipeline

Historical OHLCV data is downloaded with Yahoo Finance and transformed into model-ready time-series features:

- Data validation and schema checks
- Duplicate removal and chronological sorting
- Missing-value handling
- Log returns
- Rolling volatility
- Log volume change
- Normalized intraday price range
- Chronological train/test splitting

### 2. Synthetic Market Generation

An LSTM generator learns return-sequence structure from the **chronological training split only**, while an LSTM discriminator distinguishes real and generated sequences.

Synthetic sequences are validated using:

- Return distributions
- Volatility
- Tail behavior
- Autocorrelation

For PPO augmentation, synthetic returns are generated as **one continuous LSTM trajectory** rather than flattening independently generated sequences, avoiding artificial transitions between unrelated samples.

### 3. Reinforcement Learning

The trading environment uses **Gymnasium** and the agent uses **PPO** from Stable-Baselines3.

The continuous action represents portfolio exposure:

```text
-1.0  → Fully short
 0.0  → Neutral
+1.0  → Fully long
```

The environment models transaction costs, position turnover, portfolio value, and return-based rewards. The experiment uses chronological train/test separation and evaluates the trained agents on the held-out period.

### 4. Adversarial Stress Testing

The trained agents are evaluated under controlled market shocks:

| Scenario | Description |
| --- | --- |
| Real | Held-out historical market conditions |
| Volatility | 3× amplification of return deviations around the sample mean |
| Drawdown | Controlled sustained negative shock |
| Crash | Concentrated severe negative shock |
| Market amplification | 1.5× amplification of return deviations around the sample mean |

Because this repository currently uses a single asset, the final scenario is explicitly treated as **market-movement amplification rather than a literal correlation shock**. A true correlation test requires multiple assets.

All final model comparisons and stress tests use the same unseen 20% historical test period.

### 5. Experimental Comparison

Three PPO training strategies are compared:

1. **Real-only PPO** — trained on the 80% chronological real-data training split.
2. **Synthetic-only PPO** — trained on generated market sequences.
3. **Real + synthetic PPO** — trained on the real training split plus synthetic sequences.

All three models are evaluated on the same unseen 20% real-data test period and its controlled stress scenarios.

## Evaluation Metrics

### Performance

- Total Return
- Sharpe Ratio
- Sortino Ratio

### Risk

- Maximum Drawdown
- Value at Risk (VaR)
- Conditional Value at Risk (CVaR)

Financial metrics are computed from **actual portfolio-period returns**, while the RL reward remains an optimization signal for PPO.

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
│   │   └── trading_env.py
│   ├── evaluation/
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

The default configuration downloads 10 years of AAPL data into `data/raw/`.

### 2. Train the synthetic market generator

```bash
python -m src.generator.train
```

The generator is trained only on the chronological training split.

### 3. Validate generated sequences

```bash
python -m src.generator.validate
```

### 4. Train PPO variants

```bash
python -m src.agent.train
python -m src.agent.train_synthetic
python -m src.agent.train_combined
```

### 5. Compare all training strategies

```bash
python -m src.evaluation.compare_models
```

### 6. Evaluate robustness

```bash
python -m src.evaluation.robustness
```

Generated comparison outputs are written to `results/` and are intentionally ignored by Git.

## Testing

Run the automated tests with:

```bash
pytest -q
```

The test suite checks feature generation, numerical validity, environment behavior, action bounds, reset/step behavior, and invalid-input handling.

## Configuration

Core experiment settings are centralized in `configs/default.yaml`, including:

- Asset and historical period
- Sequence length and train split
- Generator architecture and optimization settings
- Trading capital and transaction costs
- PPO training budget
- Evaluation confidence level

The current default experiment uses **AAPL**, a 30-step sequence length, an 80/20 chronological split, a 50,000-step PPO budget, and a 95% risk confidence level.

## Research Questions

1. Can an adversarial generative model produce statistically useful market return sequences?
2. How does PPO trained on synthetic data perform on real market conditions?
3. Does combining real and synthetic data improve robustness?
4. How sensitive are trading agents to volatility, drawdown, and crash scenarios?
5. Do risk metrics reveal weaknesses hidden by total return alone?

## Limitations

This is an experimental research and portfolio project rather than a production trading platform.

- Single-asset experiments
- Simplified market microstructure
- Synthetic returns rather than full OHLCV generation
- Controlled stress scenarios rather than learned attacks
- Single-asset stress testing cannot measure true cross-asset correlation
- No live trading or execution infrastructure
- Limited historical universe

## Future Work

- Multi-asset market generation
- Regime-conditioned generative models
- Learned adversarial policies
- Portfolio-level reinforcement learning
- Walk-forward evaluation
- Hyperparameter optimization
- More realistic transaction and slippage models
- Genuine cross-asset correlation stress testing

## Disclaimer

This project is for **educational and research purposes only**. It is not financial advice and should not be used as the sole basis for real-world investment decisions.
