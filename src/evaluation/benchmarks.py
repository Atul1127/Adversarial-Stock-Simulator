import numpy as np

from src.evaluation.metrics import (
    cvar,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    var,
)


def evaluate_buy_and_hold(data):
    """Evaluate an unlevered buy-and-hold strategy on the same periods as PPO."""
    if "return" not in data.columns:
        raise ValueError("DataFrame must contain 'return'.")

    log_returns = data["return"].to_numpy(dtype=np.float64)
    if len(log_returns) < 2 or not np.isfinite(log_returns).all():
        raise ValueError("Return series must contain at least two finite observations.")

    # TradingEnv observes row t and applies the selected exposure to t -> t+1,
    # so the first stored return is not realized by the PPO evaluation either.
    period_returns = np.expm1(log_returns[1:])
    period_returns = np.maximum(period_returns, -0.999999)
    portfolio_values = np.concatenate(
        ([1.0], np.cumprod(1.0 + period_returns))
    )

    return {
        "return": total_return(portfolio_values),
        "sharpe": sharpe_ratio(period_returns),
        "sortino": sortino_ratio(period_returns),
        "max_drawdown": max_drawdown(portfolio_values),
        "var_95": var(period_returns),
        "cvar_95": cvar(period_returns),
    }
