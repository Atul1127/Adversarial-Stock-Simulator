import numpy as np


def _as_1d(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("Metric input cannot be empty.")
    if not np.isfinite(values).all():
        raise ValueError("Metric input contains non-finite values.")
    return values


def total_return(values):
    values = _as_1d(values)
    if values[0] <= 0:
        raise ValueError("Portfolio values must be positive.")
    return values[-1] / values[0] - 1


def max_drawdown(values):
    values = _as_1d(values)
    if np.any(values <= 0):
        raise ValueError("Portfolio values must be positive.")

    running_max = np.maximum.accumulate(values)
    drawdowns = values / running_max - 1
    return float(drawdowns.min())


def sharpe_ratio(returns, periods_per_year=252):
    returns = _as_1d(returns)
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")
    if returns.size < 2 or np.std(returns, ddof=1) == 0:
        return 0.0

    return float(
        np.sqrt(periods_per_year)
        * returns.mean()
        / np.std(returns, ddof=1)
    )


def sortino_ratio(returns, periods_per_year=252, target=0.0):
    """Annualized Sortino using root-mean-square downside deviation."""
    returns = _as_1d(returns)
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")

    downside = np.minimum(returns - target, 0.0)
    downside_deviation = np.sqrt(np.mean(downside**2))

    if downside_deviation == 0:
        return 0.0

    return float(
        np.sqrt(periods_per_year)
        * (returns.mean() - target)
        / downside_deviation
    )


def var(returns, confidence=0.95):
    returns = _as_1d(returns)
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")
    return float(np.quantile(returns, 1 - confidence))


def cvar(returns, confidence=0.95):
    returns = _as_1d(returns)
    threshold = var(returns, confidence)
    tail = returns[returns <= threshold]

    if len(tail) == 0:
        return threshold

    return float(tail.mean())
