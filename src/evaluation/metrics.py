import numpy as np


def total_return(values):
    return values[-1] / values[0] - 1


def max_drawdown(values):
    values = np.asarray(values)

    running_max = np.maximum.accumulate(values)
    drawdowns = values / running_max - 1

    return drawdowns.min()


def sharpe_ratio(returns, periods_per_year=252):
    returns = np.asarray(returns)

    if returns.std() == 0:
        return 0.0

    return (
        np.sqrt(periods_per_year)
        * returns.mean()
        / returns.std()
    )


def sortino_ratio(returns, periods_per_year=252):
    returns = np.asarray(returns)

    downside = returns[returns < 0]

    if len(downside) == 0 or downside.std() == 0:
        return 0.0

    return (
        np.sqrt(periods_per_year)
        * returns.mean()
        / downside.std()
    )


def var(returns, confidence=0.95):
    return np.quantile(
        np.asarray(returns),
        1 - confidence,
    )


def cvar(returns, confidence=0.95):
    returns = np.asarray(returns)

    threshold = var(returns, confidence)
    tail = returns[returns <= threshold]

    if len(tail) == 0:
        return threshold

    return tail.mean()
