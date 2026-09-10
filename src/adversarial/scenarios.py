import numpy as np
import pandas as pd


def _require_returns(df: pd.DataFrame) -> None:
    if "return" not in df.columns:
        raise ValueError("DataFrame must contain 'return'.")


def volatility_shock(
    df: pd.DataFrame,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """Scale deviations around the sample mean to create volatility stress.

    Centering before scaling avoids turning an existing positive/negative drift
    into an exaggerated directional bet simply because volatility was raised.
    """
    if multiplier <= 0:
        raise ValueError("multiplier must be positive.")

    data = df.copy()
    _require_returns(data)

    returns = data["return"].to_numpy(dtype=np.float64)
    mean_return = returns.mean()
    data["return"] = mean_return + multiplier * (returns - mean_return)
    return data


def drawdown_shock(
    df: pd.DataFrame,
    magnitude: float = 0.05,
    duration: int = 10,
) -> pd.DataFrame:
    """Inject a controlled negative cumulative shock over ``duration`` steps."""
    data = df.copy()
    _require_returns(data)

    if magnitude < 0:
        raise ValueError("magnitude must be non-negative.")
    if duration <= 0:
        raise ValueError("duration must be positive.")

    duration = min(duration, len(data))
    shock = np.zeros(len(data), dtype=np.float64)
    shock[:duration] = -magnitude / duration
    data["return"] = data["return"].to_numpy(dtype=np.float64) + shock
    return data


def market_crash(
    df: pd.DataFrame,
    magnitude: float = 0.20,
    duration: int = 5,
) -> pd.DataFrame:
    """Inject a concentrated negative shock over ``duration`` steps."""
    data = df.copy()
    _require_returns(data)

    if magnitude < 0:
        raise ValueError("magnitude must be non-negative.")
    if duration <= 0:
        raise ValueError("duration must be positive.")

    duration = min(duration, len(data))
    shock = np.zeros(len(data), dtype=np.float64)
    shock[:duration] = -magnitude / duration
    data["return"] = data["return"].to_numpy(dtype=np.float64) + shock
    return data


def market_amplification(
    df: pd.DataFrame,
    multiplier: float = 1.5,
) -> pd.DataFrame:
    """Amplify market-return deviations for a single-asset stress test.

    A true correlation shock requires multiple assets. For the current
    single-asset simulator this scenario is explicitly modeled as broader
    market-movement amplification instead of claiming to measure correlation.
    """
    if multiplier <= 0:
        raise ValueError("multiplier must be positive.")

    data = df.copy()
    _require_returns(data)

    returns = data["return"].to_numpy(dtype=np.float64)
    mean_return = returns.mean()
    data["return"] = mean_return + multiplier * (returns - mean_return)
    return data


def generate_adversarial_scenarios(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Generate controlled stress-test scenarios for the current single asset."""
    return {
        "volatility": volatility_shock(df),
        "drawdown": drawdown_shock(df),
        "crash": market_crash(df),
        "market_amplification": market_amplification(df),
    }
