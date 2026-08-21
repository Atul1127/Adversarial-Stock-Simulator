import numpy as np
import pandas as pd


def volatility_shock(
    df: pd.DataFrame,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """Increase return volatility while preserving the original timeline."""
    data = df.copy()

    if "return" not in data.columns:
        raise ValueError("DataFrame must contain 'return'.")

    data["return"] = data["return"] * multiplier
    return data


def drawdown_shock(
    df: pd.DataFrame,
    magnitude: float = 0.05,
    duration: int = 10,
) -> pd.DataFrame:
    """Inject a controlled negative return shock."""
    data = df.copy()

    if "return" not in data.columns:
        raise ValueError("DataFrame must contain 'return'.")

    duration = min(duration, len(data))

    shock = np.zeros(len(data), dtype=np.float64)
    shock[:duration] = -magnitude / duration

    data["return"] = data["return"].to_numpy() + shock

    return data


def market_crash(
    df: pd.DataFrame,
    magnitude: float = 0.20,
    duration: int = 5,
) -> pd.DataFrame:
    """Create a concentrated market crash scenario."""
    data = df.copy()

    if "return" not in data.columns:
        raise ValueError("DataFrame must contain 'return'.")

    duration = min(duration, len(data))

    shock = np.zeros(len(data), dtype=np.float64)
    shock[:duration] = -magnitude / duration

    data["return"] = data["return"].to_numpy() + shock

    return data


def correlation_shock(
    df: pd.DataFrame,
    multiplier: float = 1.5,
) -> pd.DataFrame:
    """Amplify market movements to simulate correlated stress."""
    data = df.copy()

    if "return" not in data.columns:
        raise ValueError("DataFrame must contain 'return'.")

    data["return"] = data["return"] * multiplier

    return data


def generate_adversarial_scenarios(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Generate the project's controlled stress-test scenarios."""

    return {
        "volatility": volatility_shock(df),
        "drawdown": drawdown_shock(df),
        "crash": market_crash(df),
        "correlation": correlation_shock(df),
    }
