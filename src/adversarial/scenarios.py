import numpy as np
import pandas as pd


def _require_returns(df: pd.DataFrame) -> None:
    if "return" not in df.columns:
        raise ValueError("DataFrame must contain 'return'.")
    if len(df) < 2:
        raise ValueError("Stress scenarios require at least two observations.")


def _window_bounds(n: int, duration: int, start_fraction: float) -> tuple[int, int]:
    if duration <= 0:
        raise ValueError("duration must be positive.")
    if not 0 <= start_fraction < 1:
        raise ValueError("start_fraction must be in [0, 1).")

    start = min(int(n * start_fraction), n - 1)
    end = min(start + duration, n)
    return start, end


def _log_loss_per_step(cumulative_simple_loss: float, steps: int) -> float:
    if not 0 <= cumulative_simple_loss < 1:
        raise ValueError("cumulative_simple_loss must be in [0, 1).")
    if steps <= 0:
        return 0.0

    per_step_simple_loss = 1.0 - (1.0 - cumulative_simple_loss) ** (1.0 / steps)
    return float(np.log1p(-per_step_simple_loss))


def volatility_shock(
    df: pd.DataFrame,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """Increase return dispersion without changing the sample mean."""
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
    magnitude: float = 0.10,
    duration: int = 20,
    start_fraction: float = 0.25,
) -> pd.DataFrame:
    """Inject a controlled cumulative loss over a sustained mid-path window."""
    data = df.copy()
    _require_returns(data)

    if magnitude < 0 or magnitude >= 1:
        raise ValueError("magnitude must be in [0, 1).")

    start, end = _window_bounds(len(data), duration, start_fraction)
    log_loss = _log_loss_per_step(magnitude, end - start)

    shock = np.zeros(len(data), dtype=np.float64)
    shock[start:end] = log_loss
    data["return"] = data["return"].to_numpy(dtype=np.float64) + shock
    return data


def market_crash(
    df: pd.DataFrame,
    magnitude: float = 0.20,
    duration: int = 5,
    start_fraction: float = 0.50,
) -> pd.DataFrame:
    """Inject a concentrated cumulative loss over a short mid-path window."""
    data = df.copy()
    _require_returns(data)

    if magnitude < 0 or magnitude >= 1:
        raise ValueError("magnitude must be in [0, 1).")

    start, end = _window_bounds(len(data), duration, start_fraction)
    log_loss = _log_loss_per_step(magnitude, end - start)

    shock = np.zeros(len(data), dtype=np.float64)
    shock[start:end] = log_loss
    data["return"] = data["return"].to_numpy(dtype=np.float64) + shock
    return data


def generate_adversarial_scenarios(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Generate the three core single-asset stress scenarios."""
    return {
        "volatility": volatility_shock(df),
        "drawdown": drawdown_shock(df),
        "crash": market_crash(df),
    }
