import yfinance as yf
from pathlib import Path


def download_stock(ticker="AAPL", period="10y"):
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {ticker} data...")

    df = yf.download(
        ticker,
        period=period,
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise ValueError(f"No data downloaded for {ticker}")

    # Handle yfinance multi-level columns
    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    required = ["Date", "Open", "High", "Low", "Close", "Volume"]

    df = df[required].dropna()

    output_path = output_dir / f"{ticker}.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} rows to {output_path}")

    return df


if __name__ == "__main__":
    download_stock()
