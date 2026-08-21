from src.data.loader import (
    load_stock_data,
    create_features,
    train_test_split_time_series,
)


def main():

    print("=" * 60)
    print("REAL MARKET DATA PIPELINE")
    print("=" * 60)

    df = load_stock_data("data/raw/AAPL.csv")

    print("\nRaw data:")
    print(df.head())

    print(f"\nTotal rows: {len(df)}")

    features = create_features(df)

    print("\nFeatures:")
    print(features.head())

    print("\nFeature columns:")
    print(list(features.columns))

    train, test = train_test_split_time_series(features)

    print(f"\nTraining rows: {len(train)}")
    print(f"Testing rows:  {len(test)}")

    print("\nDate ranges:")
    print(f"Train: {train.index.min()} → {train.index.max()}")
    print(f"Test:  {test.index.min()} → {test.index.max()}")

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
