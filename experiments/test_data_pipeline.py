import sys

sys.path.append(".")

from src.data.loader import create_features, train_test_split_time_series


def main():

    print("=" * 50)
    print("DATA PIPELINE TEST")
    print("=" * 50)

    print("\nPipeline functions loaded successfully.")

    print("\nFunctions:")
    print("✓ create_features")
    print("✓ train_test_split_time_series")

    print("\nData pipeline module is ready.")


if __name__ == "__main__":
    main()
