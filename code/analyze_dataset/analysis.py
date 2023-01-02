"""Prints the most common results to test cases"""
import sys

import pandas as pd


def main() -> None:
    """Main Loop"""

    # Load `preprocessed_filtered.tsv` or raw data
    dataframe = pd.read_csv(sys.argv[1], sep="\t", header=None, on_bad_lines="skip")

    for column in dataframe:
        if column < 18:
            print(f"Test case (HTTP/80): {column-2}")
        else:
            print(f"Test case (HTTPS/443): {column - 18}")
        print(dataframe[column].value_counts().nlargest(50))
        print("")


if __name__ == "__main__":
    main()
