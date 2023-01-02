"""Dumps the most frequent values in a tsv"""
import sys

import pandas as pd


def main() -> None:
    """Main Loop"""

    # Load `preprocessed_.tsv`
    dataframe = pd.read_csv(sys.argv[1], sep="\t", header=None, on_bad_lines="skip")

    print(dataframe[1].value_counts().nlargest(50))
    print("")

    print(dataframe[2].value_counts().nlargest(50))
    print("")


if __name__ == "__main__":
    main()
