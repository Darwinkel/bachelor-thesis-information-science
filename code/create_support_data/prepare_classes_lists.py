"""Creates a list of unique classes in the filtered dataset, which are used by HuggingFace"""
import sys
import time

import pandas as pd


def main() -> None:
    """Main Loop"""

    # Load `preprocessed_filtered.tsv`
    dataframe = pd.read_csv(
        sys.argv[1], sep="\t", header=None, usecols=[1, 2], on_bad_lines="skip"
    )

    minor_classes_set: set[str] = set()
    major_classes_set: set[str] = set()

    minor_classes_set.update(map(str, dataframe[1].unique()))
    major_classes_set.update(map(str, dataframe[2].unique()))

    print(len(minor_classes_set))
    print(len(major_classes_set))

    timestamp = time.time()

    with open(f"minor_classes_{timestamp}.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(map(str, minor_classes_set)))

    with open(f"major_classes_{timestamp}.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(map(str, major_classes_set)))


if __name__ == "__main__":
    main()
