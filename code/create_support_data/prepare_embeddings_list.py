"""Creates a list of unique sentences in the dataset which can be used to train embeddings on"""
import sys
import time

import pandas as pd


def main() -> None:
    """Main Loop"""

    # Load dataframe of domains
    dataframe = pd.read_csv(sys.argv[1], sep="\t", header=0, on_bad_lines="skip")

    unique_values_set: set[str] = set()
    html_list = ("HTTP/", "<html>", "<ERROR>", "<EMPTY>", "nan")

    for i in range(0, 32):
        unique_values_set.update(map(str, dataframe[f"test{i}"].unique()))

    unique_values_set_new: set[str] = set()
    for element in unique_values_set:
        if element.startswith(html_list):
            unique_values_set_new.add(element)

    print(len(unique_values_set_new))

    with open(f"embeddings_list_{time.time()}.tsv", "w", encoding="utf-8") as file:
        file.write("\n".join(map(str, unique_values_set_new)))


if __name__ == "__main__":
    main()
