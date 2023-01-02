"""Filters bad samples from the raw data"""
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd


def satisfies_quality_criteria(array: list[str]) -> bool:
    """Determines if a sample passes quality criteria"""
    elements_count = Counter(array)

    total_invalid_elements = (
        elements_count[np.nan]
        + elements_count[""]
        + elements_count["<ERROR>"]
        + elements_count["<EMPTY>"]
    )

    if total_invalid_elements > len(array) / 2:
        return False

    return True


def get_true_label(array: list[str]) -> str | None:
    """Calculates a reasonable true label from a sample, if any"""
    elements_count = Counter(array)
    most_common = elements_count.most_common(1)[0]

    if (
        most_common[0]
        not in ["nan", None, "<ERROR>", "<EMPTY>", " ", "", np.nan, "None"]
        and most_common[1] > len(array) / 2
    ):
        return most_common[0]
    return None


def main() -> None:
    """Main Loop"""

    # Load `concatenated_data_withheaders.tsv`
    dataframe = pd.read_csv(sys.argv[1], sep="\t", header=0, on_bad_lines="skip")

    html_list = ("HTTP/", "<html>", "<html><", "<ERROR>", "<EMPTY>", "nan")

    with open(f"preprocessed_{time.time()}.tsv", "w", encoding="utf-8") as file:

        for _, row in dataframe.iterrows():

            labels_80 = []
            tests_80 = []
            labels_443 = []
            tests_443 = []

            for i in range(0, 15):

                # <html> responses
                test = str(row[f"test{i}"])
                if i == 3:
                    if not test.startswith(html_list):
                        test = "<html>"

                # labels
                labels_80.append(row[f"label{i}"])

                # tests
                tests_80.append(test)
            for i in range(15, 32):

                # <html> responses
                test = str(row[f"test{i}"])
                if i == 19:
                    if not test.startswith(html_list):
                        test = "<html>"

                # labels
                labels_443.append(row[f"label{i}"])

                # tests
                tests_443.append(test)

            valid_80 = False
            valid_443 = False

            true_label_80 = get_true_label(labels_80)
            if satisfies_quality_criteria(tests_80) and true_label_80:
                valid_80 = True

            true_label_443 = get_true_label(labels_443)
            if satisfies_quality_criteria(tests_443) and true_label_443:
                valid_443 = True

            joined_tests_80 = "\t".join(map(str, tests_80))
            joined_tests_443 = "\t".join(map(str, tests_443))

            # if a response on both ports: both labels must be the same
            # if not: we are dealing with two different web servers
            if valid_80 and valid_443:
                if true_label_80 == true_label_443:
                    file.write(
                        f"{row['domain']}\t{true_label_80}\t{joined_tests_80}\t{joined_tests_443}\n"
                    )


if __name__ == "__main__":
    main()
