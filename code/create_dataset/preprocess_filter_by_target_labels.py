"""Reads an already pre-processed file and saves samples with appropriate target labels"""
import re
import sys
import time

import pandas as pd


def parse_label(label: str) -> tuple[str, str] | None:
    """Parses desired target labels from a string"""
    result = re.search(r"(apache/[0-9\.]+)", label, flags=re.IGNORECASE)
    if result:
        return result.group(1).lower(), "Apache"

    result = re.search(r"(nginx/[0-9\.]+)", label, flags=re.IGNORECASE)
    if result:
        return result.group(1).lower(), "nginx"

    result = re.search(r"(microsoft-*iis/[0-9\.]+)", label, flags=re.IGNORECASE)
    if result:
        return result.group(1).lower(), "Microsoft-IIS"

    result = re.search(r"(openresty/[0-9\.]+)", label, flags=re.IGNORECASE)
    if result:
        return result.group(1).lower(), "openresty"

    result = re.search("(^apache)", label, flags=re.IGNORECASE)
    if result:
        return "<MAJOR>", "Apache"

    result = re.search("(^nginx)", label, flags=re.IGNORECASE)
    if result:
        return "<MAJOR>", "nginx"

    result = re.search("(^litespeed)", label, flags=re.IGNORECASE)
    if result:
        return "<MAJOR>", "LiteSpeed"

    result = re.search("(^microsoft-*iis)", label, flags=re.IGNORECASE)
    if result:
        return "<MAJOR>", "Microsoft-IIS"

    result = re.search("(^openresty)", label, flags=re.IGNORECASE)
    if result:
        return "<MAJOR>", "openresty"

    return None


def main() -> None:
    """Main Loop"""

    # Load `preprocessed_.tsv`
    dataframe = pd.read_csv(sys.argv[1], sep="\t", header=0, on_bad_lines="skip")

    with open(
        f"preprocessed_filtered_{time.time()}.tsv", "w", encoding="utf-8"
    ) as file:

        for _, row in dataframe.iterrows():

            parsed_label = parse_label(row[1])

            if parsed_label is not None:

                file.write(f"{row[0]}\t{parsed_label[0]}\t{parsed_label[1]}")

                for column in row[2:]:
                    file.write(f"\t{column}")

                file.write("\n")


if __name__ == "__main__":
    main()
