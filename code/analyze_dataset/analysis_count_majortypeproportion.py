"""Performs some data analysis; calulates major types by proportion"""
import re
import sys
from collections import Counter


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

    result = re.search(r"(^apache)", label, flags=re.IGNORECASE)
    if result:
        return "<MAJOR>", "Apache"

    result = re.search(r"(^nginx)", label, flags=re.IGNORECASE)
    if result:
        return "<MAJOR>", "nginx"

    result = re.search(r"(^litespeed)", label, flags=re.IGNORECASE)
    if result:
        return "<MAJOR>", "LiteSpeed"

    result = re.search(r"(^microsoft-*iis)", label, flags=re.IGNORECASE)
    if result:
        return "<MAJOR>", "Microsoft-IIS"

    result = re.search(r"(^openresty)", label, flags=re.IGNORECASE)
    if result:
        return "<MAJOR>", "openresty"

    return None


def main() -> None:
    """Main Loop"""

    counter = Counter()

    # Load `preprocessed_.tsv`
    with open(sys.argv[1], "r", encoding="utf-8") as file:
        for line in file:
            try:
                split = line.split("\t")
                parsed_label = parse_label(split[2])
                if parsed_label:
                    counter[parsed_label[1]] += 1
            except IndexError:
                pass

    print(counter.most_common(20))


if __name__ == "__main__":
    main()
