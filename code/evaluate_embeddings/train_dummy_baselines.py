"""Calculates the dummy baselines in the dataset"""

from datasets import load_from_disk
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report


def main() -> None:
    """Main Loop"""

    dataset = load_from_disk("datasets/http-header-split-embedded-data-v1")

    dataset["train"] = dataset["train"].select(range(78321))
    dataset["test"] = dataset["test"].select(range(97902))

    major_class_names = dataset["train"].features["major_class"].names

    dataset_minor = dataset.filter(
        lambda x: x["minor_class"]
        != dataset["train"].features["minor_class"].str2int("<MAJOR>"),
        num_proc=12,
    )

    minor_class_names = dataset["train"].features["minor_class"].names
    minor_class_names.remove("<MAJOR>")

    dummy_majortypes = DummyClassifier(strategy="most_frequent")
    dummy_minortypes = DummyClassifier(strategy="most_frequent")

    dummy_majortypes.fit(dataset["train"]["array"], dataset["train"]["major_class"])
    dummy_minortypes.fit(
        dataset_minor["train"]["array"], dataset_minor["train"]["minor_class"]
    )

    major_pred = dummy_majortypes.predict(dataset["test"]["array"])
    minor_pred = dummy_minortypes.predict(dataset_minor["test"]["array"])

    print(
        classification_report(
            dataset["test"]["major_class"], major_pred, target_names=major_class_names
        )
    )
    print(
        classification_report(
            dataset_minor["test"]["minor_class"],
            minor_pred,
            zero_division=0,
            labels=range(len(minor_class_names)),
            target_names=minor_class_names,
        )
    )


if __name__ == "__main__":
    main()
