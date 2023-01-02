"""Train and evaluate a Random Forest for major type classification, and creates Confusion Matrix"""
import math

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report


def main() -> None:
    """Main Loop"""

    dataset = load_from_disk("datasets/http-header-split-embedded-data-v1")

    dataset["train"] = dataset["train"].select(range(78321))
    dataset["test"] = dataset["test"].select(range(97902))

    class_names = dataset["train"].features["major_class"].names
    amount_of_classes = len(class_names)

    print(amount_of_classes)
    print(class_names)

    rfc = RandomForestClassifier(n_jobs=-1)

    rfc.fit(dataset["train"]["array"], dataset["train"]["major_class"])

    y_pred = rfc.predict(dataset["test"]["array"])

    avg_importance = {}
    for index, value in enumerate(rfc.feature_importances_):
        if math.floor(index / 64) not in avg_importance:
            avg_importance[math.floor(index / 64)] = []
        avg_importance[math.floor(index / 64)].append(value)

    avg_importance_list = []
    for index, value in avg_importance.items():
        avg_importance_list.append(np.mean(value))

    for index, value in sorted(
        enumerate(avg_importance_list), reverse=True, key=lambda x: x[1]
    ):
        print(f"Feature {index:2.0f}: {value:.4f}")

    print(
        classification_report(
            dataset["test"]["major_class"], y_pred, target_names=class_names
        )
    )

    cmd = ConfusionMatrixDisplay.from_predictions(
        dataset["test"]["major_class"], y_pred, display_labels=class_names
    )

    fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
    cmd.plot(ax=ax)

    cmd.figure_.savefig("plots/rf_majortypes_testset_abs")

    cmd = ConfusionMatrixDisplay.from_predictions(
        dataset["test"]["major_class"],
        y_pred,
        display_labels=class_names,
        normalize="true",
    )

    fig, axis = plt.subplots(figsize=(10, 10), dpi=200)
    cmd.plot(ax=axis)

    cmd.figure_.savefig("plots/rf_majortypes_testset_normalized")


if __name__ == "__main__":
    main()
