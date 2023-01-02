"""Train and evaluate a FFN for major type classification, and creates Confusion Matrix"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from datasets import load_from_disk
from keras import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

tf.config.experimental.enable_tensor_float_32_execution(True)


def main() -> None:
    """Main Loop"""

    dataset = load_from_disk("datasets/http-header-split-embedded-data-v1")

    class_names = dataset["train"].features["major_class"].names
    amount_of_classes = len(class_names)

    print(amount_of_classes)
    print(class_names)

    trainset = dataset["train"].to_tf_dataset(
        batch_size=1000, columns="array", label_cols="major_class"
    )
    valset = dataset["valid"].to_tf_dataset(
        batch_size=1000, columns="array", label_cols="major_class"
    )

    testset = dataset["test"].to_tf_dataset(
        batch_size=1000, columns="array", label_cols="major_class"
    )

    print(trainset)

    model = Sequential()
    model.add(Dense(1024, input_shape=(2048,), activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(amount_of_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(model.summary())

    model.fit(trainset, validation_data=valset, epochs=20)

    y_pred = model.predict(testset, batch_size=64, verbose=1)
    y_pred_bool = np.argmax(y_pred, axis=1)

    print(
        classification_report(
            dataset["test"]["major_class"], y_pred_bool, target_names=class_names
        )
    )

    cmd = ConfusionMatrixDisplay.from_predictions(
        dataset["test"]["major_class"], y_pred_bool, display_labels=class_names
    )

    fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
    cmd.plot(ax=ax)

    cmd.figure_.savefig("plots/ffnn_majortypes_testset_abs")

    cmd = ConfusionMatrixDisplay.from_predictions(
        dataset["test"]["major_class"],
        y_pred_bool,
        display_labels=class_names,
        normalize="true",
    )

    fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
    cmd.plot(ax=ax)

    cmd.figure_.savefig("plots/ffnn_majortypes_testset_normalized")


if __name__ == "__main__":
    main()
