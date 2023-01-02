"""Train and evaluate a FFN for minor type classification"""

import numpy as np
import tensorflow as tf
from datasets import load_from_disk
from keras import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import classification_report

tf.config.experimental.enable_tensor_float_32_execution(True)


def main() -> None:
    """Main Loop"""

    dataset = load_from_disk("datasets/http-header-split-embedded-data-v1")

    dataset = dataset.filter(
        lambda x: x["minor_class"]
        != dataset["train"].features["minor_class"].str2int("<MAJOR>"),
        num_proc=12,
    )

    class_names = dataset["train"].features["minor_class"].names
    class_names.remove("<MAJOR>")

    amount_of_classes = len(dataset["test"].features["minor_class"].names)

    trainset = dataset["train"].to_tf_dataset(
        batch_size=1000, columns="array", label_cols="minor_class"
    )
    valset = dataset["valid"].to_tf_dataset(
        batch_size=1000, columns="array", label_cols="minor_class"
    )

    testset = dataset["test"].to_tf_dataset(
        batch_size=1000, columns="array", label_cols="minor_class"
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

    model.fit(trainset, validation_data=valset, epochs=30)

    y_pred = model.predict(testset, batch_size=64, verbose=1)
    y_pred_bool = np.argmax(y_pred, axis=1)

    print(
        classification_report(
            dataset["test"]["minor_class"],
            y_pred_bool,
            zero_division=0,
            labels=range(len(class_names)),
            target_names=class_names,
        )
    )


if __name__ == "__main__":
    main()
