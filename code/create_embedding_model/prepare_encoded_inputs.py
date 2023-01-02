"""Creates a dimensionality-reduced, usable HuggingFace dataset"""
import queue
import sys
import threading
import time
from itertools import islice

import numpy as np
import torch
from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value, config
from sklearn.decomposition import PCA
from transformers import RobertaModel, RobertaTokenizerFast


def infer():
    """Tokenizes and embeds input samples"""
    with torch.no_grad():
        with open(sys.argv[1], "r", encoding="utf-8") as file:
            while True:

                batch = list(
                    map(
                        lambda x: x.rstrip().split("\t"), list(islice(file, BATCH_SIZE))
                    )
                )

                if not batch:
                    break

                batch_array = np.array(batch)

                encoded_input_on_gpu = tokenizer(
                    batch_array[:, 3:].flatten().tolist(),
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to("cuda:0")

                encoded_output_on_gpu = mean_pooling(
                    model(**encoded_input_on_gpu),
                    encoded_input_on_gpu["attention_mask"],
                ).reshape(BATCH_SIZE, 32, 768)

                GLOBAL_QUEUE.put(
                    (batch_array[:, 1], batch_array[:, 2], encoded_output_on_gpu.cpu())
                )


def infer_pca_embeddings():
    """Creates a fit PCA model from the embedding list"""
    sentence_embeddings = []
    with torch.no_grad():
        with open(sys.argv[2], "r", encoding="utf-8") as file:
            while True:
                batch = list(map(lambda x: x.rstrip(), list(islice(file, 750))))
                if not batch:
                    break

                encoded_input_on_gpu = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to("cuda:0")

                encoded = mean_pooling(
                    model(**encoded_input_on_gpu),
                    encoded_input_on_gpu["attention_mask"],
                ).cpu()

                for element in encoded:
                    sentence_embeddings.append(np.asarray(element).astype("float32"))

    print(len(sentence_embeddings))
    print(len(sentence_embeddings[1]))
    print("FITTING PCA OBJECT...")
    pca_local = PCA(n_components=64)
    pca_local.fit(sentence_embeddings)
    print("Finished fitting PCA object...")
    return pca


def mean_pooling(model_output, attention_mask):
    """SentenceTransformers - Mean Pooling - Take attention mask into account for correct averaging"""
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def generator_from_queue():
    """A HuggingFace dataset generator that retrieves samples from a global queue"""
    while True:
        try:
            minor_classes, major_classes, reshaped_features = GLOBAL_QUEUE.get(
                block=True, timeout=10
            )
            for i in range(0, BATCH_SIZE):
                yield {
                    "minor_class": minor_classes[i],
                    "major_class": major_classes[i],
                    "array": pca.transform(reshaped_features[i]).flatten(),
                }

        except queue.Empty:
            break


BATCH_SIZE = 5

GLOBAL_QUEUE = queue.Queue(32000)
config.IN_MEMORY_MAX_SIZE = 1e9

print("LOADING TOKENIZER AND MODEL...")
tokenizer = RobertaTokenizerFast.from_pretrained("tokenizers/http-header-tokenizer-v1")
model = RobertaModel.from_pretrained("models/http-header-roberta-v1").to("cuda:0")

print("INFERRING PCA EMBEDDINGS...")
pca = infer_pca_embeddings()
torch.cuda.empty_cache()

# Spawn inference thread
inference_thread = threading.Thread(target=infer, daemon=True)
inference_thread.start()

MAJOR_CLASS_FILENAME = "major_classes_1667386531.43243.txt"
MINOR_CLASS_FILENAME = "minor_classes_1667386531.43243.txt"

features = Features(
    {
        "major_class": ClassLabel(names_file=MAJOR_CLASS_FILENAME),
        "minor_class": ClassLabel(names_file=MINOR_CLASS_FILENAME),
        "array": Sequence(Value("float32")),
    }
)
t1 = time.time()
dataset = Dataset.from_generator(generator_from_queue, features=features)
print(time.time() - t1 - 5)
inference_thread.join(timeout=5)

# We want 90% train, 10% test + validation
train_testvalid = dataset.train_test_split(test_size=0.2)
# Split the 10% test + valid in half test, half valid
test_valid = train_testvalid["test"].train_test_split(test_size=0.5)
# gather everyone if you want to have a single DatasetDict
train_test_valid_dataset = DatasetDict(
    {
        "train": train_testvalid["train"],
        "test": test_valid["test"],
        "valid": test_valid["train"],
    }
)

train_test_valid_dataset.save_to_disk("datasets/http-header-split-embedded-data-v1")
print(train_test_valid_dataset)
print((train_test_valid_dataset["train"][0]["major_class"]))

print((train_test_valid_dataset["test"][0]["minor_class"]))
print((train_test_valid_dataset["valid"][0]["minor_class"]))
