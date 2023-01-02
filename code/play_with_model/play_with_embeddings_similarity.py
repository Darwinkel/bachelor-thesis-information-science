"""Fun with embedding cosine similarity"""
import os
import sys

import torch
from scipy import spatial
from transformers import RobertaModel, RobertaTokenizerFast

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}


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


def main() -> None:
    """Main Loop"""

    tokenizer = RobertaTokenizerFast.from_pretrained(
        "tokenizers/http-header-tokenizer-v1"
    )
    model = RobertaModel.from_pretrained("models/http-header-roberta-v1")

    print("Model loading complete. Enter query when ready!")
    print("Seperate two sentences with a <SPLIT> token")
    print("---")
    for line in sys.stdin:
        try:
            text = line.rstrip().split("<SPLIT>")
            encoded_input = tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            with torch.no_grad():
                model_output = model(**encoded_input)
            sentence_embeddings = mean_pooling(
                model_output, encoded_input["attention_mask"]
            )
            cosine_similarity = spatial.distance.cosine(
                sentence_embeddings[0], sentence_embeddings[1]
            )

            print(f"Sentence 1 Input: {text[0]}")
            print("Sentence 1 Embedding:")
            print(sentence_embeddings[0])
            print(f"Sentence 2 Input: {text[1]}")
            print("Sentence 2 Embedding")
            print(sentence_embeddings[1])
            print(f"Cosine similarity: {cosine_similarity}")
            print("")
        except Exception as error:
            print(error)


if __name__ == "__main__":
    main()
