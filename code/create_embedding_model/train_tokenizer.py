"""Trains and saves a tokenizer from scratch, based on file given as argument"""
import sys

from transformers import RobertaTokenizerFast


def main() -> None:
    """Main Loop"""

    old_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    with open(sys.argv[1], "r", encoding="utf-8") as file:
        vocabulary = file.read().splitlines()

    new_tokenizer = old_tokenizer.train_new_from_iterator(
        text_iterator=vocabulary, vocab_size=2048
    )

    new_tokenizer.save_pretrained("tokenizers/http-header-tokenizer-v2")


if __name__ == "__main__":
    main()
