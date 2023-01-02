"""Fun with masked language modelling"""
import os
import sys

from transformers import RobertaForMaskedLM, RobertaTokenizerFast, pipeline
from transformers.pipelines import PipelineException

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}


def main() -> None:
    """Main Loop"""

    tokenizer = RobertaTokenizerFast.from_pretrained(
        "tokenizers/http-header-tokenizer-v1"
    )
    model = RobertaForMaskedLM.from_pretrained("models/http-header-roberta-v1")

    mask_filler = pipeline(task="fill-mask", model=model, tokenizer=tokenizer, top_k=5)

    print("Model loading complete. Enter query when ready!")
    print("Don't forget to add a <mask> token to your sentence!")
    print("---")
    for line in sys.stdin:
        try:
            text = line.rstrip()
            preds = mask_filler(text)
            print(f"Query: {text}")
            print("Score | Sequence")
            for pred in preds:
                print(f"{pred['score']:.3f} | {pred['sequence']}")
            print("")
        except PipelineException as error:
            print(error)


if __name__ == "__main__":
    main()
