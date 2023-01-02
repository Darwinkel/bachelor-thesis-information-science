"""Initializes and trains a model from scratch"""
import sys

from datasets import load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
)


def main() -> None:
    """Main Loop"""

    tokenizer = RobertaTokenizerFast.from_pretrained(
        "tokenizers/http-header-tokenizer-v1"
    )

    dataset = load_dataset("text", data_files=sys.argv[1])
    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(examples["text"], truncation=True, padding=True),
        batched=True,
        num_proc=12,
    )

    config = RobertaConfig(
        vocab_size=2048,
    )
    model = RobertaForMaskedLM(config=config)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir="checkpoints/http-header-roberta-v1",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        num_train_epochs=100,
        learning_rate=1e-4,
        weight_decay=0.01,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        save_steps=500,
        save_total_limit=4,
        tf32=True,
    )
    # Create the trainer for our model
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["train"],
    )
    # Train the model
    trainer.train()
    trainer.save_model("models/http-header-roberta-v1")


if __name__ == "__main__":
    main()
