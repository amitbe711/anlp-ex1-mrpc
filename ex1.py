"""Fine-tune bert-base-uncased on the MRPC paraphrase detection task (GLUE).

Implements the requirements of Advanced NLP Exercise 1 (Section 2):
- Train / eval / predict on the MRPC splits.
- Use AutoModelForSequenceClassification with the default model configuration.
- Truncate inputs to the model's maximum length and use dynamic padding.
- Log every training step to Weights & Biases.
- Append the validation accuracy of each configuration to res.txt.
- Generate predictions.txt for the test set in the required format.
"""

import argparse
import os
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

MODEL_NAME = "bert-base-uncased"
DATASET_NAME = "nyu-mll/glue"
DATASET_CONFIG = "mrpc"
NUM_LABELS = 2
SEED = 42

RES_FILE = "res.txt"
PREDICTIONS_FILE = "predictions.txt"
SAVED_MODELS_DIR = "saved_models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune bert-base-uncased on the MRPC paraphrase detection task."
    )
    parser.add_argument("--max_train_samples", type=int, default=-1,
                        help="Number of training samples to use (-1 for all).")
    parser.add_argument("--max_eval_samples", type=int, default=-1,
                        help="Number of validation samples to use (-1 for all).")
    parser.add_argument("--max_predict_samples", type=int, default=-1,
                        help="Number of test samples to use for prediction (-1 for all).")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Train batch size.")
    parser.add_argument("--do_train", action="store_true",
                        help="Run training.")
    parser.add_argument("--do_predict", action="store_true",
                        help="Run prediction and generate predictions.txt.")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to a fine-tuned model to use for prediction.")
    return parser.parse_args()


def build_tokenize_fn(tokenizer):
    # MRPC samples are pairs of sentences. We pass them as a pair so the
    # tokenizer adds the proper [SEP] separator and segment ids.
    # Truncation is enabled and the maximum length is the model's max length;
    # padding is intentionally disabled here so that DataCollatorWithPadding
    # can perform dynamic padding per batch.
    def tokenize_fn(examples):
        return tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            truncation=True,
            max_length=tokenizer.model_max_length,
        )
    return tokenize_fn


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = float((predictions == labels).mean())
    return {"accuracy": accuracy}


def select_first_n(dataset, n: int):
    if n is None or n == -1:
        return dataset
    n = min(n, len(dataset))
    return dataset.select(range(n))


def run_name_for(args: argparse.Namespace) -> str:
    return (
        f"epoch_num_{args.num_train_epochs}"
        f"_lr_{args.lr}"
        f"_batch_size_{args.batch_size}"
    )


def append_result_line(num_train_epochs: int, lr: float, batch_size: int, eval_acc: float) -> None:
    line = (
        f"epoch_num: {num_train_epochs}, "
        f"lr: {lr}, "
        f"batch_size: {batch_size}, "
        f"eval_acc: {eval_acc:.4f}\n"
    )
    with open(RES_FILE, "a", encoding="utf-8") as f:
        f.write(line)


def train(args: argparse.Namespace, tokenizer, tokenized_datasets, data_collator) -> str:
    # The Trainer reports to wandb automatically when the package is installed
    # and report_to="wandb" is set; setting WANDB_PROJECT here makes sure all
    # runs land in the same project so the train/loss curves can be exported
    # together for the required train_loss.png plot.
    os.environ.setdefault("WANDB_PROJECT", "anlp-ex1-mrpc")

    run_name = run_name_for(args)
    output_dir = os.path.join(SAVED_MODELS_DIR, run_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(args.batch_size, 8),
        logging_steps=1,
        logging_strategy="steps",
        eval_strategy="epoch",
        save_strategy="no",
        report_to=["wandb"],
        run_name=run_name,
        seed=SEED,
        load_best_model_at_end=False,
    )

    train_dataset = select_first_n(tokenized_datasets["train"], args.max_train_samples)
    eval_dataset = select_first_n(tokenized_datasets["validation"], args.max_eval_samples)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    eval_metrics = trainer.evaluate(eval_dataset=eval_dataset)
    eval_acc = float(eval_metrics["eval_accuracy"])
    print(f"Final validation accuracy: {eval_acc:.4f}")

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    append_result_line(args.num_train_epochs, args.lr, args.batch_size, eval_acc)

    try:
        import wandb

        if wandb.run is not None:
            wandb.run.summary["eval_accuracy"] = eval_acc
            wandb.finish()
    except ImportError:
        pass

    return output_dir


def predict(
    args: argparse.Namespace,
    tokenizer,
    tokenized_datasets,
    raw_datasets,
    data_collator,
    model_path: Optional[str],
) -> None:
    if model_path is None:
        raise ValueError(
            "--model_path must be provided for prediction (or run with --do_train so a model is trained first)."
        )

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # The instructions explicitly require switching to eval mode so layers
    # like dropout behave correctly during inference.
    model.eval()

    raw_test = select_first_n(raw_datasets["test"], args.max_predict_samples)
    tokenized_test = select_first_n(tokenized_datasets["test"], args.max_predict_samples)
    tokenized_test = tokenized_test.remove_columns(
        [c for c in ["idx", "label", "sentence1", "sentence2"] if c in tokenized_test.column_names]
    )

    pred_args = TrainingArguments(
        output_dir="prediction_tmp",
        per_device_eval_batch_size=max(args.batch_size, 8),
        report_to="none",
    )
    predictor = Trainer(
        model=model,
        args=pred_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    prediction_output = predictor.predict(tokenized_test)
    predicted_labels = np.argmax(prediction_output.predictions, axis=-1)

    with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
        for example, label in zip(raw_test, predicted_labels):
            sentence1 = example["sentence1"].replace("\n", " ").strip()
            sentence2 = example["sentence2"].replace("\n", " ").strip()
            f.write(f"{sentence1}###{sentence2}###{int(label)}\n")

    print(f"Wrote {len(predicted_labels)} predictions to {PREDICTIONS_FILE}")


def main() -> None:
    args = parse_args()
    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of --do_train or --do_predict must be specified.")

    set_seed(SEED)

    raw_datasets = load_dataset(DATASET_NAME, DATASET_CONFIG)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenize_fn = build_tokenize_fn(tokenizer)
    tokenized_datasets = raw_datasets.map(tokenize_fn, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trained_model_path: Optional[str] = None
    if args.do_train:
        trained_model_path = train(args, tokenizer, tokenized_datasets, data_collator)

    if args.do_predict:
        model_path = args.model_path or trained_model_path
        predict(args, tokenizer, tokenized_datasets, raw_datasets, data_collator, model_path)


if __name__ == "__main__":
    main()
