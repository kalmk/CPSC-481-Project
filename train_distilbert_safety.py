import inspect
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    recall_score,
)
from sklearn.model_selection import train_test_split

from data_loading import append_supplemental_training_data, load_base_dataset
from semantic_safety import DISTILBERT_MODEL_DIR, DISTILBERT_THRESHOLD_PATH
from training_utils import (
    evaluate_thresholds,
    print_threshold_results,
    select_threshold,
)

BASE_MODEL_NAME = "distilbert-base-uncased"
CHECKPOINT_DIR = Path("distilbert_checkpoints")
MAX_LENGTH = 256
RANDOM_STATE = 42


def get_model_source():
    if (DISTILBERT_MODEL_DIR / "config.json").exists():
        return str(DISTILBERT_MODEL_DIR)

    return BASE_MODEL_NAME


def require_distilbert_dependencies():
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing DistilBERT dependencies. Install them with:\n"
            "  .\\venv\\Scripts\\python.exe -m pip install -r requirements.txt"
        ) from exc

    return (
        torch,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )


class ReviewDataset:
    def __init__(self, reviews, labels, tokenizer, torch):
        self.encodings = tokenizer(
            list(reviews),
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        )
        self.labels = list(labels)
        self.torch = torch

    def __getitem__(self, index):
        item = {
            key: self.torch.tensor(values[index])
            for key, values in self.encodings.items()
        }
        item["labels"] = self.torch.tensor(self.labels[index])

        return item

    def __len__(self):
        return len(self.labels)


def load_split_data():
    df = load_base_dataset()
    X = df["review"]
    y = df["final_label"]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    X_train, y_train, supplemental_count = append_supplemental_training_data(
        X_train, y_train
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, supplemental_count


def softmax(logits):
    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted_logits)

    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def compute_metrics(eval_prediction):
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "recall_suitable": recall_score(
            labels, predictions, pos_label=1, zero_division=0
        ),
        "recall_not_suitable": recall_score(
            labels, predictions, pos_label=0, zero_division=0
        ),
    }


def build_training_arguments(TrainingArguments, model_source):
    num_train_epochs = 2 if model_source != BASE_MODEL_NAME else 3

    kwargs = {
        "output_dir": str(CHECKPOINT_DIR),
        "num_train_epochs": num_train_epochs,
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 16,
        "weight_decay": 0.01,
        "logging_steps": 25,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "recall_not_suitable",
        "greater_is_better": True,
        "report_to": "none",
        "seed": RANDOM_STATE,
    }

    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    return TrainingArguments(**kwargs)


def print_dataset_sizes(X_train, X_val, X_test, supplemental_count):
    print("Train size:", len(X_train))
    print("Supplemental training examples:", supplemental_count)
    print("Validation size:", len(X_val))
    print("Test size:", len(X_test))


def main():
    (
        torch,
        model_class,
        tokenizer_class,
        Trainer,
        TrainingArguments,
    ) = require_distilbert_dependencies()

    X_train, X_val, X_test, y_train, y_val, y_test, supplemental_count = (
        load_split_data()
    )
    print_dataset_sizes(X_train, X_val, X_test, supplemental_count)

    model_source = get_model_source()
    print("Starting DistilBERT fine-tuning from:", model_source)

    tokenizer = tokenizer_class.from_pretrained(model_source)
    model = model_class.from_pretrained(
        model_source,
        num_labels=2,
        id2label={0: "not_suitable", 1: "suitable"},
        label2id={"not_suitable": 0, "suitable": 1},
    )

    train_dataset = ReviewDataset(X_train, y_train, tokenizer, torch)
    val_dataset = ReviewDataset(X_val, y_val, tokenizer, torch)
    test_dataset = ReviewDataset(X_test, y_test, tokenizer, torch)

    trainer = Trainer(
        model=model,
        args=build_training_arguments(TrainingArguments, model_source),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    val_predictions = trainer.predict(val_dataset)
    val_probabilities = softmax(val_predictions.predictions)[:, 1]
    threshold_results = evaluate_thresholds(y_val, val_probabilities)
    selected_threshold = select_threshold(y_val, val_probabilities)
    threshold = selected_threshold["threshold"]

    print_threshold_results(threshold_results)
    print("\nBest threshold chosen from validation set:", threshold)

    test_predictions = trainer.predict(test_dataset)
    test_probabilities = softmax(test_predictions.predictions)[:, 1]
    y_pred = (test_probabilities >= threshold).astype(int)

    print("\nRecall (Suitable):", round(recall_score(y_test, y_pred, pos_label=1), 2))
    print(
        "Recall (Not Suitable):",
        round(recall_score(y_test, y_pred, pos_label=0), 2),
    )
    print("\nTest Evaluation")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    trainer.save_model(DISTILBERT_MODEL_DIR)
    tokenizer.save_pretrained(DISTILBERT_MODEL_DIR)
    joblib.dump(threshold, DISTILBERT_THRESHOLD_PATH)

    print("\nSaved DistilBERT model to:", DISTILBERT_MODEL_DIR)
    print("Saved DistilBERT threshold to:", DISTILBERT_THRESHOLD_PATH)


if __name__ == "__main__":
    main()
