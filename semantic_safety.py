from dataclasses import dataclass
from pathlib import Path
import re

import joblib

from model_explanations import NOT_SUITABLE_LABEL, SUITABLE_LABEL


DISTILBERT_MODEL_DIR = Path("distilbert_safety_model")
DISTILBERT_THRESHOLD_PATH = Path("distilbert_threshold.pkl")
DEFAULT_DISTILBERT_THRESHOLD = 0.8


@dataclass
class SafetyPrediction:
    label: str
    suitable_probability: float
    reason: str
    source: str


_tokenizer = None
_model = None
_torch = None
_threshold = None


def distilbert_artifacts_exist():
    return (
        DISTILBERT_MODEL_DIR.exists()
        and (DISTILBERT_MODEL_DIR / "config.json").exists()
        and DISTILBERT_THRESHOLD_PATH.exists()
    )


def load_distilbert_dependencies():
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        return None

    return torch, AutoTokenizer, AutoModelForSequenceClassification


def load_distilbert_safety_model():
    global _model, _threshold, _tokenizer, _torch

    if _model is not None and _tokenizer is not None:
        return True

    if not distilbert_artifacts_exist():
        return False

    dependencies = load_distilbert_dependencies()
    if dependencies is None:
        return False

    _torch, tokenizer_class, model_class = dependencies
    _tokenizer = tokenizer_class.from_pretrained(DISTILBERT_MODEL_DIR)
    _model = model_class.from_pretrained(DISTILBERT_MODEL_DIR)
    _model.eval()
    _threshold = joblib.load(DISTILBERT_THRESHOLD_PATH)

    return True


def predict_suitable_probability(review):
    encoded_review = _tokenizer(
        review,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt",
    )

    with _torch.no_grad():
        outputs = _model(**encoded_review)
        probabilities = _torch.softmax(outputs.logits, dim=1)[0]

    return float(probabilities[1].item())


def split_review_for_explanation(review, max_units=8):
    normalized_review = re.sub(r"\s+", " ", review.strip())
    split_pattern = (
        r"(?<=[.!?;])\s+|"
        r"\s+\b(?:but|yet|however|although|though|provided|assuming|because)\b\s+"
    )
    units = [
        unit.strip(" ,.;:!?")
        for unit in re.split(split_pattern, normalized_review)
        if len(unit.strip().split()) >= 3
    ]

    if len(units) <= max_units:
        return units

    grouped_units = []
    group_size = max(1, round(len(units) / max_units))

    for index in range(0, len(units), group_size):
        grouped_units.append(" ".join(units[index : index + group_size]))

    return grouped_units[:max_units]


def get_decision_support_change(label, original_probability, changed_probability):
    if label == SUITABLE_LABEL:
        return original_probability - changed_probability

    return changed_probability - original_probability


def get_influential_phrases(review, label, suitable_probability, top_n=2):
    units = split_review_for_explanation(review)

    if len(units) < 2:
        return []

    phrase_impacts = []

    for index, unit in enumerate(units):
        changed_review = " ".join(
            candidate_unit
            for candidate_index, candidate_unit in enumerate(units)
            if candidate_index != index
        )

        changed_probability = predict_suitable_probability(changed_review)
        support_change = get_decision_support_change(
            label, suitable_probability, changed_probability
        )

        if support_change > 0.01:
            phrase_impacts.append((unit, support_change))

    phrase_impacts.sort(key=lambda item: item[1], reverse=True)

    return [phrase for phrase, _ in phrase_impacts[:top_n]]


def format_phrases(phrases):
    quoted_phrases = [f'"{phrase}"' for phrase in phrases]

    if len(quoted_phrases) == 1:
        return quoted_phrases[0]

    return " and ".join(quoted_phrases)


def explain_distilbert_prediction(review, label, suitable_probability):
    influential_phrases = get_influential_phrases(
        review, label, suitable_probability
    )

    if label == NOT_SUITABLE_LABEL:
        not_suitable_probability = 1 - suitable_probability

        if influential_phrases:
            return (
                "DistilBERT's decision was most influenced by "
                f"{format_phrases(influential_phrases)}, which lowered the "
                "model's child-suitability score."
            )

        if not_suitable_probability < 0.6:
            return (
                "The DistilBERT safety model was not confident enough to mark "
                "this review suitable, so the app applied the conservative "
                "child-safety threshold."
            )

        return (
            "The DistilBERT safety model interpreted the review as describing "
            "child-unsuitable content."
        )

    if influential_phrases:
        return (
            "DistilBERT's decision was most influenced by "
            f"{format_phrases(influential_phrases)}, which supported the "
            "child-suitable classification."
        )

    return (
        "The DistilBERT safety model did not detect mature-content patterns "
        "in the review."
    )


def classify_with_distilbert(review):
    if not load_distilbert_safety_model():
        return None

    suitable_probability = predict_suitable_probability(review)

    if suitable_probability >= _threshold:
        label = SUITABLE_LABEL
    else:
        label = NOT_SUITABLE_LABEL

    return SafetyPrediction(
        label=label,
        suitable_probability=suitable_probability,
        reason=explain_distilbert_prediction(review, label, suitable_probability),
        source="distilbert",
    )
