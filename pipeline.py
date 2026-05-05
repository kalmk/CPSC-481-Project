import joblib

from model_explanations import (
    NOT_SUITABLE_LABEL,
    SUITABLE_LABEL,
    generate_model_reason,
)
from semantic_safety import classify_with_distilbert

# load everything
# model = joblib.load("model.pkl")
# vectorizer = joblib.load("vectorizer.pkl")
# threshold = joblib.load("threshold.pkl")

model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("svm_vectorizer.pkl")
threshold = joblib.load("svm_threshold.pkl")


def predict_review(review):
    review_vec = vectorizer.transform([review])
    prob = model.predict_proba(review_vec)[0][1]

    prediction = 1 if prob >= threshold else 0

    return prediction, prob, review_vec


def classify_review(review):
    distilbert_result = classify_with_distilbert(review)
    if distilbert_result is not None:
        return {
            "label": distilbert_result.label,
            "confidence": distilbert_result.suitable_probability,
            "reason": distilbert_result.reason,
            "source": distilbert_result.source,
        }

    prediction, prob, review_vec = predict_review(review)

    if prediction == 1:
        label = SUITABLE_LABEL
    else:
        label = NOT_SUITABLE_LABEL

    reason = generate_model_reason(review_vec, vectorizer, model, prediction)
    source = "svm"

    return {
        "label": label,
        "confidence": prob,
        "reason": reason,
        "source": source,
    }
