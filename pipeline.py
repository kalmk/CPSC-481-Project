import joblib

from model_explanations import (
    NOT_SUITABLE_LABEL,
    SUITABLE_LABEL,
    generate_model_reason,
)

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
    prediction, prob, review_vec = predict_review(review)
    reason = generate_model_reason(review_vec, vectorizer, model, prediction)

    if prediction == 1:
        label = SUITABLE_LABEL
    else:
        label = NOT_SUITABLE_LABEL

    return {"label": label, "confidence": prob, "reason": reason}
