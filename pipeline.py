import joblib

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

    return prediction, prob


def generate_reason(review, prediction):
    review_lower = review.lower()

    unsafe_keywords = [
        "violence",
        "gore",
        "blood",
        "kill",
        "sexual",
        "nudity",
        "drug",
        "abuse",
        "horror",
        "disturbing",
        "intense",
    ]

    safe_keywords = [
        "family",
        "kids",
        "children",
        "animated",
        "wholesome",
        "fun",
        "lighthearted",
    ]

    if prediction == 0:
        for word in unsafe_keywords:
            if word in review_lower:
                return f"The review mentions {word}, which may be inappropriate for children."
        return "The review suggests mature or intense themes."

    else:
        for word in safe_keywords:
            if word in review_lower:
                return f"The review highlights {word} content suitable for children."
        return "The review suggests generally appropriate content for children."


def classify_review(review):
    prediction, prob = predict_review(review)
    reason = generate_reason(review, prediction)

    if prediction == 1:
        label = "Suitable for Children"
    else:
        label = "Not Suitable for Children"

    return {"label": label, "confidence": prob, "reason": reason}


review = "The movie has intense battle scenes and disturbing imagery."

result = classify_review(review)

print(result)
