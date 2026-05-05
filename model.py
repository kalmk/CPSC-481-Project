from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    recall_score,
)

import joblib

from data_loading import append_supplemental_training_data, load_base_dataset
from text_features import build_text_vectorizer
from training_utils import (
    evaluate_thresholds,
    print_threshold_results,
    select_threshold,
)


df = load_base_dataset()

# input and output variables
X = df["review"]
y = df["final_label"]

# split for the training and testing sets
# first split: keep test set separate
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# second split: split temp into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

X_train, y_train, supplemental_count = append_supplemental_training_data(
    X_train, y_train
)

print("Train size:", len(X_train))
print("Supplemental training examples:", supplemental_count)
print("Validation size:", len(X_val))
print("Test size:", len(X_test))

# vectorize the text data using TF-IDF
vectorizer = build_text_vectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

# train a logistic regression model
model = LogisticRegression(max_iter=2000, random_state=42)

model.fit(X_train_vec, y_train)

# attempt to minimize the false positives
probs_val = model.predict_proba(X_val_vec)[:, 1]
threshold_results = evaluate_thresholds(y_val, probs_val)
selected_threshold = select_threshold(y_val, probs_val)
threshold = selected_threshold["threshold"]

print_threshold_results(threshold_results)
print("\nBest threshold chosen from validation set:", threshold)

probs_test = model.predict_proba(X_test_vec)[:, 1]
y_pred = (probs_test >= threshold).astype(int)

# finding the best threshold based on validation set
recall_safe = recall_score(y_test, y_pred, pos_label=1)
recall_unsafe = recall_score(y_test, y_pred, pos_label=0)


print(f"Recall (Suitable): {recall_safe:.2f}")
print(f"Recall (Not Suitable): {recall_unsafe:.2f}")


print("\nTest Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))


joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(threshold, "threshold.pkl")
