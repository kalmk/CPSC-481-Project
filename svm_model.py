import joblib

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    recall_score,
)

from data_loading import append_supplemental_training_data, load_base_dataset
from text_features import build_text_vectorizer
from training_utils import (
    evaluate_thresholds,
    print_threshold_results,
    select_threshold,
)


# load dataset
df = load_base_dataset()

# input and output variables
X = df["review"]
y = df["final_label"]

# -----------------------------
# 1. Split into train / val / test
# -----------------------------
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,  # 0.25 of 0.8 = 0.2 total
    random_state=42,
    stratify=y_temp,
)

X_train, y_train, supplemental_count = append_supplemental_training_data(
    X_train, y_train
)

print("Train size:", len(X_train))
print("Supplemental training examples:", supplemental_count)
print("Validation size:", len(X_val))
print("Test size:", len(X_test))

# -----------------------------
# 2. Vectorize the text data using TF-IDF
# -----------------------------
vectorizer = build_text_vectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# 3. Train calibrated SVM model
# -----------------------------
base_svm = LinearSVC(random_state=42)

model = CalibratedClassifierCV(estimator=base_svm, cv=5)

model.fit(X_train_vec, y_train)

# -----------------------------
# 4. Tune threshold on validation set
# -----------------------------
probs_val = model.predict_proba(X_val_vec)[:, 1]

threshold_results = evaluate_thresholds(y_val, probs_val)
selected_threshold = select_threshold(y_val, probs_val)
best_threshold = selected_threshold["threshold"]

print_threshold_results(threshold_results)
print("\nBest threshold chosen from validation set:", best_threshold)

# -----------------------------
# 5. Final evaluation on test set
# -----------------------------
probs_test = model.predict_proba(X_test_vec)[:, 1]
y_pred = (probs_test >= best_threshold).astype(int)

recall_suitable = recall_score(y_test, y_pred, pos_label=1)
recall_not_suitable = recall_score(y_test, y_pred, pos_label=0)

print("\nRecall (Suitable):", round(recall_suitable, 2))
print("Recall (Not Suitable):", round(recall_not_suitable, 2))

print("\nTest Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))


joblib.dump(model, "svm_model.pkl")
joblib.dump(vectorizer, "svm_vectorizer.pkl")
joblib.dump(best_threshold, "svm_threshold.pkl")
