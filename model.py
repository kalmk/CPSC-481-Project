from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score
 
import numpy as np
import pandas as pd
import joblib


df = pd.read_csv('training_dataset.csv')

# input and output variables
X = df['review']
y = df['final_label']

# split for the training and testing sets
# first split: keep test set separate
X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# second split: split temp into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,   # 0.25 of 0.8 = 0.2 total
    random_state=42,
    stratify=y_temp
)

print("Train size:", len(X_train))
print("Validation size:", len(X_val))
print("Test size:", len(X_test))

# vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

# train a logistic regression model
model = LogisticRegression(max_iter=2000, random_state=42)

model.fit(X_train_vec, y_train)

# attempt to minimize the false positives
probs_val = model.predict_proba(X_val_vec)[:, 1]

threshold = 0.55

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