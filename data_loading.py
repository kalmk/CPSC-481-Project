from pathlib import Path

import pandas as pd


BASE_DATASET_PATH = Path("training_dataset.csv")
SUPPLEMENTAL_DATASET_PATH = Path("supplemental_training_examples.csv")
REQUIRED_COLUMNS = ["review", "final_label"]


def load_labeled_reviews(path):
    df = pd.read_csv(path)
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{path} is missing required columns: {missing_columns}"
        )

    df = df.dropna(subset=REQUIRED_COLUMNS).copy()
    df["review"] = df["review"].astype(str)
    df["final_label"] = df["final_label"].astype(int)

    return df


def load_base_dataset():
    return load_labeled_reviews(BASE_DATASET_PATH)


def load_supplemental_dataset():
    if not SUPPLEMENTAL_DATASET_PATH.exists():
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    return load_labeled_reviews(SUPPLEMENTAL_DATASET_PATH)


def append_supplemental_training_data(X_train, y_train):
    supplemental_df = load_supplemental_dataset()

    if supplemental_df.empty:
        return X_train, y_train, 0

    X_train = pd.concat(
        [
            X_train.reset_index(drop=True),
            supplemental_df["review"].reset_index(drop=True),
        ],
        ignore_index=True,
    )
    y_train = pd.concat(
        [
            y_train.reset_index(drop=True),
            supplemental_df["final_label"].reset_index(drop=True),
        ],
        ignore_index=True,
    )

    return X_train, y_train, len(supplemental_df)
