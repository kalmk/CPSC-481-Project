import numpy as np
from sklearn.metrics import confusion_matrix, recall_score


def evaluate_thresholds(y_true, suitable_probabilities, thresholds=None):
    if thresholds is None:
        thresholds = [round(threshold, 2) for threshold in np.arange(0.5, 0.91, 0.05)]

    results = []

    for threshold in thresholds:
        predictions = (suitable_probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).ravel()

        results.append(
            {
                "threshold": float(threshold),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "true_negatives": int(tn),
                "recall_suitable": recall_score(
                    y_true, predictions, pos_label=1, zero_division=0
                ),
                "recall_not_suitable": recall_score(
                    y_true, predictions, pos_label=0, zero_division=0
                ),
            }
        )

    return results


def select_threshold(y_true, suitable_probabilities, min_suitable_recall=0.5):
    results = evaluate_thresholds(y_true, suitable_probabilities)
    viable_results = [
        result
        for result in results
        if result["recall_suitable"] >= min_suitable_recall
    ]

    if not viable_results:
        viable_results = results

    return min(
        viable_results,
        key=lambda result: (
            result["false_positives"],
            -result["recall_not_suitable"],
            -result["threshold"],
            -result["recall_suitable"],
        ),
    )


def print_threshold_results(results):
    print("\nValidation threshold search:")

    for result in results:
        print(f"\nThreshold: {result['threshold']:.2f}")
        print(
            "FP: {false_positives}, FN: {false_negatives}, "
            "TP: {true_positives}, TN: {true_negatives}".format(**result)
        )
        print(f"Recall (Suitable): {result['recall_suitable']:.2f}")
        print(f"Recall (Not Suitable): {result['recall_not_suitable']:.2f}")
