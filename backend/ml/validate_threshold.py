import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from train_model import build_model, get_dataset


def main():
    texts, labels = get_dataset()
    labels_array = np.array(labels)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Each prediction is produced by a model that did not train on that example.
    probabilities = cross_val_predict(
        build_model(),
        texts,
        labels,
        cv=cv,
        method="predict_proba",
    )

    # Recover the class order from a fitted pipeline.
    reference_model = build_model().fit(texts, labels)
    class_order = reference_model.classes_
    predictions = class_order[probabilities.argmax(axis=1)]
    confidence = probabilities.max(axis=1)
    correct = predictions == labels_array

    print(f"Validation examples: {len(labels)}")
    print(f"Out-of-fold accuracy: {correct.mean():.3f}")
    print("\nThreshold analysis:")

    for threshold in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55):
        reviewed = confidence < threshold
        auto_mask = ~reviewed
        coverage = auto_mask.mean()
        auto_errors = int((~correct & auto_mask).sum())

        if auto_mask.any():
            auto_accuracy = correct[auto_mask].mean()
            accuracy_text = f"{auto_accuracy:.3f}"
        else:
            accuracy_text = "N/A"

        print(
            f"threshold={threshold:.2f} | "
            f"review={reviewed.mean():.1%} | "
            f"auto-coverage={coverage:.1%} | "
            f"auto-accuracy={accuracy_text} | "
            f"auto-errors={auto_errors}"
        )


if __name__ == "__main__":
    main()
