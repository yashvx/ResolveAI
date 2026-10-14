import numpy as np
from sklearn.model_selection import train_test_split

from train_model import build_model, get_dataset


def expected_calibration_error(confidence, correct, bins=5):
    """Calculate top-label ECE using equal-width confidence bins."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0

    for i in range(bins):
        lower, upper = edges[i], edges[i + 1]

        if i == bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)

        if not np.any(mask):
            continue

        average_confidence = confidence[mask].mean()
        accuracy = correct[mask].mean()
        weight = mask.mean()

        ece += weight * abs(average_confidence - accuracy)

        print(
            f"{lower:.1f}-{upper:.1f} | "
            f"count={mask.sum():2d} | "
            f"confidence={average_confidence:.3f} | "
            f"accuracy={accuracy:.3f}"
        )

    return float(ece)


def main():
    texts, labels = get_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    model = build_model()
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)
    predicted_indices = probabilities.argmax(axis=1)
    predictions = model.classes_[predicted_indices]
    confidence = probabilities.max(axis=1)

    actual = np.asarray(y_test)
    correct = predictions == actual

    # Multiclass Brier score: mean sum of squared probability errors.
    one_hot = np.column_stack([
        (actual == category).astype(float)
        for category in model.classes_
    ])
    brier = np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))

    print(f"Training examples: {len(X_train)}")
    print(f"Evaluation examples: {len(X_test)}")
    print(f"Accuracy: {correct.mean():.3f}")
    print(f"Multiclass Brier score: {brier:.3f}")

    print("\nTop-label confidence calibration:")
    ece = expected_calibration_error(confidence, correct, bins=5)
    print(f"Expected Calibration Error (ECE): {ece:.3f}")

    print("\nHuman-review trade-off:")
    for threshold in (0.30, 0.40, 0.45, 0.50, 0.55):
        review = confidence < threshold
        auto = ~review
        errors_without_review = int(np.sum((~correct) & auto))

        print(
            f"Threshold={threshold:.2f} | "
            f"review={review.mean():.1%} | "
            f"auto-handled={auto.mean():.1%} | "
            f"errors without review={errors_without_review}"
        )


if __name__ == "__main__":
    main()
