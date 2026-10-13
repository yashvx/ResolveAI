import json
from pathlib import Path

import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

MODEL_PATH = Path(__file__).parent / "ticket_classifier.joblib"
DATA_PATH = Path(__file__).parent / "holdout_examples.json"


def main():
    with DATA_PATH.open(encoding="utf-8") as file:
        examples = json.load(file)

    model = joblib.load(MODEL_PATH)
    messages = [item["message"] for item in examples]
    actual = [item["category"] for item in examples]

    probabilities = model.predict_proba(messages)
    predictions = model.predict(messages)
    confidence = probabilities.max(axis=1)

    print(f"Independent examples: {len(examples)}")
    print(f"Accuracy: {accuracy_score(actual, predictions):.3f}")

    print("\nClassification report:")
    print(
        classification_report(
            actual,
            predictions,
            labels=sorted(set(actual)),
            zero_division=0,
        )
    )

    categories = sorted(set(actual))
    print("Confusion matrix (rows = actual, columns = predicted):")
    print("Category order:", categories)
    print(confusion_matrix(actual, predictions, labels=categories))

    print("\nIndividual predictions:")
    for message, expected, predicted, score in zip(
        messages, actual, predictions, confidence
    ):
        result = "CORRECT" if expected == predicted else "WRONG"
        print(f"\n[{result}] {message}")
        print(f"  Expected: {expected}")
        print(f"  Predicted: {predicted}")
        print(f"  Confidence: {score:.3f}")

    print("\nReview threshold analysis:")
    for threshold in (0.30, 0.40, 0.45, 0.50, 0.55, 0.60):
        needs_review = confidence < threshold
        coverage = (~needs_review).mean()
        errors_without_review = sum(
            (predictions != actual) & (~needs_review)
        )
        print(
            f"Threshold={threshold:.2f} | "
            f"review={needs_review.mean():.1%} | "
            f"no-review coverage={coverage:.1%} | "
            f"wrong predictions without review={errors_without_review}"
        )


if __name__ == "__main__":
    main()
