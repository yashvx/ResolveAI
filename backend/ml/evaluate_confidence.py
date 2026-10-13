import numpy as np
from sklearn.model_selection import train_test_split

from train_model import build_model, get_dataset

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
correct = predictions == np.array(y_test)

print(f"Holdout examples: {len(y_test)}")
print(f"Overall accuracy: {correct.mean():.3f}")
print("\nConfidence-band results:")

bands = [
    ("0–20%", 0.0, 0.2),
    ("20–40%", 0.2, 0.4),
    ("40–60%", 0.4, 0.6),
    ("60–80%", 0.6, 0.8),
    ("80–100%", 0.8, 1.000001),
]

for name, lower, upper in bands:
    mask = (confidence >= lower) & (confidence < upper)
    count = int(mask.sum())

    if count:
        print(
            f"{name:>8}: {count:2d} examples, "
            f"accuracy={correct[mask].mean():.3f}"
        )
    else:
        print(f"{name:>8}: 0 examples")

print("\nCandidate review thresholds:")
for threshold in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
    review = confidence < threshold
    print(
        f"Threshold={threshold:.2f} | "
        f"review={review.mean():.1%} | "
        f"no-review coverage={(~review).mean():.1%}"
    )
