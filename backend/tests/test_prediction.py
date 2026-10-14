import numpy as np
import pytest

from app import main


class FakeModel:
    """A predictable model for testing confidence-based review logic."""

    classes_ = ["Account", "Payment", "Refund", "Shipping", "Technical"]

    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict_proba(self, messages):
        assert len(messages) == 1
        return np.array([self.probabilities])


@pytest.mark.parametrize(
    ("probabilities", "expected_category", "expected_review"),
    [
        ([0.54, 0.20, 0.10, 0.08, 0.08], "Account", True),
        ([0.55, 0.15, 0.10, 0.10, 0.10], "Account", False),
        ([0.10, 0.65, 0.10, 0.08, 0.07], "Payment", False),
    ],
)
def test_analyze_message_confidence_threshold(
    monkeypatch,
    probabilities,
    expected_category,
    expected_review,
):
    monkeypatch.setattr(main, "model", FakeModel(probabilities))

    result = main.analyze_message("Example support ticket")

    assert result["category"] == expected_category
    assert 0 <= result["confidence"] <= 1
    assert result["needs_review"] is expected_review
