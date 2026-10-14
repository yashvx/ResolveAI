import pytest

from ml.train_model import predict_category


@pytest.mark.parametrize(
    ("message", "expected_category"),
    [
        ("I forgot my password and cannot log in", "Account"),
        ("My debit card payment was declined", "Payment"),
        ("My refund has not arrived", "Refund"),
        ("My parcel has not been delivered", "Shipping"),
        ("The application crashes whenever I open it", "Technical"),
    ],
)
def test_saved_model_classifies_representative_messages(
    message,
    expected_category,
):
    result = predict_category(message)

    assert result["category"] == expected_category
    assert 0 <= result["confidence"] <= 1
    assert isinstance(result["needs_review"], bool)
