import pytest
from app.main import recommend_priority


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("My payment was deducted twice but my order failed", "High"),
        ("My account was hacked", "High"),
        ("My account has been hacked and I cannot log in", "High"),
        ("I have a question about my account settings", "Medium"),
    ],
)
def test_recommend_priority(message, expected):
    assert recommend_priority(message) == expected
