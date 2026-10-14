import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.main import app, get_db


@pytest.fixture
def client():
    # Each test gets an isolated in-memory SQLite database.
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestSession = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=test_engine,
    )
    Base.metadata.create_all(bind=test_engine)

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=test_engine)
    test_engine.dispose()


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_returns_classification(client):
    response = client.post(
        "/predict",
        json={"message": "My account has been hacked and I cannot log in"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["category"] in {
        "Account", "Payment", "Refund", "Shipping", "Technical"
    }
    assert 0 <= data["confidence"] <= 1
    assert isinstance(data["needs_review"], bool)
    assert data["message"] == (
        "My account has been hacked and I cannot log in"
    )


def test_create_ticket_and_list(client):
    response = client.post(
        "/tickets",
        json={"message": "My payment was deducted twice but my order failed"},
    )

    assert response.status_code == 201
    ticket = response.json()
    assert ticket["id"] > 0
    assert ticket["category"] in {
        "Account", "Payment", "Refund", "Shipping", "Technical"
    }
    assert ticket["priority"] == "High"
    assert ticket["status"] == "Open"

    listed = client.get("/tickets")
    assert listed.status_code == 200
    assert len(listed.json()) == 1
    assert listed.json()[0]["id"] == ticket["id"]


def test_update_ticket_status(client):
    created = client.post(
        "/tickets",
        json={"message": "Please help me with my account settings"},
    )
    ticket_id = created.json()["id"]

    response = client.patch(
        f"/tickets/{ticket_id}",
        json={"status": "Resolved"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "Resolved"


def test_missing_ticket_returns_404(client):
    response = client.patch(
        "/tickets/99999",
        json={"status": "Resolved"},
    )
    assert response.status_code == 404


def test_analytics_counts_created_ticket(client):
    client.post(
        "/tickets",
        json={"message": "My payment was deducted twice but my order failed"},
    )

    response = client.get("/analytics/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["total_tickets"] == 1
    assert data["open_tickets"] == 1
    assert data["high_priority_tickets"] == 1
    assert sum(data["categories"].values()) == 1


@pytest.mark.parametrize(
    ("confidence", "expected_review"),
    [
        (0.54, True),
        (0.55, False),
    ],
)
def test_predict_api_review_threshold(
    client,
    monkeypatch,
    confidence,
    expected_review,
):
    import numpy as np
    from app import main

    class FakeModel:
        classes_ = [
            "Account", "Payment", "Refund", "Shipping", "Technical"
        ]

        def predict_proba(self, messages):
            return np.array([[
                confidence,
                (1 - confidence) / 4,
                (1 - confidence) / 4,
                (1 - confidence) / 4,
                (1 - confidence) / 4,
            ]])

    monkeypatch.setattr(main, "model", FakeModel())

    response = client.post(
        "/predict",
        json={"message": "Example support ticket"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["needs_review"] is expected_review
    assert data["confidence"] == confidence
