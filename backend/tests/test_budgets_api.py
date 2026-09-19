import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_current_user
from app.main import app
from app.models import User


@pytest.fixture
def client():
    app.dependency_overrides[get_current_user] = lambda: User(
        id=1, name="Test User", email="test@example.com", password_hash="x"
    )
    yield TestClient(app)
    app.dependency_overrides.clear()


def _payload(**overrides):
    body = {
        "category": "Dining", "amount": "100", "period": "monthly",
        "start_date": "2026-09-01", "end_date": "2026-09-30",
    }
    body.update(overrides)
    return body


def test_budget_rejects_invalid_period(client):
    assert client.post("/budgets", json=_payload(period="yearly")).status_code == 422


def test_budget_rejects_end_before_start(client):
    r = client.post("/budgets", json=_payload(start_date="2026-09-30", end_date="2026-09-01"))
    assert r.status_code == 422


def test_budget_rejects_non_positive_amount(client):
    assert client.post("/budgets", json=_payload(amount="0")).status_code == 422


def test_budgets_require_auth():
    app.dependency_overrides.clear()
    with TestClient(app) as c:
        assert c.get("/budgets").status_code in (401, 403)
