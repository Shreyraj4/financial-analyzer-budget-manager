import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_current_user
from app.main import app
from app.models import User


@pytest.fixture
def client():
    def override_get_current_user():
        return User(id=1, name="Test User", email="test@example.com", password_hash="x")

    app.dependency_overrides[get_current_user] = override_get_current_user
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_upload_rejects_non_csv_file(client):
    response = client.post(
        "/transactions/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


def test_upload_rejects_unauthenticated_request():
    app.dependency_overrides.clear()
    with TestClient(app) as unauth_client:
        response = unauth_client.post(
            "/transactions/upload",
            files={"file": ("data.csv", b"date,description,amount\n", "text/csv")},
        )
    assert response.status_code in (401, 403)


def test_upload_returns_preview_with_valid_and_invalid_rows(client):
    content = (
        b"date,description,amount\n"
        b"2026-09-01,SWIGGY,-450.00\n"
        b"bad-date,ZOMATO,-300.00\n"
    )
    response = client.post(
        "/transactions/upload",
        files={"file": ("data.csv", content, "text/csv")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total_rows"] == 2
    assert body["valid_count"] == 1
    assert body["error_count"] == 1
    assert body["rows"][0]["valid"] is True
    assert body["rows"][1]["valid"] is False
