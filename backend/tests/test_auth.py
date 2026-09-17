import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.session import Base, get_db
from app.main import app
from app.models import User


@pytest.fixture
def client():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine, tables=[User.__table__])
    TestSessionLocal = sessionmaker(bind=engine)

    def override_get_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


def _register(client, email="user@example.com", password="testpass123"):
    return client.post(
        "/auth/register",
        json={"name": "Test User", "email": email, "password": password},
    )


def test_register_creates_user(client):
    response = _register(client)
    assert response.status_code == 201
    body = response.json()
    assert body["email"] == "user@example.com"
    assert "password" not in body
    assert "password_hash" not in body


def test_register_rejects_duplicate_email(client):
    _register(client)
    response = _register(client)
    assert response.status_code == 409


def test_login_returns_token_for_correct_credentials(client):
    _register(client)
    response = client.post(
        "/auth/login", json={"email": "user@example.com", "password": "testpass123"}
    )
    assert response.status_code == 200
    assert response.json()["token_type"] == "bearer"
    assert len(response.json()["access_token"]) > 0


def test_login_rejects_wrong_password(client):
    _register(client)
    response = client.post(
        "/auth/login", json={"email": "user@example.com", "password": "wrongpass"}
    )
    assert response.status_code == 401


def test_login_rejects_unknown_email(client):
    response = client.post(
        "/auth/login", json={"email": "nobody@example.com", "password": "testpass123"}
    )
    assert response.status_code == 401


def test_me_requires_token(client):
    response = client.get("/auth/me")
    assert response.status_code in (401, 403)


def test_me_returns_current_user_with_valid_token(client):
    _register(client)
    login = client.post(
        "/auth/login", json={"email": "user@example.com", "password": "testpass123"}
    )
    token = login.json()["access_token"]

    response = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["email"] == "user@example.com"
