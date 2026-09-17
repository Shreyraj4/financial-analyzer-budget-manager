"""Integration tests that hit the real Supabase database.

These exist because the import service uses Postgres-specific
`ON CONFLICT DO NOTHING` (see app/services/transaction_import.py), which
SQLite cannot emulate - so unlike the rest of the suite, this file needs
a real Postgres connection. Each test creates its own throwaway user and
deletes all rows it created in a fixture teardown, so re-running this
file never leaves residue in the dev database.
"""

import pytest
from fastapi.testclient import TestClient

from app.database.session import SessionLocal
from app.main import app
from app.models import Transaction, User
from app.utils.security import hash_password

client = TestClient(app)


@pytest.fixture
def test_user():
    db = SessionLocal()
    user = User(
        name="Integration Test User",
        email="integration-test@example.com",
        password_hash=hash_password("testpass123"),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    user_id = user.id
    db.close()

    yield user_id

    db = SessionLocal()
    db.query(Transaction).filter(Transaction.user_id == user_id).delete()
    db.query(User).filter(User.id == user_id).delete()
    db.commit()
    db.close()


def _auth_headers(email: str, password: str) -> dict:
    response = client.post("/auth/login", json={"email": email, "password": password})
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_import_then_reimport_deduplicates(test_user):
    headers = _auth_headers("integration-test@example.com", "testpass123")
    payload = {
        "transactions": [
            {
                "transaction_date": "2026-09-01",
                "description": "SWIGGY",
                "merchant": "SWIGGY",
                "amount": "-450.00",
                "transaction_type": "debit",
                "source": "csv_upload",
            }
        ]
    }

    first = client.post("/transactions/import", json=payload, headers=headers)
    assert first.status_code == 200
    assert first.json() == {"submitted_count": 1, "imported_count": 1, "duplicate_count": 0}

    second = client.post("/transactions/import", json=payload, headers=headers)
    assert second.status_code == 200
    assert second.json() == {"submitted_count": 1, "imported_count": 0, "duplicate_count": 1}


def test_imported_transactions_appear_in_list(test_user):
    headers = _auth_headers("integration-test@example.com", "testpass123")
    payload = {
        "transactions": [
            {
                "transaction_date": "2026-09-02",
                "description": "DMART",
                "merchant": "DMART",
                "amount": "-2300.50",
                "transaction_type": "debit",
                "source": "csv_upload",
            }
        ]
    }
    client.post("/transactions/import", json=payload, headers=headers)

    response = client.get("/transactions", headers=headers)
    assert response.status_code == 200
    descriptions = [t["description"] for t in response.json()]
    assert "DMART" in descriptions


def test_list_transactions_only_returns_own_transactions(test_user):
    other_headers = None
    db = SessionLocal()
    other_user = User(
        name="Other User", email="integration-test-other@example.com",
        password_hash=hash_password("testpass123"),
    )
    db.add(other_user)
    db.commit()
    db.refresh(other_user)
    other_id = other_user.id
    db.close()

    try:
        my_headers = _auth_headers("integration-test@example.com", "testpass123")
        client.post(
            "/transactions/import",
            json={"transactions": [{
                "transaction_date": "2026-09-03", "description": "MINE",
                "merchant": "MINE", "amount": "-1", "transaction_type": "debit",
            }]},
            headers=my_headers,
        )

        other_headers = _auth_headers("integration-test-other@example.com", "testpass123")
        response = client.get("/transactions", headers=other_headers)
        descriptions = [t["description"] for t in response.json()]
        assert "MINE" not in descriptions
    finally:
        db = SessionLocal()
        db.query(Transaction).filter(Transaction.user_id == other_id).delete()
        db.query(User).filter(User.id == other_id).delete()
        db.commit()
        db.close()
