"""Deleting transactions in a date range (undoing one upload).

Uses in-memory SQLite (no Supabase needed) via a get_db override, same pattern as
test_category_labeling.py.
"""
from datetime import date

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.deps import get_current_user
from app.database.session import Base, get_db
from app.main import app
from app.models import Transaction, User


@pytest.fixture
def db():
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine, tables=[User.__table__, Transaction.__table__])
    with sessionmaker(bind=engine)() as session:
        session.add_all([
            User(id=1, name="A", email="a@x.com", password_hash="x"),
            User(id=2, name="B", email="b@x.com", password_hash="x"),
        ])
        session.commit()
        yield session


@pytest.fixture
def client(db):
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_current_user] = lambda: db.get(User, 1)
    yield TestClient(app)
    app.dependency_overrides.clear()


def _txn(db, user_id, date_str, source, description="X") -> Transaction:
    t = Transaction(
        user_id=user_id, transaction_date=date.fromisoformat(date_str), description=description, amount=-100,
        transaction_type="debit", source=source,
    )
    db.add(t)
    return t


def test_delete_only_removes_this_users_rows_in_range_and_source(client, db):
    _txn(db, 1, "2026-01-05", "csv_upload", "A")
    _txn(db, 1, "2026-02-05", "csv_upload", "B")   # outside the range
    _txn(db, 1, "2026-01-06", "pdf_upload", "C")   # in range, wrong source
    _txn(db, 2, "2026-01-07", "csv_upload", "D")   # someone else's row, in range
    db.commit()

    count = client.get("/transactions/count", params={"start_date": "2026-01-01", "end_date": "2026-01-31"}).json()
    assert count == 2  # A and C: both mine, both in range, source not filtered yet

    resp = client.request(
        "DELETE", "/transactions",
        json={"start_date": "2026-01-01", "end_date": "2026-01-31", "source": "csv_upload"},
    )
    assert resp.status_code == 200
    assert resp.json()["deleted_count"] == 1  # only A

    remaining = {(t.user_id, t.description) for t in db.query(Transaction).all()}
    assert remaining == {(1, "B"), (1, "C"), (2, "D")}


def test_delete_with_no_matches_returns_zero(client, db):
    _txn(db, 1, "2026-01-05", "csv_upload")
    db.commit()
    resp = client.request("DELETE", "/transactions", json={"start_date": "2026-06-01", "end_date": "2026-06-30"})
    assert resp.json()["deleted_count"] == 0


def test_end_date_before_start_date_is_rejected(client):
    resp = client.request("DELETE", "/transactions", json={"start_date": "2026-02-01", "end_date": "2026-01-01"})
    assert resp.status_code == 422
