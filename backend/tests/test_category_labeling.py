"""User-labeling loop: confidence gate, review queue, label propagation, retraining.

Uses in-memory SQLite (no Supabase needed) via a get_db override.
"""
from datetime import date

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.deps import get_current_user
from app.database.session import Base, get_db
from app.main import app
from app.ml.categorization.model import MLCategorizer
from app.ml.categorization.retrain import retrain, user_labeled_frame
from app.ml.dataset import load_labeled_transactions
from app.models import CategoryRule, Transaction, User
from app.services.categorization import Categorizer, load_user_labels


@pytest.fixture(scope="module")
def trained_model():
    df = load_labeled_transactions()
    return MLCategorizer("logreg", C=10).fit(df.sample(4000, random_state=1))


@pytest.fixture
def db():
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    # Only the tables this feature touches (agent_reports uses Postgres-only JSONB).
    Base.metadata.create_all(engine, tables=[User.__table__, Transaction.__table__, CategoryRule.__table__])
    with sessionmaker(bind=engine)() as session:
        session.add_all([User(id=1, name="A", email="a@x.com", password_hash="x"),
                         User(id=2, name="B", email="b@x.com", password_hash="x")])
        session.commit()
        yield session


@pytest.fixture
def client(db):
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_current_user] = lambda: db.get(User, 1)
    yield TestClient(app)
    app.dependency_overrides.clear()


def _txn(db: Session, user_id: int, description: str, amount=-500.0, **kw) -> Transaction:
    t = Transaction(user_id=user_id, transaction_date=date(2026, 1, 1), description=description,
                    amount=amount, transaction_type="debit" if amount < 0 else "credit", **kw)
    db.add(t)
    db.commit()
    return t


# ---- Categorizer priority + confidence gate ---------------------------------

def test_priority_user_then_rule_then_model_then_review(trained_model):
    rules = [CategoryRule(merchant_pattern="ZOMATO", category="Rule Food")]
    user_labels = {"SWIGGY": ("My Food", None)}
    cat = Categorizer(rules, user_labels, trained_model, threshold=0.7)
    out = cat.categorize_rows([("UPI/1234567890/SWIGGY/ybl", -400), ("ZOMATO ORDER", -300),
                               ("DMART", -2000), ("XQZJ KLMNOP", -50)])
    assert (out[0].category, out[0].source) == ("My Food", "user")
    assert (out[1].category, out[1].source) == ("Rule Food", "rule")
    assert out[2].source == "model" and out[2].confidence >= 0.7
    assert out[3].needs_review and out[3].suggested_category is not None


def test_low_confidence_prediction_is_not_auto_applied(trained_model):
    out = Categorizer([], {}, trained_model, threshold=0.999999).categorize_rows([("DMART", -2000)])[0]
    assert out.category is None and out.needs_review and out.suggested_category


def test_without_model_falls_back_to_review():
    out = Categorizer([], {}, None).categorize_rows([("SOMETHING NEW", -10)])[0]
    assert out.needs_review and out.suggested_category is None


# ---- API: label, propagate, review queue ------------------------------------

def test_label_propagates_to_same_merchant_only(client, db):
    a = _txn(db, 1, "UPI/111111111/NEWSHOP/ybl")
    b = _txn(db, 1, "NEWSHOP PUNE")
    other_merchant = _txn(db, 1, "DIFFERENT STORE")
    other_user = _txn(db, 2, "NEWSHOP DELHI")
    already_user = _txn(db, 1, "POS 1234XXXXXX5678 NEWSHOP", category="Keep Me", category_source="user")

    r = client.patch(f"/transactions/{a.id}/category", json={"category": "Gadgets"})
    assert r.status_code == 200
    assert r.json()["updated_count"] == 2

    for t in (a, b, other_merchant, other_user, already_user):
        db.refresh(t)
    assert (a.category, a.category_source) == ("Gadgets", "user")
    assert b.category == "Gadgets"
    assert other_merchant.category is None
    assert other_user.category is None            # other users are never touched
    assert already_user.category == "Keep Me"      # earlier user labels are not overwritten


def test_label_single_transaction_when_not_applying_to_merchant(client, db):
    a = _txn(db, 1, "NEWSHOP A")
    b = _txn(db, 1, "NEWSHOP B")
    r = client.patch(f"/transactions/{a.id}/category", json={"category": "X", "apply_to_merchant": False})
    assert r.json()["updated_count"] == 1
    db.refresh(b)
    assert b.category is None


def test_cannot_label_someone_elses_transaction(client, db):
    t = _txn(db, 2, "NEWSHOP")
    assert client.patch(f"/transactions/{t.id}/category", json={"category": "X"}).status_code == 404


def test_blank_category_rejected(client, db):
    t = _txn(db, 1, "NEWSHOP")
    assert client.patch(f"/transactions/{t.id}/category", json={"category": "   "}).status_code == 422


def test_review_queue_lists_only_uncategorized_own_transactions(client, db):
    _txn(db, 1, "UNKNOWN A")
    _txn(db, 1, "KNOWN", category="Food", category_source="rule")
    _txn(db, 2, "UNKNOWN B")
    r = client.get("/transactions/review")
    assert [t["description"] for t in r.json()] == ["UNKNOWN A"]


def test_categories_endpoint_includes_custom_labels(client, db):
    db.add(CategoryRule(merchant_pattern="X", category="Rule Cat"))
    _txn(db, 1, "SHOP", category="Pets", category_source="user")
    cats = client.get("/transactions/categories").json()
    assert "Pets" in cats and "Rule Cat" in cats and cats == sorted(cats)


def test_user_labels_remembered_for_future_uploads(db):
    _txn(db, 1, "UPI/222222222/NEWSHOP/ybl", category="Gadgets", category_source="user")
    labels = load_user_labels(db, 1)
    assert labels == {"NEWSHOP": ("Gadgets", None)}
    assert load_user_labels(db, 2) == {}


# ---- Retraining ---------------------------------------------------------------

def test_retrain_learns_a_new_user_defined_category(db, tmp_path):
    for i in range(6):
        _txn(db, 1, f"UPI/33333333{i}/PETCO PET SUPPLIES/ybl", -800 - i, category="Pets", category_source="user")
    _txn(db, 1, "AUTO LABELED", category="Food", category_source="model")  # not a user label

    frame = user_labeled_frame(db)
    assert len(frame) == 6 and set(frame["category"]) == {"Pets"}

    base = load_labeled_transactions().sample(2000, random_state=0)
    summary = retrain(db, base=base, save_to=tmp_path / "m.joblib")
    assert summary["user_labeled_rows"] == 6

    model = MLCategorizer.load(tmp_path / "m.joblib")
    pred = model.predict(pd.DataFrame({"description": ["POS 1234XXXXXX5678 PETCO PET SUPPLIES"], "amount": [-820.0]}))
    assert pred[0] == "Pets"
