from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.deps import get_current_user
from app.database.session import Base, get_db
from app.features.transaction_features import build_transaction_features
from app.main import app
from app.ml.anomaly.detectors import (
    AnomalyDetector,
    IsolationForestDetector,
    RuleDetector,
    UnionDetector,
    ZScoreDetector,
    default_rule_detector,
)
from app.ml.anomaly.evaluate import best_f1_threshold, evaluate, month_splits
from app.ml.anomaly.explain import explain_flags
from app.ml.dataset import load_labeled_transactions
from app.models import CategoryRule, Transaction, User
from app.services import anomalies as anomalies_service


def _normal_history(n=60, user_id=1):
    start = date(2026, 1, 1)
    rows = [(user_id, start + timedelta(days=3 * i), f"SWIGGY ORDER {i}", -(400.0 + (i % 7) * 15), "Food & Dining") for i in range(n)]
    return pd.DataFrame(rows, columns=["user_id", "date", "description", "amount", "category"])


@pytest.fixture(scope="module")
def sample():
    df = load_labeled_transactions()
    df = df[df["user_id"].isin([1, 2, 3, 4])].reset_index(drop=True)
    feats = build_transaction_features(df)
    mask = feats["is_credit"] == 0
    return df[mask], feats[mask]


# ---- detectors ------------------------------------------------------------------

def test_zscore_flags_amount_spike_only():
    df = _normal_history()
    df.loc[len(df)] = (1, date(2026, 7, 1), "SWIGGY BIG", -9000.0, "Food & Dining")
    feats = build_transaction_features(df)
    scores = ZScoreDetector().score(feats)
    assert scores[-1] > 3.5 and (scores[:-1] < 3.5).all()


def test_rules_flag_duplicates_and_bursts():
    df = _normal_history()
    day = date(2026, 8, 1)
    extra = [(1, day, "AMAZON", -999.0, "Shopping"), (1, day + timedelta(days=1), "AMAZON", -999.0, "Shopping")]
    extra += [(1, date(2026, 8, 10), "UBER", -200.0 - i, "Transport") for i in range(4)]
    df = pd.concat([df, pd.DataFrame(extra, columns=df.columns)], ignore_index=True)
    feats = build_transaction_features(df)
    flagged = RuleDetector().score(feats) >= 1.0
    assert flagged[-6:].all()      # both duplicates and all 4 burst rows
    assert not flagged[:60].any()  # ordinary history is quiet


def test_isolation_forest_fits_scores_and_ranks_an_outlier_highest(sample):
    df, feats = sample
    det = IsolationForestDetector(n_estimators=100).fit(feats, df["user_id"])
    extreme = feats.iloc[:1].copy()
    extreme["amount_robust_z"], extreme["amount_share_of_monthly_spend"] = 80.0, 3.0
    extreme["same_day_merchant_count"] = 6.0
    scores = det.score(pd.concat([feats.iloc[1:200], extreme]), pd.concat([df["user_id"].iloc[1:200], df["user_id"].iloc[:1]]))
    assert np.isfinite(scores).all() and scores.argmax() == len(scores) - 1


def test_per_user_model_falls_back_for_unseen_user(sample):
    df, feats = sample
    det = IsolationForestDetector(per_user=True, n_estimators=50).fit(feats, df["user_id"])
    scores = det.score(feats.head(10), pd.Series([999] * 10, index=feats.head(10).index))
    assert np.isfinite(scores).all()


def test_union_flags_what_either_flags():
    class Const:
        def score(self, X, users=None):
            return np.array([0.9, 0.1, 0.1])

    X = pd.DataFrame({"amount_robust_z": [0.0, 10.0, 0.0]})
    scores = UnionDetector(Const(), base_threshold=0.5).score(X)
    assert (scores >= 1.0).tolist() == [True, True, False]


def test_anomaly_detector_save_load_roundtrip(sample, tmp_path):
    df, feats = sample
    model = AnomalyDetector(IsolationForestDetector(n_estimators=50).fit(feats, df["user_id"]), threshold=0.6)
    model.save(tmp_path / "a.joblib")
    loaded = AnomalyDetector.load(tmp_path / "a.joblib")
    a, b = model.flag(feats.head(50), df["user_id"].head(50)), loaded.flag(feats.head(50), df["user_id"].head(50))
    assert np.allclose(a["score"], b["score"]) and (a["is_anomaly"] == b["is_anomaly"]).all()


# ---- evaluation helpers ----------------------------------------------------------

def test_month_splits_are_chronological_and_partition_the_data():
    dates = pd.Series(pd.date_range("2024-01-01", "2025-12-31", freq="7D"))
    train, val, test = month_splits(dates)
    assert (train.astype(int) + val.astype(int) + test.astype(int) == 1).all()
    assert dates[train].max() < dates[val].min() and dates[val].max() < dates[test].min()
    assert dates[test].dt.to_period("M").nunique() == 6


def test_best_f1_threshold_separates_clean_scores():
    y = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.15, 0.3, 0.8, 0.9])
    thr = best_f1_threshold(y, scores)
    assert 0.3 < thr <= 0.8


def test_evaluate_reports_recall_by_type():
    y = np.array([1, 1, 1, 0, 0, 0])
    types = np.array(["spike", "spike", "burst", "", "", ""])
    scores = np.array([0.9, 0.8, 0.1, 0.2, 0.1, 0.1])
    m = evaluate(y, scores, 0.5, types)
    assert m["recall_by_type"] == {"burst": 0.0, "spike": 1.0}
    assert m["precision"] == 1.0 and round(m["recall"], 2) == 0.67


# ---- explanations ------------------------------------------------------------------

def test_explanations_are_specific_and_fall_back_honestly():
    df = _normal_history()
    df.loc[len(df)] = (1, date(2026, 7, 1), "SWIGGY BIG", -9000.0, "Food & Dining")
    feats = build_transaction_features(df)
    flagged = pd.Series(False, index=feats.index)
    flagged.iloc[-1] = True
    reasons = explain_flags(df, feats, flagged)
    assert list(reasons) == [len(df) - 1]
    assert reasons[len(df) - 1][0]["code"] == "amount_spike"
    assert "usual Food & Dining spend" in reasons[len(df) - 1][0]["text"]

    flagged.iloc[:] = False
    flagged.iloc[3] = True  # ordinary row flagged by a model -> honest generic reason
    assert explain_flags(df, feats, flagged)[3][0]["code"] == "unusual_pattern"


# ---- API (SQLite) -------------------------------------------------------------------

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(anomalies_service, "get_detector", default_rule_detector)
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine, tables=[User.__table__, Transaction.__table__, CategoryRule.__table__])
    db = sessionmaker(bind=engine)()
    db.add_all([User(id=1, name="A", email="a@x.com", password_hash="x"), User(id=2, name="B", email="b@x.com", password_hash="x")])
    for i in range(60):
        db.add(Transaction(user_id=1, transaction_date=date(2026, 1, 1) + timedelta(days=3 * i), description=f"SWIGGY {i}",
                           amount=-(400.0 + (i % 7) * 15), transaction_type="debit", category="Food & Dining"))
    db.add(Transaction(user_id=1, transaction_date=date(2026, 7, 1), description="SWIGGY BIG", amount=-9000.0,
                       transaction_type="debit", category="Food & Dining"))
    db.add(Transaction(user_id=2, transaction_date=date(2026, 7, 1), description="OTHER USERS BIG", amount=-90000.0,
                       transaction_type="debit", category="Food & Dining"))
    db.commit()
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_current_user] = lambda: db.get(User, 1)
    yield TestClient(app)
    app.dependency_overrides.clear()
    db.close()


def test_endpoint_returns_flagged_transaction_with_reasons(client):
    body = client.get("/analytics/transaction-anomalies").json()
    assert [b["description"] for b in body] == ["SWIGGY BIG"]
    assert body[0]["reasons"][0]["code"] == "amount_spike"


def test_endpoint_needs_history_and_respects_the_window(client):
    assert client.get("/analytics/transaction-anomalies?months=1").json()[0]["description"] == "SWIGGY BIG"
    app.dependency_overrides[get_current_user] = lambda: User(id=2, name="B", email="b@x.com", password_hash="x")
    assert client.get("/analytics/transaction-anomalies").json() == []  # one transaction: too little history


def test_endpoint_requires_auth():
    app.dependency_overrides.clear()
    with TestClient(app) as c:
        assert c.get("/analytics/transaction-anomalies").status_code in (401, 403)
