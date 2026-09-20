from datetime import date

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.deps import get_current_user
from app.database.session import Base, get_db
from app.features.monthly_features import monthly_category_panel, monthly_share_vectors
from app.main import app
from app.ml.clustering.evaluate import (
    feature_matrix,
    internal_and_external,
    kmeans_stability,
    leave_one_user_out,
    spend_level_baseline,
)
from app.ml.clustering.model import TOTAL_COL, SpendingProfiler
from app.ml.dataset import load_labeled_transactions
from app.models import CategoryRule, Transaction, User
from app.services import profile as profile_service


@pytest.fixture(scope="module")
def data():
    df = load_labeled_transactions()
    panel = monthly_category_panel(df)
    vectors = monthly_share_vectors(panel, window=12).reset_index(drop=True)
    personas = vectors["user_id"].map(df.groupby("user_id")["persona"].first())
    return df, panel, vectors, personas


def _blobs():
    """Three clearly different spending styles, 20 months each."""
    rng = np.random.default_rng(0)
    rows = []
    styles = {1: ("Food", "Travel", 0.7, 9.0), 2: ("Housing", "Food", 0.6, 11.0), 3: ("Shopping", "Food", 0.65, 10.0)}
    for uid, (main, other, share, log_total) in styles.items():
        for m in range(20):
            s = share + rng.normal(0, 0.02)
            rows.append({"user_id": uid, "month": pd.Timestamp("2025-01-01") + pd.DateOffset(months=m),
                         main: s, other: 1 - s, TOTAL_COL: log_total + rng.normal(0, 0.05)})
    return pd.DataFrame(rows).fillna(0.0)


# ---- features -----------------------------------------------------------------

def test_window_pools_months_and_shares_sum_to_one(data):
    _, panel, _, _ = data
    v1, v12 = monthly_share_vectors(panel), monthly_share_vectors(panel, window=12)
    cats = [c for c in v12.columns if c not in {"user_id", "month", TOTAL_COL}]
    assert np.allclose(v12[cats].sum(axis=1), 1.0)
    assert len(v12) < len(v1)  # the first 11 months per user lack a full window
    assert v12.groupby("user_id")["month"].min().gt(v1.groupby("user_id")["month"].min()).all()


def test_window_average_is_a_trailing_mean_without_future_data():
    months = pd.date_range("2025-01-01", periods=4, freq="MS")
    panel = pd.DataFrame({"user_id": 1, "category": "Food", "month": months, "spend": [10.0, 20.0, 30.0, 1000.0]})
    v = monthly_share_vectors(panel, window=3)
    assert np.isclose(np.expm1(v[TOTAL_COL].iloc[0]), 20.0)  # mean(10, 20, 30); month 4 is not included


# ---- model ----------------------------------------------------------------------

def test_profiler_separates_distinct_styles_and_names_them_relatively():
    v = _blobs()
    profiler = SpendingProfiler(k=3).fit(v)
    labels = profiler.predict(v)
    per_style = [set(labels[(v["user_id"] == u).to_numpy()]) for u in (1, 2, 3)]
    assert all(len(s) == 1 for s in per_style) and len(set().union(*per_style)) == 3
    names = " ".join(profiler.names_.values())
    assert "Higher-spend" in names and "Lower-spend" in names and "Housing" in names


def test_assign_margin_is_high_for_typical_and_low_for_borderline_points():
    v = _blobs()
    profiler = SpendingProfiler(k=3).fit(v)
    typical = profiler.assign(v.head(3))["margin"]
    midpoint = v[v["user_id"] == 1].head(1).copy()
    other = v[v["user_id"] == 2].head(1)
    for col in [c for c in v.columns if c not in {"user_id", "month"}]:
        midpoint[col] = (midpoint[col].to_numpy() + other[col].to_numpy()) / 2
    border = profiler.assign(midpoint)["margin"]
    assert typical.min() > border.max() and (typical >= 0).all() and (typical <= 1).all()


def test_predict_ignores_unknown_categories_and_fills_missing_ones():
    v = _blobs()
    profiler = SpendingProfiler(k=3).fit(v)
    odd = v.head(2).drop(columns=["Travel"]).assign(Brand_New=0.2)
    assert profiler.predict(odd).shape == (2,)


def test_save_load_roundtrip(tmp_path):
    v = _blobs()
    profiler = SpendingProfiler(k=3).fit(v)
    profiler.save(tmp_path / "p.joblib")
    assert (SpendingProfiler.load(tmp_path / "p.joblib").predict(v) == profiler.predict(v)).all()


# ---- evaluation helpers ---------------------------------------------------------

def test_metrics_are_perfect_when_clusters_equal_truth():
    v = _blobs()
    X = feature_matrix(v)
    truth = v["user_id"].to_numpy()
    m = internal_and_external(X, truth, truth)
    assert m["ari_vs_personas"] == 1.0 and m["nmi_vs_personas"] == 1.0 and m["silhouette"] > 0.8
    assert kmeans_stability(X, 3, seeds=range(4)) > 0.95


def test_spend_level_baseline_ignores_shares():
    v = _blobs()
    bins = spend_level_baseline(v, 3)
    assert set(bins) == {0, 1, 2}
    # Spend level alone separates these styles, so each user falls in a single bin.
    assert all(len(set(bins[(v["user_id"] == u).to_numpy()])) == 1 for u in (1, 2, 3))


def test_leave_one_user_out_reports_all_users(data):
    _, _, vectors, personas = data
    r = leave_one_user_out(vectors, personas, k=6)
    assert r["n_users"] == 12 and 0.0 <= r["user_level_persona_accuracy"] <= 1.0


# ---- API (SQLite) -----------------------------------------------------------------

@pytest.fixture
def client(monkeypatch, data):
    _, _, vectors, _ = data
    fitted = SpendingProfiler(k=4).fit(vectors)
    monkeypatch.setattr(profile_service, "get_profiler", lambda: fitted)
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine, tables=[User.__table__, Transaction.__table__, CategoryRule.__table__])
    db = sessionmaker(bind=engine)()
    db.add_all([User(id=1, name="A", email="a@x.com", password_hash="x"), User(id=2, name="B", email="b@x.com", password_hash="x")])
    for m in range(1, 5):  # user 1: four months of history
        for cat, amt in (("Housing", 20000.0), ("Food & Dining", 6000.0), ("Shopping", 3000.0)):
            db.add(Transaction(user_id=1, transaction_date=date(2026, m, 5), description=cat, amount=-amt,
                               transaction_type="debit", category=cat))
    db.add(Transaction(user_id=2, transaction_date=date(2026, 1, 5), description="X", amount=-100.0,
                       transaction_type="debit", category="Food & Dining"))  # user 2: one month only
    db.commit()
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_current_user] = lambda: db.get(User, 1)
    yield TestClient(app), db
    app.dependency_overrides.clear()
    db.close()


def test_profile_endpoint_returns_profile_with_window_flag(client):
    c, _ = client
    body = c.get("/analytics/spending-profile").json()
    assert body["window_months"] == 4 and body["is_full_window"] is False
    assert 0 <= body["assignment_margin"] <= 1
    assert body["your_top_categories"][0]["category"] == "Housing"
    assert round(body["your_typical_monthly_spend"]) == 29000


def test_profile_is_null_with_too_little_history(client):
    c, db = client
    app.dependency_overrides[get_current_user] = lambda: db.get(User, 2)
    assert c.get("/analytics/spending-profile").json() is None


def test_profile_requires_auth():
    app.dependency_overrides.clear()
    with TestClient(app) as c:
        assert c.get("/analytics/spending-profile").status_code in (401, 403)
