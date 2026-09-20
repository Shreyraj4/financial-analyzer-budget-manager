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
from app.features.monthly_features import build_forecast_features, monthly_category_panel
from app.main import app
from app.ml.anomaly.detectors import default_rule_detector
from app.ml.dataset import load_labeled_transactions
from app.ml.forecasting.model import GlobalForecaster
from app.ml.recommendation.budget import BudgetCalibrator, BudgetModel, conformal_quantile
from app.ml.recommendation.evaluate import budget_metrics, conformal_budgets, split_months, uncalibrated_budgets
from app.models import Budget, CategoryRule, Transaction, User
from app.services import anomalies as anomalies_service
from app.services import recommendations as rec_service
from app.services.recommendations import advise


# ---- conformal margin ---------------------------------------------------------------

def test_conformal_quantile_uses_finite_sample_rank():
    r = np.arange(1, 10, dtype=float)             # 1..9
    assert conformal_quantile(r, 0.8) == 8.0      # ceil(0.8 * 10) = 8th smallest
    assert conformal_quantile(r, 0.99) == 9.0     # rank is capped at n
    assert conformal_quantile(np.array([]), 0.8) == 0.0


def test_calibrator_uses_category_margin_only_with_enough_data():
    rng = np.random.default_rng(0)
    res = pd.DataFrame({"category": ["Big"] * 60 + ["Small"] * 5, "r": np.r_[rng.normal(0, 1, 60), rng.normal(5, 1, 5)]})
    cal = BudgetCalibrator(min_pool=30).fit(res, 0.8)
    assert cal.uses_category_margin("Big") and not cal.uses_category_margin("Small")
    assert cal.margin_ratio(["Small"])[0] == cal.global_q_ == cal.margin_ratio(["Unknown"])[0]


def _synthetic_preds(n_months=12, n_series=60, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_months):
        for s in range(n_series):
            typical = 1000.0 * (1 + s % 5)
            forecast = typical
            rows.append({"user_id": s // 5, "category": f"c{s % 5}", "month": pd.Timestamp("2025-01-01") + pd.DateOffset(months=m),
                         "actual": typical * max(rng.normal(1, noise), 0), "hist_mean": typical, "f": forecast})
    return pd.DataFrame(rows)


def test_conformal_budgets_achieve_roughly_the_target_coverage():
    preds = _synthetic_preds()
    b = conformal_budgets(preds, "f", 0.8, pd.Timestamp("2025-04-01"))
    assert 0.12 < budget_metrics(b, 0.8)["breach_rate"] < 0.28


def test_margin_for_a_month_never_uses_that_month_or_later():
    preds = _synthetic_preds()
    month = pd.Timestamp("2025-07-01")
    baseline = conformal_budgets(preds, "f", 0.8, month)
    tampered = preds.copy()
    tampered.loc[tampered["month"] >= month, "actual"] *= 100     # wreck the future, including the target month
    after = conformal_budgets(tampered, "f", 0.8, month)
    first = baseline["month"] == month
    assert np.allclose(baseline.loc[first, "budget"].to_numpy(), after.loc[after["month"] == month, "budget"].to_numpy())


def test_uncalibrated_budget_is_breached_about_half_the_time():
    preds = _synthetic_preds()
    b = uncalibrated_budgets(preds, "f", pd.Timestamp("2025-04-01"))
    assert 0.35 < budget_metrics(b, 0.8)["breach_rate"] < 0.65


def test_budget_metrics_known_values():
    rows = pd.DataFrame({"actual": [100.0, 100.0, 100.0, 100.0], "budget": [120.0, 80.0, 100.0, 150.0]})
    m = budget_metrics(rows, 0.8)
    assert m["breach_rate"] == 0.25                              # only the 80 budget is exceeded (strictly)
    assert m["slack"] == round((20 + 0 + 50) / 400, 4) and m["shortfall"] == round(20 / 400, 4)
    # pinball at tau=.8: under-budget by 20 costs .8*20=16; over-budget by 20 costs .2*20=4 ...
    assert m["pinball_loss"] == round((4 + 16 + 0 + 10) / 4, 2)


def test_split_months_orders_origins():
    months = [pd.Timestamp("2025-01-01") + pd.DateOffset(months=i) for i in range(12)][::-1]
    assert split_months(months, 3, 3) == (pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-01"))


# ---- advice rules ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "existing, expected",
    [
        (None, ("set_budget", 1200.0)),
        (900.0, ("raise", 300.0)),       # existing below the forecast: likely to be exceeded
        (1100.0, ("keep", 0.0)),         # between forecast and recommendation * 1.25
        (2000.0, ("tighten", 800.0)),    # far above what is needed
    ],
)
def test_advice_rules(existing, expected):
    got = advise(recommended=1200.0, forecast=1000.0, existing=existing)
    assert (got["action"], got["action_amount"]) == expected


# ---- BudgetModel + API (SQLite) ---------------------------------------------------------

@pytest.fixture(scope="module")
def budget_model():
    df = load_labeled_transactions()
    df = df[df["user_id"].isin([1, 2, 3])]
    panel = monthly_category_panel(df)
    forecaster = GlobalForecaster("ridge").fit(build_forecast_features(panel, lags=(1, 2, 3)))
    residuals = pd.DataFrame({"category": ["Food & Dining"] * 40, "r": np.linspace(-0.5, 0.9, 40)})
    return BudgetModel(forecaster, BudgetCalibrator().fit(residuals, 0.8), use_clean_history=True, name="test")


def test_recommendation_is_forecast_plus_margin(budget_model):
    months = pd.date_range("2025-01-01", periods=8, freq="MS")
    panel = pd.DataFrame({"user_id": 1, "category": "Food & Dining", "month": months, "spend": [1000.0] * 8})
    row = budget_model.recommend(panel).iloc[0]
    assert row["basis"] == "ml_conformal_category"
    assert np.isclose(row["recommended_budget"], row["forecast"] + row["margin"]) and row["margin"] > 0


@pytest.fixture
def client(monkeypatch, budget_model):
    monkeypatch.setattr(rec_service, "get_budget_model", lambda: budget_model)
    monkeypatch.setattr(anomalies_service, "get_detector", default_rule_detector)
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine, tables=[User.__table__, Transaction.__table__, CategoryRule.__table__, Budget.__table__])
    db = sessionmaker(bind=engine)()
    db.add_all([User(id=1, name="A", email="a@x.com", password_hash="x"), User(id=2, name="B", email="b@x.com", password_hash="x")])
    start = date(2026, 1, 1)
    for i in range(120):   # 4 months of steady Food & Dining spending, plus one absurd one-off in month 4
        db.add(Transaction(user_id=1, transaction_date=start + timedelta(days=i), description=f"SWIGGY {i}",
                           amount=-(300.0 + (i % 5) * 20), transaction_type="debit", category="Food & Dining"))
    db.add(Transaction(user_id=1, transaction_date=date(2026, 4, 20), description="SWIGGY HUGE", amount=-90000.0,
                       transaction_type="debit", category="Food & Dining"))
    db.add(Budget(user_id=1, category="Food & Dining", amount=1000, period="monthly", start_date=date(2026, 4, 1), end_date=date(2026, 4, 30)))
    db.commit()
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_current_user] = lambda: db.get(User, 1)
    yield TestClient(app), db
    app.dependency_overrides.clear()
    db.close()


def test_endpoint_excludes_anomalous_spend_and_advises_against_existing_budget(client):
    c, _ = client
    body = c.get("/recommendations/budgets").json()
    assert body["for_month"] == "2026-05-01" and body["target_coverage"] == 0.8
    assert body["excluded_anomalous_transactions"] >= 1 and body["excluded_anomalous_spend"] >= 90000
    item = body["recommendations"][0]
    assert item["category"] == "Food & Dining"
    assert item["forecast"] < 20000                       # the one-off Rs 90,000 did not poison the forecast
    assert item["last_month_spend"] > 90000               # ...but the user's real spending is still reported
    assert item["existing_budget"] == 1000 and item["action"] == "raise"
    assert item["recommended_budget"] >= item["forecast"]


def test_endpoint_returns_null_for_user_without_data(client):
    c, db = client
    app.dependency_overrides[get_current_user] = lambda: db.get(User, 2)
    assert c.get("/recommendations/budgets").json() is None


def test_endpoint_null_when_no_model_and_requires_auth(monkeypatch):
    monkeypatch.setattr(rec_service, "get_budget_model", lambda: None)
    app.dependency_overrides.clear()
    with TestClient(app) as c:
        assert c.get("/recommendations/budgets").status_code in (401, 403)
