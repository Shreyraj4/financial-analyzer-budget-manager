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
from app.features.monthly_features import build_forecast_features, monthly_category_panel
from app.main import app
from app.ml.dataset import load_labeled_transactions
from app.ml.forecasting.evaluate import (
    compute_metrics,
    naive_scales,
    rolling_origin_predictions,
    split_validation_test,
)
from app.ml.forecasting.model import GlobalForecaster, moving_average_3, naive, seasonal_naive
from app.ml.forecasting.predict import predict_next_month
from app.models import CategoryRule, Transaction, User


@pytest.fixture(scope="module")
def panel():
    df = load_labeled_transactions()
    return monthly_category_panel(df[df["user_id"].isin([1, 2, 3])])


@pytest.fixture(scope="module")
def features(panel):
    return build_forecast_features(panel, lags=(1, 2, 3))


def _series(values, category="Food", user_id=1):
    months = pd.date_range("2025-01-01", periods=len(values), freq="MS")
    return pd.DataFrame({"user_id": user_id, "category": category, "month": months, "spend": values})


# ---- baselines & model --------------------------------------------------------

def test_baselines_read_the_right_history():
    f = build_forecast_features(_series([10.0 * i for i in range(1, 15)]))
    last = f.iloc[-1]  # month 14 (spend 140): lag_1=130, lag_12=20, mean of 100,110,120... uses months 11-13
    assert naive(f)[-1] == 130.0
    assert moving_average_3(f)[-1] == np.mean([110.0, 120.0, 130.0])
    assert seasonal_naive(f)[-1] == 20.0
    assert last["spend"] == 140.0


def test_global_forecaster_predicts_non_negative_finite_values(features):
    preds = GlobalForecaster("ridge").fit(features).predict(features)
    assert np.isfinite(preds).all() and (preds >= 0).all()


@pytest.mark.parametrize("kind", ["ridge", "gbm"])
def test_forecaster_gives_zero_when_series_has_no_history(features, kind):
    model = GlobalForecaster(kind).fit(features)
    blank = features.iloc[:3].copy()
    blank["hist_mean"] = 0.0
    assert (model.predict(blank) == 0.0).all()


def test_unseen_category_does_not_crash(features):
    model = GlobalForecaster("ridge").fit(features)
    row = features.iloc[:2].copy()
    row["category"] = "Brand New Category"
    assert np.isfinite(model.predict(row)).all()


def test_save_load_roundtrip(features, tmp_path):
    model = GlobalForecaster("gbm", max_depth=2).fit(features)
    model.save(tmp_path / "f.joblib")
    loaded = GlobalForecaster.load(tmp_path / "f.joblib")
    assert np.allclose(loaded.predict(features.head(20)), model.predict(features.head(20)))


# ---- evaluation protocol ------------------------------------------------------

def test_rolling_origin_never_trains_on_the_test_month_or_later(features, panel):
    seen = []

    class Spy(GlobalForecaster):
        def fit(self, frame):
            seen.append(frame["month"].max())
            return super().fit(frame)

        def predict(self, frame):
            seen.append(("test", frame["month"].min(), frame["month"].max()))
            return super().predict(frame)

    preds = rolling_origin_predictions(features, panel, {"spy": lambda: Spy("ridge")})
    fits = [x for x in seen if not isinstance(x, tuple)]
    tests = [x for x in seen if isinstance(x, tuple)]
    for max_train_month, (_, test_min, test_max) in zip(fits, tests):
        assert max_train_month < test_min == test_max
    assert len(preds["month"].unique()) == len(fits) == 12


def test_validation_precedes_test(features, panel):
    preds = rolling_origin_predictions(features, panel, {})
    val, test = split_validation_test(preds)
    assert val["month"].max() < test["month"].min()
    assert len(val) + len(test) == len(preds)


def test_metrics_perfect_and_naive_reference(panel, features):
    preds = rolling_origin_predictions(features, panel, {})
    scales = naive_scales(panel)
    preds["perfect"] = preds["actual"]
    m = compute_metrics(preds, ["perfect", "naive_last_month"], scales)
    assert m["perfect"] == {"mae": 0.0, "rmse": 0.0, "wape": 0.0, "mase": 0.0}
    assert m["naive_last_month"]["mae"] > 0


# ---- prediction -----------------------------------------------------------------

def test_predict_next_month_uses_ml_for_long_series_and_fallback_for_short(features):
    model = GlobalForecaster("ridge").fit(features)
    long_series = _series([100.0] * 8, category="Food")
    short_series = _series([50.0, 70.0], category="Travel")
    out = {f.category: f for f in predict_next_month(pd.concat([long_series, short_series]), model)}
    assert out["Food"].method == "ml_global" and out["Food"].history_months == 8
    assert 50 < float(out["Food"].predicted_next_month_spend) < 200
    assert out["Travel"].method == "average" and float(out["Travel"].predicted_next_month_spend) == 60.0


def test_predict_next_month_without_model_falls_back():
    out = predict_next_month(_series([100.0, 110.0, 120.0, 130.0]), None)
    assert out[0].method == "linear_regression"


def test_predict_on_empty_panel():
    assert predict_next_month(pd.DataFrame(columns=["user_id", "category", "month", "spend"]), None) == []


# ---- API (SQLite, no Supabase) ---------------------------------------------------

@pytest.fixture
def client():
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine, tables=[User.__table__, Transaction.__table__, CategoryRule.__table__])
    db = sessionmaker(bind=engine)()
    db.add(User(id=1, name="A", email="a@x.com", password_hash="x"))
    for month in range(1, 6):
        db.add(Transaction(user_id=1, transaction_date=date(2026, month, 5), description="SWIGGY", amount=-500.0 - 20 * month,
                           transaction_type="debit", category="Food & Dining"))
    db.commit()
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_current_user] = lambda: db.get(User, 1)
    yield TestClient(app)
    app.dependency_overrides.clear()
    db.close()


def test_forecast_endpoint_returns_per_category_prediction(client):
    body = client.get("/forecast").json()
    assert [f["category"] for f in body["forecasts"]] == ["Food & Dining"]
    f = body["forecasts"][0]
    assert f["history_months"] == 5 and float(f["predicted_next_month_spend"]) > 0
