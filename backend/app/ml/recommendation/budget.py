"""Budget recommendation = ML forecast + a calibrated safety margin.

A point forecast is what we expect to be spent, so a budget equal to it is
breached about half the time. To promise "you should stay within this ~80% of
months" we add a margin taken from the model's own past errors (split
conformal prediction):

    scaled residual  r = (actual - forecast) / typical_monthly_spend
    margin quantile  q = the ceil(coverage * (n + 1))-th smallest r
    budget           = forecast + q * typical_monthly_spend

Under mild assumptions (past and future errors are exchangeable) this covers
about ``coverage`` of future months, whatever forecaster is used. The margin is
learned per category when there is enough data (travel is far less predictable
than utilities), otherwise from all categories pooled.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.config import REPO_ROOT
from app.ml.forecasting.model import GlobalForecaster
from app.ml.forecasting.predict import predict_next_month

MODEL_PATH = REPO_ROOT / "backend" / "models" / "budget_model.joblib"
DEFAULT_COVERAGE = 0.8
MIN_POOL_PER_CATEGORY = 30


def conformal_quantile(residuals: np.ndarray, coverage: float) -> float:
    """Finite-sample-corrected empirical quantile of the scaled residuals."""
    r = np.sort(np.asarray(residuals, dtype=float))
    if len(r) == 0:
        return 0.0
    rank = int(np.ceil(coverage * (len(r) + 1)))
    return float(r[min(rank, len(r)) - 1])


class BudgetCalibrator:
    """Per-category (with global fallback) margin quantiles of scaled residuals."""

    def __init__(self, min_pool: int = MIN_POOL_PER_CATEGORY):
        self.min_pool = min_pool

    def fit(self, residuals: pd.DataFrame, coverage: float) -> "BudgetCalibrator":
        """``residuals``: columns category, r (scaled residual)."""
        self.coverage_ = coverage
        self.global_q_ = conformal_quantile(residuals["r"].to_numpy(), coverage)
        self.by_category_ = {
            cat: conformal_quantile(g["r"].to_numpy(), coverage)
            for cat, g in residuals.groupby("category")
            if len(g) >= self.min_pool
        }
        return self

    def margin_ratio(self, categories) -> np.ndarray:
        return np.array([self.by_category_.get(c, self.global_q_) for c in categories])

    def uses_category_margin(self, category: str) -> bool:
        return category in self.by_category_


class BudgetModel:
    """The deployable unit: a forecaster, its calibrated margins, and the history convention it was trained on."""

    def __init__(self, forecaster: GlobalForecaster, calibrator: BudgetCalibrator, use_clean_history: bool, name: str = ""):
        self.forecaster = forecaster
        self.calibrator = calibrator
        self.use_clean_history = use_clean_history
        self.name = name

    @property
    def coverage(self) -> float:
        return self.calibrator.coverage_

    def recommend(self, panel: pd.DataFrame) -> pd.DataFrame:
        """One row per category of a single user's monthly panel."""
        forecasts = predict_next_month(panel, self.forecaster)
        if not forecasts:
            return pd.DataFrame(columns=["category", "forecast", "typical_monthly_spend", "margin", "recommended_budget", "basis", "history_months"])
        typical = panel.groupby("category")["spend"].mean()
        rows = []
        for f in forecasts:
            base = float(f.predicted_next_month_spend)
            scale = float(typical[f.category])
            ml = f.method == "ml_global"
            q = float(self.calibrator.margin_ratio([f.category])[0])
            margin = max(q, 0.0) * scale
            basis = ("ml_conformal_category" if self.calibrator.uses_category_margin(f.category) else "ml_conformal_pooled") if ml else "fallback_pooled_margin"
            rows.append({
                "category": f.category, "forecast": base, "typical_monthly_spend": scale, "margin": margin,
                "recommended_budget": base + margin, "basis": basis, "history_months": f.history_months,
            })
        return pd.DataFrame(rows)

    def save(self, path: Path | str = MODEL_PATH) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path | str = MODEL_PATH) -> "BudgetModel":
        return joblib.load(path)
