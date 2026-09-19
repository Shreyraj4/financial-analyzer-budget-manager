"""Next-month spend forecasting models.

All models predict spend for one (user, category, month) row from features
built out of *earlier* months only (see features/monthly_features.py).
Baselines use those features directly; the ML models are one *global* model
shared by every user and category. To make that possible, spend is divided
by the series' own historical mean, so the model learns shapes ("this month
is usually 1.3x the recent average") instead of rupee amounts.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.config import REPO_ROOT

MODEL_PATH = REPO_ROOT / "backend" / "models" / "forecaster.joblib"
RATIO_CAP = 10.0  # winsorize ratios: intermittent categories (travel, fees) spike to many x their mean

RATIO_FEATURES = ["lag_1", "lag_2", "lag_3", "roll_mean_3", "roll_std_3"]


def _scale(frame: pd.DataFrame) -> np.ndarray:
    return frame["hist_mean"].to_numpy(dtype=float)


# ------------------------------------------------------------------ baselines
def naive(frame: pd.DataFrame) -> np.ndarray:
    """Next month = last month."""
    return frame["lag_1"].to_numpy(dtype=float)


def moving_average_3(frame: pd.DataFrame) -> np.ndarray:
    return frame["roll_mean_3"].to_numpy(dtype=float)


def seasonal_naive(frame: pd.DataFrame) -> np.ndarray:
    """Next month = same month last year (falls back to naive if unavailable)."""
    return frame["lag_12"].fillna(frame["lag_1"]).to_numpy(dtype=float)


# ------------------------------------------------------------------ ML models
class GlobalForecaster:
    """Ridge or gradient-boosting regressor on history-scaled features."""

    def __init__(self, kind: str = "ridge", **params):
        if kind not in {"ridge", "gbm"}:
            raise ValueError(f"Unknown forecaster kind: {kind}")
        self.kind = kind
        self.params = params

    def _design(self, frame: pd.DataFrame) -> np.ndarray:
        s = _scale(frame)[:, None]
        ratios = np.clip(frame[RATIO_FEATURES].to_numpy(dtype=float) / s, 0.0, RATIO_CAP)
        sin = frame[["month_sin"]].to_numpy(dtype=float)
        cos = frame[["month_cos"]].to_numpy(dtype=float)
        cats = frame["category"].to_numpy()
        codes = np.array([self.categories_.get(c, -1) for c in cats])

        if self.kind == "gbm":
            return np.hstack([ratios, sin, cos, codes[:, None].astype(float)])

        # Ridge is linear, so give it category-specific seasonality explicitly
        # via one-hot(category) x (sin, cos) interaction terms.
        onehot = np.zeros((len(frame), len(self.categories_)))
        known = codes >= 0
        onehot[np.flatnonzero(known), codes[known]] = 1.0
        return np.hstack([ratios, sin, cos, onehot, onehot * sin, onehot * cos])

    def fit(self, frame: pd.DataFrame) -> "GlobalForecaster":
        # A series with no spend so far has no scale to learn ratios against.
        frame = frame[frame["hist_mean"] > 0]
        self.categories_ = {c: i for i, c in enumerate(sorted(frame["category"].unique()))}
        X = self._design(frame)
        y = np.clip(frame["spend"].to_numpy(dtype=float) / _scale(frame), 0.0, RATIO_CAP)
        if self.kind == "ridge":
            self.model_ = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(**{"alpha": 10.0, **self.params}))])
        else:
            self.model_ = HistGradientBoostingRegressor(
                categorical_features=[X.shape[1] - 1],
                **{"max_depth": 3, "learning_rate": 0.05, "max_iter": 150, "random_state": 0, **self.params},
            )
        self.model_.fit(X, y)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        scale = _scale(frame)
        has_history = scale > 0
        out = np.zeros(len(frame))
        if has_history.any():
            sub = frame[has_history]
            ratio = self.model_.predict(self._design(sub))
            out[has_history] = np.clip(ratio, 0.0, RATIO_CAP) * scale[has_history]
        return out  # no history at all -> nothing to base a forecast on -> 0

    def save(self, path: Path | str = MODEL_PATH) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path | str = MODEL_PATH) -> "GlobalForecaster":
        return joblib.load(path)
