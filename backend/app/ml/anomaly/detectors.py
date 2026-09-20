"""Transaction anomaly detectors.

Every detector exposes ``fit(X, users)`` and ``score(X, users)``; a higher
score means "more anomalous". ``X`` is the feature frame from
``features.transaction_features``; ``users`` is the aligned user_id Series.

  ZScoreDetector        amount vs the user's category history (simple baseline)
  RuleDetector          hand-written rules over the same features (strong baseline)
  IsolationForest       unsupervised ML: anomalies are easy to isolate with random splits
  LocalOutlierFactor    unsupervised ML: low local density relative to neighbours
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from app.config import REPO_ROOT
from app.features.transaction_features import FEATURE_COLUMNS

MODEL_PATH = REPO_ROOT / "backend" / "models" / "anomaly_detector.joblib"

Z_THRESHOLD = 3.5
BURST_SAME_DAY = 3
BURST_3D = 4
LARGE_SHARE = 0.30
RARE_CATEGORY = 0.03


class ZScoreDetector:
    """Robust z-score of the amount within (user, category). Catches only amount spikes."""

    name = "robust_zscore"
    default_threshold = Z_THRESHOLD

    def fit(self, X: pd.DataFrame, users: pd.Series | None = None) -> "ZScoreDetector":
        return self

    def score(self, X: pd.DataFrame, users: pd.Series | None = None) -> np.ndarray:
        return X["amount_robust_z"].to_numpy(dtype=float)


class RuleDetector:
    """Score = the strongest triggered rule, scaled so that >= 1.0 means 'flag'.

    Caution when reading its results: these rules were written by someone who
    knows what the synthetic anomalies look like, so they are close to an
    oracle. The ML detectors must discover the same patterns without labels.
    """

    name = "hand_written_rules"
    default_threshold = 1.0

    def fit(self, X: pd.DataFrame, users: pd.Series | None = None) -> "RuleDetector":
        return self

    def score(self, X: pd.DataFrame, users: pd.Series | None = None) -> np.ndarray:
        parts = np.column_stack(
            [
                X["amount_robust_z"].clip(lower=0) / Z_THRESHOLD,
                X["duplicate_within_1d"],
                (X["same_day_merchant_count"] >= BURST_SAME_DAY).astype(float),
                (X["merchant_count_3d"] >= BURST_3D).astype(float),
                ((X["amount_share_of_monthly_spend"] >= LARGE_SHARE) & (X["category_freq"] < RARE_CATEGORY)).astype(float),
            ]
        )
        return parts.max(axis=1)


class IsolationForestDetector:
    """Isolation Forest, globally or one model per user (with a global fallback)."""

    default_threshold = None  # chosen on validation data

    def __init__(self, per_user: bool = False, n_estimators: int = 300, seed: int = 0):
        self.per_user = per_user
        self.n_estimators = n_estimators
        self.seed = seed
        self.name = "isolation_forest_per_user" if per_user else "isolation_forest"

    def _new(self) -> IsolationForest:
        return IsolationForest(n_estimators=self.n_estimators, random_state=self.seed, n_jobs=1)

    def fit(self, X: pd.DataFrame, users: pd.Series | None = None) -> "IsolationForestDetector":
        self.global_ = self._new().fit(X[FEATURE_COLUMNS])
        self.by_user_ = {}
        if self.per_user:
            for uid, idx in X.groupby(users.to_numpy()).groups.items():
                self.by_user_[uid] = self._new().fit(X.loc[idx, FEATURE_COLUMNS])
        return self

    def score(self, X: pd.DataFrame, users: pd.Series | None = None) -> np.ndarray:
        if not self.per_user:
            return -self.global_.score_samples(X[FEATURE_COLUMNS])
        out = np.empty(len(X))
        u = users.to_numpy()
        for uid in np.unique(u):
            mask = u == uid
            model = self.by_user_.get(uid, self.global_)
            out[mask] = -model.score_samples(X.loc[mask, FEATURE_COLUMNS])
        return out


class LOFDetector:
    name = "local_outlier_factor"
    default_threshold = None

    def __init__(self, n_neighbors: int = 35):
        self.n_neighbors = n_neighbors

    def fit(self, X: pd.DataFrame, users: pd.Series | None = None) -> "LOFDetector":
        self.model_ = make_pipeline(
            RobustScaler(), LocalOutlierFactor(n_neighbors=self.n_neighbors, novelty=True)
        ).fit(X[FEATURE_COLUMNS])
        return self

    def score(self, X: pd.DataFrame, users: pd.Series | None = None) -> np.ndarray:
        return -self.model_.score_samples(X[FEATURE_COLUMNS])


class UnionDetector:
    """Flags what either the base ML detector or the amount z-score would flag.

    Complementary strengths: Isolation Forest finds unusual *combinations*
    (bursts, duplicates, rare categories) but can miss a pure amount spike
    that looks ordinary in every other feature; the z-score is exactly the
    opposite. Each score is divided by its own threshold, so 1.0 = 'flag'.
    """

    name = "iforest_plus_zscore"
    default_threshold = 1.0

    def __init__(self, base, base_threshold: float, z_threshold: float = Z_THRESHOLD):
        self.base = base
        self.base_threshold = base_threshold
        self.z_threshold = z_threshold

    def fit(self, X: pd.DataFrame, users: pd.Series | None = None) -> "UnionDetector":
        self.base.fit(X, users)
        return self

    def score(self, X: pd.DataFrame, users: pd.Series | None = None) -> np.ndarray:
        return np.maximum(self.base.score(X, users) / self.base_threshold, X["amount_robust_z"].to_numpy(dtype=float) / self.z_threshold)


class AnomalyDetector:
    """A fitted detector plus its decision threshold - the unit that is saved and served."""

    def __init__(self, detector, threshold: float):
        self.detector = detector
        self.threshold = threshold

    def flag(self, X: pd.DataFrame, users: pd.Series) -> pd.DataFrame:
        scores = self.detector.score(X, users)
        return pd.DataFrame({"score": scores, "is_anomaly": scores >= self.threshold}, index=X.index)

    def save(self, path: Path | str = MODEL_PATH) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path | str = MODEL_PATH) -> "AnomalyDetector":
        return joblib.load(path)


def default_rule_detector() -> AnomalyDetector:
    """Used when no ML model has been trained yet."""
    detector = RuleDetector()
    return AnomalyDetector(detector, detector.default_threshold)
