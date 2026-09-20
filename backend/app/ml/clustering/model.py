"""Spending-behaviour profiles via KMeans.

One point = one (user, month): the share of that month's spend going to each
category, plus log total spend. Shares make people comparable regardless of
income; log total keeps overall spending level as its own signal.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from app.config import REPO_ROOT

MODEL_PATH = REPO_ROOT / "backend" / "models" / "spending_profiler.joblib"
TOTAL_COL = "log_total_spend"
LEVELS = {0: "Lower-spend", 1: "Mid-spend", 2: "Higher-spend"}  # relative to the other profiles


class SpendingProfiler:
    def __init__(self, k: int = 4, seed: int = 0):
        self.k = k
        self.seed = seed

    def _matrix(self, vectors: pd.DataFrame) -> pd.DataFrame:
        """Aligns to the training categories: unseen categories are dropped,
        categories the user never used become 0."""
        return vectors.reindex(columns=self.categories_ + [TOTAL_COL], fill_value=0.0)

    def fit(self, vectors: pd.DataFrame) -> "SpendingProfiler":
        self.categories_ = sorted(c for c in vectors.columns if c not in {"user_id", "month", TOTAL_COL})
        X = self._matrix(vectors)
        self.scaler_ = StandardScaler().fit(X)
        self.kmeans_ = KMeans(n_clusters=self.k, n_init=20, random_state=self.seed).fit(self.scaler_.transform(X))
        self.centroids_ = pd.DataFrame(self.scaler_.inverse_transform(self.kmeans_.cluster_centers_), columns=X.columns)
        self._name_clusters()
        return self

    def predict(self, vectors: pd.DataFrame) -> np.ndarray:
        return self.kmeans_.predict(self.scaler_.transform(self._matrix(vectors)))

    def _name_clusters(self) -> None:
        """Human-readable names from the centroids alone (deterministic, no LLM):
        spend level by rank, plus the categories that stand out versus the average cluster."""
        totals = self.centroids_[TOTAL_COL].rank(method="first") - 1
        bins = np.floor(totals / self.k * 3).astype(int).clip(0, 2)
        mean_share = self.centroids_[self.categories_].mean()
        self.names_ = {}
        for c in range(self.k):
            lift = (self.centroids_.loc[c, self.categories_] - mean_share).sort_values(ascending=False)
            top = [cat for cat, v in lift.items() if v > 0.03][:2]
            emphasis = " / ".join(top) + "-heavy" if top else "balanced"
            self.names_[c] = f"{LEVELS[int(bins[c])]}, {emphasis}"

    def assign(self, vectors: pd.DataFrame) -> pd.DataFrame:
        """Cluster plus a 0-1 margin: how much closer the point is to its cluster
        than to the runner-up (near 0 = on the border between two profiles)."""
        d = self.kmeans_.transform(self.scaler_.transform(self._matrix(vectors)))
        order = np.sort(d, axis=1)
        margin = (order[:, 1] - order[:, 0]) / np.maximum(order[:, 1], 1e-9)
        return pd.DataFrame({"cluster": d.argmin(axis=1), "margin": margin}, index=vectors.index)

    def describe(self) -> list[dict]:
        out = []
        for c in range(self.k):
            shares = self.centroids_.loc[c, self.categories_].sort_values(ascending=False)
            out.append({
                "cluster": c,
                "name": self.names_[c],
                "typical_monthly_spend": round(float(np.expm1(self.centroids_.loc[c, TOTAL_COL])), 2),
                "top_categories": [{"category": k, "share": round(float(v), 3)} for k, v in shares.head(3).items()],
            })
        return out

    def save(self, path: Path | str = MODEL_PATH) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path | str = MODEL_PATH) -> "SpendingProfiler":
        return joblib.load(path)
