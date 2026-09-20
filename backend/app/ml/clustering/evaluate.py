"""Evaluating clusterings.

Internal metrics (need no labels; what you would have on real data):
  silhouette      how much closer a point is to its own cluster than to the next one (-1..1, higher better)
  Davies-Bouldin  average cluster similarity, lower is better
External metrics (compare with the true personas, available only because the data is synthetic):
  ARI   adjusted Rand index: agreement with true groups, 0 = random, 1 = identical
  NMI   normalized mutual information
Stability: does re-running with a different random seed give the same groups (mean pairwise ARI)?
"""
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from app.ml.clustering.model import SpendingProfiler, TOTAL_COL


def feature_matrix(vectors: pd.DataFrame) -> np.ndarray:
    cols = [c for c in vectors.columns if c not in {"user_id", "month"}]
    return StandardScaler().fit_transform(vectors[cols])


def internal_and_external(X: np.ndarray, labels: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    return {
        "silhouette": round(float(silhouette_score(X, labels)), 4),
        "davies_bouldin": round(float(davies_bouldin_score(X, labels)), 4),
        "ari_vs_personas": round(float(adjusted_rand_score(truth, labels)), 4),
        "nmi_vs_personas": round(float(normalized_mutual_info_score(truth, labels)), 4),
    }


def kmeans_stability(X: np.ndarray, k: int, seeds=range(10)) -> float:
    runs = [KMeans(n_clusters=k, n_init=1, random_state=s).fit_predict(X) for s in seeds]
    pairs = [adjusted_rand_score(runs[i], runs[j]) for i in range(len(runs)) for j in range(i + 1, len(runs))]
    return round(float(np.mean(pairs)), 4)


def spend_level_baseline(vectors: pd.DataFrame, k: int) -> np.ndarray:
    """Naive baseline: split months into k equal-size bins by total spend alone."""
    return pd.qcut(vectors[TOTAL_COL].rank(method="first"), k, labels=False).to_numpy()


def algorithm_labels(name: str, X: np.ndarray, k: int) -> np.ndarray:
    if name == "kmeans":
        return KMeans(n_clusters=k, n_init=20, random_state=0).fit_predict(X)
    if name == "ward_hierarchical":
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    if name == "gaussian_mixture":
        return GaussianMixture(n_components=k, random_state=0, n_init=5).fit(X).predict(X)
    raise ValueError(name)


def leave_one_user_out(vectors: pd.DataFrame, personas: pd.Series, k: int) -> dict[str, float]:
    """Held-out generalization: for each user, fit on all OTHER users, map each cluster to the
    persona most common among its training months, then check whether the held-out user's
    months land in clusters that map to their true persona."""
    month_ok, user_ok, users = [], [], vectors["user_id"].unique()
    for uid in users:
        train, test = vectors[vectors["user_id"] != uid], vectors[vectors["user_id"] == uid]
        model = SpendingProfiler(k=k).fit(train)
        train_clusters = pd.Series(model.predict(train), index=train.index)
        cluster_to_persona = personas.loc[train.index].groupby(train_clusters).agg(lambda s: s.value_counts().idxmax())
        predicted = pd.Series(model.predict(test)).map(cluster_to_persona)
        true = personas.loc[test.index].iloc[0]
        month_ok.extend((predicted == true).tolist())
        user_ok.append(predicted.value_counts().idxmax() == true)
    return {
        "month_level_persona_accuracy": round(float(np.mean(month_ok)), 4),
        "user_level_persona_accuracy": round(float(np.mean(user_ok)), 4),
        "n_users": int(len(users)),
    }
