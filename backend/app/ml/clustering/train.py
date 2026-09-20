"""Choose the pooling window and k, compare clustering methods, save the production profiler.

Choices are made WITHOUT the persona labels (silhouette only); personas are used
afterwards purely to grade the result, which is only possible because the data
is synthetic.

Usage (from backend/): python -m app.ml.clustering.train
"""
import json

import pandas as pd

from app.config import REPO_ROOT
from app.features.monthly_features import monthly_category_panel, monthly_share_vectors
from app.ml.clustering.evaluate import (
    algorithm_labels,
    feature_matrix,
    internal_and_external,
    kmeans_stability,
    leave_one_user_out,
    spend_level_baseline,
)
from app.ml.clustering.model import SpendingProfiler
from app.ml.dataset import load_labeled_transactions

METRICS_PATH = REPO_ROOT / "docs" / "metrics" / "clustering.json"
WINDOWS = (1, 3, 6, 12)
MIN_K = 3  # fewer than 3 profiles would only separate "low" from "high" spenders


def vectors_and_truth(panel: pd.DataFrame, persona_of_user: pd.Series, window: int):
    vectors = monthly_share_vectors(panel, window=window).reset_index(drop=True)
    personas = vectors["user_id"].map(persona_of_user)
    return vectors, personas, personas.astype("category").cat.codes.to_numpy()


def main() -> None:
    df = load_labeled_transactions()
    panel = monthly_category_panel(df)
    persona_of_user = df.groupby("user_id")["persona"].first()
    n_users = int(persona_of_user.size)
    # With few users, more clusters just means "one cluster per user": require >= 2 users per cluster on average.
    k_max = n_users // 2
    k_range = range(2, k_max + 1)
    print(f"users={n_users} personas={persona_of_user.nunique()} k range=2..{k_max}")

    # 1. Choose the pooling window (label-free): best achievable silhouette.
    window_table = {}
    print(f"\n{'window':>6s} {'points':>6s} {'best sil':>9s} {'at k':>5s}   (ARI at that k, for information)")
    for w in WINDOWS:
        v, _, truth = vectors_and_truth(panel, persona_of_user, w)
        X = feature_matrix(v)
        scored = {k: internal_and_external(X, algorithm_labels("kmeans", X, k), truth) for k in range(MIN_K, k_max + 1)}
        k_best = max(scored, key=lambda k: scored[k]["silhouette"])
        window_table[w] = {"points": len(v), "best_silhouette": scored[k_best]["silhouette"], "k": k_best, "ari": scored[k_best]["ari_vs_personas"]}
        print(f"{w:6d} {len(v):6d} {scored[k_best]['silhouette']:9.3f} {k_best:5d}   ARI={scored[k_best]['ari_vs_personas']:.3f}")
    window = max(window_table, key=lambda w: window_table[w]["best_silhouette"])
    print(f"\nChosen window: {window} months")

    vectors, personas, truth = vectors_and_truth(panel, persona_of_user, window)
    X = feature_matrix(vectors)

    # 2. Choose k (label-free) at that window.
    by_k = {}
    print(f"\n{'k':>2s} {'silhouette':>10s} {'DB':>6s} {'ARI':>6s} {'NMI':>6s} {'stability':>9s}")
    for k in k_range:
        m = internal_and_external(X, algorithm_labels("kmeans", X, k), truth)
        m["stability"] = kmeans_stability(X, k)
        by_k[k] = m
        print(f"{k:2d} {m['silhouette']:10.3f} {m['davies_bouldin']:6.2f} {m['ari_vs_personas']:6.3f} {m['nmi_vs_personas']:6.3f} {m['stability']:9.3f}")
    chosen_k = max((k for k in k_range if k >= MIN_K), key=lambda k: by_k[k]["silhouette"])
    print(f"\nChosen k (best silhouette, {MIN_K} <= k <= {k_max}): {chosen_k}")

    # 3. Algorithms and a naive baseline at the chosen k.
    comparison = {name: internal_and_external(X, algorithm_labels(name, X, chosen_k), truth)
                  for name in ("kmeans", "ward_hierarchical", "gaussian_mixture")}
    comparison["baseline_spend_level_bins"] = internal_and_external(X, spend_level_baseline(vectors, chosen_k), truth)
    print(f"\nAlgorithms at k={chosen_k}:")
    for name, m in comparison.items():
        print(f"  {name:28s} silhouette={m['silhouette']:6.3f} ARI={m['ari_vs_personas']:6.3f} NMI={m['nmi_vs_personas']:6.3f}")

    # 4. The fair test: users the model never saw.
    louo = leave_one_user_out(vectors, personas, chosen_k)
    print("\nLeave-one-user-out persona recovery:", json.dumps(louo))

    # 5. Production model.
    profiler = SpendingProfiler(k=chosen_k).fit(vectors)
    profiler.save()
    crosstab = pd.crosstab(pd.Series(profiler.predict(vectors), name="cluster"), personas.rename("persona"))
    print("\nProfiles:")
    for d in profiler.describe():
        print(f"  [{d['cluster']}] {d['name']}: ~Rs {d['typical_monthly_spend']:,.0f}/month; top {[t['category'] for t in d['top_categories']]}")
    print("\nCluster x persona counts:\n", crosstab.to_string())

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps({
        "n_users": n_users, "window_months": window, "windows_tried": {str(w): v for w, v in window_table.items()},
        "n_points": int(len(vectors)), "k_range": [2, k_max], "by_k": {str(k): v for k, v in by_k.items()},
        "chosen_k": chosen_k, "algorithms_at_chosen_k": comparison, "leave_one_user_out": louo,
        "profiles": profiler.describe(), "cluster_vs_persona_counts": crosstab.to_dict(),
    }, indent=2))
    print(f"\nSaved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
