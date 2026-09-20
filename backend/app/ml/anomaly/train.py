"""Compare anomaly detectors, then train and save the production one.

Protocol (same spirit as forecasting):
  * chronological split: months 1-12 train, 13-18 validation, 19-24 test
  * detectors are FIT without labels (unsupervised) on the training months
  * each detector's decision threshold is chosen on validation labels
  * numbers reported on the untouched test months
  * production model = best validation PR-AUC among the ML detectors, refit on train+validation

Only debits are scored; salary/income credits are not spend anomalies.

Usage (from backend/): python -m app.ml.anomaly.train
"""
import json

from app.config import REPO_ROOT
from app.features.transaction_features import build_transaction_features
from app.ml.anomaly.detectors import (
    AnomalyDetector,
    IsolationForestDetector,
    LOFDetector,
    RuleDetector,
    UnionDetector,
    ZScoreDetector,
)
from app.ml.anomaly.evaluate import best_f1_threshold, evaluate, month_splits
from app.ml.dataset import load_labeled_transactions

METRICS_PATH = REPO_ROOT / "docs" / "metrics" / "anomaly_detection.json"


def main() -> None:
    df = load_labeled_transactions()
    feats = build_transaction_features(df)
    debit = feats["is_credit"] == 0
    df, feats = df[debit], feats[debit]

    train_m, val_m, test_m = month_splits(df["date"])
    users = df["user_id"]
    y = df["is_anomaly"].to_numpy()
    types = df["anomaly_type"].fillna("").to_numpy()

    detectors = [ZScoreDetector(), RuleDetector(), IsolationForestDetector(), IsolationForestDetector(per_user=True), LOFDetector()]
    print(f"debit rows: train={train_m.sum()} val={val_m.sum()} test={test_m.sum()} | anomaly rate test={y[test_m.to_numpy()].mean():.3f}")

    results, fitted = {}, {}
    for det in detectors:
        det.fit(feats[train_m], users[train_m])
        val_scores = det.score(feats[val_m], users[val_m])
        test_scores = det.score(feats[test_m], users[test_m])
        threshold = det.default_threshold if det.default_threshold is not None else best_f1_threshold(y[val_m.to_numpy()], val_scores)
        results[det.name] = {
            "threshold": round(float(threshold), 4),
            "validation": evaluate(y[val_m.to_numpy()], val_scores, threshold, types[val_m.to_numpy()]),
            "test": evaluate(y[test_m.to_numpy()], test_scores, threshold, types[test_m.to_numpy()]),
        }
        fitted[det.name] = (det, threshold)

    # Union of the winning Isolation Forest and the z-score (its threshold comes from validation).
    base, base_thr = fitted["isolation_forest"]
    union = UnionDetector(base, base_thr)
    val_scores, test_scores = union.score(feats[val_m], users[val_m]), union.score(feats[test_m], users[test_m])
    results[union.name] = {
        "threshold": 1.0,
        "validation": evaluate(y[val_m.to_numpy()], val_scores, 1.0, types[val_m.to_numpy()]),
        "test": evaluate(y[test_m.to_numpy()], test_scores, 1.0, types[test_m.to_numpy()]),
    }
    fitted[union.name] = (union, 1.0)

    kinds = ["amount_spike", "duplicate_charge", "burst", "unusual_high_value"]
    print(f"\n{'detector (TEST)':28s} {'PR-AUC':>7s} {'ROC-AUC':>8s} {'prec':>6s} {'recall':>7s} {'F1':>6s}   recall by type: " + " ".join(k[:9] for k in kinds))
    for name, r in results.items():
        t = r["test"]
        by = " ".join(f"{t['recall_by_type'].get(k, float('nan')):9.2f}" for k in kinds)
        print(f"{name:28s} {t['pr_auc']:7.3f} {t['roc_auc']:8.3f} {t['precision']:6.3f} {t['recall']:7.3f} {t['f1']:6.3f}   {by}")

    ml_names = [n for n in results if n.startswith(("isolation", "local", "iforest"))]
    winner = max(ml_names, key=lambda n: results[n]["validation"]["pr_auc"])
    print(f"\nProduction model (best validation PR-AUC among ML): {winner}")

    det, threshold = fitted[winner]
    train_val = (train_m | val_m).to_numpy()
    if isinstance(det, UnionDetector):
        final = UnionDetector(IsolationForestDetector(), det.base_threshold)
    else:
        final = type(det)(**({"per_user": True} if getattr(det, "per_user", False) else {}))
    final.fit(feats[train_val], users[train_val])
    AnomalyDetector(final, threshold).save()

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps({
        "protocol": "unsupervised fit on months 1-12; threshold chosen on months 13-18; reported on months 19-24; debits only",
        "results": results, "production_model": winner,
    }, indent=2))
    print(f"Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
