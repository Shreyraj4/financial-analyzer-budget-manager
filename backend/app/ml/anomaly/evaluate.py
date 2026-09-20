"""Evaluation for anomaly detectors on labeled data.

Anomalies are rare (~6%), so accuracy is meaningless ("always normal" scores
94%). We report:

  PR-AUC     area under the precision-recall curve; threshold-free; the headline metric
  ROC-AUC    ranking quality; optimistic when positives are rare, so secondary
  P / R / F1 at one fixed threshold (chosen on VALIDATION data, never on test)
  recall by anomaly type   which kinds of anomaly each detector actually catches
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def month_splits(dates: pd.Series, train_months: int = 12, val_months: int = 6) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Boolean masks: first ``train_months`` -> train, next ``val_months`` -> validation, rest -> test."""
    periods = dates.dt.to_period("M")
    ordered = sorted(periods.unique())
    train_end, val_end = ordered[train_months - 1], ordered[train_months + val_months - 1]
    return periods <= train_end, (periods > train_end) & (periods <= val_end), periods > val_end


def best_f1_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def evaluate(y: np.ndarray, scores: np.ndarray, threshold: float, types: np.ndarray) -> dict:
    pred = scores >= threshold
    tp = int((pred & (y == 1)).sum())
    precision = tp / max(int(pred.sum()), 1)
    recall = tp / max(int(y.sum()), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    by_type = {
        t: round(float(pred[(types == t)].mean()), 3) for t in sorted(set(types[y == 1]))
    }
    return {
        "pr_auc": round(float(average_precision_score(y, scores)), 4),
        "roc_auc": round(float(roc_auc_score(y, scores)), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "flag_rate": round(float(pred.mean()), 4),
        "base_rate": round(float(y.mean()), 4),
        "recall_by_type": by_type,
    }
