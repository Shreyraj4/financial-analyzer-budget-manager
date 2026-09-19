"""Cross-validated evaluation of categorization models.

Two scenarios, because they answer different questions:

* ``known_merchants`` (stratified k-fold over transactions): the everyday
  case - most of a user's transactions are at merchants seen before.
* ``unseen_merchants`` (GroupKFold by merchant): every test merchant is
  absent from training. This measures real generalization, not memorization,
  and is deliberately much harder.
"""
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

N_SPLITS = 5


def _folds(df: pd.DataFrame, scenario: str):
    if scenario == "known_merchants":
        return StratifiedKFold(N_SPLITS, shuffle=True, random_state=0).split(df, df["category"])
    if scenario == "unseen_merchants":
        return GroupKFold(N_SPLITS).split(df, df["category"], groups=df["merchant_key"])
    raise ValueError(scenario)


def cross_val_predict_df(model, df: pd.DataFrame, scenario: str) -> pd.Series:
    """Out-of-fold predictions: each row is predicted by a model that never saw it."""
    preds = pd.Series(index=df.index, dtype=object)
    for train_idx, test_idx in _folds(df, scenario):
        fitted = clone(model).fit(df.iloc[train_idx]) if hasattr(model, "get_params") else _refit(model, df.iloc[train_idx])
        preds.iloc[test_idx] = fitted.predict(df.iloc[test_idx])
    return preds


def _refit(model, train: pd.DataFrame):
    # The plain-Python baselines aren't sklearn estimators; rebuild a fresh copy per fold.
    fresh = type(model).__new__(type(model))
    fresh.__dict__.update({k: v for k, v in model.__dict__.items() if not k.endswith("_")})
    return fresh.fit(train)


def score(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
    }


def evaluate_models(models: dict[str, object], df: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    results: dict[str, dict[str, dict[str, float]]] = {}
    for name, model in models.items():
        results[name] = {}
        for scenario in ("known_merchants", "unseen_merchants"):
            preds = cross_val_predict_df(model, df, scenario)
            results[name][scenario] = score(df["category"], preds)
    return results


def detailed_report(y_true: pd.Series, y_pred: pd.Series) -> dict:
    labels = sorted(y_true.unique())
    return {
        "per_class": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def confidence_tradeoff(y_true: pd.Series, confidence: pd.Series, y_pred: pd.Series, thresholds=(0.3, 0.5, 0.7, 0.9)):
    """For each threshold: share of transactions auto-labeled, and accuracy on those.
    Low-confidence rows would be sent to the user for review."""
    rows = []
    for t in thresholds:
        kept = confidence >= t
        rows.append(
            {
                "threshold": t,
                "coverage": round(float(kept.mean()), 4),
                "accuracy_on_covered": round(float((y_true[kept] == y_pred[kept]).mean()), 4) if kept.any() else None,
            }
        )
    return rows
