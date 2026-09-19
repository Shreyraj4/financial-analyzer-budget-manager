"""Transaction categorization models.

Two kinds of baseline and three ML variants share one interface
(``fit(df)`` / ``predict(df)``) so evaluation code can treat them alike.
``df`` needs ``description`` and ``amount``; ``category`` is needed to fit.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC

from app.features.text_features import build_text_vectorizer
from app.preprocessing.text import extract_merchant

UNKNOWN = "Unknown"


def _amount_features(frame: pd.DataFrame) -> np.ndarray:
    """log magnitude and direction: rent-sized debits and salary credits are
    strong hints that text alone can't give (module-level so it pickles)."""
    amount = frame["amount"].to_numpy(dtype=float)
    return np.column_stack([np.log1p(np.abs(amount)), (amount > 0).astype(float)])


def _make_pipeline(classifier, use_amount: bool) -> Pipeline:
    transformers = [("text", build_text_vectorizer(), "description")]
    if use_amount:
        transformers.append(
            ("amount", Pipeline([("f", FunctionTransformer(_amount_features)), ("s", StandardScaler())]), ["amount"])
        )
    return Pipeline([("features", ColumnTransformer(transformers)), ("clf", classifier)])


class MerchantLookupBaseline:
    """What a well-maintained rule table does: remember the most common
    category of every merchant seen in training, and give up on new ones."""

    def fit(self, df: pd.DataFrame) -> "MerchantLookupBaseline":
        merchants = df["description"].map(extract_merchant)
        self.table_ = df.groupby(merchants)["category"].agg(lambda s: s.value_counts().idxmax()).to_dict()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["description"].map(extract_merchant).map(self.table_).fillna(UNKNOWN).to_numpy()


class SubstringRulesBaseline:
    """The project's original approach: fixed merchant-substring rules,
    longest pattern first (see ingestion/categorize.py)."""

    def __init__(self, rules: list[tuple[str, str]]):
        self.rules = sorted(rules, key=lambda r: len(r[0]), reverse=True)

    def fit(self, df: pd.DataFrame) -> "SubstringRulesBaseline":
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        def match(description: str) -> str:
            merchant = extract_merchant(description)
            return next((cat for pattern, cat in self.rules if pattern in merchant), UNKNOWN)

        return df["description"].map(match).to_numpy()


class MLCategorizer(BaseEstimator, ClassifierMixin):
    """TF-IDF character n-grams (+ optional amount features) with a linear classifier."""

    def __init__(self, classifier: str = "logreg", C: float = 10.0, use_amount: bool = True):
        self.classifier = classifier
        self.C = C
        self.use_amount = use_amount

    def _build(self) -> Pipeline:
        if self.classifier == "logreg":
            clf = LogisticRegression(C=self.C, max_iter=2000)
        elif self.classifier == "linearsvc":
            clf = LinearSVC(C=self.C)
        else:
            raise ValueError(f"Unknown classifier: {self.classifier}")
        return _make_pipeline(clf, self.use_amount)

    def fit(self, df: pd.DataFrame, y=None) -> "MLCategorizer":
        self.pipeline_ = self._build().fit(df[["description", "amount"]], df["category"])
        self.classes_ = self.pipeline_.classes_
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline_.predict(df[["description", "amount"]])

    def predict_with_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Category plus the model's probability for it (LogisticRegression only:
        LinearSVC has no probabilities, which is why it is a comparison model)."""
        if self.classifier != "logreg":
            raise ValueError("Confidence scores require the logreg classifier")
        proba = self.pipeline_.predict_proba(df[["description", "amount"]])
        best = proba.argmax(axis=1)
        return pd.DataFrame(
            {"category": self.classes_[best], "confidence": proba[np.arange(len(best)), best]},
            index=df.index,
        )

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path | str) -> "MLCategorizer":
        return joblib.load(path)
