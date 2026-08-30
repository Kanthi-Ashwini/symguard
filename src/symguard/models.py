"""The four model families.

All four use the same underlying classifiers, so any difference between them is
caused by the REPRESENTATION, not the estimator.

    naive      raw abc measurements.  What every public notebook trains on.
    augmented  raw abc, with the three C3 rotations applied as label-consistent
               augmentation.  This is the known "phase switching" trick from the
               literature -- the baseline to beat, NOT this project's idea.
    canonical  rotate each snapshot into a data-derived canonical phase frame,
               then classify.  Correct on unseen phase positions BY
               CONSTRUCTION, with no augmentation at all.
    invariant  use only the strictly C3-invariant Clarke features.  Phase-blind
               by construction: it cannot represent WHICH phase faulted, which
               is exactly right when the target is the (invariant) fault type.
"""

from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import features_abc, invariants
from .symmetry import canonicalise, rotate

FAMILIES = ["naive", "augmented", "canonical", "invariant"]
BASES = ["dummy", "logreg", "rf"]


def _make_base(name: str, seed: int):
    if name == "dummy":
        return DummyClassifier(strategy="most_frequent")
    if name == "logreg":
        return LogisticRegression(max_iter=2000, random_state=seed)
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=seed
        )
    raise ValueError(f"unknown base estimator: {name}")


class FaultClassifier:
    """One (family, base) combination, with a scikit-learn style interface."""

    def __init__(self, family: str = "naive", base: str = "rf", seed: int = 42):
        if family not in FAMILIES:
            raise ValueError(f"unknown family: {family}")
        if base not in BASES:
            raise ValueError(f"unknown base estimator: {base}")
        self.family = family
        self.base = base
        self.seed = seed
        self.pipeline = Pipeline(
            [("scale", StandardScaler()), ("clf", _make_base(base, seed))]
        )
        self.tie_ = None

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self.family == "invariant":
            return invariants(X).to_numpy()
        if self.family == "canonical":
            Xc, _, tie = canonicalise(X)
            self.tie_ = tie
            return features_abc(Xc).to_numpy()
        return features_abc(X).to_numpy()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FaultClassifier":
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y)
        if self.family == "augmented":
            # the fault TYPE is invariant under phase rotation, so the label is
            # simply repeated for each of the three group elements
            X = np.vstack([rotate(X, k) for k in range(3)])
            y = np.tile(y, 3)
        self.pipeline.fit(self._transform(X), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(self._transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(self._transform(X))

    @property
    def classes_(self) -> np.ndarray:
        return self.pipeline.named_steps["clf"].classes_

    def __repr__(self) -> str:
        return f"FaultClassifier(family={self.family!r}, base={self.base!r})"
