"""Metrics, the phase-bias gap, and selective prediction.

Macro F1 is the selection metric throughout: it weights every fault class
equally, so a model cannot hide a missed class behind a common one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def report(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> str:
    return classification_report(
        y_true, y_pred, labels=labels, zero_division=0, digits=3
    )


def confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)


def phase_bias_gap(macro_f1_s2: float, macro_f1_s3: float) -> float:
    """The headline number.

    How much of a model's in-distribution score evaporates when the fault moves
    to a phase position the training data never contained.  A model that learned
    fault physics scores near zero here.  One that learned "watch phase A"
    scores close to its entire macro F1.
    """
    return macro_f1_s2 - macro_f1_s3


def risk_coverage(
    proba: np.ndarray,
    classes: np.ndarray,
    y_true: np.ndarray,
    abstain: np.ndarray | None = None,
    grid: np.ndarray | None = None,
) -> pd.DataFrame:
    """Accuracy as a function of the fraction of cases the model will answer.

    A wrong trip is expensive, so the useful question is not "how accurate is
    it" but "how accurate is it on the cases it is willing to answer".

    `abstain` marks rows the model itself flags as unreliable -- for the
    canonical family, the near-tie rows where the phase frame is unstable.
    Those are pushed to the bottom of the ranking rather than being silently
    scored as confident predictions.

    NOTE: the score used here is an uncalibrated max class probability.  It is
    not a calibrated probability and must not be reported as one.
    """
    conf = proba.max(axis=1)
    pred = np.asarray(classes)[proba.argmax(axis=1)]
    correct = (pred == np.asarray(y_true)).astype(float)

    if abstain is not None:
        # rank flagged rows below every unflagged row, keeping order within each
        conf = np.where(np.asarray(abstain, dtype=bool), conf - 10.0, conf)

    order = np.argsort(-conf)
    correct = correct[order]
    n = len(correct)
    grid = np.arange(0.1, 1.01, 0.05) if grid is None else grid

    rows = []
    for cov in grid:
        k = max(1, int(round(cov * n)))
        rows.append({"coverage": k / n, "accuracy": float(correct[:k].mean())})
    return pd.DataFrame(rows)
