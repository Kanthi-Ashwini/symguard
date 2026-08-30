"""The four evaluation splits.

The point of this project is not another model leaderboard.  It is that the
number you report depends almost entirely on which split you chose, and the
split every public notebook uses is the most flattering one available.

    S1  random row split   - what the public notebooks do.  Reported ONLY to
                             show it is inflated: adjacent samples from one
                             simulated event land on both sides.
    S2  grouped split      - GroupShuffleSplit on recovered scenario blocks.
                             The honest in-distribution number.
    S3  symmetry held out  - train on phase-A-anchored faults (all the public
                             data contains), test on B- and C-anchored ones.
                             The headline: macroF1(S2) - macroF1(S3) is the
                             phase-bias gap.
    S4  cross-dataset      - SPECIFIED BUT NOT RUN.  See PROPOSAL.md 5.4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .simulate import HELD_OUT_CLASSES, KAGGLE_CLASSES

# Classes unchanged by cyclic phase rotation.  They appear on both sides of S3,
# split by scenario so no event is ever shared.
ROTATION_INVARIANT = ["NoFault", "ABC", "ABCG"]


def kaggle_pool(df: pd.DataFrame) -> np.ndarray:
    """Row mask for the six classes the public CSV actually contains."""
    return df["fault_class"].isin(KAGGLE_CLASSES).to_numpy()


def s1_random(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Random row split -- deliberately leaky, for comparison only."""
    idx = np.flatnonzero(kaggle_pool(df))
    return train_test_split(
        idx,
        test_size=test_size,
        random_state=seed,
        stratify=df["fault_type"].to_numpy()[idx],
    )


def s2_grouped(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Grouped split: no scenario block spans the train/test boundary."""
    idx = np.flatnonzero(kaggle_pool(df))
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(splitter.split(idx, groups=df["scenario_id"].to_numpy()[idx]))
    return idx[tr], idx[te]


def s3_symmetry_holdout(df: pd.DataFrame, seed: int = 42):
    """Train on what the public data has; test on phase positions it never shows.

    Training : NoFault / AG / AB / ABG / ABC / ABCG   (all phase-A-anchored)
    Testing  : BG / CG / BC / AC / BCG / ACG          (never seen)
               plus held-out SCENARIOS of the rotation-invariant classes, so
               every fault type is represented on both sides.

    A model that learned fault physics transfers.  One that learned "watch phase
    A" collapses.  Nothing in the public benchmark can tell those two apart.
    """
    rng = np.random.default_rng(seed)
    cls = df["fault_class"].to_numpy()
    sid = df["scenario_id"].to_numpy()

    train_mask = np.isin(cls, KAGGLE_CLASSES)
    test_mask = np.isin(cls, HELD_OUT_CLASSES)

    for name in ROTATION_INVARIANT:
        scen = np.unique(sid[cls == name])
        held = rng.permutation(scen)[: len(scen) // 2]
        moved = (cls == name) & np.isin(sid, held)
        train_mask &= ~moved
        test_mask |= moved

    return np.flatnonzero(train_mask), np.flatnonzero(test_mask)


def assert_no_group_overlap(df: pd.DataFrame, train_idx, test_idx) -> None:
    """Guard used by tests/test_no_leakage.py."""
    sid = df["scenario_id"].to_numpy()
    shared = set(sid[train_idx]) & set(sid[test_idx])
    if shared:
        raise AssertionError(
            f"{len(shared)} scenario block(s) span the split: {sorted(shared)[:5]}"
        )
