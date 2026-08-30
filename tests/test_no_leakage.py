"""No scenario block may span a train/test boundary in S2 or S3.

The earlier review of this project conceded that event-level leakage "cannot be
ruled out" because the public CSV ships no event identifier.  It can be: the
label is piecewise constant, so contiguous runs recover the scenarios.  These
tests hold that guarantee in place (verdict C5).
"""

import numpy as np
import pytest

from symguard.simulate import HELD_OUT_CLASSES, KAGGLE_CLASSES, simulate_dataset
from symguard.splits import (
    assert_no_group_overlap,
    s1_random,
    s2_grouped,
    s3_symmetry_holdout,
)

DF = simulate_dataset(n_scenarios=12, n_samples=24, seed=5)


@pytest.mark.parametrize("split_fn", [s2_grouped, s3_symmetry_holdout])
def test_grouped_splits_share_no_scenario(split_fn):
    tr, te = split_fn(DF)
    assert_no_group_overlap(DF, tr, te)


def test_random_split_does_leak():
    """Documented, not fixed: S1 exists precisely to show the inflation."""
    tr, te = s1_random(DF)
    sid = DF["scenario_id"].to_numpy()
    assert set(sid[tr]) & set(sid[te]), "S1 is supposed to leak; if it does not, it is not S1"


def test_s3_trains_only_on_phase_a_anchored_faults():
    tr, te = s3_symmetry_holdout(DF)
    cls = DF["fault_class"].to_numpy()
    assert set(cls[tr]) <= set(KAGGLE_CLASSES)
    assert set(HELD_OUT_CLASSES) <= set(cls[te]), "S3 must test the unseen phase positions"


def test_s3_covers_every_fault_type_on_both_sides():
    tr, te = s3_symmetry_holdout(DF)
    typ = DF["fault_type"].to_numpy()
    assert set(typ[tr]) == set(typ[te]), "macro F1 is only comparable if both sides share classes"


def test_splits_are_deterministic():
    a = s3_symmetry_holdout(DF, seed=1)
    b = s3_symmetry_holdout(DF, seed=1)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


# --- scenario-block recovery: the two real limits ----------------------------

def test_blocks_merge_consecutive_same_label_scenarios():
    """Over-merging is SAFE, and worth pinning so nobody 'fixes' it into a leak.

    Two distinct scenarios that happen to be adjacent and share a label collapse
    into one block.  Both then land on the same side of any grouped split, which
    is stricter than necessary -- never leakier.
    """
    from symguard.data import reconstruct_scenario_blocks

    labels = ["AG"] * 4 + ["AB"] * 4          # 2 labels, but could be 4 scenarios
    blocks = reconstruct_scenario_blocks(labels)
    assert len(set(blocks)) == 2, "adjacent same-label runs merge; that is expected"


def test_block_health_flags_degenerate_recovery():
    """The dangerous case: alternating labels give one block per row, and the
    grouped split silently becomes a random split."""
    from symguard.data import block_health, reconstruct_scenario_blocks

    alternating = ["AG", "AB"] * 50
    health = block_health(reconstruct_scenario_blocks(alternating))
    assert health["degenerate"] is True
    assert health["singleton_fraction"] == 1.0

    healthy = ["AG"] * 50 + ["AB"] * 50
    assert block_health(reconstruct_scenario_blocks(healthy))["degenerate"] is False
