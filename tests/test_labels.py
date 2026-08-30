"""The six G/C/B/A patterns must decode correctly, and nothing else may decode.

The original project idea lists FIVE classes and drops LLLG.  These tests pin
the six that the public source actually contains (verdict C1 in PROPOSAL.md).
"""

import numpy as np
import pytest

from symguard.data import (
    KAGGLE_LABELS,
    class_name,
    decode_flags,
    fault_type,
    participation,
    reconstruct_scenario_blocks,
)

EXPECTED = {
    (0, 0, 0, 0): ("NoFault", "NoFault"),
    (1, 0, 0, 1): ("AG", "LG"),
    (0, 0, 1, 1): ("AB", "LL"),
    (1, 0, 1, 1): ("ABG", "LLG"),
    (0, 1, 1, 1): ("ABC", "LLL"),
    (1, 1, 1, 1): ("ABCG", "LLLG"),
}


def test_exactly_six_classes():
    assert len(KAGGLE_LABELS) == 6, "the source has six classes, not the five in the idea doc"
    assert "ABCG" in KAGGLE_LABELS.values(), "LLLG (ABCG) must not be dropped"


@pytest.mark.parametrize("flags,expected", EXPECTED.items())
def test_decode_and_type(flags, expected):
    name, ftype = expected
    assert decode_flags(*flags) == name
    p, g = participation(name)
    assert fault_type(int(p.sum()), g) == ftype
    assert class_name(p, g) == name


def test_every_kaggle_class_involves_phase_a():
    """The finding this whole project rests on (PROPOSAL.md 1.2)."""
    for name in KAGGLE_LABELS.values():
        if name == "NoFault":
            continue
        p, _ = participation(name)
        assert p[0] == 1, f"{name} does not involve phase A -- assumption broken"


@pytest.mark.parametrize("flags", [(1, 0, 1, 0), (0, 1, 0, 1), (1, 1, 0, 0), (0, 0, 0, 1)])
def test_unknown_pattern_raises(flags):
    """An unexpected combination must fail loudly rather than be guessed."""
    with pytest.raises(ValueError):
        decode_flags(*flags)


def test_single_phase_without_ground_rejected():
    with pytest.raises(ValueError):
        fault_type(1, 0)


def test_scenario_blocks_from_runs():
    labels = ["AG", "AG", "AG", "AB", "AB", "AG"]
    assert list(reconstruct_scenario_blocks(labels)) == [0, 0, 0, 1, 1, 2]
