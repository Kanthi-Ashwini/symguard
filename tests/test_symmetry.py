"""The C3 group action must actually be a C3 group action."""

import numpy as np
import pytest

from symguard.data import class_name, fault_type, participation
from symguard.simulate import ALL_CLASSES, KAGGLE_CLASSES
from symguard.symmetry import canonicalise, rotate, rotate_participation

rng = np.random.default_rng(0)
X = rng.normal(size=(500, 6)) * np.array([200, 200, 200, 1, 1, 1])


def test_rotation_has_order_three():
    assert np.allclose(rotate(rotate(rotate(X, 1), 1), 1), X)
    assert np.allclose(rotate(X, 3), X)
    assert np.allclose(rotate(X, 1), rotate(X, 4))


def test_rotations_are_inverse():
    assert np.allclose(rotate(rotate(X, 1), 2), X)


def test_currents_and_voltages_move_together():
    r = rotate(X, 1)
    assert np.allclose(r[:, 0:3], np.roll(X[:, 0:3], 1, axis=1))
    assert np.allclose(r[:, 3:6], np.roll(X[:, 3:6], 1, axis=1))


def test_participation_rotates_with_the_data():
    """A fault on A becomes a fault on B, and so on."""
    assert list(rotate_participation(np.array([1, 0, 0]), 1)) == [0, 1, 0]
    assert list(rotate_participation(np.array([1, 1, 0]), 1)) == [0, 1, 1]
    assert list(rotate_participation(np.array([1, 1, 1]), 1)) == [1, 1, 1]


def test_fault_type_is_invariant_under_rotation():
    """The core claim: type is invariant, phase set is equivariant."""
    for name in ALL_CLASSES:
        p, g = participation(name)
        base = fault_type(int(p.sum()), g)
        for k in (1, 2):
            pr = rotate_participation(p, k)
            assert fault_type(int(pr.sum()), g) == base


def test_rotating_kaggle_classes_reaches_the_held_out_ones():
    """Twelve physical states are reachable from the six in the data."""
    reached = set()
    for name in KAGGLE_CLASSES:
        p, g = participation(name)
        for k in range(3):
            reached.add(class_name(rotate_participation(p, k), g))
    assert reached == set(ALL_CLASSES)
    assert len(reached) == 12


def test_canonicalisation_is_rotation_invariant():
    """The whole point: a rotated input yields the same canonical form."""
    base, _, _ = canonicalise(X)
    for k in (1, 2):
        rot, _, _ = canonicalise(rotate(X, k))
        assert np.allclose(base, rot, atol=1e-9)


def test_canonicalisation_puts_largest_current_first():
    Xc, _, _ = canonicalise(X)
    assert np.all(np.abs(Xc[:, 0]) >= np.abs(Xc[:, 1]) - 1e-12)
    assert np.all(np.abs(Xc[:, 0]) >= np.abs(Xc[:, 2]) - 1e-12)
