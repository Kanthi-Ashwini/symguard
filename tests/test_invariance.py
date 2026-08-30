"""THE PROOF.

This is the test the whole method rests on.  If the Clarke-derived features are
genuinely invariant under cyclic phase relabelling, then a classifier built on
them cannot possibly care which phase faulted -- and the phase-bias failure mode
documented in PROPOSAL.md 1.2 is structurally impossible for it.

If this test fails, the physics claim in the write-up is decorative and the
project's central argument does not stand.
"""

import numpy as np
import pytest

from symguard.features import INVARIANT_COLUMNS, clarke, features_clarke, invariants
from symguard.simulate import simulate_dataset
from symguard.symmetry import rotate

RTOL = 1e-10

rng = np.random.default_rng(7)
RANDOM_X = rng.normal(size=(2000, 6)) * np.array([500, 500, 500, 1, 1, 1])
PHYSICAL_X = simulate_dataset(n_scenarios=6, n_samples=16, seed=3)[
    ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]
].to_numpy()


def _max_rel_drift(X, k):
    base = invariants(X).to_numpy()
    rot = invariants(rotate(X, k)).to_numpy()
    scale = np.maximum(np.abs(base).max(axis=0), 1e-9)
    return float((np.abs(base - rot) / scale).max())


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("name,X", [("random", RANDOM_X), ("physical", PHYSICAL_X)])
def test_invariants_survive_rotation(name, X, k):
    drift = _max_rel_drift(X, k)
    assert drift < RTOL, f"{name} data, k={k}: relative drift {drift:.2e} exceeds {RTOL:.0e}"


def test_every_declared_invariant_column_is_covered():
    cols = list(invariants(RANDOM_X).columns)
    assert cols == INVARIANT_COLUMNS, "declared invariant column list is out of sync"


def test_zero_sequence_is_the_ground_discriminator():
    """LL and LLL have no ground path, so i0 stays near zero; LG and LLG do not.

    This is the physical fact that makes ground involvement separable, and it is
    why the confusion the idea document predicts between LL and LLG (verdict C8)
    does not actually materialise.
    """
    df = simulate_dataset(n_scenarios=30, n_samples=16, seed=11)
    X = df[["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]].to_numpy()
    inv = invariants(X)
    inv["t"] = df["fault_type"].to_numpy()
    ratio = inv.groupby("t")["I_zero_ratio"].mean()

    grounded = max(ratio["LG"], ratio["LLG"])
    floating = max(ratio["LL"], ratio["LLL"])
    assert floating < 0.05, f"floating faults leak zero-sequence current: {floating:.3f}"
    assert grounded > 0.15, f"ground faults show too little zero-sequence: {grounded:.3f}"
    assert grounded > 5 * floating


def test_space_vector_angle_is_equivariant_not_invariant():
    """Theta must ROTATE by 120 degrees -- it carries phase identity, so it is
    the one quantity that is deliberately not invariant."""
    base = features_clarke(RANDOM_X)["I_theta"].to_numpy()
    rot = features_clarke(rotate(RANDOM_X, 1))["I_theta"].to_numpy()
    shift = np.angle(np.exp(1j * (rot - base)))
    assert np.allclose(np.abs(shift), 2 * np.pi / 3, atol=1e-9)


def test_clarke_zero_sequence_matches_definition():
    x3 = rng.normal(size=(50, 3))
    _, _, zero = clarke(x3)
    assert np.allclose(zero, x3.sum(axis=1) / 3.0)
