"""The C3 phase-relabelling symmetry.

A balanced three-phase system is invariant under the cyclic group C3 acting on
the phase labels (a -> b -> c -> a).  Applying g in C3 to a snapshot permutes
(Ia, Ib, Ic) and (Va, Vb, Vc) identically, permutes the phase-participation bits
of the label the same way, and leaves the FAULT TYPE unchanged.

    fault type       is INVARIANT   under C3
    faulted phase set is EQUIVARIANT under C3

Every array here uses the column order of the public Kaggle CSV:

    [Ia, Ib, Ic, Va, Vb, Vc]
"""

from __future__ import annotations

import numpy as np

PHASES = ("a", "b", "c")
COLUMNS = ("Ia", "Ib", "Ic", "Va", "Vb", "Vc")

# Relative margin below which the two largest |i| are treated as a tie and the
# canonical phase frame is reported as unstable.  Feeds the P7 abstention path.
TIE_RTOL = 0.05


def rotate(X: np.ndarray, k: int) -> np.ndarray:
    """Relabel phases by k steps: a fault on A becomes a fault on B for k=1.

    Currents and voltages are rolled together so the two triplets stay aligned.
    """
    X = np.asarray(X, dtype=float)
    k = int(k) % 3
    if k == 0:
        return X.copy()
    single = X.ndim == 1
    X2 = np.atleast_2d(X)
    out = np.empty_like(X2)
    out[:, 0:3] = np.roll(X2[:, 0:3], k, axis=1)
    out[:, 3:6] = np.roll(X2[:, 3:6], k, axis=1)
    return out[0] if single else out


def rotate_participation(p: np.ndarray, k: int) -> np.ndarray:
    """Apply the same relabelling to phase-participation bits (pa, pb, pc)."""
    p = np.asarray(p)
    single = p.ndim == 1
    p2 = np.atleast_2d(p)
    out = np.roll(p2, int(k) % 3, axis=1)
    return out[0] if single else out


def canonicalise(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate each snapshot into a data-derived canonical phase frame.

    The frame is chosen so the phase carrying the largest instantaneous |i| lands
    in position a.  Because that choice is itself equivariant, a rotated input
    yields an identical canonical form -- which is what makes a classifier built
    on top of it correct on B- and C-anchored faults with no augmentation.

    Returns
    -------
    X_canon : the snapshots in canonical frame
    k       : the rotation applied to each row
    tie     : True where the top two |i| are within TIE_RTOL, so the frame is
              unstable.  This is not a defect to hide: for a balanced LLL fault
              the phase frame is genuinely arbitrary, and near a current zero
              crossing it is genuinely ambiguous (see C6 in PROPOSAL.md).
    """
    X = np.asarray(X, dtype=float)
    single = X.ndim == 1
    X2 = np.atleast_2d(X)

    mag = np.abs(X2[:, 0:3])
    lead = np.argmax(mag, axis=1)

    order = np.sort(mag, axis=1)
    top, second = order[:, 2], order[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(top > 0, (top - second) / top, 0.0)
    tie = rel < TIE_RTOL

    # rotate(X, k) moves old index (j - k) mod 3 to position j, so to bring the
    # leading phase to position 0 we need k = -lead (mod 3).
    k = (-lead) % 3

    out = np.empty_like(X2)
    for kk in range(3):
        m = k == kk
        if m.any():
            out[m] = rotate(X2[m], kk)

    if single:
        return out[0], k[0], tie[0]
    return out, k, tie
