"""Feature bases computed from a single instantaneous snapshot.

The original project idea (project_Idea.pdf, section 7) proposes mean / RMS /
standard deviation / peak / peak-to-peak features.  Those are undefined here:
one row is one instantaneous six-sensor reading, not a waveform segment, and
averaging across the six columns would mix kV with A.  See C3 in PROPOSAL.md.

What IS computable from one row is the Clarke (alpha-beta-zero) transform, which
is an instantaneous algebraic identity requiring no phasor estimate.  Cyclic
phase relabelling is a 120-degree rotation about the (1,1,1) axis, so it
preserves the zero-sequence component and the magnitude of the alpha-beta space
vector while rotating its angle.  That gives an exactly C3-invariant feature set
and one equivariant angle -- the split this whole project is built on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12

ABC_COLUMNS = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]

INVARIANT_COLUMNS = [
    "I_mag", "V_mag", "I_zero", "V_zero",
    "I_zero_ratio", "V_zero_ratio", "IV_ratio",
    "I_absum", "V_absum", "I_spread", "V_spread",
]


def clarke(x3: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Instantaneous Clarke transform of a three-phase triplet.

    Returns (alpha, beta, zero) for input shaped (n, 3).
    """
    x3 = np.atleast_2d(np.asarray(x3, dtype=float))
    a, b, c = x3[:, 0], x3[:, 1], x3[:, 2]
    alpha = (2.0 * a - b - c) / 3.0
    beta = (b - c) / np.sqrt(3.0)
    zero = (a + b + c) / 3.0
    return alpha, beta, zero


def invariants(X: np.ndarray) -> pd.DataFrame:
    """The strictly C3-invariant features.  Every column here must survive
    `rotate()` unchanged -- that is what tests/test_invariance.py asserts.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    ia_, ib_, i0 = clarke(X[:, 0:3])
    va_, vb_, v0 = clarke(X[:, 3:6])

    i_mag = np.hypot(ia_, ib_)
    v_mag = np.hypot(va_, vb_)
    cur, vol = np.abs(X[:, 0:3]), np.abs(X[:, 3:6])

    return pd.DataFrame({
        "I_mag": i_mag,
        "V_mag": v_mag,
        "I_zero": np.abs(i0),
        "V_zero": np.abs(v0),
        # ground-return indicators: near zero for LL and LLL, non-zero when a
        # ground path exists.  This is the physical LL-vs-LLG discriminator.
        "I_zero_ratio": np.abs(i0) / (i_mag + EPS),
        "V_zero_ratio": np.abs(v0) / (v_mag + EPS),
        # unbalance / severity proxy
        "IV_ratio": i_mag / (v_mag + EPS),
        # symmetric functions of the per-phase magnitudes
        "I_absum": cur.sum(axis=1),
        "V_absum": vol.sum(axis=1),
        "I_spread": cur.max(axis=1) - cur.min(axis=1),
        "V_spread": vol.max(axis=1) - vol.min(axis=1),
    })


def features_abc(X: np.ndarray) -> pd.DataFrame:
    """The raw six measurements, exactly as the public notebooks use them."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return pd.DataFrame(X, columns=ABC_COLUMNS)


def features_clarke(X: np.ndarray) -> pd.DataFrame:
    """Invariants plus the equivariant space-vector angles.

    The angles rotate by 120 degrees under phase relabelling, so they carry
    phase identity.  They are only safe to use inside a canonical frame.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    ia_, ib_, _ = clarke(X[:, 0:3])
    va_, vb_, _ = clarke(X[:, 3:6])
    out = invariants(X)
    out["I_theta"] = np.arctan2(ib_, ia_)
    out["V_theta"] = np.arctan2(vb_, va_)
    return out
