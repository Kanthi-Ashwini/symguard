"""A physics-shaped three-phase fault generator.

WHY THIS EXISTS.  It is not a substitute for the public Kaggle data.  It is
required apparatus: the Kaggle source contains only phase-A-anchored faults
(AG, AB, ABG, ABC, ABCG), so it can NEVER supply the B- and C-anchored faults
that the symmetry-held-out evaluation (S3) needs.  See PROPOSAL.md section 1.2.

HONESTY RULE.  Numbers produced from this generator demonstrate that the
apparatus works.  They are NOT results about the Kaggle data and must never be
quoted as such.

MODEL.  A simplified single-source lumped model solved with phasors: a balanced
source behind a source impedance, a line, a fault at fractional distance `loc`
through fault impedance Z_f, and a load.  It is deliberately NOT a full
sequence-network solution -- it reproduces the qualitative signatures that
matter here (ground-return current, voltage sag, fault-current magnitude) and
nothing more.

Every scenario emits a contiguous run of samples with a shared scenario_id, so
the leakage demonstration (S1 vs S2) has real groups to work with.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import class_name, fault_type, participation

F_NOM = 50.0
SAMPLES_PER_CYCLE = 128
I_BASE = 100.0          # A per pu, so fault currents land in the hundreds
V_BASE = 0.6            # pu peak, matching the scale of the public CSV

# The six classes the public Kaggle CSV actually contains.  All are anchored on
# phase A -- that is the whole finding.
KAGGLE_CLASSES = ["NoFault", "AG", "AB", "ABG", "ABC", "ABCG"]

# The full set of physically distinct shunt-fault states.  The six above plus
# their non-trivial C3 rotations.  NoFault, ABC and ABCG are rotation-invariant.
ALL_CLASSES = [
    "NoFault",
    "AG", "BG", "CG",
    "AB", "BC", "AC",
    "ABG", "BCG", "ACG",
    "ABC", "ABCG",
]

HELD_OUT_CLASSES = [c for c in ALL_CLASSES if c not in KAGGLE_CLASSES]


def _source_phasors(v_mag: float, phi: float) -> np.ndarray:
    """Balanced positive-sequence source, phases a/b/c at 0/-120/+120 degrees."""
    k = np.arange(3)
    return v_mag * np.exp(1j * (phi - 2.0 * np.pi * k / 3.0))


def simulate_scenario(
    name: str,
    rng: np.random.Generator,
    n_samples: int = 64,
) -> tuple[np.ndarray, dict]:
    """Generate one fault scenario as a contiguous run of `n_samples` snapshots.

    Returns (X, meta) where X has columns [Ia, Ib, Ic, Va, Vb, Vc].
    """
    p, ground = participation(name)
    faulted = np.flatnonzero(p)

    # --- randomised operating point -------------------------------------
    phi = rng.uniform(0.0, 2.0 * np.pi)          # fault inception angle
    loc = rng.uniform(0.1, 0.9)                  # fractional distance to fault
    r_f = rng.uniform(0.005, 0.25)               # fault resistance, pu
    z_src = complex(0.01, rng.uniform(0.05, 0.12))
    z_line = complex(0.02, rng.uniform(0.20, 0.40))
    z_load = complex(rng.uniform(1.6, 3.2), rng.uniform(0.5, 1.1))
    noise = rng.uniform(0.002, 0.012)

    v_src = _source_phasors(V_BASE, phi)

    # --- load current always flows --------------------------------------
    i_ph = v_src / (z_src + z_line + z_load)

    # --- superimpose the fault ------------------------------------------
    z_to_fault = z_src + loc * z_line

    if len(faulted) and ground:
        # each faulted phase has its own path to ground -> non-zero i0.
        # a small per-phase spread in Z_f keeps a three-phase-to-ground fault
        # from being perfectly balanced, which is what makes LLL vs LLLG hard
        # rather than trivially separable.
        for k in faulted:
            zf = complex(r_f * rng.uniform(0.85, 1.15), 0.0)
            i_ph[k] += v_src[k] / (z_to_fault + zf)

    elif len(faulted) == 2:
        # phase-to-phase: the current circulates between the two phases and
        # sums to zero, so i0 stays ~0.  This is the physical LL/LLG separator.
        j, k = faulted
        i_circ = (v_src[j] - v_src[k]) / (2.0 * z_to_fault + complex(r_f, 0.0))
        i_ph[j] += i_circ
        i_ph[k] -= i_circ

    elif len(faulted) == 3:
        # balanced three-phase: the three fault currents sum to zero -> i0 ~ 0.
        for k in faulted:
            i_ph[k] += v_src[k] / (z_to_fault + complex(r_f, 0.0))

    # --- measured voltage at the relay = source minus drop on Z_src ------
    v_meas = v_src - i_ph * z_src

    # --- phasors to waveform --------------------------------------------
    t = np.arange(n_samples) / (F_NOM * SAMPLES_PER_CYCLE)
    rot = np.exp(1j * 2.0 * np.pi * F_NOM * t)[:, None]
    currents = np.real(i_ph[None, :] * rot) * I_BASE
    voltages = np.real(v_meas[None, :] * rot)

    X = np.hstack([currents, voltages])
    X += rng.normal(0.0, noise, size=X.shape) * np.array(
        [I_BASE, I_BASE, I_BASE, 1.0, 1.0, 1.0]
    )

    meta = {
        "fault_class": name,
        "fault_type": fault_type(int(p.sum()), ground),
        "ground": ground,
        "n_phases": int(p.sum()),
        "loc": loc,
        "r_f": r_f,
        "phi": phi,
    }
    return X, meta


def simulate_dataset(
    classes: list[str] | None = None,
    n_scenarios: int = 60,
    n_samples: int = 64,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a dataset with the same schema `data.load_raw_csv` produces."""
    classes = list(ALL_CLASSES if classes is None else classes)
    rng = np.random.default_rng(seed)

    rows, meta_rows, ids = [], [], []
    sid = 0
    for name in classes:
        for _ in range(n_scenarios):
            X, meta = simulate_scenario(name, rng, n_samples=n_samples)
            rows.append(X)
            meta_rows.extend([meta] * len(X))
            ids.extend([sid] * len(X))
            sid += 1

    df = pd.DataFrame(np.vstack(rows), columns=["Ia", "Ib", "Ic", "Va", "Vb", "Vc"])
    meta_df = pd.DataFrame(meta_rows).reset_index(drop=True)
    for col in ["fault_class", "fault_type", "ground", "n_phases", "loc", "r_f", "phi"]:
        df[col] = meta_df[col]
    df["scenario_id"] = ids
    return df
