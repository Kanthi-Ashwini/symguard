"""Label decoding, fault taxonomy, and scenario-block recovery.

The public Kaggle CSV encodes the target as four binary flags [G, C, B, A].
The original project idea lists FIVE classes and silently drops LLLG; the source
actually carries SIX.  See C1 in PROPOSAL.md.

It ships no event identifier, which is why the earlier review conceded that
event-level leakage "cannot be ruled out".  It can: the rows are ordered
simulation samples and the label is piecewise constant, so contiguous runs of an
identical flag pattern reconstruct the scenario blocks.  That is what
`reconstruct_scenario_blocks` does, and it is what makes a grouped split
possible.  See C5.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .symmetry import PHASES

RAW_COLUMNS = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]
FLAG_COLUMNS = ["G", "C", "B", "A"]

# The six patterns actually present in the public source, confirmed against the
# dataset README.  Note every fault involves phase A -- that is the finding in
# PROPOSAL.md section 1.2, and the reason this project exists.
KAGGLE_LABELS: dict[tuple[int, int, int, int], str] = {
    (0, 0, 0, 0): "NoFault",
    (1, 0, 0, 1): "AG",
    (0, 0, 1, 1): "AB",
    (1, 0, 1, 1): "ABG",
    (0, 1, 1, 1): "ABC",
    (1, 1, 1, 1): "ABCG",
}

FAULT_TYPES = ["NoFault", "LG", "LL", "LLG", "LLL", "LLLG"]


def fault_type(n_phases: int, ground: int) -> str:
    """Map (number of involved phases, ground flag) to the invariant fault type."""
    ground = int(bool(ground))
    if n_phases == 0:
        if ground:
            raise ValueError("ground flag set with no faulted phase")
        return "NoFault"
    if n_phases == 1:
        if not ground:
            raise ValueError("single phase without ground is an open conductor, not a shunt fault")
        return "LG"
    if n_phases == 2:
        return "LLG" if ground else "LL"
    if n_phases == 3:
        return "LLLG" if ground else "LLL"
    raise ValueError(f"impossible phase count: {n_phases}")


def participation(name: str) -> tuple[np.ndarray, int]:
    """Split a class name such as 'ABG' into (participation bits, ground flag)."""
    if name == "NoFault":
        return np.zeros(3, dtype=int), 0
    ground = int(name.endswith("G"))
    letters = name[:-1] if ground else name
    p = np.zeros(3, dtype=int)
    for ch in letters:
        idx = PHASES.index(ch.lower())
        p[idx] = 1
    return p, ground


def class_name(p: np.ndarray, ground: int) -> str:
    """Inverse of `participation`: build the canonical class name."""
    p = np.asarray(p, dtype=int)
    if p.sum() == 0:
        return "NoFault"
    letters = "".join(PHASES[i].upper() for i in range(3) if p[i])
    return letters + ("G" if ground else "")


def decode_flags(g: int, c: int, b: int, a: int) -> str:
    """Decode one [G, C, B, A] flag pattern to a class name.

    Raises on any pattern outside the six the source is documented to contain,
    so an unexpected combination fails loudly rather than being guessed.
    """
    key = (int(g), int(c), int(b), int(a))
    if key not in KAGGLE_LABELS:
        raise ValueError(f"unexpected G/C/B/A pattern {key}; refusing to guess a label")
    return KAGGLE_LABELS[key]


def reconstruct_scenario_blocks(labels: pd.Series) -> np.ndarray:
    """Recover scenario identifiers from a piecewise-constant label column.

    Each maximal run of an identical label is one block.  Grouping on the result
    prevents adjacent samples from the same event landing on both sides of a
    train/test split.

    TWO LIMITS, both real:

    1. It CANNOT separate consecutive scenarios that share a label -- they merge
       into one block.  That is conservative, not dangerous: an over-merged group
       sends both scenarios to the same side of a split, which is stricter than
       necessary and never leakier.  It costs granularity, not correctness.

    2. If the labels alternate every row, every block has length 1 and the
       grouped split silently degenerates to a random one.  That IS dangerous,
       and it is what `block_health` exists to catch.
    """
    s = pd.Series(labels).reset_index(drop=True)
    return (s != s.shift()).cumsum().to_numpy() - 1


def block_health(scenario_id: np.ndarray) -> dict:
    """Diagnose recovered blocks.  Call this before trusting a grouped split."""
    sizes = pd.Series(scenario_id).value_counts()
    return {
        "n_blocks": int(sizes.size),
        "median_size": float(sizes.median()),
        "min_size": int(sizes.min()),
        "singleton_fraction": float((sizes == 1).mean()),
        "degenerate": bool(sizes.median() <= 1),
    }


def load_raw_csv(path: str) -> pd.DataFrame:
    """Load the public Kaggle classification CSV through the project's schema.

    The generator in simulate.py emits the same frame, so everything downstream
    is identical whether the data is real or synthetic.
    """
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    missing = [c for c in FLAG_COLUMNS + RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing expected columns: {missing}")

    df["fault_class"] = [
        decode_flags(*row) for row in df[FLAG_COLUMNS].to_numpy(dtype=int)
    ]
    parts = [participation(n) for n in df["fault_class"]]
    df["ground"] = [g for _, g in parts]
    df["n_phases"] = [int(p.sum()) for p, _ in parts]
    df["fault_type"] = [fault_type(n, g) for n, g in zip(df["n_phases"], df["ground"])]
    df["scenario_id"] = reconstruct_scenario_blocks(df["fault_class"])

    health = block_health(df["scenario_id"].to_numpy())
    if health["degenerate"]:
        warnings.warn(
            "Scenario-block recovery is degenerate: median block size "
            f"{health['median_size']} over {health['n_blocks']} blocks. The rows "
            "are probably not in simulation order, so a grouped split is NOT "
            "meaningfully different from a random one and the leakage argument "
            "in PROPOSAL.md C5 does not hold for this file.",
            RuntimeWarning,
            stacklevel=2,
        )

    # The flags ARE the target.  Keeping them in X would be direct leakage (C2).
    return df.drop(columns=FLAG_COLUMNS)
