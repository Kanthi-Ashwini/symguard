"""Run the SymGuard baseline end to end and write reports/.

    python -m symguard.run_baseline                  # synthetic apparatus
    python -m symguard.run_baseline --csv data/raw/classData.csv

The synthetic path proves the METHOD works.  It says nothing about the Kaggle
data -- see the honesty rule in simulate.py and README.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import FAULT_TYPES, load_raw_csv
from .evaluate import confusion, phase_bias_gap, report, risk_coverage, score
from .models import BASES, FAMILIES, FaultClassifier
from .simulate import simulate_dataset
from .splits import assert_no_group_overlap, s1_random, s2_grouped, s3_symmetry_holdout

RAW = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]
REPORTS = Path(__file__).resolve().parents[2] / "reports"


def build_dataset(csv: str | None, n_scenarios: int, seed: int) -> tuple[pd.DataFrame, str]:
    if csv:
        df = load_raw_csv(csv)
        # the real CSV is phase-A anchored only, so S3 has no held-out phase
        # positions to test against.  That is the point, and it is why the
        # synthetic apparatus is needed alongside it.
        return df, "kaggle"
    return simulate_dataset(n_scenarios=n_scenarios, seed=seed), "synthetic"


def main() -> None:
    ap = argparse.ArgumentParser(description="SymGuard baseline")
    ap.add_argument("--csv", default=None, help="path to the real Kaggle classData.csv")
    ap.add_argument("--scenarios", type=int, default=60, help="scenarios per class")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df, source = build_dataset(args.csv, args.scenarios, args.seed)
    X = df[RAW].to_numpy()
    y = df["fault_type"].to_numpy()

    print(f"source={source}  rows={len(df)}  scenarios={df.scenario_id.nunique()}")
    print(f"classes present: {sorted(df.fault_class.unique())}\n")

    splits = {
        "S1_random": s1_random(df, seed=args.seed),
        "S2_grouped": s2_grouped(df, seed=args.seed),
        "S3_symmetry": s3_symmetry_holdout(df, seed=args.seed),
    }
    for name in ("S2_grouped", "S3_symmetry"):
        assert_no_group_overlap(df, *splits[name])

    rows, rc_curves, confusions = [], {}, {}
    for family in FAMILIES:
        for base in BASES:
            per_split = {}
            for sname, (tr, te) in splits.items():
                if len(np.unique(y[tr])) < 2:
                    continue
                model = FaultClassifier(family=family, base=base, seed=args.seed).fit(X[tr], y[tr])
                pred = model.predict(X[te])
                m = score(y[te], pred)
                per_split[sname] = m["macro_f1"]
                rows.append({"family": family, "base": base, "split": sname, **m})

                if sname == "S3_symmetry" and base == "rf":
                    proba = model.predict_proba(X[te])
                    abstain = model.tie_ if family == "canonical" else None
                    rc_curves[family] = risk_coverage(proba, model.classes_, y[te], abstain)
                    confusions[family] = confusion(y[te], pred, FAULT_TYPES)

            if {"S2_grouped", "S3_symmetry"} <= per_split.keys():
                rows.append({
                    "family": family, "base": base, "split": "phase_bias_gap",
                    "accuracy": np.nan, "balanced_accuracy": np.nan,
                    "macro_f1": phase_bias_gap(per_split["S2_grouped"], per_split["S3_symmetry"]),
                })

    res = pd.DataFrame(rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    res.to_csv(REPORTS / "results.csv", index=False)

    # ---- console summary ------------------------------------------------
    pivot = res[res.split != "phase_bias_gap"].pivot_table(
        index=["family", "base"], columns="split", values="macro_f1"
    )
    gaps = res[res.split == "phase_bias_gap"].set_index(["family", "base"])["macro_f1"]
    pivot["phase_bias_gap"] = gaps
    pivot = pivot.reindex(FAMILIES, level="family")
    print("macro F1 by split\n")
    print(pivot.round(3).to_string(), "\n")

    for fam, cm in confusions.items():
        print(f"--- confusion, {fam}/rf on S3 (rows = truth) ---")
        print(cm.to_string(), "\n")

    _figures(pivot, rc_curves, source)
    print(f"wrote {REPORTS/'results.csv'} and figures to {REPORTS}")


def _figures(pivot: pd.DataFrame, rc_curves: dict, source: str) -> None:
    note = f"data source: {source}" + (
        "  -- synthetic apparatus, NOT a result about the Kaggle data"
        if source == "synthetic" else ""
    )

    sub = pivot.xs("rf", level="base").reindex(FAMILIES)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    idx = np.arange(len(sub))
    w = 0.38
    ax.bar(idx - w / 2, sub["S2_grouped"], w, label="S2 grouped (in-distribution)")
    ax.bar(idx + w / 2, sub["S3_symmetry"], w, label="S3 symmetry held out (B/C-anchored)")
    ax.set_xticks(idx); ax.set_xticklabels(sub.index)
    ax.set_ylabel("macro F1"); ax.set_ylim(0, 1)
    ax.set_title("Phase-bias gap by model family (Random Forest)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.text(0.5, 0.005, note, ha="center", fontsize=7, style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(REPORTS / "phase_bias_gap.png", dpi=150)
    plt.close(fig)

    if rc_curves:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for fam in [f for f in FAMILIES if f in rc_curves]:
            curve = rc_curves[fam]
            ax.plot(curve["coverage"], curve["accuracy"], marker="o", ms=3, label=fam)
        ax.set_xlabel("coverage (fraction answered)"); ax.set_ylabel("accuracy on answered")
        ax.set_title("Risk-coverage on S3 (Random Forest)")
        ax.set_ylim(0, 1.02); ax.legend(); ax.grid(alpha=0.3)
        fig.text(0.5, 0.005, note, ha="center", fontsize=7, style="italic")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(REPORTS / "risk_coverage.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
