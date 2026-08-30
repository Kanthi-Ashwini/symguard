# SymGuard

**Phase-equivariant three-phase fault classification, and a measured critique of a
widely-used public benchmark.**

Every fault in the public Kaggle transmission-line dataset involves phase A.
There is no BG, no CG, no BC, no BCG. A classifier can therefore score ~99% by
learning *"watch phase A"* — and no experiment the dataset supports can tell that
apart from a model that learned fault physics.

SymGuard builds the test the benchmark cannot, and measures the gap.

---

## The result

Random Forest, macro F1, 46,080 snapshots across 720 scenarios:

| model family | S1 random rows | S2 grouped | S3 unseen phase positions | phase-bias gap |
|---|---|---|---|---|
| **naive** (raw abc — what the public notebooks train on) | 0.928 | 0.819 | **0.177** | **0.643** |
| augmented (C3 rotations as augmentation) | 0.936 | 0.818 | 0.845 | −0.027 |
| canonical (rotate into a canonical phase frame) | 0.933 | 0.817 | 0.844 | −0.027 |
| **invariant** (Clarke invariants only) | 0.961 | 0.868 | **0.888** | −0.020 |

Four things fall out of that table:

1. **Random splitting inflates the score.** The naive model reads 0.928 on a random
   row split and 0.819 once whole scenarios are kept together — 0.109 of the
   headline number is adjacent-sample leakage.
2. **The naive model collapses on unseen phase positions**, 0.819 → 0.177, barely
   above the 0.070 majority-class floor. It learned the dataset, not the physics.
3. **Symmetry-aware models do not collapse** — and none of them ever saw a B- or
   C-anchored fault during training.
4. **The representation matters more than the estimator.** Logistic regression
   scores 0.165 on raw `abc` and 0.879 on Clarke invariants — a 5.3× swing from
   changing the features alone, larger than any model swap in the table.

`reports/phase_bias_gap.png`, `reports/risk_coverage.png`, `reports/results.csv`.

---

## ⚠️ Honesty rule

**The numbers above come from the bundled physics generator, not from the Kaggle
CSV.** They demonstrate that the *apparatus* works. They are **not** results about
the public dataset and must never be quoted as such.

This is not a workaround for missing data. The Kaggle source **structurally
cannot** provide B- and C-anchored faults — that absence is the entire finding —
so producing them is required apparatus, not a substitute. `src/symguard/data.py`
reads the real CSV through the identical schema the moment it is placed in
`data/raw/`, and every module downstream is unchanged.

---

## Run it

No downloads, no credentials, runs in about a minute.

```bash
pip install -r requirements.txt

cd src
python -m pytest ../tests -q            # 36 tests
python -m symguard.run_baseline         # writes reports/

# once the public CSV is in data/raw/
python -m symguard.run_baseline --csv ../data/raw/classData.csv
```

`make test` / `make baseline` / `make real` do the same where `make` is available.

---

## How it works

A balanced three-phase system is invariant under the cyclic group **C₃** acting on
phase labels (a→b→c→a). Applying `g ∈ C₃` permutes the current and voltage
triplets identically and permutes the phase-participation bits the same way:

> fault **type** is **invariant** under C₃ · faulted **phase set** is **equivariant**

Everything follows from those two lines.

- **`symmetry.py`** — the group action, and `canonicalise()`, which rotates each
  snapshot into a data-derived canonical frame so a rotated input yields an
  identical canonical form.
- **`features.py`** — the Clarke (αβ0) transform. Cyclic relabelling is a 120°
  rotation about the (1,1,1) axis, so it preserves the zero-sequence component
  and the space-vector magnitude while rotating its angle. `|I|`, `|i₀|`, `|I|/|V|`
  are therefore **exactly** C₃-invariant; `θ` carries phase identity.
  `tests/test_invariance.py` asserts this to a relative 1e-10 — that test is the
  project's proof, not a formality.
- **`data.py`** — decodes the six `[G,C,B,A]` patterns, and recovers scenario
  blocks from runs of a constant label, which is what makes a leakage-free
  grouped split possible at all.
- **`simulate.py`** — a phasor-solved lumped model producing the correct
  signatures: ground faults carry zero-sequence current, LL and LLL do not.
- **`splits.py` / `models.py` / `evaluate.py`** — the four splits, the four
  families, and the phase-bias gap.

---

## What is and is not novel

**Not new**, and not claimed: the dataset, the six inputs and the LR/SVM/RF
comparison are all public; phase-permutation ("phase switching") augmentation is
already in the literature — it is the `augmented` baseline here, not the idea;
Clarke and symmetrical components are textbook.

**The contribution** is using the phase symmetry as an *evaluation axis* rather
than only a training trick, on a benchmark that cannot otherwise test it, and
quantifying how much of the widely-reported 99%+ is phase bias.

The honest one-line claim is *"a measured critique and a symmetry-grounded fix for
a widely-used public benchmark"* — **not** "a novel fault-detection algorithm."

---

## Status

| | |
|---|---|
| Verification of the original project idea | done — `PROPOSAL.md` §1 |
| C₃ symmetry, Clarke basis, invariance proof | done — 36 tests green |
| Four model families, splits S1/S2/S3 | done |
| Selective prediction (risk–coverage) | done |
| **S4 cross-dataset transfer to PROTECT-90** | **specified, NOT RUN** — `PROPOSAL.md` §5.4 |
| Window ablation / Bayes floor | specified, not run — `PROPOSAL.md` §5.5 |
| Run against the real Kaggle CSV | pending the download |

Read `PROPOSAL.md` for the full design and the claim-by-claim verification,
`data_card.md` for the data and its limits, `model_card.md` for what the numbers
do and do not support.
