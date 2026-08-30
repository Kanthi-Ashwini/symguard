# SymGuard — Verification of the Project Idea, and a Symmetry-Grounded Proposal

*Companion to `project_Idea (1).pdf` and `three_phase_fault_classifier_5_day_plan.pdf`.*

Part 1 verifies the original idea document claim by claim. Part 2 proposes the
replacement approach. Part 3 reports what the implemented baseline actually
measured. Part 4 states the novelty claim honestly. Part 5 is the work that is
specified but deliberately not run.

---

## 1. Verification of the original project idea

### 1.1 Sources checked

| Source | What it established |
|---|---|
| Kaggle `esathyaprakash/electrical-fault-detection-and-classification` | The public dataset under discussion |
| GitHub `sathyaprakash000/Electrical-Fault-detection-and-classification` | **The `[G,C,B,A]` encoding and the exact six classes**; MATLAB Simulink origin — four 11 kV generators, fault at the line midpoint |
| STMicroelectronics ST Edge-AI case study | An independent industry treatment of the same dataset — evidence the problem is well-trodden |
| `arXiv:2606.24298` — *PROTECT-90*, IEEE PES ISGT Europe 2026, Zenodo `10.5281/zenodo.18418330` | A CC BY 4.0 waveform benchmark that fixes every limitation identified here |
| Literature on phase-permutation augmentation | **"Phase switching" augmentation already exists** — so it cannot be claimed as novel |

### 1.2 Verdicts

| # | Claim | Verdict | What is actually true |
|---|---|---|---|
| **C1** | §3: five classes — No Fault, LG, LL, LLG, LLL | **Wrong — there are six** | `[G C B A]`: `0000` NoFault, `1001` AG, `0011` AB, `1011` ABG, `0111` ABC, `1111` ABCG. The proposal silently drops **LLLG (ABCG)**. Pinned by `tests/test_labels.py`. |
| **C2** | §4: inputs `[Va,Vb,Vc,Ia,Ib,Ic]` | **Correct** | File order is `Ia,Ib,Ic,Va,Vb,Vc`. `G/C/B/A` must never enter `X` — they *are* the target. `data.py` drops them on load. |
| **C3** | §7: extract mean, RMS, sigma, peak, peak-to-peak "from the signals" | **Invalid as written** | Each row is one *instantaneous* six-sensor snapshot; there is no per-row time window. Computing these across the six columns averages kV with A. Legitimate only at W >= 1 cycle — see §5.5. |
| **C4** | §7.5: "sequence components may be included" | **Half true** | Zero-sequence and the Clarke alpha-beta-zero transform *are* instantaneous and valid from one row — SymGuard is built on exactly that. Positive/negative-sequence (Fortescue) need **phasors**, so magnitude *and* angle, so a time window. Impossible at W=1. |
| **C5** | §9: "event-level split where possible" | **Not possible as stated — but recoverable** | No event-ID column, which is why the earlier review conceded leakage "cannot be ruled out". It can: rows are ordered samples and the label is piecewise constant, so contiguous runs of an identical pattern reconstruct the scenarios. `data.reconstruct_scenario_blocks`. **Measured cost of ignoring this: 0.109 macro F1.** |
| **C6** | §6: different faults produce distinguishable patterns | **True, with a missing caveat** | Near a current zero crossing a faulted phase is momentarily indistinguishable from a healthy one, so single-snapshot classification has an irreducible error floor the document never mentions. Visible here as the 10.6% canonical-frame tie rate. |
| **C7** | §12: PCA as a meaningful extension | **Low value, and the wrong tool** | Six correlated features present no dimensionality problem, and PCA is a *data-fitted* rotation that destroys phase structure. Clarke is a *fixed physical* rotation that preserves it. **Measured: logistic regression 0.165 on raw `abc` vs 0.879 on Clarke invariants.** |
| **C8** | §14: "confusion between LL and LLG will be informative" | **Will not happen** | Ground involvement is near-perfectly separable by zero-sequence current. Asserted in `test_invariance.py`: floating faults hold `abs(i0)/abs(I) < 0.05`, ground faults `> 0.15`. The genuinely hard pair is **LLL vs LLLG**, which the document never mentions. |
| **C9** | §19: "research basis — published studies" | **True but uncited**, and the specific project is already public | Multiple GitHub repositories plus the ST case study use this exact dataset and model set. |
| **C10** | 4–5 day feasibility | **True** for the scoped version | Confirmed by the earlier review. SymGuard is larger; three weeks is budgeted. |

### 1.3 The flaw neither document caught

**Every fault class in the dataset is anchored on phase A: AG, AB, ABG, ABC, ABCG.**
There is no BG, CG, BC, BCG, CA or CAG anywhere in it.
Asserted in `tests/test_labels.py::test_every_kaggle_class_involves_phase_a`.

Three consequences:

1. A model can reach ~100% by learning **"watch phase A"** and fail completely on a
   real BG or CG fault. The 99%+ figures in public notebooks are fully consistent
   with a model that learned *dataset phase bias*. **Measured here: 0.819 -> 0.177.**
2. It explains C8. With ground separable by the zero-sequence current and every
   fault A-anchored, the task is far easier than it appears and the confusion
   matrix is uninformative.
3. It answers the novelty question. The dataset **cannot** test phase
   generalisation. Constructing that test is the contribution.

---

## 2. The approach

> **Thesis.** A three-phase fault classifier must be equivariant to cyclic phase
> relabelling. The public benchmark cannot test this because every fault it
> contains involves phase A. SymGuard builds the equivariance in by construction
> and measures the gap it closes.

### P1 — The symmetry

A balanced three-phase system is invariant under the cyclic group **C3** on phase
labels (a -> b -> c -> a). For `g` in C3, applying `g` permutes `(Ia,Ib,Ic)` and
`(Va,Vb,Vc)` identically and permutes the phase-participation bits the same way.

- Fault **type** is **invariant** under C3.
- Faulted **phase set** is **equivariant** under C3.

`symmetry.py`; verified by `test_symmetry.py` (order 3, inverse, type invariance,
and that rotating the six Kaggle classes reaches all **twelve** physical states).

*The S3 extension — the reflection swapping two phases — is exact only with a
consistent phase-sequence flip. Left as an ablation, §5.6.*

### P2 — Factorised label

`g` in {0,1} ground involvement; `p` in {0,1}^3 phase participation; type
`= f(|p|, g)`. Under rotation `p` rotates and `g` is fixed, which is what makes
equivariance expressible and testable. `data.participation` / `data.fault_type`.

### P3 — The invariant feature basis

```
i_alpha = (2*ia - ib - ic)/3     i_beta = (ib - ic)/sqrt(3)     i0 = (ia + ib + ic)/3
|I| = hypot(i_alpha, i_beta)     theta_I = atan2(i_beta, i_alpha)        (same for V)

ground:    |i0|, |v0|, |i0|/|I|, |v0|/|V|
unbalance: |I|/|V|, sum|i_x|, (max - min)|i_x|
```

Cyclic relabelling is a 120-degree rotation about the (1,1,1) axis, so it
preserves the zero-sequence component and the space-vector magnitude while
rotating the angle. `|I|`, `|i0|`, `|I|/|V|` are therefore **exactly** invariant
and `theta` carries phase identity. `test_invariance.py` asserts a relative drift
below **1e-10** on both random and physically simulated data. **That test is the
proof the physics claim is real rather than decorative.** If it fails, the
argument does not stand.

### P4 — Four model families

Identical estimators throughout, so any difference is caused by the
**representation**, not the model.

| Family | Representation | Role |
|---|---|---|
| `naive` | raw `abc` | What the public notebooks train on |
| `augmented` | raw `abc` plus the three C3 rotations as augmentation | **Already known** — "phase switching" from the literature. The baseline to beat, not the idea. |
| `canonical` | rotate into a data-derived canonical frame, classify there | Correct on unseen phase positions **by construction, zero augmentation** |
| `invariant` | Clarke invariants only | Phase-blind by construction — it *cannot* represent which phase faulted, which is exactly right when the target is the invariant type |

### P5 — Four splits

| Split | What it is | What it measures |
|---|---|---|
| **S1** random rows | what every public notebook does | reported **only** to show it is inflated |
| **S2** grouped | `GroupShuffleSplit` on recovered scenario blocks | the honest in-distribution number |
| **S3** symmetry held out | train A-anchored, test B/C-anchored | **the headline: the phase-bias gap** |
| **S4** cross-dataset | Kaggle -> PROTECT-90 | **specified, NOT RUN** — §5.4 |

S3 tests on BG/CG/BC/AC/BCG/ACG plus held-out *scenarios* of the
rotation-invariant classes, so all six types appear on both sides and macro F1 is
comparable. `test_no_leakage.py` holds every guarantee in place.

### P6 — Selective prediction

Risk–coverage on S3, with canonical-frame ties routed to explicit abstention. The
score is an **uncalibrated** max class probability and is labelled as such.

---

## 3. What the baseline measured

46,080 snapshots, 720 scenarios, Random Forest, macro F1:

| family | S1 random | S2 grouped | S3 unseen phases | phase-bias gap |
|---|---|---|---|---|
| **naive** | 0.928 | 0.819 | **0.177** | **0.643** |
| augmented | 0.936 | 0.818 | 0.845 | -0.027 |
| canonical | 0.933 | 0.817 | 0.844 | -0.027 |
| **invariant** | 0.961 | 0.868 | **0.888** | -0.020 |

Majority-class floor: 0.070. Canonical tie rate on S3: **10.6%**.
Risk–coverage (canonical): 0.995 at 50% coverage, 0.933 at 90%, 0.927 at full.

**Findings**

1. **Leakage is real and costs 0.109 macro F1** (0.928 -> 0.819). Verdict C5 confirmed.
2. **The naive model collapses to near-chance on unseen phase positions**
   (0.819 -> 0.177). The finding of §1.3, measured.
3. **Every symmetry-aware family holds**, none having seen a B/C-anchored fault.
   `invariant` is best at 0.888 and also wins in-distribution.
4. **Representation beats estimator.** Logistic regression: 0.165 on raw `abc`,
   0.879 on Clarke invariants — a 5.3x swing, larger than any model change.
   Verdict C7 confirmed.
5. **LLL vs LLLG is the genuinely hard pair**, not LL vs LLG as the idea document
   predicted. Both are balanced three-phase faults with near-zero zero-sequence
   current, so the ground path is nearly invisible from a snapshot. Verdict C8 confirmed.

> **These numbers come from the bundled generator, not the Kaggle CSV.** They
> demonstrate the apparatus. See the honesty rule in `README.md`.

---

## 4. The novelty claim

**Not new, and not claimed:**

- The dataset, the six inputs, the LR/SVM/RF comparison — all public.
- Phase-permutation augmentation — **already in the literature**; it is the
  `augmented` baseline here.
- Clarke and symmetrical components — textbook power systems.

**Defensible:**

1. **The symmetry-held-out protocol (S3)** — phase symmetry as an *evaluation
   axis*, not only a training trick, on a benchmark that cannot otherwise test it.
2. **Quantifying the phase-bias gap** in the widely-reported 99%+ results.
3. **A hard-equivariant classifier needing zero augmentation**, with a numerical
   invariance proof.
4. **Scenario-block recovery**, giving a leakage-free grouped split where the
   earlier review concluded none was possible.
5. *(specified, not run)* Cross-simulator transfer — §5.4.

Write **"a measured critique and a symmetry-grounded fix for a widely-used public
benchmark"** — not "a novel fault-detection algorithm."

---

## 5. Specified but not run

### 5.4 Cross-dataset transfer to PROTECT-90 — **STATUS: NOT RUN**

Train on Kaggle (11 kV, Simulink), test on PROTECT-90 (90 kV, DIgSILENT
PowerFactory EMT). Real out-of-distribution generalisation across simulator,
voltage level and topology.

**Verified specifications.** 9,022 EMT episodes on a 90 kV double-line topology;
6.4 kHz (128 samples per 50 Hz cycle); 1 s per episode (6,400 steps); 48
synchronised channels across 8 relay locations; metadata CSV carrying `sc_type`,
`phase_select`, `sc_location`, fault resistance and inception time.
**All phase combinations present.** CC BY 4.0, Zenodo `10.5281/zenodo.18418330`.

**Three constraints that must be stated with any future result:**

- **12.5 GB compressed, ~31 GB uncompressed.** Do not pull the archive. Take
  ~300–600 stratified episodes at one relay location, roughly 1 GB.
- **SLG / LL / LLG / LLL only — no LLLG and no no-fault class.** Any transfer
  result is on the **four-class intersection**. Do not fudge the label map.
- The paper independently warns that sliding windows from one episode must not
  cross the train/test boundary — **outside corroboration of verdict C5.**

### 5.5 Window ablation and the error floor — **STATUS: NOT RUN**

Sweep `W` in {1,2,4,8,16,32,64} samples under S2 and plot macro F1 against `W`. At
`W=1`, stratify accuracy by `|I|` and show it collapses in the low-magnitude bin —
verdict C6 measured rather than asserted. At `W >= 1 cycle` the RMS / sigma /
peak-to-peak features of §7 and the true sequence components of §7.5 become
**legitimately computable**: the original feature set is correct, but only there.
`simulate.py` already emits an ordered time axis, so this is the first thing to
run next.

### 5.6 Reflection ablation — **STATUS: NOT RUN**

Extend C3 to the full symmetric group S3 by adding the reflection that swaps two
phases, with the consistent phase-sequence flip that makes it physically exact.

---

## 6. Status and schedule

| Week | Work | State |
|---|---|---|
| 1 | Verification, label decoding, scenario blocks, C3 action, Clarke basis, 36 tests | **done** |
| 2 | Four families, splits S1/S2/S3, the phase-bias gap, risk–coverage | **done** |
| 3 | Real Kaggle CSV through `data.py`; §5.5 window ablation; §5.6 ablation | pending |

**Honest failure modes**, carried into `model_card.md`:

- The generator is not the Kaggle data. Largest caveat on everything in §3.
- Canonicalisation is unstable near ties — measured at 10.6%, reported, not hidden.
- Rotated S3 data is exact only under the balanced-system assumption. Without §5.4,
  **no real B- or C-phase fault is ever seen.**
- LLL vs LLLG remains unresolved at W=1 and may be irreducible from a snapshot.
