# Model card

## What this is

A study of how much a three-phase fault classifier's reported score depends on
the evaluation split, and a demonstration that a symmetry-grounded representation
removes the largest failure mode. Four model families, three classifiers, three splits.

## What this is not

**Not a protection relay, and not a candidate for one.** It must not control
electrical equipment, inform a trip decision, or be described as real-time
protection. It classifies simulated snapshots offline.

---

## Intended use

- Teaching and demonstrating how evaluation-protocol choices change a headline number.
- A reference implementation of C3 phase-equivariant fault classification.
- A worked critique of a widely-used public benchmark.

## Out of scope

Fault location, fault-clearing action, relay coordination, real-time performance
claims, field deployment, any use on real measurements.

---

## Model families

| Family | Representation | Phase-equivariant? |
|---|---|---|
| `naive` | raw `Ia, Ib, Ic, Va, Vb, Vc` | no |
| `augmented` | raw, plus the three C3 rotations at fit time | approximately, by training |
| `canonical` | rotated into a data-derived canonical phase frame | yes, by construction |
| `invariant` | Clarke invariants only | yes, exactly — cannot represent phase identity at all |

Classifiers: `DummyClassifier(most_frequent)`, `LogisticRegression`,
`RandomForestClassifier(300 trees, class_weight=balanced)`. Identical across
families, so any difference is caused by the representation.

Target: fault **type** — NoFault / LG / LL / LLG / LLL / LLLG — which is invariant
under phase rotation. Phase *identity* is deliberately not a target here.

---

## Results

Random Forest, macro F1. 46,080 snapshots, 720 scenarios, seed 42.

| family | S1 random | S2 grouped | S3 unseen phases | phase-bias gap |
|---|---|---|---|---|
| **naive** | 0.928 | 0.819 | **0.177** | **0.643** |
| augmented | 0.936 | 0.818 | 0.845 | -0.027 |
| canonical | 0.933 | 0.817 | 0.844 | -0.027 |
| **invariant** | 0.961 | 0.868 | **0.888** | -0.020 |

Logistic regression, same splits: `naive` 0.269 / 0.165 / 0.223 versus
`invariant` 0.884 / 0.879 / 0.876.

Majority-class floor: 0.070.

### What each number does and does not support

| Number | Supports | Does **not** support |
|---|---|---|
| **S1 random, 0.928** | Nothing. Published only as the inflated figure to argue against. | Any claim of accuracy. Adjacent samples from one scenario appear on both sides. |
| **S2 grouped, 0.819** | In-distribution performance with no event leakage. | Performance on phase positions absent from training. |
| **S3 unseen phases, 0.177 (naive)** | That the naive model learned phase position, not fault physics. | A claim about real hardware — the rotated data is synthetic. |
| **S3, 0.888 (invariant)** | That an invariant representation transfers to phase positions never trained on. | Transfer across simulator, voltage level or topology — that is §5.4, not run. |
| **phase-bias gap, 0.643** | The size of the failure mode the public benchmark cannot detect. | A number about the Kaggle CSV. It is measured on the generator. |

---

## Selective prediction

Confidence is the **max class probability from the Random Forest. It is not
calibrated** and must not be reported as a probability. No calibration study was run.

Canonical-frame ties — where the two largest instantaneous phase currents are
within 5% and the frame is unstable — are routed to explicit abstention rather
than scored as confident predictions.

| Coverage | Accuracy on answered |
|---|---|
| 50% | 0.995 |
| 75% | 0.973 |
| 90% | 0.933 |
| 100% | 0.927 |

Tie rate on S3: **10.6%**. Reported, not hidden. The rate is not a defect: for a
balanced LLL fault the phase frame is genuinely arbitrary, and near a current
zero crossing it is genuinely ambiguous.

---

## Limitations

**1. The results above come from the bundled generator, not the Kaggle CSV.**
The largest caveat on everything in this card. The generator is required
apparatus — the Kaggle data cannot supply B/C-anchored faults — but it is a
simplified lumped model, not a validated network simulation. See `data_card.md`.

**2. No real B- or C-phase fault has ever been seen.** The S3 test data is
produced by rotating simulated A-anchored faults. That rotation is algebraically
exact *under the balanced-system assumption*, and no more. The check that would
close this is §5.4, which is specified and not run.

**3. LLL vs LLLG is unresolved.** Both are balanced three-phase faults with
near-zero zero-sequence current, so the ground path is nearly invisible from a
single snapshot. Every family confuses them. This may be irreducible at W=1 and
is the first thing the window ablation (§5.5) should test.

**4. Single snapshots have an error floor.** Near a current zero crossing a
faulted phase is momentarily indistinguishable from a healthy one. Not yet
quantified — that is §5.5.

**5. Canonicalisation is discontinuous.** Small perturbations near a tie flip the
frame. Mitigated by the abstention path, not eliminated.

**6. No hyperparameter search.** Defaults throughout, deliberately: the argument
is about representation and evaluation protocol, and tuning would confound it.

**7. Not evaluated on the real Kaggle CSV yet.** When it is, S3 will be
unavailable on that data alone — it contains no B/C-anchored faults to test with.

---

## Reproducing

```bash
pip install -r requirements.txt
cd src
python -m pytest ../tests -q          # 36 tests
python -m symguard.run_baseline --scenarios 60 --seed 42
```

All randomness is seeded. `reports/results.csv` carries every number in this card.

## Ethical and safety note

Simulated data, offline classification, prototype status. Presenting this as
validated protection equipment would be unsafe. Protection engineering requires
hardware-in-the-loop validation, standards compliance and field testing, none of
which is in scope here.
