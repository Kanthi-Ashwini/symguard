# Data card

Two data sources are involved: the public Kaggle CSV this project critiques, and
the bundled generator that supplies what the Kaggle CSV structurally cannot.

---

## 1. Public source — Kaggle "Electrical Fault detection and classification"

| | |
|---|---|
| URL | `https://www.kaggle.com/datasets/esathyaprakash/electrical-fault-detection-and-classification` |
| Origin | **Simulated**, MATLAB Simulink — four 11 kV generators (a pair at each line end), transformers between them, fault injected at the line midpoint |
| Files | `classData.csv` (classification), `detect_dataset` (binary detection) |
| Size | roughly 12,000 rows across the two files |
| Measurements | `Ia, Ib, Ic` (line currents, A) and `Va, Vb, Vc` (line voltages, pu) |
| Target | four binary flags `[G, C, B, A]` |
| Licence | see the Kaggle page. **Not redistributed in this repository** — `data/raw/` is gitignored. Download it yourself and credit the source. |

### Label map — six classes, not five

| `G C B A` | Class | Fault type |
|---|---|---|
| `0 0 0 0` | NoFault | NoFault |
| `1 0 0 1` | AG | LG |
| `0 0 1 1` | AB | LL |
| `1 0 1 1` | ABG | LLG |
| `0 1 1 1` | ABC | LLL |
| `1 1 1 1` | ABCG | LLLG |

The originating project idea lists **five** classes and drops LLLG. Any pattern
outside these six raises in `data.decode_flags` rather than being guessed.

### Known gaps — read before using this data

**1. Every fault involves phase A.**
AG, AB, ABG, ABC, ABCG. There is no BG, CG, BC, BCG, CA or CAG anywhere in the
dataset. A model can therefore score ~100% by learning "watch phase A", and
nothing the dataset supports will detect that. This is the central finding of the
project; see `PROPOSAL.md` §1.3.

**2. There is no event identifier.**
Rows are ordered simulation samples with no scenario column, so a naive
`train_test_split` puts samples microseconds apart on both sides of the boundary.
`data.reconstruct_scenario_blocks` recovers the scenarios from runs of a constant
label, which makes a grouped split possible. Measured cost of ignoring this:
**0.109 macro F1**.

**3. A row is a snapshot, not a waveform.**
Each row is one instantaneous six-sensor reading. Mean, RMS, standard deviation,
peak and peak-to-peak are undefined per row and meaningless across the six
columns, which mix kV with A. See `PROPOSAL.md` C3.

**4. The flags are the target.**
`G/C/B/A` must never appear in `X`. `data.load_raw_csv` drops them on load.

---

## 2. Bundled generator — `src/symguard/simulate.py`

### Why it exists

Not because the Kaggle data is unavailable. Because the Kaggle data **cannot
supply B- and C-anchored faults**, and the symmetry-held-out evaluation needs
them. It is required apparatus, not a substitute.

### Honesty rule

> Numbers produced from the generator demonstrate that the **apparatus** works.
> They are **not** results about the Kaggle data and must never be quoted as such.

Stated in `README.md`, in `simulate.py`, in `model_card.md`, and printed into
every figure caption by `run_baseline.py`.

### Model

A simplified single-source lumped model solved with phasors: a balanced
positive-sequence source behind a source impedance, a line, a fault at fractional
distance `loc` through fault impedance `Z_f`, and a load. Measured voltage is the
source voltage minus the drop across the source impedance.

Randomised per scenario: inception angle, fault location (0.1–0.9), fault
resistance (0.005–0.25 pu), source/line/load impedances, and measurement noise.
Each scenario emits 64 consecutive samples at 128 per 50 Hz cycle, sharing one
`scenario_id` — which is what gives the grouped split real groups.

### What it deliberately is not

- **Not a sequence-network solution.** It reproduces the qualitative signatures
  that matter here — ground-return current, voltage sag, fault-current magnitude —
  and nothing more.
- **No CT/VT saturation, no arc dynamics, no travelling waves, no topology
  variation.**
- Scaled to resemble the Kaggle ranges (`I_BASE = 100 A/pu`, `V_BASE = 0.6 pu`),
  not to match any specific network.

### Signatures it produces (measured, mean over 30 scenarios/class)

| Fault type | `I_mag` | `\|i0\|/\|I\|` | Physical reading |
|---|---|---|---|
| NoFault | 22.5 | 0.015 | load current only |
| LG | 118.6 | **0.422** | ground return path present |
| LL | 203.8 | 0.003 | circulating current, no ground path |
| LLG | 176.1 | **0.331** | ground return path present |
| LLL | 246.0 | 0.001 | balanced, sums to zero |
| LLLG | 235.6 | 0.013 | balanced; ground path nearly invisible |

The last row is deliberate. LLL and LLLG are genuinely hard to separate from a
snapshot because a balanced three-phase-to-ground fault carries almost no
zero-sequence current. The generator does not cheat that away, and the models
confirm it is the hardest pair.

### Classes

- Kaggle set (training pool): `NoFault, AG, AB, ABG, ABC, ABCG`
- Held out for S3: `BG, CG, BC, AC, BCG, ACG`
- Rotation-invariant (`NoFault, ABC, ABCG`) are split by scenario across both sides.

Twelve physically distinct states, reachable from the six by cyclic rotation.
Asserted in `tests/test_symmetry.py`.

---

## 3. Not used — PROTECT-90

Specified as future work in `PROPOSAL.md` §5.4 and **not downloaded or run**.
9,022 EMT episodes, 90 kV, 6.4 kHz, all phase combinations present, CC BY 4.0,
Zenodo `10.5281/zenodo.18418330`. It carries no LLLG and no no-fault class, and
the archive is 12.5 GB compressed, so any future use must take a stratified
subset and evaluate on the four-class intersection.
