# Temporal N170

Question: on genuine concurrent device measurements, does selective correction
reduce source accessibility while retaining the physiological task?

Neuroscan and Flex record the same face/watch events. The representation is
240 fixed temporal features (16 channels × 15 bins), not a foundation-model
embedding and not the separate cross-task reuse experiment. Continuous
0.1–30 Hz filtering, CAR, 128 Hz resampling, baseline subtraction and a 100 μV
paired amplitude criterion precede feature construction. Event timing,
including the independently corroborated 1034 impedance gap, is audited.

FIT/VAL/HEAD/EVAL contain respectively 622/288/716/1381 clean paired events
from 4/2/4/10 fixed participants. All roles remain participant-disjoint.
FIT determines standardization and the normalized-contrast measurement gate
(90% energy, rank 56). No task labels enter adapter fitting. This cohort overlaps
the N170 task-reuse cohort; the two protocols are not independent replications.

## Comparisons

Native outputs: Identity, LEACE, both orientations of CORAL and FEATMAP,
Closed-form MGPA, Iterative MGPA, its linear-only critic control, and IGBP. IGBP
uses the published longer-training source-only preset with 100 stages.
A separate matched-movement panel compares Iterative MGPA and IGBP at the first
VAL crossings of 0.10 and 0.25; endpoint selection never uses HEAD/EVAL.
All methods share the 11 independent source readers and frozen/refitted Ridge
task heads. CORAL/FEATMAP require source ID at application; the others do not.

## Running

Install the package from the repository root. Run the following commands from
this experiment directory:

```sh
python run.py plan
python run.py all --paths configs/paths.local.toml --run-id reproduction_v1
```

Create `paths.local.toml` from `configs/paths.example.toml` and supply verified
prepared data. Alternatively use `configs/raw.example.toml` for the original
EEG collection. This experiment computes temporal features directly; it needs
no encoder checkpoint. Preprocessing precedes FIT-only coordinate and gate
estimation, and acquisition receipts record input hashes and numerical versions.

Phases can run separately: `prepare`, `fit`, `evaluate`, `report`, `verify`.
Only preparation needs `--paths`; subsequent phases use the sealed run inputs.
All declared fits finish and are sealed before the first HEAD/EVAL evaluation.
Fit seeds 17/29/43 are training repetitions on fixed participant roles, not new
subjects. Deterministic maps are fitted once and their reuse is explicit.

Fresh outputs live under `artifacts/runs/<run-id>/`: portable prepared inputs,
numeric model arrays, raw prediction-bearing cells, tables, and short progress
logs. Scientific results are immutable; a changed recipe needs a new run ID.
`verify` replays maps and hashes without retraining readers; it is an output
integrity check, not a second implementation of the complete experiment.

For the shipped numerical results, use `python run.py report --published`.
This regenerates summaries without data/model fitting. It is separate from
`python run.py report --run-id reproduction_v1`, which summarizes a fresh run.
The central paper-asset builder supplies the manuscript layouts.

Source accessibility is the strongest participant-mean reader AUROC within
each fit, averaged across fits. Frozen/refitted task AUROCs average the two
cross-device transfer directions. Intervals use 20,000 shared participant
bootstrap draws, seed 1101, recomputing the source maximum in each draw. They
condition on fitted coordinates, adapters and readers and are not corrected
for multiple comparisons.
