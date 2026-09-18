# Controlled SSVEP

Question: which edits are supported by measurement contrasts, and what should
those edits target? The experiment has three separately identifiable panels.

| Comparison | Corrections | Evaluation |
| --- | --- | --- |
| Closed form | Identity, LEACE, both CORAL/FEATMAP routes, Closed-form MGPA | Native maps; one deterministic fit per rotation |
| Edit permission | Measurement, same-rank PCA, five fixed random orientations, ungated, identity | First VAL movement crossing at 0.05 |
| Target | Conditional score mean, empirical FIT score mean, zero, identity | Native 48-stage maps |

The gate comparison conditions every arm on the same original measurement
complement Q0. PCA/random still receive measurement-derived rank and Q0; they
are permission controls, not calibration-free competitors. An unattained
movement budget is reported as unavailable, never replaced by a native map.
Five random orientations are averaged after each orientation's source-reader
maximum. They are not selected using EVAL.

## Data

Frozen EEGPT encodes full CAR/Oz rereferences of the same wet trials. The
observed endpoints use the fixed γ=0.005 chord; calibration uses the full
contrasts. FIT/VAL source assignment is 0.9-associated with binary stimulus
frequency; independent HEAD/EVAL see both endpoints. Task labels construct
this fixed assay but are removed before adapter fitting. The four rotations
have 81 distinct EVAL participants in total. Folds use seed 20260903; assignment
replica r02 is not a fit seed. Source readers use the fixed 11-reader bank;
task heads are frozen and refitted Ridge readouts. The measurement gate captures
90% of normalized contrast energy, with ranks 3, 3, 3 and 2 across rotations.
Controlled gate and target comparisons share these participants.

## Running

Install the package from the repository root. Run the following commands from
this experiment directory:

```sh
python run.py plan
python run.py all --paths configs/paths.local.toml --run-id reproduction_v1
```

Create `paths.local.toml` from `configs/paths.example.toml` and supply verified
prepared data. Alternatively use `configs/raw.example.toml` for raw EEG and a
separately acquired EEGPT checkpoint where required. Raw features are built
before FIT-only coordinates and gates. They may differ in low bits across
encoder backends; freshly generated receipts record this rather than claiming
bitwise equality to a published feature cache.

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

Source accessibility takes the reader maximum within each fit before averaging
fits. Uncertainty uses 20,000 shared participant-bootstrap draws, seed 1101,
recomputing the reader maximum and averaging the five random orientations within
each draw. Intervals condition on the fitted maps/readers; neither fit seeds nor
random orientations increase the participant count.
