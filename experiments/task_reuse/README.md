# P300-selected correction reused on N170 and MMN

This experiment compares the six manuscript families: Identity,
LEACE, Closed-form MGPA, Iterative MGPA, CORAL and FEATMAP. The two MGPA methods
use pooled conditional targets and a rank-eight measurement gate. Iterative
MGPA uses `max_step=.025`. P300 DEV chooses operating points; those maps are
applied unchanged to N170/MMN. Result identifiers are `mgpa_cf` and `mgpa_iter`.

Run the commands below from the repository root. Install the optional raw-data
dependencies when preparing EEG epochs:

```sh
python -m pip install -e '.[raw]'
```

Saved participant results and DEV candidate curves are in `artifacts/tables/`.
Rebuild the table, CSV and figure without EEG or fitting; the second command
independently checks the saved DEV selections:

```sh
python experiments/task_reuse/run.py report --published
python experiments/task_reuse/run.py verify-selection --published
```

## Fresh reproduction

Obtain the required [Flex–Neuroscan files](../../DATASETS.md#flex-neuroscan) and
[EEGPT checkpoint](../../DATASETS.md#eegpt). The download guide includes the
[raw-input setup](../../DATASETS.md#run-from-raw-data).

Use a new run directory. The raw collection root must contain the original
`raw/Saline Raw Data/` layout. Supply the published EEGPT safetensors checkpoint
(SHA256 `fb34c20609983324679b9534f9d17a2289a232d0276dd2b8b7d5b088f876f621`).

```sh
python experiments/task_reuse/run.py all \
  --raw-root /data/flex_collection \
  --checkpoint /data/weights/model.safetensors \
  --run-dir /data/runs/mgpa-task-reuse
```

Raw ingestion uses simultaneous event correspondence, the 129.05 Hz Flex
physical sampling rate and eight-sample offset, common 16 channels, continuous
0.1–30 Hz filtering, common-average reference and 128 Hz resampling. One-second
epochs begin at −.2 s and subtract the prestimulus baseline. All technically
valid pairs are retained, including amplitude-flagged epochs; this differs
from the temporal N170 amplitude-QC protocol. Acquisition receipts record raw
input hashes and event timing. EEGPT uses fixed 20 μV scaling and all 7×4×512
tokens. Raw EVAL recordings are not opened until the selection lock exists.

For existing paired epochs or frozen full-token features, provide a portable
manifest instead; its schema is documented in [DATA.md](DATA.md):

```sh
python experiments/task_reuse/run.py prepare \
  --manifest /data/reuse/inputs.json --run-dir /data/runs/mgpa-task-reuse
python experiments/task_reuse/run.py fit --run-dir /data/runs/mgpa-task-reuse
python experiments/task_reuse/run.py evaluate \
  --manifest /data/reuse/inputs.json --run-dir /data/runs/mgpa-task-reuse
python experiments/task_reuse/run.py report --run-dir /data/runs/mgpa-task-reuse
python experiments/task_reuse/run.py verify --run-dir /data/runs/mgpa-task-reuse
```

`prepare` fits PCA256 and featurewise standardization on Neuroscan endpoints of
the remaining P300 observed FIT events only. It materializes P300 FIT, CAL,
HEAD and DEV, preserving event IDs and timestamps. `fit` builds both frozen
P300 logistic heads, fits all declared candidates, evaluates all declared DEV
endpoints, then writes `selection_lock.json`. It can resume completed candidate
cells after checking their saved model hashes. Interrupted incomplete model
directories require a new run directory. `evaluate` authenticates that lock,
fits each transfer task's heads on its original HEAD data, and evaluates the
same selected maps on EVAL. `verify` checks hashes and recomputes saved
association coefficients from saved probabilities. It does not refit.

## Fixed protocol

| Role | Participants |
|---|---|
| FIT/CAL | 1013, 1014, 1016, 1018 |
| HEAD | 1019, 1020, 1024, 1025 |
| DEV | 1022, 1023 |
| EVAL | 1026–1033 |

For each FIT participant, the first 100 consecutive P300 events are CAL.
Remove other events whose one-second epoch overlaps CAL on either device.
The common bank contains 2,260 observed endpoints and both endpoints of 400
CAL pairs: 3,060 unique rows. Source validation uses 1,332 DEV endpoints.
Labels enter endpoint assignment and task-head fitting/selection only; adapter
fitters receive features and source labels. N170/MMN supply no coordinate or
adapter fitting/selection data.

Every task has aligned-association and independent-association logistic heads.
HEAD-only leave-one-participant-out validation selects a shared C from
`1e-6, 1e-5, …, 10`. Participant weights are equalized. Both heads and their
standardizers stay frozen under correction. Data-assignment seed is 17;
task-head seed is 1017. Iterative fitting alone varies over 17, 29, 43.

Within each candidate, P300 DEV maximizes worst association AUROC subject to
at most .01 loss in each head's independent-regime AUROC. Ties prefer less
movement, then earlier endpoints. All iterative prefixes 0–48 and affine
strengths `0,.02,…,1,1.1,1.2,1.3` are tested. Across baseline recipes, candidates
within .002 of the best admissible value are ordered by movement and a fixed
candidate-name tie rule. The exact grids are in [configs/paper.json](configs/paper.json).
The strict endpoint comparison permits only four float64 machine epsilons of
rounding slack. The .002 family threshold permits `1e-15` rounding slack; it
is a declared DEV parsimony choice, not a significance threshold.

The published choices are CF strength 1.1; Iterative stages 13, 11, 19 for
seeds 17, 29, 43; LEACE ridge 0/strength .98; CORAL toward Flex, ridge 10/strength
.98; and FEATMAP toward Neuroscan, ridge 1000/strength 1. Fresh runs perform
the declared DEV selection rather than hard-coding these recorded outcomes.

`artifacts/tables/dev_cells/` retains all 28 declared candidates and
1,444 endpoint rows: Identity, pooled CF at rank eight, all five LEACE ridges,
eight CORAL and ten FEATMAP recipes, and each of the three 49-prefix Iterative
trajectories. Each row preserves the participant AUROCs, movement, two utility
guardrails, and endpoint summary needed for exact selection replay.
`artifacts/logs/dev_selection_verification.json` records a successful replay:
every selected summary is exactly equal, and family selection recovers the
manuscript settings. Numerical provenance and file hashes are recorded in the
repository's [release manifest](../../release_manifest.json).

CORAL and FEATMAP require the actual source ID at application and preserve the
reference device. The other methods apply without source routing.

## Numerical outputs and uncertainty

Each run keeps resolved configuration, preparation receipts, coordinates,
frozen heads, candidate models, complete DEV curves, locked selections,
prediction arrays, per-task cells and logs. `report` reads these saved cells
and produces `results.json`, `table.csv`, `table.md`, and `worst_auroc.png/pdf`.
Reporting from the supplied participant cells is separate from fresh-run outputs.

The participant-mean biased-head AUROC is exactly quadratic in association
ξ on [−1,1]. Worst AUROC includes the interior minimum when present. Ordinary
AUROC uses the separate control head at ξ=0. For each Iterative fit, average
participant curves and minimize first; then average the three minima.
Bootstrap intervals use 20,000 common participant draws, seed 1101, across
methods, tasks and fits, repeating minimization within every draw. They are
paired percentile intervals conditional on fitted coordinates, maps and heads,
without multiplicity correction. Previously exposed cohorts are not an
independent confirmation set. Fit seeds do not add participants.

Saved-model replay, statistic reconstruction and standalone execution checks
are documented in [VALIDATION.md](../../VALIDATION.md). These checks do not
establish equivalence of a complete fresh raw-data-to-paper training run.
