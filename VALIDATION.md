# Validation

Validation covers saved-model replay, serialization, reported statistics and
standalone execution. It does not establish numerical equivalence of a complete
fresh raw-data-to-paper training run. No new adapter or reader fits were required
for the checks below.

## Numerical replay

The following saved states were compared with their reference computations
using two numerical threads. Every stated replay comparison had maximum
absolute error zero.

| Check | Scope |
| --- | --- |
| Closed-form MGPA | Published rotation-1 state on four controlled VAL rows |
| Iterative MGPA | All 48 controlled stages on the same four VAL rows |
| PCA edit-permission control | All 48 stages, conditioning on the fixed measurement complement |
| P300-selected Iterative MGPA | All 48 stages on four P300 endpoints in the saved PCA coordinates |
| IGBP | All 100 projections on four temporal N170 VAL rows |
| Target and stage calculations | Conditional, empirical FIT-mean and zero targets, with fixed and ordinary complements, on synthetic inputs |
| Serialization | Identical outputs after JSON and numerical-NPZ round trips for MGPA, baselines and the common-token-offset interface |

These are checks on the stated input slices, not full-dataset replay. Stage
fitting uses explicit isolated seeds: `seed + 20000 + 1009*k` for MGPA and
`seed + 1009*k` for IGBP. P300 reuse selects neural-critic epochs by source-VAL
binary cross-entropy with the configured patience. The linear-only control
uses half the two-critic damping.

## Data and reported statistics

All arrays, dtypes and coordinate states in the four controlled rotations,
temporal N170 and recorded SSVEP prepared contexts were unchanged by relocation.
FIT/VAL bundles contain no downstream task labels. Details are recorded in
[the data checks](verification/data_preparation_verification.json).

The paper verifier recomputes 107 full-precision statistics from saved
participant cells, including source-reader maxima, random-orientation averaging,
task scores, movement and paired bootstrap intervals. Maximum discrepancy
from the reference values is 2.22e-16. Reuse point estimates and intervals agree
within 3.61e-15, recomputing each fitted map's quadratic worst-association minimum
inside each participant-bootstrap draw.

Reuse DEV selection replays 28 candidates and 1,444 endpoints. It reproduces
the selected summaries exactly: Closed-form strength 1.1, Iterative stages
13/11/19 and the reported baseline settings. The
[selection receipt](experiments/task_reuse/artifacts/logs/dev_selection_verification.json)
records this check. It validates the selection calculation, not independence
of the cohorts used during protocol development.

Table and figure builders read saved numerical results only. The measurement-path
and theory illustrations in the paper are schematic figures supplied as static
assets. File hashes are recorded in
[release_manifest.json](release_manifest.json).

## Standalone execution

Wheel installation and imports of all 37 package modules passed. Execution from
a relocated directory, including a path containing spaces, passed launcher
help, the three component-study job plans, all four saved-result reports, reuse
DEV selection verification and the central statistic and figure builders.
[The standalone receipt](verification/standalone_verification.json) records these
checks.

Fresh-run reporting was also exercised on synthetic saved summaries, including
missing endpoints and percentile intervals that do not contain the point
estimate. These validate reporting behavior; they are not new scientific results.
