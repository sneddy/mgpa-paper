# Measurement-Gated Provenance Attenuation

MGPA corrects a frozen representation within directions supported by measurement
calibration. The retained coordinates guide the correction target. This release
contains the two constructions used in the manuscript:

- **Closed-form MGPA:** one analytic correction with a direction estimated from
  measurement pairs and a pooled conditional-mean target.
- **Iterative MGPA:** learned source critics, conditional score anchors, and
  bounded trust-region updates within a fixed measurement gate.

Both apply without encoder retraining or source identity. The experiments also
include the manuscript's component controls and Identity, LEACE, CORAL, FEATMAP
and IGBP comparisons. CORAL and FEATMAP require source identity at application.
The result identifiers for the two MGPA methods are `mgpa_cf` and `mgpa_iter`.

## Manuscript

The self-contained manuscript project is in [writing/](writing/README.md):
[LaTeX source](writing/merged_v5.tex) · [PDF](writing/merged_v5.pdf).
It includes all figures, table sources, bibliography, and local ICLR styles.
Compilation instructions are in [writing/README.md](writing/README.md).

## Reproduction guide

Choose one of two workflows after [installation](#install):

- **Saved results:** [rebuild and verify the reported statistics](#rebuild-the-reported-results-without-training)
  from the supplied participant-level cells, without downloading EEG or fitting
  models. The experimental tables and figures below use these saved results.
- **Fresh experiments:** obtain the inputs in [DATASETS.md](DATASETS.md), then
  follow [fresh numerical reproduction](#fresh-numerical-reproduction) and the
  relevant protocol guide and configuration. This separately prepares data,
  fits adapters/readers and evaluates new outputs.

Raw recordings, EEGPT weights and fitted adapter/head checkpoints are not shipped.
[VALIDATION.md](VALIDATION.md) describes the saved-model replay, statistic and
execution checks; a complete fresh raw-data-to-paper training run has not been
verified. The protocol guides specify participant roles, preprocessing, endpoint
selection and uncertainty; configurations fix the published numerical settings.

| Protocol guide | Published configuration |
|---|---|
| [Controlled SSVEP](experiments/controlled/README.md) | [paper.toml](experiments/controlled/configs/paper.toml) |
| [Temporal N170](experiments/temporal_n170/README.md) | [paper.toml](experiments/temporal_n170/configs/paper.toml) |
| [Recorded SSVEP](experiments/recorded_ssvep/README.md) | [paper.toml](experiments/recorded_ssvep/configs/paper.toml) |
| [P300-selected task reuse](experiments/task_reuse/README.md) | [paper.json](experiments/task_reuse/configs/paper.json) |

### Paper-to-artifact map

These are the eight external tables and four figures included in the paper.
Saved participant cells are under each experiment's `artifacts/`; task reuse
also retains DEV selection evidence. [release_manifest.json](release_manifest.json)
records their hashes.

| Paper content | Supplied artifacts |
|---|---|
| Measurement paths and construction | [Measurement-path illustration](paper_assets/measurement_paths.png); [theory figure](paper_assets/theory_story.pdf) |
| Controlled Closed-form MGPA | [Closed-form comparison table](paper_assets/controlled_cf_main.tex) |
| Controlled gate and target | [Gate/target figure](paper_assets/controlled_mechanisms_compact.pdf); [target-control table](paper_assets/controlled_target_controls.tex) |
| Native N170 and recorded SSVEP | [Native comparison table](paper_assets/real_native_main.tex) |
| N170 critic and matched-budget controls | [Critic comparison](paper_assets/n170_critic_compact.tex); [IGBP budget comparison](paper_assets/n170_igbp_budget.tex) |
| P300-selected reuse on N170/MMN | [Worst/control table](paper_assets/reuse_main.tex); [absolute outcomes](paper_assets/reuse_absolute_compact.tex); [paired gains](paper_assets/reuse_paired_ci.tex); [transfer figure](paper_assets/reuse_worst_preview.pdf) |

## Install

From this directory, using Python 3.11 or 3.12 on macOS or Linux:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
# Only when preparing raw EEG:
python -m pip install -e '.[raw]'
```

The dependency versions match the numerical environment used for the paper.
Experiment launchers use Unix file locks; Windows execution has not been validated.
EEG recordings and EEGPT weights are external inputs, not bundled or downloaded
implicitly. Their required formats, sources and preparation options are documented
in each experiment and in [THIRD_PARTY.md](THIRD_PARTY.md).
See [DATASETS.md](DATASETS.md) for official download links, required files and
directory layouts, and the EEGPT checkpoint.

## Use a correction in another study

The classes accept feature matrices, not raw EEG. Supply FIT-standardized
features and an orthonormal measurement basis `B` of shape `[features, rank]`.
Calibration pairs have shape `[pairs, 2, features]` in the same coordinates.
Coordinate fitting and gate estimation must exclude downstream evaluation data.

```python
from mgpa import ClosedFormMGPA, IterativeMGPA, save_model, load_model

analytic = ClosedFormMGPA().fit(x_fit, source_fit, B, calibration_pairs)
x_corrected = analytic.transform(x_new)

iterative = IterativeMGPA().fit(x_fit, source_fit, B, seed=17)
x_corrected = iterative.transform(x_new)

save_model(analytic, 'adapter')
restored = load_model('adapter')
```

`source_fit` labels the acquisition condition, not the downstream task.
The encoder remains frozen. Standardization is deliberately explicit: reuse
the same FIT-fitted coordinate transform for new observations. Methods expose
their fitted components, settings and training history. Serialization uses JSON
and numerical NPZ, not executable pickle objects. The optional
`TokenOffsetAdapter` applies a fitted pooled correction as a common token offset.

## Four experiments, four questions

| Directory | Question | Representation |
|---|---|---|
| [controlled](experiments/controlled/README.md) | Do measurement permissions and conditional targets explain the correction? | Controlled CAR/Oz SSVEP; frozen EEGPT |
| [temporal_n170](experiments/temporal_n170/README.md) | Does nonlinear correction help on real paired devices? | Neuroscan/Flex; 240 temporal features |
| [recorded_ssvep](experiments/recorded_ssvep/README.md) | Does source attenuation survive a full-input audit while retaining task utility? | Real unpaired wet/dry; 15×512 EEGPT tokens |
| [task_reuse](experiments/task_reuse/README.md) | Can a P300-selected correction be reused on N170 and MMN? | Neuroscan/Flex; frozen EEGPT and P300-only PCA256 |

Each directory contains `run.py`, experiment-specific code, configurations and
saved numerical results under `artifacts/`. Reusable methods, data preparation
and evaluation live in `mgpa/`. Configurations expose the published settings;
adapter fitting, HEAD/EVAL evaluation and reporting are separate phases.

## Rebuild the reported results without training

The saved participant-level cells include all reported fit repetitions and
source-reader scores. Reuse cells retain the coefficients needed to recompute
worst-association AUROC. The manuscript tables and figures are in `paper_assets/`.
These commands use the supplied results: no EEG recordings, feature downloads
or pretrained weights are needed.

```sh
python paper_assets/verify.py
python paper_assets/build.py
python experiments/controlled/run.py report --published
python experiments/temporal_n170/run.py report --published
python experiments/recorded_ssvep/run.py report --published
python experiments/task_reuse/run.py report --published
```

`build.py` writes tables, CSVs, experimental figures and the theory illustration
to `paper_assets/generated/`; it does not overwrite the supplied manuscript assets.
Generated reports are local outputs and are not needed to run the package.
`verify.py` authenticates the released code, configurations and saved numerical
cells, then checks participant-derived means and intervals against the manuscript.
The reuse reporter recomputes the exact quadratic minima with the common
participant bootstrap. Repetitions and random orientations are not treated as
additional participants.

## Fresh numerical reproduction

For raw inputs, follow the [download and setup guide](DATASETS.md#run-from-raw-data).
First inspect the configuration and job plan. For the controlled study:

```sh
python experiments/controlled/run.py plan
# Create experiments/controlled/configs/paths.local.toml from an example.
python experiments/controlled/run.py all --run-id reproduction_v1 \
  --paths experiments/controlled/configs/paths.local.toml
```

The native N170 and recorded SSVEP runners use the same phases:
`prepare`, `fit`, `evaluate`, `report`, `verify`, or `all`.
The reuse README gives its raw-data and prepared-input invocation.
New outputs go under `artifacts/runs/<run-id>/`, apart from the frozen published
results. Reuse preserves every declared DEV candidate and the locked selection.
There is no implicit rerun, EVAL-based endpoint selection, or fallback replacing
an unattained movement budget with a native endpoint.

## Interpretation and validation

Participant roles are disjoint within each split, but these cohorts informed
protocol development. The controlled comparisons share participants, and the
two N170 protocols overlap. Reported intervals are conditional on fitted
coordinates, adapters and readers; they are not an independent confirmation
study. Source-reader scores measure accessibility within the declared reader
bank, not complete information erasure or anonymity.

[VALIDATION.md](VALIDATION.md) describes numerical replay, saved-result and
standalone-execution checks. A complete fresh raw-data-to-paper training run
has not been verified; the validation claims are limited to those checks.
[release_manifest.json](release_manifest.json) records file hashes and numerical
provenance. See [mgpa/data/README.md](mgpa/data/README.md) for the prepared-data API.

Code is distributed under the [MIT license](LICENSE). Data and pretrained weights
remain subject to their providers' terms; see [THIRD_PARTY.md](THIRD_PARTY.md).
