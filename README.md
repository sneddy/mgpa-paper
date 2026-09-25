# Measurement-Gated Provenance Attenuation (MGPA)

MGPA corrects a frozen representation only within directions supported by paired
measurement calibration, and anchors the correction to the conditional mean given
the preserved coordinates. This repository contains the two constructions used in
the paper, **Closed-form MGPA** (`mgpa_cf`) and **Iterative MGPA** (`mgpa_iter`),
the four experiments, and the participant-level results behind every reported number.

## Quick check: saved results (no EEG, no GPU, a few minutes)

Python 3.11 or 3.12 on macOS or Linux:

```sh
python -m venv .venv && source .venv/bin/activate
python -m pip install -e .

python paper_assets/verify.py    # PASS: 374 released files ... ; PASS: 107 full-precision statistics
python paper_assets/build.py     # rebuilds tables and experimental figures in paper_assets/generated/
```

`verify.py` checks file hashes against [release_manifest.json](release_manifest.json)
and recomputes the reported means and bootstrap intervals from the saved
participant-level results. Per-experiment reports:

```sh
python experiments/controlled/run.py report --published
python experiments/temporal_n170/run.py report --published
python experiments/recorded_ssvep/run.py report --published
python experiments/task_reuse/run.py report --published
```

## Where each result comes from

| Paper | Supplied file (`paper_assets/`) | Experiment and configuration |
|---|---|---|
| Figure 1, Figure 2 (schematics) | `measurement_paths_v2.png`, `mgpa_figure.png` | static illustrations |
| Table 1 | `controlled_cf_main.tex` | [controlled](experiments/controlled/README.md) · [paper.toml](experiments/controlled/configs/paper.toml) |
| Figure 3, Table 7 | `controlled_mechanisms_compact.pdf`, `controlled_target_controls.tex` | [controlled](experiments/controlled/README.md) · [paper.toml](experiments/controlled/configs/paper.toml) |
| Table 2A, Tables 8–9 | `real_native_main.tex`, `n170_critic_compact.tex`, `n170_igbp_budget.tex` | [temporal_n170](experiments/temporal_n170/README.md) · [paper.toml](experiments/temporal_n170/configs/paper.toml) |
| Table 2B | `real_native_main.tex` | [recorded_ssvep](experiments/recorded_ssvep/README.md) · [paper.toml](experiments/recorded_ssvep/configs/paper.toml) |
| Table 3, Figure 4, Tables 10–11 | `reuse_main.tex`, `reuse_worst_preview.pdf`, `reuse_absolute_compact.tex`, `reuse_paired_ci.tex` | [task_reuse](experiments/task_reuse/README.md) · [paper.json](experiments/task_reuse/configs/paper.json) |

Saved participant-level results are in each experiment's `artifacts/`; task reuse also
keeps every DEV candidate used for operating-point selection.

## Use MGPA on your own features

The classes take feature matrices, not raw EEG: FIT-standardized features `x`, source
labels (acquisition condition, not the task), an orthonormal measurement basis `B` of
shape `[features, rank]`, and calibration pairs of shape `[pairs, 2, features]`.

```python
from mgpa import ClosedFormMGPA, IterativeMGPA, save_model, load_model

analytic = ClosedFormMGPA().fit(x_fit, source_fit, B, calibration_pairs)
x_corrected = analytic.transform(x_new)

iterative = IterativeMGPA().fit(x_fit, source_fit, B, seed=17)
x_corrected = iterative.transform(x_new)

save_model(analytic, 'adapter'); restored = load_model('adapter')
```

The encoder stays frozen, and correction needs no source label at application.
Fit coordinates and the gate without downstream evaluation data. Models serialize
to JSON and NPZ.

## Re-run from raw data

Recordings, EEGPT weights and fitted checkpoints are not shipped. Obtain them as
described in [DATASETS.md](DATASETS.md) (see also [THIRD_PARTY.md](THIRD_PARTY.md)),
install the raw-data extras with `python -m pip install -e '.[raw]'`, then follow the
experiment's README (copy `configs/paths.example.toml` to `paths.local.toml` first;
the prepared-data API is in [mgpa/data/README.md](mgpa/data/README.md)). For example:

```sh
python experiments/controlled/run.py plan
python experiments/controlled/run.py all --run-id reproduction_v1 \
  --paths experiments/controlled/configs/paths.local.toml
```

New outputs go to `artifacts/runs/<run-id>/` and never overwrite the saved results.

## Scope

The saved-result checks, model replays and standalone execution are described in
[VALIDATION.md](VALIDATION.md); a complete fresh raw-data-to-paper run has not been
verified. Source-reader scores measure accessibility within the declared reader bank,
not complete erasure. Code is MIT-licensed ([LICENSE](LICENSE)); data and pretrained
weights remain under their providers' terms.
