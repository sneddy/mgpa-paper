# Obtaining the data and frozen encoder

You do **not** need EEG recordings or model weights to rebuild the supplied
tables and figures. For that route, follow
[Rebuild the reported results](README.md#rebuild-the-reported-results-without-training).
The instructions below are for fresh runs from raw recordings.

The four experiments use **two source datasets and one EEGPT checkpoint**.
P300, N170 and MMN are tasks within the same Neuroscan/Flex collection, not
three separate datasets. ERP CORE, OpenBMI and other discovery datasets are
not required by this release.

| Experiment | Raw input | EEGPT needed? |
|---|---|---|
| [Controlled](experiments/controlled/README.md) | Wet recordings from the SSVEP collection; CAR/Oz views are constructed locally | Yes |
| [Recorded SSVEP](experiments/recorded_ssvep/README.md) | Wet and dry recordings from that same collection | Yes |
| [Temporal N170](experiments/temporal_n170/README.md) | Simultaneous Neuroscan/Flex N170 | No; fixed temporal features |
| [Task reuse](experiments/task_reuse/README.md) | Simultaneous Neuroscan/Flex P300, N170 and MMN | Yes |

## Suggested local layout

Run commands in this guide from the repository root. The root `data/` directory
is ignored by Git; recordings and weights must not be committed. An external
disk works equally well if you supply its absolute paths in the configurations.

```text
data/
  ssvep/
    S001.mat ... S102.mat
    Subjects_Information.mat
    Impedance.mat                 # optional provider metadata
    Readme.pdf                    # provider documentation
    stimulation_information.pdf  # provider documentation
  flex_neuroscan/
    raw/
      Saline Raw Data/
        N170/
          behav/
          eeg/emotiv/
          eeg/neuroscan/
        oddball_active/           # P300; same three subdirectories
        oddball_passive/          # MMN; same three subdirectories
      scripts/                   # optional provider code, not executed
  weights/
    model.safetensors
```

Published source-file metadata and our existing download inventory give these
payload sizes (decimal GB; ZIP sizes may differ):

| Input | Size and scope |
|---|---|
| SSVEP v4 | 973,933,622 bytes, about 0.97 GB; 106 files including documentation |
| Flex selected collection | About 4.71 GB for all three ERP paradigms and the optional author scripts; the exact minimum consumed by both published protocols is about 4.05 GB / 300 files |
| EEGPT | 101,160,080 bytes, about 101 MB |

The convenient full selections above occupy about 5.8 GB before generated
outputs. Keep additional room for download archives if retained, and separately
for feature caches, model states and evaluation outputs. These payload sizes
are **not** a bound on the storage needed for complete experimental runs.

<a id="ssvep"></a>
## 1. Wet/dry SSVEP

Source: Zhu et al., *An Open Dataset for Wearable SSVEP-Based Brain-Computer
Interfaces*, [Figshare record 13560281, version 4](https://doi.org/10.6084/m9.figshare.13560281.v4).
The [provider record](https://figshare.com/articles/dataset/An_Open_Dataset_for_Wearable_SSVEP-Based_Brain-Computer_Interfaces/13560281)
declares **CC BY 4.0**; cite the dataset and retain its attribution.

### Download and unpack

1. Open the Figshare record, select **version 4**, and choose **Download all**.
   Individual file downloads are also available. Public downloads do not require
   credentials.
2. Extract the files into `data/ssvep/`. If the ZIP creates an extra enclosing
   folder, use the directory that directly contains the `.mat` files as
   `raw_root`.
3. Keep **all `S001.mat` through `S102.mat` and `Subjects_Information.mat`**.
   Both raw-input runners require this complete inventory, even though the
   controlled experiment constructs its views from wet trials only. The other
   three files are useful provider documentation/metadata, not additional
   required recordings.

Each participant's `data` tensor has shape `(8, 710, 2, 10, 12)`, including both
electrode conditions. Do not split, rename, rereference or pre-filter these
files before running this package. The two experiments perform their own
different preprocessing/calibration steps.

The [versioned Figshare metadata](https://api.figshare.com/v2/articles/13560281/versions/4)
lists every file's `download_url`, byte size and `computed_md5`, if you need to
verify a download against the provider rather than only checking its structure.

<a id="flex-neuroscan"></a>
## 2. Simultaneous Neuroscan/Flex recordings

Source: [A Validation of Emotiv Flex, OSF project `zj3f5`](https://osf.io/zj3f5/),
specifically **Emotiv EPOC Flex Saline Validation**. See
[Williams et al. (2020), PeerJ 8:e9713](https://doi.org/10.7717/peerj.9713)
for the collection and the authors' data-availability statement.

### Download the three task folders

1. In the OSF file browser open **Emotiv EPOC Flex Saline Validation**, then
   **Saline Raw Data**. Select `N170`, `oddball_active` (P300), and
   `oddball_passive` (MMN). Download the folders or their individual files.
2. Preserve each task's `behav`, `eeg/emotiv` and `eeg/neuroscan` subdirectories,
   including all companion files. Place the three task folders under
   `data/flex_neuroscan/raw/Saline Raw Data/` as shown above. Parentheses naming
   P300/MMN are explanations, not part of the loader's folder names.
3. The author's `scripts` folder is optional background material. Do not run it
   as a preprocessing prerequisite: this package implements the paper's own
   fixed preprocessing. Resting-state, the separate Flex SSVEP task, processed
   derivatives and the Gel Validation collection are unnecessary.

For every required participant/task, retain:

- `behav/<participant>_*.csv` (event/behavior records);
- `eeg/emotiv/<participant>_*.edf`;
- `eeg/neuroscan/<participant>_*.dat` **and the same-stem `.dap`, `.rs3`, `.ceo`**.

Keep original filenames. There must be exactly one matching EEG/behavior file
for a participant in each required branch; duplicate exports cause an explicit
error. The Neuroscan sidecars are essential, not optional documentation.

Downloading the complete three folders is simplest. If downloading only the
files actually read by the final protocols, use this inventory:

| Protocol/task | Required participants |
|---|---|
| Temporal N170 | 1013–1034, excluding 1015 and 1017 (20 people) |
| Reuse P300 | 1013, 1014, 1016, 1018, 1019, 1020, 1022, 1023, 1024, 1025, 1026–1033 (18 people) |
| Reuse N170 and MMN, each | 1019, 1020, 1024, 1025, 1026–1033 (12 people) |

N170 files are shared by the two protocols; do not download duplicate copies.
The 300-file minimum and its approximately 4.05 GB size are deduplicated across
these rows, based on our verified source inventory. Available subjects differ
by task; do not replace the fixed roles with a new complete-case intersection.

**Path convention:** both Flex runners expect the **collection root**
`data/flex_neuroscan`, which contains `raw/Saline Raw Data/`. Do not pass its
`raw/` child. In contrast, SSVEP's `raw_root` directly contains the `.mat` files.

The [OSF project metadata](https://api.osf.io/v2/nodes/zj3f5/) is public but did
not specify a node-level data license when checked on 2026-09-18. Consult the
provider's current notices, or ask the authors if your intended reuse requires
clarification. This repository's MIT license does not license these recordings;
we do not redistribute them.

<a id="eegpt"></a>
## 3. The exact frozen EEGPT checkpoint

Use the Braindecode-compatible
[`braindecode/eegpt-pretrained`](https://huggingface.co/braindecode/eegpt-pretrained)
checkpoint, not a different original-training checkpoint or the moving `main`
revision. The [provider model card](https://huggingface.co/braindecode/eegpt-pretrained/blob/main/README.md)
declares BSD-3-Clause. Model weights remain separate from this repository.

- Revision: `e41cb3ae2ce4fd9eb736862292c91f8128d15618`
- File: `model.safetensors`
- Size: `101160080` bytes
- SHA-256: `fb34c20609983324679b9534f9d17a2289a232d0276dd2b8b7d5b088f876f621`

Download once and reuse the same file for controlled, recorded SSVEP and task
reuse. Temporal N170 does not load EEGPT.

```sh
mkdir -p data/weights
curl --fail --location --retry 3 \
  'https://huggingface.co/braindecode/eegpt-pretrained/resolve/e41cb3ae2ce4fd9eb736862292c91f8128d15618/model.safetensors' \
  --output data/weights/model.safetensors
```

The checksum identifies the actual model bytes, not a Git LFS pointer file.
The loaders reject a mismatched checkpoint. No training or implicit weight
download occurs during installation.

## Check the inputs before a long run

Install the package and optional EEG dependencies first:

```sh
python -m pip install -e '.[raw]'
```

With the suggested layout, this checks SSVEP metadata/tensor headers, all
required Flex file paths, and the checkpoint checksum. It does not filter EEG,
encode recordings, train a model or validate cross-device event alignment;
the latter checks happen during preparation.

```sh
python - <<'PY'
from pathlib import Path
from mgpa.data.raw.ssvep_data import load_subject_metadata, audit_raw_files
from mgpa.data.raw.flex_ingest import source_files
from mgpa.data.reuse import FIT, HEAD, DEV, EVAL, sha256, CHECKPOINT_SHA256

ssvep = Path('data/ssvep')
assert len(load_subject_metadata(ssvep)) == 102
assert len(audit_raw_files(ssvep, check_finite=False)) == 102

flex = Path('data/flex_neuroscan')
temporal = {str(p) for p in range(1013, 1035)} - {'1015', '1017'}
records = {(p, 'n170') for p in temporal}
records.update((p, 'p300') for p in (*FIT, *HEAD, *DEV, *EVAL))
records.update((p, task) for p in (*HEAD, *EVAL) for task in ('n170', 'mmn'))
files = {path for person, task in sorted(records)
         for path in source_files(flex, person, task).values()}
assert len(files) == 300
assert all(path.is_file() for path in files)

checkpoint = Path('data/weights/model.safetensors')
assert checkpoint.stat().st_size == 101160080
assert sha256(checkpoint) == CHECKPOINT_SHA256
print('PASS: SSVEP headers/metadata, 300 Flex files, and EEGPT checksum')
PY
```

<a id="run-from-raw-data"></a>
## Connect the inputs to the experiment runners

For controlled, temporal N170 and recorded SSVEP, copy that experiment's
`configs/raw.example.toml` to `configs/paths.local.toml` and replace the paths
with **absolute paths** to your inputs. Local path files are ignored by Git.
Do not leave the default prepared-data example in place when starting from raw
EEG. For example:

```sh
cp experiments/controlled/configs/raw.example.toml \
   experiments/controlled/configs/paths.local.toml
# Edit raw_root and checkpoint in paths.local.toml, then:
python experiments/controlled/run.py prepare --run-id reproduction_v1 \
  --paths experiments/controlled/configs/paths.local.toml
```

| Runner | `raw_root` / `--raw-root` value in the suggested layout | Other input |
|---|---|---|
| `experiments/controlled/run.py` | Absolute path to `data/ssvep` | `checkpoint`, `device` in TOML |
| `experiments/recorded_ssvep/run.py` | Absolute path to `data/ssvep` | `checkpoint`, `device` in TOML |
| `experiments/temporal_n170/run.py` | Absolute path to `data/flex_neuroscan` | No checkpoint |
| `experiments/task_reuse/run.py` | Absolute path to `data/flex_neuroscan`, via `--raw-root` | Absolute checkpoint path, via `--checkpoint` |

For reuse, the raw-data entry point is:

```sh
python experiments/task_reuse/run.py prepare \
  --raw-root "$PWD/data/flex_neuroscan" \
  --checkpoint "$PWD/data/weights/model.safetensors" \
  --run-dir experiments/task_reuse/artifacts/runs/reproduction_v1
```

These commands only prepare inputs. Continue with the `fit`, `evaluate`,
`report`, and `verify` phases in each experiment's README, or use its `all`
entry point. Keep generated outputs outside the raw input folders. In
particular, do not apply temporal N170's amplitude exclusion to task reuse;
each runner handles its own published event/QC policy.

If you already have compatible prepared inputs, use the alternative schemas in
[the prepared-data guide](mgpa/data/README.md) or
[the reuse input specification](experiments/task_reuse/DATA.md). Those files are
formats for locally prepared inputs, not a separately hosted feature dataset.
Raw preparation records input hashes and versions; exact floating-point feature
parity across encoder backends is not assumed. See [VALIDATION.md](VALIDATION.md)
for the checks performed on this release.
