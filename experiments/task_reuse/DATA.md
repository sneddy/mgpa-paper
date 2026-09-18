# Portable inputs

For official downloads and required files, see the
[Flex–Neuroscan data guide](../../DATASETS.md#flex-neuroscan) and
[EEGPT checkpoint guide](../../DATASETS.md#eegpt). This page describes the input
manifest; [raw-input setup commands](../../DATASETS.md#run-from-raw-data) are in
the download guide.

`--raw-root` builds a manifest automatically and prepares original recordings
lazily through the shared raw-data loader. To reuse prepared epochs or frozen
EEGPT tokens, create an explicit JSON manifest. Relative paths resolve against
the manifest's own directory, independently of the working directory.

```json
{
  "schema": "mgpa-task-reuse-inputs-v1",
  "source_order": ["neuroscan", "flex"],
  "normalization": "fixed_20uv",
  "checkpoint_sha256": "fb34c20609983324679b9534f9d17a2289a232d0276dd2b8b7d5b088f876f621",
  "records": [
    {
      "person": "1013",
      "task": "p300",
      "metadata": "metadata/1013_p300.npz",
      "tokens": "tokens/1013_p300.npy"
    }
  ]
}
```

Include every required participant/task record: P300 FIT/CAL, HEAD, DEV and
EVAL; N170/MMN HEAD and EVAL only. `prepare` opens only P300 non-EVAL records.
`evaluate` opens remaining records after authenticating the P300 lock.

Each metadata NPZ must contain:

- `y`: binary task labels, shape `[event]`;
- `event_id` (or `ordinal`): strictly increasing integer event ordinals;
- `onset_s`: original physical event onset seconds, shape `[event,2]`, ordered
  Neuroscan then Flex. Instead, the manifest record may contain `acquisition`
  pointing to a raw-loader JSON with `event_audit[source].event_ids` and
  `event_times_physical_s`;
- optional `subject` and `source_order`, validated when present;
- `channels`: ordered channel names when raw epochs are encoded.

Frozen token NPY files have shape `[event,2,7,4,512]`. Alternatively, `features`
may name an NPZ whose `x` array has that shape or `[event,2,14336]`; set
`feature_key` for another array name. These are full frozen encoder outputs,
not a prior PCA or a token average. The manifest's normalization/checkpoint
contract is required. The release fits its own P300-only PCA256.

For paired raw-epoch input, replace `tokens` with `epochs`, pointing to an NPZ
with `x[event,2,16,128]` in volts. Supply `--checkpoint`, or a `checkpoint` path
in the manifest. The shared raw loader's NPZ can serve as both `metadata` and
`epochs`; its companion JSON serves as `acquisition`. Preserve all technically
valid paired events and do not apply the temporal N170 amplitude exclusion.

Each path can be accompanied by `<field>_sha256`, such as
`tokens_sha256`, `metadata_sha256`, `epochs_sha256` or `acquisition_sha256`.
Supplied hashes are checked, and hashes of every accessed file are retained in
run receipts. The frozen EEGPT checkpoint is always checked against the paper
hash. No automatic download or alternate encoder is selected.

Array input never uses pickle. Missing events, changed source order, invalid
labels/timestamps, different feature widths or unexpected FIT/CAL/DEV counts
fail before adapter fitting. The original input files remain read-only.
