# Data preparation API

The component experiments use four participant-disjoint roles:

- `fit`: coordinates, calibration gate, adapter fitting;
- `val`: source-only critic checks or prespecified movement endpoints;
- `head`: independent source readers and downstream task heads;
- `eval`: held-out participant scoring.

FIT/VAL archives contain no `y` or `y_*` arrays. A task label can help define
the fixed controlled source assignment upstream, but is never passed to an
adapter. A prepared manifest records each role's participants and SHA256.

## Portable inputs

`load_prepared(manifest, roles=('fit', 'val'))` opens exactly those archives,
not HEAD/EVAL. A portable bundle has relative, contained paths:

```text
manifest.json
coordinate_state.npz  # center, scale, basis; FIT-only
fit.npz
val.npz
head.npz
eval.npz
```

Paired studies store `x[event,2,feature]`, Unicode `subject` and `event_id`.
Controlled SSVEP additionally stores the assigned `x_observed[event,512]`,
binary source `u`, and full-reference `x_calibration` in FIT/VAL. Its primary
task labels in HEAD/EVAL are `y=y_binary` and `y_multiclass` is retained as
metadata. Temporal N170 uses two 240-coordinate vectors and binary face/watch
`y`; sources are event-major Neuroscan=0/Flex=1.

Recorded SSVEP stores `x_observed[recording,15,512]`, dry=0/wet=1 `u`, matching
observed metadata aliases, and twelve-class `y=y_multiclass` only for readers.
FIT/VAL additionally contain `calibration_reference[N,15,512]`,
`calibration_alternatives[N,6,15,512]`, their participant/event identifiers,
and the six ordered view names. These are virtual same-trial contrasts,
not physical wet/dry trial pairs. `interfaces.recorded_calibration_pairs`
orients them consistently for Closed-form MGPA.

## Three explicit acquisition routes

1. `mode='prepared'`: verify and relocate a supplied coordinate bundle without
   refitting coordinates. The source manifest needs an explicit SHA256.
2. `mode='frozen_features'`: reconstruct FIT-only coordinates and gates from
   the supplied frozen feature grids and participant metadata. Input hashes
   and exact split settings are in `feature_protocols.py`; no encoder runs.
3. `mode='raw'` in an experiment paths file: ingest local raw EEG, optionally
   run the separately acquired frozen EEGPT checkpoint, record a portable
   acquisition receipt, then construct the same role-separated coordinates.
   The low-level `prepare_context` equivalent is `mode='raw_receipt'` with
   a receipt path and SHA256.

Raw EEG and checkpoint weights are never downloaded implicitly. Optional
dependencies are imported only when raw preparation is requested. The raw
submodule has no adapter fitting or source-reader code. Fresh encoding records
backend/version hashes and does not promise bitwise equality across hardware.

`runner.py` implements the prepare, fit, evaluate, report and verify phases
shared by controlled SSVEP, temporal N170 and recorded SSVEP. Task reuse uses
its own CAL/HEAD/DEV protocol in `reuse.py`; see the
[task-reuse guide](../../experiments/task_reuse/README.md).
