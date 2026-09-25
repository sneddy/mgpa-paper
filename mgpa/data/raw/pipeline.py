"""Raw EEG to frozen features, with explicit inputs and a portable receipt.

This optional stage is intentionally independent of correction-model fitting. Raw
data/checkpoints are not downloaded implicitly. Existing output attempts are
never overwritten; completed acquisitions are verified and reusable.
"""
from __future__ import annotations

from copy import deepcopy
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from mgpa.data import sha256
from mgpa.data.feature_protocols import CONTROLLED, FLEX, RECORDED_SSVEP
from mgpa.data.prepared import _write_json, pinned_path
from .ssvep_encoder import CHECKPOINT_SHA256


def _save_npz(path, arrays):
    """Atomically finish a new numeric feature archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".npz.partial").open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    path.with_suffix(".npz.partial").replace(path)


def _versions():
    """Record installed optional acquisition dependency versions."""
    versions = {}
    for name in ("numpy", "scipy", "pandas", "scikit-learn", "mne", "torch", "braindecode", "safetensors", "curryreader"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def source_hashes():
    """Bind the local acquisition and preparation implementation."""
    root = Path(__file__).parent.parent
    return {str(path.relative_to(root)): sha256(path) for path in sorted(root.rglob("*.py"))
            if not path.name.startswith("reuse")}


def _flex_people(dataset):
    """Return every participant in the fixed temporal N170 roles."""
    context = next(row for row in FLEX["contexts"] if row["id"].startswith("flex_"+dataset+"_"))
    return sorted({person for people in context["roles"].values() for person in people})


def _source_inventory(dataset, raw_root):
    """Hash the explicit raw data dependencies before acquisition."""
    if dataset in ("controlled", "recorded_ssvep"):
        paths = [raw_root/"Subjects_Information.mat", *[raw_root/f"S{person:03d}.mat" for person in range(1, 103)]]
    else:
        from .flex_ingest import source_files
        tasks = [(person, dataset) for person in _flex_people(dataset)]
        paths = sorted({path for person, task in tasks for path in source_files(raw_root, person, task).values()})
    return {str(path.relative_to(raw_root)): sha256(path) for path in paths}


def _ssvep(dataset, raw_root, output, checkpoint, device, rotations):
    """Encode fixed SSVEP views and optional FIT-local calibration transports."""
    from .ssvep_data import build_trial_index, deterministic_subsample, load_subject_metadata, load_trials, make_outer_splits
    from .ssvep_encoder import FrozenEEGPTEncoder, EMBED_DIM, N_PATCHES
    from .ssvep_references import load_or_create_reference_grid
    from .ssvep_calibration import load_or_create_view_bundle
    from mgpa.data.features import METADATA_SHA

    index = build_trial_index(raw_root)
    columns = ["sample_id", "subject", "gender", "first_electrode", "source", "block", "target", "frequency_binary", "phase_class"]
    metadata = output/"metadata.csv"
    index[columns].to_csv(metadata, index=False, lineterminator="\n")
    if sha256(metadata) != METADATA_SHA:
        raise ValueError("Raw source metadata differs from the fixed SSVEP cohort/order")
    encoder = FrozenEEGPTEncoder(checkpoint, device=device)
    if dataset == "controlled":
        feature_path = output/"features/eegpt_exact_reference_grid_v1.npy"
        load_or_create_reference_grid(cache_path=feature_path, index=index, encoder=encoder)
        return {"metadata": "metadata.csv", "feature_grid": str(feature_path.relative_to(output))}, {}, encoder.manifest
    feature_path = output/"features/eegpt_patch_mean_grid_v1.npy"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    grid = np.lib.format.open_memmap(feature_path, mode="w+", dtype=np.float16, shape=(len(index), N_PATCHES*EMBED_DIM))
    for subject, rows in index.groupby("subject", sort=True):
        grid[rows.index.to_numpy()] = encoder.encode_patch_mean_grid(load_trials(rows)).astype(np.float16)
        grid.flush()
        print(f"Encoded recorded SSVEP S{int(subject):03d}/102", flush=True)
    sequence = grid.reshape(len(index), N_PATCHES, EMBED_DIM)
    seed = RECORDED_SSVEP["seed"]
    splits = make_outer_splits(load_subject_metadata(raw_root), seed=seed)
    calibration_paths = {}
    for rotation in rotations:
        split = splits[rotation]
        fit = index[index["subject"].isin(split.adapter_fit_subjects)].copy()
        val = index[index["subject"].isin(split.adapter_validation_subjects)].copy()
        partitions = {"tangent_fit": deterministic_subsample(fit, 1024, seed=seed+100*rotation+1, group_columns=("subject", "source")),
                      "validation_geometry": deterministic_subsample(val, 512, seed=seed+100*rotation+2, group_columns=("subject", "source"))}
        path = output/f"features/fold_{rotation}_selection_v1.npz"
        load_or_create_view_bundle(cache_path=path, raw_sequence=sequence, transport_fit_metadata=fit,
                                   partitions=partitions, encoder=encoder, seed=seed+100*rotation+3)
        calibration_paths[str(rotation)] = {"calibration_features": str(path.relative_to(output)),
                                            "calibration_manifest": str(path.with_suffix(".json").relative_to(output))}
    return {"metadata": "metadata.csv", "feature_grid": str(feature_path.relative_to(output))}, calibration_paths, encoder.manifest


def _flex(dataset, raw_root, output, checkpoint, device):
    """Temporal N170: exact epochs and fixed 50-ms means, no encoder or PCA."""
    from .flex_ingest import ingest_one
    from .flex_gap import ingest as ingest_gap
    raw_output, feature_output = output/"epochs", output/"features"
    for person in _flex_people("n170"):
        if person == "1034":
            ingest_gap(raw_root, raw_output)
        else:
            ingest_one(raw_root, raw_output, person, "n170", "retain_flagged")
        path = raw_output/f"{person}_n170.npz"
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        arrays = {key: value for key, value in data.items() if key != "x"}
        arrays["raw_timebins"] = raw_timebins(data["x"], data["times"])
        destination = feature_output/path.name
        _save_npz(destination, arrays)
        _write_json(destination.with_suffix(".json"), {"status": "COMPLETE",
            "raw_epoch_sha256": sha256(path), "output_sha256": sha256(destination),
            "feature": "raw_timebins", "labels": "copied metadata, never used in coordinates"})
        print(f"Prepared temporal N170 participant {person}", flush=True)
    return {"feature_directory": "features"}, {}, {"encoder": None, "representation": "raw 50ms timebins"}


def raw_timebins(x, times):
    """Fixed 50-ms bins over 0–750 ms, in microvolts."""
    times = np.asarray(times)
    edges = np.arange(0., .750001, .05)
    blocks = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        keep = (times >= lo - 1e-8) & (times < hi - 1e-8)
        if not keep.any():
            raise ValueError("Empty temporal N170 bin")
        blocks.append(np.asarray(x)[..., keep].mean(-1))
    return (np.stack(blocks, axis=-1).reshape(*x.shape[:-2], -1) * 1e6).astype(np.float32)


def acquire(dataset, raw_root, output, *, checkpoint=None, device="cpu", rotations=None):
    """Create one new acquisition attempt and return its complete receipt path."""
    dataset = {"temporal_n170": "n170"}.get(dataset, dataset)
    if dataset not in ("controlled", "recorded_ssvep", "n170"):
        raise ValueError("Unknown acquisition dataset")
    raw_root, output = Path(raw_root).resolve(), Path(output).resolve()
    if output == raw_root or output.is_relative_to(raw_root):
        raise ValueError("Keep generated artifacts outside the raw input directory")
    if dataset != "n170":
        checkpoint = pinned_path(checkpoint, CHECKPOINT_SHA256)
    rotations = tuple(rotations if rotations is not None else ((1, 2, 3, 4) if dataset == "controlled" else (1,)))
    if not rotations or not set(rotations).issubset(range(5)) or len(set(rotations)) != len(rotations):
        raise ValueError("Unique rotation indices in 0..4 required")
    receipt_path = output/"acquisition.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["dataset"] != dataset or receipt.get("status") != "COMPLETE":
            raise FileExistsError("Existing acquisition has a different or incomplete contract")
        if (receipt.get("rotations") != list(rotations)
                or receipt.get("device") != (device if dataset != "n170" else None)
                or receipt.get("source_hashes") != source_hashes()):
            raise FileExistsError("Existing acquisition has different rotations/backend/source code")
        if any(sha256(output/name) != digest for name, digest in receipt["outputs"].items()):
            raise ValueError("Existing acquisition output changed")
        return receipt_path
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserve incomplete acquisition; use a new output directory")
    inputs = _source_inventory(dataset, raw_root)
    output.mkdir(parents=True, exist_ok=True)
    contract = {"dataset": dataset, "raw_inputs": inputs, "device": device if dataset != "n170" else None,
                "checkpoint_sha256": CHECKPOINT_SHA256 if dataset != "n170" else None,
                "source_hashes": source_hashes(), "versions": _versions(), "rotations": list(rotations)}
    _write_json(output/"started.json", {**contract, "status": "PREPARING"})
    if dataset in ("controlled", "recorded_ssvep"):
        coordinate_inputs, calibrations, encoder = _ssvep(dataset, raw_root, output, checkpoint, device, rotations)
    else:
        coordinate_inputs, calibrations, encoder = _flex(dataset, raw_root, output, checkpoint, device)
    if _source_inventory(dataset, raw_root) != inputs:
        raise ValueError("Raw acquisition inputs changed during preparation")
    if checkpoint is not None and sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Frozen checkpoint changed during preparation")
    if source_hashes() != contract["source_hashes"]:
        raise ValueError("Acquisition code changed during preparation")
    outputs = {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*")) if path.is_file()}
    receipt = {**contract, "schema": "mgpa-raw-acquisition-v1", "status": "COMPLETE", "outputs": outputs,
               "inputs_for_coordinates": coordinate_inputs, "recorded_calibrations": calibrations,
               "encoder": encoder, "raw_signal_reproduction": True, "bitwise_feature_parity_claimed": False}
    _write_json(receipt_path, receipt)
    return receipt_path
