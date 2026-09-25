"""Preparation boundary: data arrays only, never fitted correction maps/readers.

Two explicit modes are supported. ``prepared`` verifies and relocates existing
coordinate archives byte-for-byte. ``frozen_features`` re-estimates coordinates
and the gate from FIT features. Neither mode claims raw-EEG re-encoding.
Portable bundles use relative dependency paths and require only this package.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

ROLES = ("fit", "val", "head", "eval")
SCHEMA = "mgpa-release-prepared-v1"


def sha256(path):
    """Compute a streaming SHA256 digest for an explicit file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(*arrays):
    """Hash array shape, dtype, and contiguous bytes in order."""
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(json.dumps({"shape": array.shape, "dtype": array.dtype.str}, sort_keys=True).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def pinned_path(path, expected):
    """Resolve a supplied input only after its explicit SHA256 matches."""
    path = Path(path).expanduser().resolve()
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"Explicit SHA256 required for acquisition input {path}")
    if sha256(path) != expected:
        raise ValueError(f"SHA256 mismatch: {path}")
    return path


def pinned_json(path, expected):
    """Read JSON from a hash-authenticated input path."""
    return json.loads(pinned_path(path, expected).read_text())


def _resolve_dependency(root, value, *, legacy=False):
    """Resolve a contained portable dependency or explicit read-only input."""
    value = Path(value)
    if value.is_absolute():
        if legacy:
            return value
        raise ValueError("Portable manifest dependency must be relative")
    resolved = (root / value).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("Manifest dependency escapes bundle")
    return resolved


def _clean_provenance(value):
    """Keep acquisition hashes and names, not machine-specific absolute paths."""
    if isinstance(value, dict):
        return {Path(k).name if str(k).startswith("/") else k: _clean_provenance(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_provenance(v) for v in value]
    if isinstance(value, str) and value.startswith("/"):
        return Path(value).name
    return value


def validate_roles(datasets, *, complete=True):
    """Reject invalid representations, task-label exposure, or participant overlap."""
    if complete and set(datasets) != set(ROLES):
        raise ValueError("Exactly FIT, VAL, HEAD and EVAL are required")
    if not set(datasets).issubset(ROLES):
        raise ValueError("Unknown data role")
    seen, width = set(), None
    for role in ROLES:
        if role not in datasets:
            continue
        data = datasets[role]
        token = "x" not in data
        key = "x_observed" if token else "x"
        x = data[key]
        if x.ndim != 3 or not len(x) or x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError(f"Invalid {role} float32 primary features")
        if not token and x.shape[1] != 2:
            raise ValueError("Paired arrays must have exactly two sources")
        if width not in (None, x.shape[-1]):
            raise ValueError("Role feature widths disagree")
        width = x.shape[-1]
        for name in ("subject", "event_id"):
            if data[name].shape != (len(x),) or data[name].dtype.kind not in "US":
                raise ValueError("Metadata must be aligned non-object string arrays")
        people = set(data["subject"].tolist())
        if people & seen:
            raise ValueError("Participant leakage across roles")
        seen |= people
        if len(set(data["event_id"].tolist())) != len(x):
            raise ValueError("Duplicate global event ID")
        if role in ("fit", "val") and any(k == "y" or k.startswith("y_") for k in data):
            raise ValueError("Task labels forbidden in FIT/VAL")
        if role in ("head", "eval") and ("y" not in data or data["y"].shape != (len(x),)):
            raise ValueError("Reader task labels missing/misaligned")
        if "x_observed" in data:
            observed = data["x_observed"]
            if observed.dtype != np.float32 or not np.isfinite(observed).all():
                raise ValueError("Invalid observed features")
            if data["u"].shape != (len(observed),) or not set(data["u"].tolist()).issubset({0, 1}):
                raise ValueError("Observed source labels must be aligned binary labels")
            for name in ("subject_observed", "event_id_observed"):
                if data[name].shape != (len(observed),):
                    raise ValueError("Observed metadata misaligned")
            if not np.array_equal(data["subject_observed"], data["subject"]):
                raise ValueError("Observed participant identity differs")
        if token and role in ("fit", "val"):
            reference, alternatives = data["calibration_reference"], data["calibration_alternatives"]
            if reference.shape[1:] != x.shape[1:] or alternatives.shape != (len(reference), 6, *x.shape[1:]):
                raise ValueError("Recorded calibration requires six same-trial token views")
            if not set(data["calibration_subject"].tolist()).issubset(people):
                raise ValueError("Calibration participant leakage")
        if token and role in ("head", "eval") and "calibration_reference" in data:
            raise ValueError("No fabricated recorded HEAD/EVAL calibration pairs")
    return True


def _read_verified(root, entry, *, legacy=False):
    """Load non-pickle arrays from a hash-verified archive."""
    path = _resolve_dependency(root, entry["path"], legacy=legacy)
    if sha256(path) != entry["sha256"]:
        raise ValueError(f"Prepared archive SHA256 mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def load_prepared(manifest_path, roles=ROLES):
    """Return ``(role_arrays, coordinate_state, manifest)`` with verified hashes.

    ``roles=('fit','val')`` never opens HEAD/EVAL archives or their task labels.
    Legacy acquisition manifests are accepted read-only; release outputs are
    strictly relative, path-contained bundles.
    """
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "COMPLETE":
        raise ValueError("Preparation is not complete")
    selected = tuple(roles)
    if not selected or len(set(selected)) != len(selected) or not set(selected).issubset(ROLES):
        raise ValueError("Select unique known roles")
    legacy = manifest.get("schema") in ("mgpa-unified-context-v1", "mgpa-unified-context-token-v1")
    if not legacy and manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported prepared schema")
    records = {role: _read_verified(path.parent, manifest["roles"][role], legacy=legacy) for role in selected}
    validate_roles(records, complete=set(selected) == set(ROLES))
    state = _read_verified(path.parent, manifest["coordinate_state"], legacy=legacy)
    basis = state["basis"]
    if basis.ndim != 2 or not np.isfinite(basis).all() or not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=2e-5):
        raise ValueError("Gate basis must be finite and orthonormal")
    if state["center"].shape != (basis.shape[0],) or state["scale"].shape != (basis.shape[0],) or np.any(state["scale"] <= 0):
        raise ValueError("Invalid coordinate standardizer")
    role_people = [set(manifest["roles"][role]["subjects"]) for role in ROLES]
    if any(a & b for i, a in enumerate(role_people) for b in role_people[i+1:]):
        raise ValueError("Manifest participant leakage")
    return records, state, manifest


def _write_json(path, value):
    """Atomically finish a newly created preparation receipt."""
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _export_prepared(paths, context_id, destination):
    """Relocate verified role archives without changing their numerical bytes."""
    source = pinned_path(paths["source_manifest"], paths["source_manifest_sha256"])
    original = json.loads(source.read_text())
    actual_context = original.get("contract", {}).get("metadata", {}).get("instance_id", original["context_id"])
    aliases = {"temporal_n170": "flex_n170", "n170": "flex_n170", "recorded_ssvep": "ssvep_recorded_wet_dry__rotation1"}
    expected = aliases.get(context_id, context_id)
    if expected != actual_context:
        raise ValueError(f"Requested {expected}, source contains {actual_context}")
    destination = Path(destination).resolve()
    final = destination / "manifest.json"
    if final.exists():
        _, _, previous = load_prepared(final, roles=("fit",))
        if previous.get("acquisition", {}).get("manifest_sha256") != paths["source_manifest_sha256"]:
            raise FileExistsError("Destination belongs to a different acquisition")
        # Verify all archives without loading their values.
        for name, digest in previous["outputs"].items():
            if sha256(_resolve_dependency(destination, name)) != digest:
                raise ValueError("Existing prepared bundle changed")
        return final
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Preserve nonempty/incomplete preparation; choose a new destination")
    # Validate one role at a time to keep recorded token preparation memory bounded.
    for role in ROLES:
        load_prepared(source, roles=(role,))
    destination.mkdir(parents=True, exist_ok=True)
    manifest = _clean_provenance(deepcopy(original))
    manifest.update(schema=SCHEMA, context_id=expected,
                    acquisition={"mode": "prepared_coordinates", "manifest_name": source.name,
                                 "manifest_sha256": paths["source_manifest_sha256"],
                                 "raw_signal_reproduction": False}, outputs={})
    for key, entry in [(role, manifest["roles"][role]) for role in ROLES] + [("coordinate_state", manifest["coordinate_state"])]:
        old = original["coordinate_state"] if key == "coordinate_state" else original["roles"][key]
        incoming = _resolve_dependency(source.parent, old["path"], legacy=True)
        name = key + ".npz"
        temporary = destination / (name + ".partial")
        shutil.copyfile(incoming, temporary)
        if sha256(temporary) != old["sha256"]:
            raise ValueError("Copied archive differs from verified source")
        temporary.replace(destination / name)
        entry["path"] = name
        manifest["outputs"][name] = entry["sha256"]
    _write_json(final, manifest)
    return final


def write_prepared(context_id, destination, datasets, state, metadata):
    """Serialize fresh FIT-only coordinates; never overwrite a prior attempt."""
    validate_roles(datasets)
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Prepared destination must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": SCHEMA, "status": "COMPLETE", "context_id": context_id,
                "feature": metadata["feature"], "roles": {}, "outputs": {},
                "contract": {"metadata": _clean_provenance(metadata)},
                "acquisition": {"mode": "frozen_features", "raw_signal_reproduction": False},
                "scientific_maps_or_readers_fitted": False}
    for role, arrays in list(datasets.items()) + [("coordinate_state", state)]:
        name = role + ".npz"
        path = destination / name
        with path.with_suffix(".npz.partial").open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        path.with_suffix(".npz.partial").replace(path)
        record = {"path": name, "sha256": sha256(path)}
        manifest["outputs"][name] = record["sha256"]
        if role == "coordinate_state":
            record.update(rank=state["basis"].shape[1], basis_sha256=array_hash(state["basis"]))
            manifest[role] = record
        else:
            primary = "x" if "x" in arrays else "x_observed"
            record.update(shape=list(arrays[primary].shape), primary_array=primary,
                          subjects=sorted(set(arrays["subject"].tolist())), task_labels_present="y" in arrays)
            manifest["roles"][role] = record
    _write_json(destination / "manifest.json", manifest)
    return destination / "manifest.json"


def prepare_context(data_root_or_paths: dict, context_id: str, destination: Path):
    """Prepare one explicit context from verified local acquisition inputs.

    Prepared mode requires ``source_manifest`` and ``source_manifest_sha256``.
    Frozen-feature mode delegates to a self-contained builder. Paths are supplied
    by a local configuration, never inferred from this package's installation.
    """
    paths = dict(data_root_or_paths)
    if "manifests" in paths:
        paths["source_manifest"] = paths["manifests"][context_id]
        paths["source_manifest_sha256"] = paths["manifest_hashes"][context_id]
    mode = paths.get("mode", "prepared")
    if mode == "prepared":
        return _export_prepared(paths, context_id, destination)
    if mode == "raw_receipt":
        from .raw.receipt import prepare_generated_features
        return prepare_generated_features(paths["receipt"], context_id, destination,
                                          receipt_sha256=paths["receipt_sha256"])
    if mode == "frozen_features":
        from .features import prepare_features
        return prepare_features(paths, context_id, destination)
    raise ValueError("mode must be prepared, frozen_features, or raw_receipt")
