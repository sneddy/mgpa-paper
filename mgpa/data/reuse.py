"""Portable input preparation for P300-selected cross-task correction.

The manifest names paired epochs or frozen full EEGPT tokens. Coordinates are
fitted on source-0 P300 observed FIT only. EVAL is opened only by an explicit
post-selection request. No paths into the research workspace are required.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

FIT = ("1013", "1014", "1016", "1018")
HEAD = ("1019", "1020", "1024", "1025")
DEV = ("1022", "1023")
EVAL = tuple(str(p) for p in range(1026, 1034))
TASKS = ("p300", "n170", "mmn")
SOURCE_ORDER = ("neuroscan", "flex")
STATE_KEYS = ("pca_mean", "pca_components", "pca_explained_variance_ratio", "center", "scale")
CHECKPOINT_SHA256 = "fb34c20609983324679b9534f9d17a2289a232d0276dd2b8b7d5b088f876f621"


def sha256(path):
    """Hash a file incrementally without loading its full contents into memory."""
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    """Read a JSON manifest or numerical receipt from an explicit path."""
    return json.loads(Path(path).read_text())


def write_json(path, value):
    """Create an immutable JSON artifact or confirm identical existing content."""
    path = Path(path)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(f"Refusing changed artifact: {path}")
        return
    with path.open("x") as stream:
        stream.write(text)


def save_arrays(path, arrays):
    """Save numeric arrays without pickle and reject conflicting existing data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as previous:
            if set(previous.files) != set(arrays) or any(not np.array_equal(previous[k], v) for k, v in arrays.items()):
                raise FileExistsError(f"Refusing changed arrays: {path}")
        return
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def load_arrays(path):
    """Load independent array copies from a non-pickle NPZ archive."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def subset(record, mask):
    """Copy the same event subset across every field in a paired record."""
    return {key: value[mask].copy() for key, value in record.items()}


def concatenate(records):
    """Combine compatible participant records while rejecting duplicate events."""
    if not records or any(set(record) != set(records[0]) for record in records):
        raise ValueError("No records or incompatible record fields")
    result = {key: np.concatenate([r[key] for r in records]) for key in records[0]}
    if len(np.unique(result["event_id"])) != len(result["event_id"]):
        raise ValueError("Duplicate event/pair identities")
    return result


def split_calibration(record, count=100):
    """First contiguous ordinal block; purge any 1s epoch overlap, either device."""
    if len(np.unique(record["subject"])) != 1 or len(np.unique(record["task_ids"])) != 1:
        raise ValueError("Split exactly one participant and one task at a time")
    ids = np.asarray(record["ordinal"])
    times = np.asarray(record["onset_s"], dtype=np.float64)
    if (len(ids) <= count or count < 1 or times.shape != (len(ids), 2) or not np.isfinite(times).all()
            or np.any(np.diff(ids) <= 0) or np.any(np.diff(times, axis=0) <= 0)):
        raise ValueError("Invalid count, ordered event IDs, or device timestamps")
    if not np.array_equal(ids[:count], np.arange(1, count + 1)):
        raise ValueError("Canonical first100 contiguous event block is not available")
    is_cal = np.arange(len(ids)) < count
    overlap = (np.abs(times[:, None, :] - times[None, :count, :]) < 1.).any(axis=(1, 2))
    keep = ~is_cal & ~overlap
    if not keep.any():
        raise ValueError("No observed FIT remains after overlap purge")
    audit = dict(cal_event_ids=record["event_id"][is_cal].tolist(),
                 fit_event_ids=record["event_id"][keep].tolist(),
                 purged_event_ids=record["event_id"][~is_cal & overlap].tolist(),
                 epoch_interval_seconds=[-.2, .8],
                 overlap_rule="strict temporal overlap in either device", labels_or_quality_used=False)
    return subset(record, keep), subset(record, is_cal), audit


def fit_coordinates(fit, leading_task="p300", *, pca_rank=256):
    """Exact full-SVD PCA and featurewise standardization on source0 observed FIT."""
    from sklearn.decomposition import PCA
    if set(fit["subject"]) != {f"flex:{p}" for p in FIT} or set(fit["task_ids"]) != {leading_task}:
        raise ValueError("Coordinates require exactly P300 FIT participants")
    x = np.asarray(fit["x"][:, 0], dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all() or not 0 < pca_rank <= min(x.shape):
        raise ValueError("Infeasible coordinate fit; no silent rank reduction")
    pca = PCA(n_components=pca_rank, svd_solver="full")
    projected = pca.fit_transform(x)
    return dict(pca_mean=pca.mean_, pca_components=pca.components_,
                pca_explained_variance_ratio=pca.explained_variance_ratio_,
                center=projected.mean(0), scale=np.maximum(projected.std(0), 1e-8))


def transform_coordinates(state, values):
    """Apply the frozen P300 PCA and standardizer, returning float32 pairs."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite paired coordinates")
    projected = (values - state["pca_mean"]) @ state["pca_components"].T
    return ((projected - state["center"]) / state["scale"]).astype(np.float32)


def calibration_basis(pairs, rank=8):
    """Published normalized contrasts: nonzero, fifth-percentile filter, eigh."""
    delta = np.asarray(pairs[:, 1], np.float64) - pairs[:, 0]
    norms = np.linalg.norm(delta, axis=1)
    nonzero = norms > 1e-10
    if not nonzero.any():
        return np.empty((delta.shape[1], 0), np.float32), {"rank": 0}
    used = nonzero & (norms >= np.quantile(norms[nonzero], .05))
    directions = delta[used] / norms[used, None]
    eig, vec = np.linalg.eigh(directions.T @ directions)
    order = np.argsort(eig)[::-1]
    eig, vec = np.maximum(eig[order], 0), vec[:, order]
    if not 1 <= rank <= min(len(directions), delta.shape[1]):
        raise ValueError("Gate rank exceeds calibration support")
    return vec[:, :rank].astype(np.float32), dict(rank=rank, pairs=len(pairs),
        used_contrasts=int(used.sum()), normalized=True, energy_retained=float(eig[:rank].sum()/eig.sum()))


class Inputs:
    """Read only explicitly requested participants from a portable manifest."""

    def __init__(self, manifest, *, checkpoint=None, device="cpu"):
        """Validate the input contract without opening any participant arrays."""
        self.path = Path(manifest).resolve()
        self.spec = read_json(self.path)
        if self.spec.get("schema") != "mgpa-task-reuse-inputs-v1":
            raise ValueError("Expected mgpa-task-reuse-inputs-v1 manifest")
        if self.spec.get("source_order") != list(SOURCE_ORDER):
            raise ValueError("Source order must be Neuroscan, Flex")
        if self.spec.get("normalization") != "fixed_20uv":
            raise ValueError("Published reuse requires fixed_20uv EEGPT normalization")
        if self.spec.get("checkpoint_sha256") != CHECKPOINT_SHA256:
            raise ValueError("Published EEGPT checkpoint hash differs")
        self.entries = {(str(r["person"]), r["task"]): r for r in self.spec["records"]}
        if len(self.entries) != len(self.spec["records"]):
            raise ValueError("Duplicate participant/task in manifest")
        self.checkpoint, self.device, self.encoder = checkpoint, device, None
        self.pins = {"manifest": sha256(self.path)}

    def resolve(self, name, expected=None):
        """Resolve a manifest-relative file and record or check its content hash."""
        path = Path(name).expanduser()
        if not path.is_absolute():
            path = self.path.parent / path
        actual = sha256(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Input hash mismatch: {path}")
        self.pins[str(name)] = actual
        return path

    def record(self, person, task):
        """Load one paired record or lazily ingest and encode its raw epochs."""
        entry = dict(self.entries[str(person), task])
        if entry.get("raw", False):
            from .raw.flex_ingest import ingest_one
            raw_root = Path(self.spec["raw_root"]).expanduser()
            cache = Path(self.spec["epoch_cache"]).expanduser()
            if not raw_root.is_absolute():
                raw_root = self.path.parent / raw_root
            if not cache.is_absolute():
                cache = self.path.parent / cache
            receipt = ingest_one(raw_root, cache, str(person), task, "retain_flagged")
            epochs = cache / f"{person}_{task}.npz"
            entry.update(metadata=str(epochs), metadata_sha256=receipt["npz_sha256"],
                epochs=str(epochs), epochs_sha256=receipt["npz_sha256"],
                acquisition=str(epochs.with_suffix(".json")))
        metadata = load_arrays(self.resolve(entry["metadata"], entry.get("metadata_sha256")))
        y = np.asarray(metadata["y"], dtype=np.int64)
        ordinal = np.asarray(metadata.get("ordinal", metadata.get("event_id")))
        if not np.issubdtype(ordinal.dtype, np.integer) or ordinal.shape != y.shape or np.any(np.diff(ordinal) <= 0):
            raise ValueError("Metadata needs strictly increasing integer event ordinals")
        if "source_order" in metadata and metadata["source_order"].tolist() != list(SOURCE_ORDER):
            raise ValueError("Metadata source order differs")
        if "subject" in metadata and any(str(p).split(":")[-1] != str(person) for p in metadata["subject"]):
            raise ValueError("Metadata participant differs")
        if "onset_s" in metadata:
            onsets = np.asarray(metadata["onset_s"], dtype=np.float64)
        else:
            audit = read_json(self.resolve(entry["acquisition"], entry.get("acquisition_sha256")))
            columns = []
            for source in SOURCE_ORDER:
                events = audit["event_audit"][source]
                lookup = dict(zip(events["event_ids"], events["event_times_physical_s"]))
                columns.append([lookup[int(i)] for i in ordinal])
            onsets = np.asarray(columns, dtype=np.float64).T
        if onsets.shape != (len(y), 2) or not np.isfinite(onsets).all() or not np.isin(y, [0, 1]).all():
            raise ValueError("Invalid paired timing or binary task labels")
        if "tokens" in entry:
            x = np.load(self.resolve(entry["tokens"], entry.get("tokens_sha256")), allow_pickle=False)
        elif "features" in entry:
            value = load_arrays(self.resolve(entry["features"], entry.get("features_sha256")))
            x = value[entry.get("feature_key", "x")]
        else:
            from .reuse_encoder import Encoder, preprocess
            raw = load_arrays(self.resolve(entry["epochs"], entry.get("epochs_sha256")))
            waveform = raw["x"]
            if waveform.shape != (len(y), 2, 16, 128) or not np.isfinite(waveform).all():
                raise ValueError("Expected paired one-second 16-channel 128Hz epochs in volts")
            checkpoint = self.checkpoint or self.spec.get("checkpoint")
            if checkpoint is None:
                raise ValueError("Raw epoch encoding requires --checkpoint or manifest checkpoint")
            checkpoint = self.resolve(checkpoint, self.spec.get("checkpoint_sha256", CHECKPOINT_SHA256))
            channels = metadata.get("channels", raw.get("channels"))
            if channels is None:
                raise ValueError("Epoch encoding requires ordered channel names")
            if self.encoder is None:
                self.encoder = Encoder(channels.tolist(), checkpoint=checkpoint, device=self.device)
                self.channels = channels.tolist()
            elif self.channels != channels.tolist():
                raise ValueError("All records must preserve ordered common channel names")
            x = self.encoder.encode(preprocess(waveform.reshape(-1, 16, 128), 128., "fixed_20uv"),
                                    batch_size=24).reshape(len(y), 2, 7, 4, 512)
        if x.shape not in ((len(y), 2, 7, 4, 512), (len(y), 2, 14336)) or not np.isfinite(x).all():
            raise ValueError("Frozen features must retain all 7x4x512 EEGPT coordinates")
        return dict(x=x.reshape(len(y), 2, 14336), y=y, ordinal=ordinal,
            subject=np.full(len(y), f"flex:{person}"), task_ids=np.full(len(y), task),
            event_id=np.asarray([f"flex:{person}:{task}:{i}" for i in ordinal]), onset_s=onsets)


def raw_manifest(raw_root, destination, *, checkpoint):
    """Declare raw records without opening recordings; ingestion stays role-lazy."""
    destination = Path(destination).resolve()
    records = [dict(person=p, task="p300", raw=True) for p in (*FIT, *DEV, *HEAD, *EVAL)]
    records += [dict(person=p, task=t, raw=True) for t in ("n170", "mmn") for p in (*HEAD, *EVAL)]
    write_json(destination, dict(schema="mgpa-task-reuse-inputs-v1", source_order=list(SOURCE_ORDER),
        raw_root=str(Path(raw_root).expanduser().resolve()), epoch_cache="raw_epochs",
        normalization="fixed_20uv",
        checkpoint=str(Path(checkpoint).expanduser().resolve()), checkpoint_sha256=CHECKPOINT_SHA256,
        records=records))
    return destination


def prepare(manifest, run_dir, *, checkpoint=None, device="cpu"):
    """Prepare only P300 FIT/CAL/HEAD/DEV, recording exact coordinate provenance."""
    root = Path(run_dir)
    inputs = Inputs(manifest, checkpoint=checkpoint, device=device)
    fits, cals, split_audit = [], [], {}
    for person in FIT:
        fit, cal, split = split_calibration(inputs.record(person, "p300"))
        fits.append(fit)
        cals.append(cal)
        split_audit[person] = split
    fit, cal = concatenate(fits), concatenate(cals)
    if len(fit["x"]) != 2260 or len(cal["x"]) != 400:
        raise ValueError("Published input contract requires 2,260 observed FIT and 400 CAL events")
    state = fit_coordinates(fit)
    save_arrays(root / "models/coordinates.npz", state)
    roles = {"fit": fit, "cal": cal}
    for name, people in (("head", HEAD), ("dev", DEV)):
        # Project each participant separately, as in the published loader.
        rows = []
        for person in people:
            record = inputs.record(person, "p300")
            rows.append({**record, "x": transform_coordinates(state, record["x"])})
        roles[name] = concatenate(rows)
    if len(roles["dev"]["x"]) != 1332:
        raise ValueError("Published DEV source-validation contract requires 1,332 events")
    for name, record in roles.items():
        if name in ("fit", "cal"):
            record = {**record, "x": transform_coordinates(state, record["x"])}
        save_arrays(root / "prepared" / f"p300_{name}.npz", record)
    record = dict(schema="mgpa-task-reuse-preparation-v1", input_manifest_sha256=sha256(manifest),
        input_pins=inputs.pins, source_order=list(SOURCE_ORDER), split_audit=split_audit,
        coordinate_sha256=sha256(root / "models/coordinates.npz"), evaluation_opened=False,
        fit_rows=2260, calibration_pairs=400, source_validation_rows=1332)
    write_json(root / "logs/preparation.json", record)
    return record


def prepare_evaluation_task(manifest, run_dir, task, *, checkpoint=None, device="cpu"):
    """Apply frozen coordinates to HEAD/EVAL after all P300 selections are locked."""
    root = Path(run_dir)
    if task not in TASKS or not (root / "selection_lock.json").is_file():
        raise ValueError("Known task and frozen P300 selection lock required before EVAL")
    inputs = Inputs(manifest, checkpoint=checkpoint, device=device)
    prep = read_json(root / "logs/preparation.json")
    if sha256(manifest) != prep["input_manifest_sha256"]:
        raise ValueError("Input manifest changed since preparation")
    if sha256(root / "models/coordinates.npz") != prep["coordinate_sha256"]:
        raise ValueError("Frozen coordinate map changed")
    state = load_arrays(root / "models/coordinates.npz")
    for name, people in (("head", HEAD), ("eval", EVAL)):
        destination = root / "prepared" / f"{task}_{name}.npz"
        if task == "p300" and name == "head":
            continue
        rows = []
        for person in people:
            record = inputs.record(person, task)
            rows.append({**record, "x": transform_coordinates(state, record["x"])})
        save_arrays(destination, concatenate(rows))
    write_json(root / "logs" / f"inputs_{task}.json", inputs.pins)
