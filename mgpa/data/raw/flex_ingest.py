"""Portable paired FLEX/Neuroscan raw ingestion. Original inputs are read-only."""
from __future__ import annotations

import argparse

import csv

import hashlib

import importlib.metadata

import json

import os

from pathlib import Path

import sys

import time

import numpy as np

CHANNELS = ("Cz", "Fz", "Fp1", "F3", "FT7", "CP3", "P7", "O1", "Pz",
            "Oz", "O2", "P8", "CP4", "FT8", "F4", "Fp2")

SOURCE_ORDER = ("neuroscan", "flex")

PARADIGMS = {"p300": "oddball_active", "mmn": "oddball_passive", "n170": "N170"}

FORMAT_VERSION = "flex-paired-erp-v3"

ARTIFACT_POLICIES = ("drop_pairs", "retain_flagged")


def package_version(name):
    """Read an installed package version without altering the import path."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None

def sha256(path: Path) -> str:
    """Compute a streaming SHA256 digest for an explicit file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def stable_digest(value: object) -> str:
    """Hash structured provenance using stable JSON encoding."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

def config(artifact_policy: str = "drop_pairs") -> dict:
    """Return the fixed continuous-EEG preprocessing and event-alignment recipe."""
    if artifact_policy not in ARTIFACT_POLICIES:
        raise ValueError(f"Unknown artifact policy: {artifact_policy}")
    return {
        "format_version": FORMAT_VERSION, "channels": list(CHANNELS),
        "source_order": list(SOURCE_ORDER), "output_sfreq_hz": 128.0,
        "flex_header_expected_hz": 128.0, "flex_physical_sfreq_hz": 129.05,
        "flex_event_offset_native_samples": 8,
        "flex_event_offset_seconds": 8 / 129.05,
        "tmin_s": -0.2, "tmax_exclusive_s": 0.8,
        "n_times": 128, "bandpass_hz": [0.1, 30.0],
        "filter": "MNE zero-phase FIR firwin; continuous; default transitions",
        "resample": "MNE continuous FFT resample to 128Hz; then linear epoch interpolation",
        "reference": "average of same 16 scalp channels, no M1/M2/EOG",
        "baseline": "subtract epoch per-channel mean of all relative times < 0",
        "artifact_peak_abs_volts": 100e-6,
        "artifact_policy": artifact_policy,
        "artifact_rule": "flag pair when either device exceeds 100uV after baseline",
        "artifact_action": ("drop flagged pairs" if artifact_policy == "drop_pairs"
                            else "retain all boundary-valid pairs and attach unchanged threshold flags"),
        "manual_ica": False, "data_unit": "V",
        "alignment": "within-device corrected triggers matched to behavioral ordinal trial IDs",
        "max_affine_timing_residual_s": 0.100,
        "max_p95_affine_timing_residual_s": 0.040,
        "max_clock_slope_deviation": 0.002,
        "timing_fit_applied_to_signal": False,
        "metadata_restart_repair": "1016/n170/neuroscan: unique contiguous behavior-timing match only",
    }

def leading_edges(marker: np.ndarray) -> np.ndarray:
    """Positive pulses following zero; initial nonzero state is not an edge."""
    marker = np.asarray(marker, dtype=float)
    if marker.ndim != 1 or not np.isfinite(marker).all():
        raise ValueError("Marker must be a finite one-dimensional array")
    positive = marker > 0.5
    return np.flatnonzero(positive & np.r_[False, ~positive[:-1]])

def flex_event_seconds(samples: np.ndarray, physical_rate: float = 129.05) -> np.ndarray:
    """Convert marker samples using the physical Flex rate and fixed offset."""
    return (np.asarray(samples, dtype=float) + 8.0) / physical_rate

def map_event_ids(n_events: int, n_behavior: int, subject: str,
                  paradigm: str, device: str) -> np.ndarray:
    """Fail closed on unexplained missing/extra markers; never guess labels."""
    skip = 0
    if device == "neuroscan" and str(subject) == "1034":
        skip = {"n170": 4, "mmn": 3}.get(paradigm, 0)
    expected = n_behavior - skip
    if n_events != expected:
        raise ValueError(
            f"Unexplained event count {subject}/{paradigm}/{device}: "
            f"observed={n_events}, expected={expected}, behavior={n_behavior}, documented_skip={skip}"
        )
    return np.arange(skip + 1, n_behavior + 1, dtype=np.int32)

def timing_audit(recorded: np.ndarray, behavior: np.ndarray) -> dict:
    """Clock/sequence diagnostic only. The fitted mapping never alters epochs."""
    recorded, behavior = np.asarray(recorded), np.asarray(behavior)
    if len(recorded) != len(behavior) or len(recorded) < 3:
        raise ValueError("Need at least three matched events for timing audit")
    if np.any(np.diff(recorded) <= 0) or np.any(np.diff(behavior) <= 0):
        raise ValueError("Event times must be strictly increasing")
    slope, intercept = np.polyfit(behavior - behavior[0], recorded - recorded[0], 1)
    residual = recorded - recorded[0] - (slope * (behavior - behavior[0]) + intercept)
    result = {
        "n_events": len(recorded), "slope_recorded_per_behavior": float(slope),
        "relative_intercept_s": float(intercept),
        "max_abs_residual_s": float(np.max(np.abs(residual))),
        "p95_abs_residual_s": float(np.quantile(np.abs(residual), .95)),
        "rms_residual_s": float(np.sqrt(np.mean(residual ** 2))),
        "first_recorded_s": float(recorded[0]), "last_recorded_s": float(recorded[-1]),
        "min_behavior_soa_s": float(np.min(np.diff(behavior))),
        "median_behavior_soa_s": float(np.median(np.diff(behavior))),
        "fit_applied": False,
    }
    cfg = config()
    result["passed"] = bool(
        result["max_abs_residual_s"] <= cfg["max_affine_timing_residual_s"]
        and result["p95_abs_residual_s"] <= cfg["max_p95_affine_timing_residual_s"]
        and abs(slope - 1) <= cfg["max_clock_slope_deviation"]
    )
    return result

def unique_contiguous_timing_match(recorded: np.ndarray, behavior: np.ndarray) -> tuple[int, list[dict]]:
    """Find an unambiguous complete behavior sequence using event times only."""
    recorded, behavior = np.asarray(recorded), np.asarray(behavior)
    if len(recorded) <= len(behavior):
        raise ValueError("Restart matching requires extra recorded events")
    checks = []
    for start in range(len(recorded) - len(behavior) + 1):
        check = timing_audit(recorded[start:start + len(behavior)], behavior)
        checks.append({"start_index_zero_based": start, **check})
    matches = [check["start_index_zero_based"] for check in checks if check["passed"]]
    if len(matches) != 1:
        raise ValueError(f"Ambiguous/unmatched event restart: {len(matches)} eligible windows")
    return matches[0], checks

def apply_metadata_restart(event_times: np.ndarray, codes: np.ndarray, behavior_times: np.ndarray,
                           subject: str, paradigm: str, device: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """One audited source exception; other unexplained count mismatches fail."""
    if (str(subject), paradigm, device) != ("1016", "n170", "neuroscan") or len(event_times) == len(behavior_times):
        return event_times, codes, {"applied": False}
    start, checks = unique_contiguous_timing_match(event_times, behavior_times)
    stop = start + len(behavior_times)
    discarded = np.r_[np.arange(start), np.arange(stop, len(event_times))]
    repair = {
        "applied": True,
        "reason": "unique complete contiguous behavioral inter-event timing match; earlier stimulus run not in behavioral/Flex recording",
        "not_an_author_documented_exception": True,
        "selection_uses": "event timestamps only, no EEG content, labels, or task accuracy",
        "original_n_stimulus_markers": len(event_times),
        "selected_marker_ordinals_one_based": list(range(start + 1, stop + 1)),
        "discarded_marker_ordinals_one_based": (discarded + 1).tolist(),
        "discarded_marker_times_s": np.asarray(event_times)[discarded].tolist(),
        "candidate_timing_audits": checks,
    }
    return event_times[start:stop], codes[start:stop], repair

def load_behavior(path: Path, paradigm: str) -> dict:
    """Read task labels and timing metadata from the paradigm's behavior file."""
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(stream)]
    ids = np.array([int(row["trial"]) for row in rows], dtype=np.int32)
    if not np.array_equal(ids, np.arange(1, len(rows) + 1)):
        raise ValueError(f"Behavior trial IDs are not contiguous and one-based: {path}")
    if paradigm == "n170":
        codes = np.array([int(row["eventCode"]) for row in rows], dtype=np.int16)
        y = np.array([int(row["stimCat"] == "face") for row in rows], dtype=np.int8)
        times = np.array([float(row["stimElapsedTime"]) for row in rows])
        orientation = np.array([row["stimOrient"] for row in rows])
        if any(row["stimCat"] not in {"face", "watch"} for row in rows):
            raise ValueError("Unknown N170 stimulus category")
    else:
        codes = np.array([int(row["tone"]) for row in rows], dtype=np.int16)
        if not set(codes).issubset({1, 3}):
            raise ValueError("Unknown auditory tone event code")
        y = (codes == 3).astype(np.int8)
        times = np.array([float(row["elapsed_time"]) for row in rows])
        orientation = np.full(len(rows), "not_applicable")
    return dict(ids=ids, codes=codes, y=y, times=times, orientation=orientation)

def one_file(directory: Path, pattern: str) -> Path:
    """Require exactly one matching raw acquisition file."""
    candidates = sorted(directory.glob(pattern))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one {directory}/{pattern}, found {len(candidates)}")
    return candidates[0]

def source_files(data_root: Path, subject: str, paradigm: str) -> dict:
    """Resolve the raw EEG and behavior files for a participant and task."""
    base = data_root / "raw" / "Saline Raw Data" / PARADIGMS[paradigm]
    edf = one_file(base / "eeg" / "emotiv", f"{subject}_*.edf")
    dat = one_file(base / "eeg" / "neuroscan", f"{subject}_*.dat")
    behav = one_file(base / "behav", f"{subject}_*.csv")
    result = {"flex_edf": edf, "neuroscan_dat": dat, "behavior_csv": behav}
    for suffix in (".dap", ".rs3", ".ceo"):
        p = dat.with_suffix(suffix)
        if not p.exists():
            raise FileNotFoundError(p)
        result[f"neuroscan{suffix}"] = p
    return result

def inspect_flex_aux(raw) -> tuple[np.ndarray, dict]:
    """EDF marks auxiliary numbers as microvolts: undo MNE's SI scaling."""
    names = ["MARKER_HARDWARE", "TIME_STAMP_s", "TIME_STAMP_ms", "COUNTER", "INTERPOLATED"]
    for name in names:
        if raw._orig_units.get(name) not in {"µV", "uV", "μV"}:
            raise ValueError(f"Unexpected EDF auxiliary unit for {name}: {raw._orig_units.get(name)}")
    values = raw.get_data(picks=names) * 1e6
    marker, seconds, millis, counter, interpolated = values
    samples = leading_edges(marker)
    timestamps = seconds + millis / 1000
    dt = np.diff(timestamps)
    good_dt = dt[(dt > 0.0076) & (dt < .0080)]
    # The original calibration uses stable timestamp intervals, not EDF labels.
    audit = {
        "header_sfreq_hz": float(raw.info["sfreq"]),
        "physical_sfreq_hz_used": 129.05,
        "initial_nonzero_marker_excluded": bool(marker[0] > .5),
        "n_leading_edges": len(samples),
        "pulse_values": np.unique(marker[samples]).tolist(),
        "timestamp_median_positive_dt_s": float(np.median(dt[dt > 0])),
        "timestamp_rate_from_mean_stable_dt_hz": float(1 / good_dt.mean()) if len(good_dt) else None,
        "timestamp_nonpositive_steps": int(np.count_nonzero(dt <= 0)),
        "timestamp_steps_gt_20ms": int(np.count_nonzero(dt > .020)),
        "interpolated_fraction": float(np.mean(interpolated > .5)),
        "counter_nonunit_steps_excluding_mod128_wrap": int(np.count_nonzero(
            np.abs(np.mod(np.diff(np.rint(counter)), 128) - 1) > .1)),
        "auxiliary_numeric_units_restored_from_SI": True,
    }
    return samples, audit

def load_device(files: dict, device: str):
    """Read, filter, rereference, and resample one physical device stream."""
    from .runtime import ensure_runtime
    ensure_runtime()
    import mne
    if device == "flex":
        raw = mne.io.read_raw_edf(files["flex_edf"], preload=False, verbose="ERROR")
        if raw.info["sfreq"] != config()["flex_header_expected_hz"]:
            raise ValueError("Unexpected Flex header sampling rate")
        samples, aux = inspect_flex_aux(raw)
        event_times = flex_event_seconds(samples)
        trigger_codes = np.ones(len(samples), dtype=int)  # Hardware pulse, not condition label.
        native_rate = 129.05
    else:
        raw = mne.io.read_raw_curry(files["neuroscan_dat"], preload=False, verbose="ERROR")
        # Author removes noise codes >10. Retain original codes for audit only.
        keep = np.array([str(x).isdigit() and 0 < int(x) <= 10 for x in raw.annotations.description])
        event_times = np.asarray(raw.annotations.onset[keep], dtype=float)
        trigger_codes = np.asarray(raw.annotations.description[keep], dtype=int)
        native_rate = float(raw.info["sfreq"])
        aux = {"header_sfreq_hz": native_rate, "physical_sfreq_hz_used": native_rate,
               "n_annotations_excluded_noise_or_noninteger": int((~keep).sum())}
    lookup = {name.lower(): name for name in raw.ch_names}
    picks = [lookup[name.lower()] for name in CHANNELS]
    data = raw.get_data(picks=picks)
    if not np.isfinite(data).all():
        raise ValueError(f"Nonfinite samples in {device}")
    aux["raw_channel_names_selected"] = picks
    aux["n_continuous_samples"] = data.shape[-1]
    aux["duration_physical_s"] = data.shape[-1] / native_rate
    aux["median_channel_std_uV"] = float(np.median(np.std(data, axis=1)) * 1e6)
    raw.close()
    info = mne.create_info(list(CHANNELS), native_rate, ch_types="eeg")
    continuous = mne.io.RawArray(data, info, verbose="ERROR")
    continuous.filter(.1, 30, method="fir", phase="zero", fir_design="firwin", n_jobs=1, verbose="ERROR")
    continuous.set_eeg_reference(ref_channels="average", projection=False, verbose="ERROR")
    continuous.resample(128.0, npad="auto", n_jobs=1, verbose="ERROR")
    return continuous, event_times, trigger_codes, aux

def extract_epochs(data: np.ndarray, events_s: np.ndarray, sfreq: float = 128.) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate epochs on exact common relative times [-.2,.8), in seconds."""
    relative = -.2 + np.arange(128) / sfreq
    positions = (events_s[:, None] + relative[None, :]) * sfreq
    valid = (positions[:, 0] >= 0) & (positions[:, -1] <= data.shape[-1] - 1)
    out = np.full((len(events_s), data.shape[0], 128), np.nan, dtype=np.float32)
    x = np.arange(data.shape[-1])
    for idx in np.flatnonzero(valid):
        values = np.stack([np.interp(positions[idx], x, channel) for channel in data])
        values -= values[:, relative < 0].mean(axis=1, keepdims=True)
        out[idx] = values
    return out, valid

def class_counts(y: np.ndarray) -> dict:
    """Summarize binary trial counts for quality-control provenance."""
    return {str(int(value)): int(np.count_nonzero(y == value)) for value in (0, 1)}

def paired_artifact_masks(in_bounds: np.ndarray, clean_by_device: np.ndarray,
                          artifact_policy: str = "drop_pairs") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return selection, per-device threshold flags, and shared clean mask.

    Rows correspond to already matched original event IDs. Boundary-invalid
    epochs are never included under either policy; flags refer only to the
    unchanged amplitude rule, not to task labels or model outputs.
    """
    if artifact_policy not in ARTIFACT_POLICIES:
        raise ValueError(f"Unknown artifact policy: {artifact_policy}")
    in_bounds = np.asarray(in_bounds, dtype=bool)
    clean_by_device = np.asarray(clean_by_device, dtype=bool)
    if in_bounds.ndim != 2 or in_bounds.shape[1] != 2 or in_bounds.shape != clean_by_device.shape:
        raise ValueError("Expected matching [n_pair, 2] boundary and clean masks")
    if np.any(clean_by_device & ~in_bounds):
        raise ValueError("Boundary-invalid epochs cannot be marked clean")
    boundary_valid = in_bounds.all(axis=1)
    artifact_by_device = in_bounds & ~clean_by_device
    clean_pair = boundary_valid & clean_by_device.all(axis=1)
    selected = clean_pair if artifact_policy == "drop_pairs" else boundary_valid
    return selected, artifact_by_device, clean_pair

def ingest_one(data_root: Path, output: Path, subject: str, paradigm: str,
               artifact_policy: str = "drop_pairs") -> dict:
    """Write aligned in-bounds device pairs with the selected artifact policy."""
    from .runtime import ensure_runtime
    ensure_runtime()
    import mne
    files = source_files(data_root, subject, paradigm)
    cfg = config(artifact_policy)
    provenance = {
        "subject": subject, "paradigm": paradigm, "config": cfg,
        "sources": {key: {"path": str(path.resolve()), "bytes": path.stat().st_size,
                          "sha256": sha256(path)} for key, path in files.items()},
        "source_code_sha256": sha256(Path(__file__)),
        "versions": {name: package_version(name) for name in ("mne", "numpy", "scipy", "curryreader")},
    }
    fingerprint = stable_digest(provenance)
    npz_path, manifest_path = output / f"{subject}_{paradigm}.npz", output / f"{subject}_{paradigm}.json"
    if manifest_path.exists() or npz_path.exists():
        if not (manifest_path.exists() and npz_path.exists()):
            raise RuntimeError(f"Partial previous output; inspect before reuse: {manifest_path}")
        old = json.loads(manifest_path.read_text())
        if old.get("provenance_digest") != fingerprint or old.get("npz_sha256") != sha256(npz_path):
            raise RuntimeError(f"Provenance/content mismatch; use a new output folder: {manifest_path}")
        print(f"REUSED {subject} {paradigm}: {old['qc']['n_retained_pairs']} pairs", flush=True)
        return old
    behavior = load_behavior(files["behavior_csv"], paradigm)
    n_behavior = len(behavior["ids"])
    device_values, device_ids, audits = {}, {}, {}
    for device in SOURCE_ORDER:
        continuous, event_times, codes, aux = load_device(files, device)
        event_times, codes, restart = apply_metadata_restart(
            event_times, codes, behavior["times"], subject, paradigm, device)
        event_ids = map_event_ids(len(event_times), n_behavior, subject, paradigm, device)
        time_check = timing_audit(event_times, behavior["times"][event_ids - 1])
        audit = {"native": aux, "timing": time_check, "event_ids": event_ids.tolist(),
                 "event_times_physical_s": event_times.tolist(),
                 "original_trigger_codes": codes.tolist(),
                 "metadata_restart_repair": restart,
                 "labels_from_behavior_not_hardware_pulse": True}
        if not time_check["passed"]:
            raise RuntimeError(f"Timing audit failed for {subject}/{paradigm}/{device}: {json.dumps(audit)}")
        epochs, in_bounds = extract_epochs(continuous.get_data(), event_times)
        continuous.close()
        peak = np.max(np.abs(epochs), axis=(1, 2))
        clean = in_bounds & (peak <= cfg["artifact_peak_abs_volts"])
        audit.update(n_events=len(event_ids), n_in_bounds=int(in_bounds.sum()),
                     n_clean_device=int(clean.sum()),
                     boundary_rejected_event_ids=event_ids[~in_bounds].tolist(),
                     amplitude_rejected_event_ids=event_ids[in_bounds & ~clean].tolist(),
                     peak_abs_uV=np.where(np.isfinite(peak), peak * 1e6, -1).tolist())
        device_values[device] = (epochs, in_bounds, clean)
        device_ids[device] = event_ids
        audits[device] = audit
    common = np.intersect1d(device_ids["neuroscan"], device_ids["flex"])
    indices = {device: np.searchsorted(device_ids[device], common) for device in SOURCE_ORDER}
    paired_timing = timing_audit(
        np.array(audits["flex"]["event_times_physical_s"])[indices["flex"]],
        np.array(audits["neuroscan"]["event_times_physical_s"])[indices["neuroscan"]])
    if not paired_timing["passed"]:
        raise RuntimeError(f"Cross-device event sequence timing failed for {subject}/{paradigm}: {paired_timing}")
    paired_bounds = np.stack([device_values[d][1][indices[d]] for d in SOURCE_ORDER], axis=1)
    paired_clean = np.stack([device_values[d][2][indices[d]] for d in SOURCE_ORDER], axis=1)
    selected, artifact_by_device, clean_pair = paired_artifact_masks(
        paired_bounds, paired_clean, artifact_policy)
    retained_ids = common[selected]
    if not len(retained_ids):
        raise RuntimeError(f"No paired epochs retained for {subject}/{paradigm}")
    paired_x = np.stack([device_values[d][0][indices[d]][selected] for d in SOURCE_ORDER], axis=1)
    y = behavior["y"][retained_ids - 1]
    soa = np.diff(behavior["times"])
    qc = {
        "n_behavior_trials": n_behavior, "n_common_device_events": len(common),
        "artifact_policy": artifact_policy,
        "n_retained_pairs": len(retained_ids), "n_rejected_pairs": int((~selected).sum()),
        "n_boundary_valid_pairs": int(paired_bounds.all(axis=1).sum()),
        "n_clean_pairs_under_original_100uv_rule": int(clean_pair.sum()),
        "n_amplitude_flagged_boundary_valid_pairs": int((paired_bounds.all(axis=1) & ~clean_pair).sum()),
        "n_flagged_retained_pairs": int((artifact_by_device.any(axis=1) & selected).sum()),
        "retained_event_ids": retained_ids.tolist(), "rejected_pair_event_ids": common[~selected].tolist(),
        "clean_pair_event_ids": common[clean_pair].tolist(),
        "artifact_flagged_pair_event_ids": common[paired_bounds.all(axis=1) & ~clean_pair].tolist(),
        "class_counts_all_behavior": class_counts(behavior["y"]),
        "class_counts_common_before_qc": class_counts(behavior["y"][common - 1]),
        "class_counts_retained": class_counts(y),
        "class_counts_clean_under_original_rule": class_counts(behavior["y"][common[clean_pair] - 1]),
        "min_soa_s": float(soa.min()), "median_soa_s": float(np.median(soa)),
        "trials_with_next_stimulus_inside_poststim_window": int(np.count_nonzero(soa < .8)),
        "adjacent_epoch_intervals_overlap_count": int(np.count_nonzero(soa < 1.0)),
        "n_same_event_pairs": len(retained_ids), "not_independent_human_replications": True,
        "scalp_channels": list(CHANNELS), "manual_ica_performed": False,
        "claim_scope": "technical pilot; acquisition-system change, not isolated amplifier effect",
    }
    if qc["trials_with_next_stimulus_inside_poststim_window"]:
        raise RuntimeError(f"Window intersects subsequent stimulus for {subject}/{paradigm}: {qc}")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, x=paired_x.astype(np.float32), y=y,
                        subject=np.full(len(y), subject), event_id=retained_ids,
                        original_event_code=behavior["codes"][retained_ids - 1],
                        orientation=behavior["orientation"][retained_ids - 1],
                        artifact_peak_100uv=artifact_by_device[selected].any(axis=1),
                        artifact_by_device=artifact_by_device[selected],
                        cleanflag=clean_pair[selected],
                        artifact_policy=np.array(artifact_policy),
                        channels=np.array(CHANNELS), source_order=np.array(SOURCE_ORDER),
                        sfreq=np.array(128.), tmin=np.array(-.2),
                        times=-.2 + np.arange(128) / 128., paradigm=np.array(paradigm),
                        provenance_digest=np.array(fingerprint))
    result = {**provenance, "provenance_digest": fingerprint, "event_audit": audits,
              "paired_event_timing_audit": paired_timing, "qc": qc,
              "npz_path": str(npz_path.resolve()), "npz_sha256": sha256(npz_path),
              "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"BUILT {subject} {paradigm}: {len(y)}/{len(common)} pairs, classes={class_counts(y)}", flush=True)
    return result
