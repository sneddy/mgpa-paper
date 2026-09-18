"""Audited timestamp-only 1034/N170 internal-gap handling."""
from __future__ import annotations

import argparse

import importlib.metadata

import json

from pathlib import Path

import time

from . import flex_ingest as base

import numpy as np

def unique_single_gap(recorded, behavior, *, min_segment=10):
    """Identify one missing contiguous behavior block; permit a clock break.

    Every candidate is audited under the SAME original r3 slope/residual bounds.
    Both nonempty segments must pass and exactly one candidate must qualify.
    No fitted affine clock transform is applied to any EEG or epoch timestamp.
    """
    recorded, behavior = np.asarray(recorded, dtype=float), np.asarray(behavior, dtype=float)
    gap = len(behavior) - len(recorded)
    if gap <= 0 or min_segment < 3 or len(recorded) < 2 * min_segment:
        raise ValueError("single missing-block model requires fewer recorded events and two valid segments")
    checks = []
    for k in range(min_segment, len(recorded) - min_segment + 1):
        before = base.timing_audit(recorded[:k], behavior[:k])
        after = base.timing_audit(recorded[k:], behavior[k + gap:])
        checks.append({"recorded_split_zero_based": k, "before": before, "after": after,
                       "passed": before["passed"] and after["passed"]})
    matches = [row for row in checks if row["passed"]]
    if len(matches) != 1:
        raise ValueError(f"single-gap timing match is not unique: {len(matches)} passing candidates")
    selected = matches[0]
    k = selected["recorded_split_zero_based"]
    event_ids = np.r_[np.arange(1, k + 1), np.arange(k + gap + 1, len(behavior) + 1)].astype(np.int32)
    return event_ids, {"selected": selected, "candidates": checks,
                      "missing_behavior_ids": list(range(k + 1, k + gap + 1)),
                      "candidate_count": len(checks), "gap_count": gap,
                      "clock_mapping_applied_to_signal": False}

def ingest(data_root, output):
    """Align 1034 N170 around its independently annotated four-event timing gap."""
    from .runtime import ensure_runtime
    ensure_runtime()
    import mne
    subject, paradigm = "1034", "n170"
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = base.source_files(Path(data_root), subject, paradigm)
    config = {**base.config("retain_flagged"), "format_version": "flex-paired-erp-v3-plus-1034-internal-gap",
              "alignment_exception": "unique single internal gap + independently documented impedance annotation",
              "author_initial_four_event_skip_not_used": True}
    provenance = {"subject": subject, "paradigm": paradigm, "config": config,
        "sources": {key: {"path": str(path.resolve()), "bytes": path.stat().st_size,
                          "sha256": base.sha256(path)} for key, path in files.items()},
        "source_code_sha256": base.sha256(Path(__file__)),
        "frozen_r3_source_sha256": base.sha256(Path(base.__file__)),
        "versions": {name: base.package_version(name) for name in ("mne", "numpy", "scipy", "curryreader")}}
    fingerprint = base.stable_digest(provenance)
    dest = output / "1034_n170.npz"
    report_path = dest.with_suffix(".json")
    if dest.exists() or report_path.exists():
        if dest.exists() and report_path.exists():
            old = json.loads(report_path.read_text())
            if old["provenance_digest"] == fingerprint and old["npz_sha256"] == base.sha256(dest):
                print(json.dumps({"status": "verified", "output": str(dest)}))
                return old
        raise RuntimeError("existing corrected cache differs; do not overwrite it")
    behavior = base.load_behavior(files["behavior_csv"], paradigm)
    raw = mne.io.read_raw_curry(files["neuroscan_dat"], preload=False, verbose="ERROR")
    markers = [{"onset": float(t), "description": str(d)} for t, d in
               zip(raw.annotations.onset, raw.annotations.description) if "impedance" in str(d).lower()]
    raw.close()
    recordings, ids, audits = {}, {}, {}
    gap_audit = None
    for device in base.SOURCE_ORDER:
        continuous, event_times, codes, native = base.load_device(files, device)
        if device == "neuroscan":
            if len(behavior["ids"]) - len(event_times) != 4:
                raise ValueError("1034 source no longer has the audited four-event deficit")
            event_ids, gap_audit = unique_single_gap(event_times, behavior["times"])
            split = gap_audit["selected"]["recorded_split_zero_based"]
            if len(markers) != 1 or not event_times[split - 1] < markers[0]["onset"] < event_times[split]:
                raise ValueError("the independently annotated impedance check does not corroborate the selected gap")
            if (event_times[split - 1] + .8 >= markers[0]["onset"] or
                    event_times[split] - .2 <= markers[0]["onset"]):
                raise ValueError("an epoch spans the impedance-check timestamp")
            checks = [gap_audit["selected"]["before"], gap_audit["selected"]["after"]]
        else:
            event_ids = base.map_event_ids(len(event_times), len(behavior["ids"]), subject, paradigm, device)
            checks = [base.timing_audit(event_times, behavior["times"][event_ids - 1])]
        if not all(check["passed"] for check in checks):
            raise ValueError(f"timing check failed: {device}")
        epochs, valid = base.extract_epochs(continuous.get_data(), event_times)
        continuous.close()
        peaks = np.max(np.abs(epochs), axis=(1, 2))
        clean = valid & (peaks <= config["artifact_peak_abs_volts"])
        recordings[device], ids[device] = (epochs, valid, clean), event_ids
        audits[device] = {"native": native, "timing_segments": checks,
                         "event_ids": event_ids.tolist(), "event_times_physical_s": event_times.tolist(),
                         "original_trigger_codes": codes.tolist(), "n_events": len(event_ids)}
    common = np.intersect1d(ids["neuroscan"], ids["flex"])
    index = {device: np.searchsorted(ids[device], common) for device in base.SOURCE_ORDER}
    split = gap_audit["selected"]["recorded_split_zero_based"]
    paired_timing = []
    ft = np.asarray(audits["flex"]["event_times_physical_s"])[index["flex"]]
    nt = np.asarray(audits["neuroscan"]["event_times_physical_s"])[index["neuroscan"]]
    for segment in (slice(None, split), slice(split, None)):
        paired_timing.append(base.timing_audit(ft[segment], nt[segment]))
    if not all(check["passed"] for check in paired_timing):
        raise ValueError("cross-device piecewise clock audit failed")
    bounds = np.stack([recordings[d][1][index[d]] for d in base.SOURCE_ORDER], axis=1)
    clean_by_device = np.stack([recordings[d][2][index[d]] for d in base.SOURCE_ORDER], axis=1)
    keep, flags, clean = base.paired_artifact_masks(bounds, clean_by_device, "retain_flagged")
    retained = common[keep]
    x = np.stack([recordings[d][0][index[d]][keep] for d in base.SOURCE_ORDER], axis=1)
    y = behavior["y"][retained - 1]
    arrays = dict(x=x, y=y, subject=np.full(len(y), subject), event_id=retained,
        original_event_code=behavior["codes"][retained - 1], orientation=behavior["orientation"][retained - 1],
        artifact_peak_100uv=flags[keep].any(axis=1), artifact_by_device=flags[keep], cleanflag=clean[keep],
        artifact_policy=np.array("retain_flagged"), channels=np.array(base.CHANNELS),
        source_order=np.array(base.SOURCE_ORDER), sfreq=np.array(128.), tmin=np.array(-.2),
        times=-.2 + np.arange(128) / 128., paradigm=np.array(paradigm), provenance_digest=np.array(fingerprint))
    np.savez_compressed(dest, **arrays)
    report = {**provenance, "provenance_digest": fingerprint, "event_audit": audits,
              "single_gap_audit": gap_audit, "corroborating_impedance_annotations": markers,
              "paired_event_timing_segments": paired_timing,
              "qc": {"n_behavior_trials": 300, "n_common_device_events": len(common),
                     "n_retained_pairs": len(y), "n_boundary_exclusions": int((~keep).sum()),
                     "n_clean_pairs_under_original_100uv_rule": int(clean[keep].sum()),
                     "class_counts_retained": base.class_counts(y)},
              "model_predictions_or_accuracy_used": False,
              "npz_path": str(dest), "npz_sha256": base.sha256(dest),
              "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": "COMPLETE", "output": str(dest), "missing_behavior_ids": gap_audit["missing_behavior_ids"],
                      "retained_pairs": len(y), "cross_device_audits_pass": True}), flush=True)
    return report
