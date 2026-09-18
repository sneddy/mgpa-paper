"""Rebuild role-separated coordinates from independently supplied frozen features.

Three input recipes are distinct: controlled reference-token means, observational
wet/dry full tokens with six synthetic calibration views, and paired FLEX
features. Frozen encoding precedes coordinate preparation.
"""
from __future__ import annotations

import csv
from pathlib import Path
import re

import numpy as np

from .geometry import (controlled_gate, fit_coordinates, fit_standardizer,
                       natural_gate, standardize, transform_coordinates)
from .prepared import ROLES, pinned_json, pinned_path, write_prepared
from .feature_protocols import CONTROLLED, FLEX, RECORDED_SELECTIONS, RECORDED_SSVEP

CONFIG_SHA = {
    "controlled": "5e9de58e41b5a4858386f6d7ddddeb6a42f4d0533f947f5fbc6ffcffc6f77f8d",
    "recorded_ssvep": "04c1c906cfe3c5075d54646e158167d8a7f88ef499664d7a8b249ae19099a6e0",
    "flex": "22915b064fc588b476d961c05f478e1b4351b9ba521a01f0e1dee5a8d6574715",
}
METADATA_SHA = "1c0421fdb710d87a524f06df0cc81d294f26127d736ec35fc2f7516d7c0950b5"
SELECTION_SHA = {
    0: "3ed23a13f92667099e0184649aea65586406c91c3bc24bbd888166f90d56dad1",
    1: "f2c4743c7d85063b50f38683c1c406a488b6c1d6eb2cfa247416caa74e662008",
    2: "dd0eefbd6d041c0806d8faa09538ea29e2b78c88a3e0f8ef3491a6a8ac5c5b3e",
    3: "f7ca5c7e738b79e4c3c95a4e27e3365bad4c09e1570727b55ee4e4d8f1a68e0b",
    4: "0c78e63db01ff000da85145152de84329da9c1be60101a825a2a8704118b433f",
}


def _protocol(paths, family):
    """Return the declared data-only recipe or authenticate an explicit copy."""
    if paths.get("protocol_config"):
        return pinned_json(paths["protocol_config"], CONFIG_SHA[family])
    return {"controlled": CONTROLLED, "recorded_ssvep": RECORDED_SSVEP, "flex": FLEX}[family]


def metadata_rows(path):
    """Load and validate the fixed 24,480-row SSVEP observation index."""
    with pinned_path(path, METADATA_SHA).open(newline="") as stream:
        records = list(csv.DictReader(stream))
    data = {key: np.array([row[key] for row in records]) for key in records[0]}
    for key in ("subject", "source", "block", "target", "frequency_binary", "phase_class"):
        data[key] = data[key].astype(np.int64)
    if len(data["sample_id"]) != 24480 or len(set(data["sample_id"])) != 24480:
        raise ValueError("SSVEP metadata size/identity mismatch")
    return data


def participant_splits(data, seed):
    """Build the published five participant folds stratified by order and gender."""
    from sklearn.model_selection import StratifiedKFold
    people = np.unique(data["subject"])
    strata = []
    for person in people:
        attrs = [(key, np.unique(data[key][data["subject"] == person])) for key in ("first_electrode", "gender")]
        if any(len(value) != 1 for _, value in attrs):
            raise ValueError("Within-person stratification mismatch")
        strata.append("|".join(value[0] for _, value in attrs))
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = [people[test] for _, test in splitter.split(people, strata)]
    return [{"fit": np.sort(np.concatenate([folds[(r+3) % 5], folds[(r+4) % 5]])),
             "val": folds[(r+2) % 5], "head": folds[(r+1) % 5], "eval": folds[r]} for r in range(5)]


def qualified(data, indices):
    """Create globally qualified SSVEP participant and observation identities."""
    return (np.array([f"ssvep:{s:03d}" for s in data["subject"][indices]]),
            np.array(["ssvep:" + value for value in data["sample_id"][indices]]))


def controlled_assignment(subject, target, binary, probability, seed):
    """Assign exact within-participant frequency-cell source counts."""
    rng = np.random.default_rng(seed)
    source = np.empty(len(subject), np.int64)
    for person in np.unique(subject):
        for task in np.unique(target[subject == person]):
            indices = np.flatnonzero((subject == person) & (target == task))
            if len(indices) != 10:
                raise ValueError("Controlled assignment requires ten blocks per subject/task")
            probability_one = probability if int(binary[indices[0]]) else 1-probability
            count = int(round(10 * probability_one))
            ordered = rng.permutation(indices)
            source[ordered[:count]], source[ordered[count:]] = 1, 0
    return source


def _controlled(paths, instance, destination, *, recipe=None):
    """Build fixed-chord CAR/Oz observations and FIT-only measurement coordinates."""
    match = re.fullmatch(r"ssvep_controlled__rotation([1-4])__(r02)", instance)
    if not match:
        raise ValueError("Controlled preparation requires explicit rotation and source-assignment replica")
    rotation, replica = int(match[1]), match[2]
    config = _protocol(paths, "controlled") if recipe is None else recipe
    # Gamma is the only deliberately exposed change to the established recipe.
    gamma = float(paths.get("gamma", config["reference_chord_amplitude"]))
    if gamma != .005:
        raise ValueError("Published controlled protocol fixes gamma=0.005")
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("Reference-chord gamma must be finite and positive")
    data = metadata_rows(paths["metadata"])
    split = participant_splits(data, config["random_seed"])[rotation]
    cache_path = pinned_path(paths["feature_grid"], config["cache_sha256"])
    cache = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    if cache.shape != tuple(config["cache_shape"]) or str(cache.dtype) != config["cache_dtype"]:
        raise ValueError("Controlled frozen feature grid changed")
    rng_key = next(row["rng_key"] for row in config["replicates"] if row["replicate_id"] == replica)
    probs = (config["fit_source_probability"], config["validation_source_probability"],
             config["evaluation_train_source_probability"], config["evaluation_test_source_probability"])
    streams = ("partition_fit", "partition_validation", "partition_evaluation_train", "partition_evaluation_test")
    datasets = {}
    for role, probability, stream in zip(ROLES, probs, streams):
        indices = np.flatnonzero(np.isin(data["subject"], split[role]) & (data["source"] == config["base_acquisition_index"]))
        pairs = np.empty((len(indices), 2, 512), np.float32)
        for start in range(0, len(indices), 256):
            for view, source_view in enumerate(config["selected_view_indices"]):
                pairs[start:start+256, view] = np.asarray(cache[indices[start:start+256], source_view], np.float32).mean(axis=1)
        seed = config["random_seed"] + 100000*rotation + 1000*rng_key + config["seed_offsets"][stream]
        source = controlled_assignment(data["subject"][indices], data["target"][indices], data["frequency_binary"][indices], probability, seed)
        midpoint, delta = pairs.mean(axis=1), pairs[:, 1] - pairs[:, 0]
        endpoints = np.stack([midpoint-.5*gamma*delta, midpoint+.5*gamma*delta], axis=1).astype(np.float32)
        observed = (midpoint+(2*source-1)[:, None]*.5*gamma*delta).astype(np.float32)
        subject, event_id = qualified(data, indices)
        role_data = {"x": endpoints, "subject": subject, "event_id": event_id,
                     "x_observed": observed, "u": source, "subject_observed": subject.copy(), "event_id_observed": event_id.copy()}
        if role in ("fit", "val"):
            role_data["x_calibration"] = pairs
        else:
            role_data.update(y=data["frequency_binary"][indices].astype(np.int8),
                             y_binary=data["frequency_binary"][indices].astype(np.int8),
                             y_multiclass=data["target"][indices].astype(np.int8))
        datasets[role] = role_data
    state = fit_standardizer(datasets["fit"]["x_observed"], config["evaluation_minimum_scale"])
    state["basis"], gate_info = controlled_gate(datasets["fit"]["x_calibration"], state, config)
    for role_data in datasets.values():
        for key in ("x", "x_observed", "x_calibration"):
            if key in role_data:
                role_data[key] = standardize(state, role_data[key])
    metadata = {"instance_id": instance, "feature": "eegpt_exact_reference_token_mean512",
                "interface": "paired_and_observed", "rotation": rotation, "replicate": replica,
                "gamma": gamma, "gate": gate_info, "task_primary": "frequency>=12Hz",
                "independent_unit": "participant", "source_order": ["gamma_minus_endpoint", "gamma_plus_endpoint"],
                "source_probability_high_task": dict(zip(ROLES, probs)),
                "task_labels_used_for_source_assignment": True, "task_labels_supplied_to_adapter": False,
                "calibration": "full same-trial CAR/Oz contrasts; gamma scales observed endpoints, not calibration",
                "feature_grid_sha256": config["cache_sha256"], "metadata_sha256": METADATA_SHA,
                "protocol_sha256": CONFIG_SHA["controlled"], "source_replica_is_not_fit_seed": True}
    return write_prepared(instance, destination, datasets, state, metadata)


def _natural(paths, instance, destination, *, recipe=None):
    """Build recorded token roles and FIT-local synthetic calibration geometry."""
    match = re.fullmatch(r"ssvep_recorded_wet_dry__rotation(1)", instance)
    if not match:
        raise ValueError("Recorded preparation requires an explicit rotation")
    rotation = int(match[1])
    config = _protocol(paths, "recorded_ssvep") if recipe is None else recipe
    selected = (pinned_json(paths["selection_config"], SELECTION_SHA[rotation])["mgpa_selected"]
                if paths.get("selection_config") else RECORDED_SELECTIONS[rotation])
    data = metadata_rows(paths["metadata"])
    split = participant_splits(data, config["seed"])[rotation]
    expected = config["canonical_inputs"]["embeddings"]["sha256"]
    cache = np.load(pinned_path(paths["feature_grid"], expected), mmap_mode="r", allow_pickle=False).reshape(24480, 15, 512)
    datasets = {}
    for role in ROLES:
        indices = np.flatnonzero(np.isin(data["subject"], split[role]))
        subject, event_id = qualified(data, indices)
        role_data = {"x_observed": np.asarray(cache[indices], np.float32), "u": data["source"][indices],
                     "subject_observed": subject, "event_id_observed": event_id,
                     "subject": subject.copy(), "event_id": event_id.copy()}
        if role in ("head", "eval"):
            role_data.update(y=data["target"][indices].astype(np.int8), y_multiclass=data["target"][indices].astype(np.int8),
                             y_binary=data["frequency_binary"][indices].astype(np.int8))
        datasets[role] = role_data
    state = fit_standardizer(datasets["fit"]["x_observed"], config["selection"]["minimum_scale"])
    for role_data in datasets.values():
        role_data["x_observed"] = standardize(state, role_data["x_observed"])
    entry = next(row for row in config["canonical_inputs"]["measurement_views"] if row["rotation"] == rotation and row["phase"] == "selection")
    transport = pinned_path(paths["calibration_features"], entry["archive"]["sha256"])
    sidecar = pinned_json(paths["calibration_manifest"], entry["manifest"]["sha256"])
    if not set(sidecar["transport_fit_subjects"]).issubset(set(split["fit"])):
        raise ValueError("Calibration transport was not fitted exclusively on FIT people")
    lookup = {value: i for i, value in enumerate(data["sample_id"])}
    for role, prefix in (("fit", "tangent_fit"), ("val", "validation_geometry")):
        with np.load(transport, allow_pickle=False) as archive:
            try:
                ids = archive[prefix+"_sample_ids"]
            except ValueError:
                # Only exact canonical archives, checked against immutable code-pinned
                # config hashes above, allow object-array string identifiers.
                with np.load(transport, allow_pickle=True) as trusted:
                    ids = trusted[prefix+"_sample_ids"]
                if not all(isinstance(value, str) for value in ids.tolist()):
                    raise ValueError("Canonical calibration IDs must be strings")
            ids = ids.astype(str)
            reference = archive[prefix+"_raw"].astype(np.float32)
            alternatives = [archive[prefix+f"_view_{i}"].astype(np.float32) for i in range(6)]
        if len(set(ids)) != len(ids) or any(value not in lookup for value in ids):
            raise ValueError("Invalid calibration event identity")
        rows = np.array([lookup[value] for value in ids], np.int64)
        if not set(data["subject"][rows]).issubset(set(split[role])):
            raise ValueError("Calibration role leakage")
        if reference.shape != (len(rows), 15, 512) or any(value.shape != reference.shape for value in alternatives):
            raise ValueError("Calibration token shape differs")
        if not np.array_equal(reference, np.asarray(cache[rows], np.float32)):
            raise ValueError("Calibration references differ from original frozen tokens")
        reference = standardize(state, reference)
        alternatives = [standardize(state, value) for value in alternatives]
        if role == "fit":
            state["basis"], gate_info = natural_gate(reference, alternatives, energy=selected["measurement_subspace_energy"])
            if recipe is None and gate_info["rank"] != selected["measurement_rank"]:
                raise ValueError("Rebuilt gate rank differs from the published context")
        subject, event_id = qualified(data, rows)
        datasets[role].update(calibration_reference=reference, calibration_alternatives=np.stack(alternatives, axis=1),
                              calibration_subject=subject, calibration_event_id=event_id,
                              calibration_view_names=np.asarray(config["view_names"]))
    metadata = {"instance_id": instance, "feature": "eegpt_15tokens_width512", "interface": "token_shared_offset",
                "rotation": rotation, "gate": gate_info, "source_order": ["dry", "wet"],
                "task_primary": "12-frequency multiclass", "independent_unit": "participant",
                "paired_physiological_views": False, "task_labels_supplied_to_adapter": False,
                "calibration": "six FIT-local synthetic same-trial transports; not paired wet/dry sessions",
                "input_protocol_phase": "Published FIT/VAL participant roles remain disjoint",
                "feature_grid_sha256": expected, "metadata_sha256": METADATA_SHA,
                "calibration_features_sha256": entry["archive"]["sha256"],
                "selection_sha256": SELECTION_SHA[rotation], "protocol_sha256": CONFIG_SHA["recorded_ssvep"]}
    return write_prepared(instance, destination, datasets, state, metadata)


def _flex(paths, context_id, destination, *, recipe=None):
    """Prepare the fixed temporal N170 participant roles without task-informed coordinates."""
    context_id = {"temporal_n170": "flex_n170", "n170": "flex_n170"}.get(context_id, context_id)
    config = _protocol(paths, "flex") if recipe is None else recipe
    full_id = {"flex_n170": "flex_n170_raw240_clean_pair_same_task"}[context_id]
    context = next(value for value in config["contexts"] if value["id"] == full_id)
    spec = context["tasks"][context["target"]]
    # Hash-pinned basenames under an explicit acquisition directory are portable;
    # recorded absolute locations in the frozen source config are never opened.
    feature_root = Path(paths["feature_directory"])
    files = []
    for original in spec["inputs"]:
        path = pinned_path(feature_root / Path(original).name, config["input_hashes"][original])
        pinned_path(path.with_suffix(".json"), config["encoding_provenance"][original]["manifest_sha256"])
        files.append(path)
    datasets, source_order, channels = {}, None, None
    for role, original_role in zip(ROLES, ("fit", "validation", "head", "test")):
        people = context["roles"][original_role]
        pieces = []
        for path in files:
            with np.load(path, allow_pickle=False) as archive:
                subject = archive["subject"].astype(str)
                indices = np.flatnonzero(np.isin(subject, people))
                if not len(indices):
                    continue
                if source_order is None:
                    source_order, channels = archive["source_order"].astype(str), archive["channels"].astype(str)
                if not np.array_equal(source_order, archive["source_order"].astype(str)) or not np.array_equal(channels, archive["channels"].astype(str)):
                    raise ValueError("FLEX source/channel semantics differ")
                row_key = "row_id" if "row_id" in archive.files else "trial_index" if "trial_index" in archive.files else "event_id"
                piece = {"x": archive[context["feature"]][indices], "subject": subject[indices],
                         "event_id": archive[row_key][indices].astype(str), "cleanflag": archive["cleanflag"][indices].astype(bool)}
                # Crucially do not even open the y member for FIT or VAL people.
                if role in ("head", "eval"):
                    piece["y"] = archive["y"][indices]
                pieces.append(piece)
        arrays = {key: np.concatenate([piece[key] for piece in pieces]) for key in pieces[0]}
        if set(arrays["subject"]) != set(people):
            raise ValueError("Requested FLEX people missing")
        mask = arrays["cleanflag"]
        arrays = {key: value[mask] for key, value in arrays.items()}
        if set(arrays["subject"]) != set(people):
            raise ValueError("A FLEX participant has no retained clean events")
        arrays["subject"] = np.array([context["dataset"]+":"+person for person in arrays["subject"]])
        arrays["event_id"] = np.array([person+":"+context["target"]+":"+event for person, event in zip(arrays["subject"], arrays["event_id"])])
        datasets[role] = arrays
    state, geometry = fit_coordinates(datasets["fit"]["x"])
    for arrays in datasets.values():
        arrays["x"] = transform_coordinates(state, arrays["x"])
    metadata = {"feature": context["feature"], "interface": "paired", "coordinate_fit": geometry,
                "source_order": source_order.tolist(), "channels": channels.tolist(),
                "quality_policy": "clean_pair", "protocol_sha256": CONFIG_SHA["flex"],
                "task_labels_supplied_to_adapter": False,
                "preexisting_feature_fit": "none"}
    return write_prepared(context_id, destination, datasets, state, metadata)


def prepare_features(paths, context_id, destination):
    """Dispatch a supported frozen-feature input to its exact coordinate recipe."""
    if context_id.startswith("ssvep_controlled__"):
        return _controlled(paths, context_id, destination)
    if context_id.startswith("ssvep_recorded_wet_dry__"):
        return _natural(paths, context_id, destination)
    if context_id in ("temporal_n170", "n170", "flex_n170"):
        return _flex(paths, context_id, destination)
    raise ValueError("Unsupported frozen-feature context")
