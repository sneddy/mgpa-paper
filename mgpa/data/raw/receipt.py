"""Fresh-encoding receipts bridge portable raw preparation and numerical data."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from mgpa.data import sha256
from mgpa.data.features import _controlled, _flex, _natural
from mgpa.data.feature_protocols import CONTROLLED, FLEX, RECORDED_SSVEP
from mgpa.data.prepared import _write_json, pinned_json


def prepare_generated_features(receipt_path, context_id, destination, *, receipt_sha256):
    """Use identical coordinate code on independently re-encoded feature arrays.

    Only acquisition hashes are replaced. Participant splits, gate estimation,
    feature definitions and every data-recipe parameter remain fixed. The caller
    pins the complete receipt; all feature dependencies are contained/verified.
    """
    receipt_path = Path(receipt_path).resolve()
    receipt = pinned_json(receipt_path, receipt_sha256)
    if receipt.get("schema") != "mgpa-raw-acquisition-v1" or receipt.get("status") != "COMPLETE":
        raise ValueError("A complete raw-acquisition receipt is required")
    for name, digest in receipt["outputs"].items():
        path = (receipt_path.parent/name).resolve()
        if not path.is_relative_to(receipt_path.parent) or sha256(path) != digest:
            raise ValueError("Fresh acquisition dependency changed or escapes receipt root")
    def contained(value):
        """Resolve a feature dependency inside the supplied acquisition receipt."""
        path = (receipt_path.parent/value).resolve()
        if Path(value).is_absolute() or not path.is_relative_to(receipt_path.parent):
            raise ValueError("Acquisition input path escapes its portable receipt root")
        return path

    paths = {key: contained(value) for key, value in receipt["inputs_for_coordinates"].items()}
    if context_id.startswith("ssvep_controlled__"):
        if receipt["dataset"] != "controlled":
            raise ValueError("Acquisition dataset mismatch")
        recipe = deepcopy(CONTROLLED)
        recipe["cache_sha256"] = sha256(paths["feature_grid"])
        output = _controlled(paths, context_id, destination, recipe=recipe)
    elif context_id.startswith("ssvep_recorded_wet_dry__"):
        if receipt["dataset"] != "recorded_ssvep":
            raise ValueError("Acquisition dataset mismatch")
        rotation = int(context_id.rsplit("rotation", 1)[1])
        calibration = receipt["recorded_calibrations"][str(rotation)]
        paths.update({key: contained(value) for key, value in calibration.items()})
        recipe = deepcopy(RECORDED_SSVEP)
        recipe["canonical_inputs"]["embeddings"]["sha256"] = sha256(paths["feature_grid"])
        entry = next(row for row in recipe["canonical_inputs"]["measurement_views"] if row["rotation"] == rotation and row["phase"] == "selection")
        entry["archive"]["sha256"] = sha256(paths["calibration_features"])
        entry["manifest"]["sha256"] = sha256(paths["calibration_manifest"])
        output = _natural(paths, context_id, destination, recipe=recipe)
    else:
        if context_id not in ("flex_n170", "temporal_n170") or receipt["dataset"] != "n170":
            raise ValueError("Acquisition dataset mismatch")
        recipe = deepcopy(FLEX)
        context_id = "flex_n170"
        context = recipe["contexts"][0]
        for name in context["tasks"][context["target"]]["inputs"]:
            path = paths["feature_directory"]/name
            recipe["input_hashes"][name] = sha256(path)
            recipe["encoding_provenance"][name]["manifest_sha256"] = sha256(path.with_suffix(".json"))
        output = _flex(paths, context_id, destination, recipe=recipe)
    # This is a newly created bundle in the caller's new destination, never an
    # existing paper artifact. Record the upstream raw/encoder provenance.
    manifest = json.loads(output.read_text())
    manifest["acquisition"] = {"mode": "raw_signal_reencoding", "raw_signal_reproduction": True,
                                "receipt_sha256": receipt_sha256,
                                "checkpoint_sha256": receipt.get("checkpoint_sha256"),
                                "source_hashes": receipt["source_hashes"],
                                "bitwise_feature_parity_claimed": False}
    _write_json(output, manifest)
    return output
