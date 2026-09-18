"""Dataset-specific fitting observations and replay interfaces.

These functions distinguish physical pairs, observed source groups, and
same-trial synthetic calibrations. A token is never an independent example.
"""
from dataclasses import dataclass

import numpy as np

from .prepared import load_prepared

CONTROLLED_CONTEXTS = tuple(f"ssvep_controlled__rotation{r}__r02" for r in (1, 2, 3, 4))
CONTEXTS = {"controlled": CONTROLLED_CONTEXTS, "temporal_n170": ("flex_n170",),
            "recorded_ssvep": ("ssvep_recorded_wet_dry__rotation1",)}
N170_PEOPLE = {
    "fit": ("1013", "1014", "1016", "1018"), "val": ("1022", "1023"),
    "head": ("1019", "1020", "1024", "1025"),
    "eval": ("1021", "1026", "1027", "1028", "1029", "1030", "1031", "1032", "1033", "1034"),
}
VIEW_NAMES = tuple(f"transport_{kind}_opposite@{strength:g}"
                  for kind in ("spectral", "spatial", "combined") for strength in (.5, 1.))


def pooled(tokens):
    """Released adapter pooling: float64 accumulation, then float32 storage."""
    x = np.asarray(tokens, np.float32)
    if x.ndim != 3 or x.shape[1:] != (15, 512) or not len(x) or not np.isfinite(x).all():
        raise ValueError("Recorded SSVEP requires finite [recording,15,512] tokens")
    return x.mean(axis=1, dtype=np.float64).astype(np.float32)


def load_dataset(dataset, manifest_path, roles):
    """Validate the experiment-specific semantics of selected portable roles."""
    records, state, manifest = load_prepared(manifest_path, roles=roles)
    meta = manifest.get("contract", {}).get("metadata", {})
    context = meta.get("instance_id", manifest["context_id"])
    if dataset not in CONTEXTS or context not in CONTEXTS[dataset]:
        raise ValueError("Manifest does not belong to this published experiment")
    width = 240 if dataset == "temporal_n170" else 512
    if state["basis"].shape[0] != width:
        raise ValueError("Prepared gate width differs from the representation")
    if dataset == "controlled":
        if meta.get("gamma") != .005 or manifest["feature"] != "eegpt_exact_reference_token_mean512":
            raise ValueError("Controlled protocol requires the fixed CAR/Oz gamma=.005 assay")
    elif dataset == "temporal_n170":
        if manifest["feature"] != "raw_timebins" or tuple(meta["source_order"]) != ("neuroscan", "flex"):
            raise ValueError("Temporal N170 is the paired raw240 experiment, not EEGPT reuse")
    else:
        if (manifest["feature"] != "eegpt_15tokens_width512" or tuple(meta["source_order"]) != ("dry", "wet")
                or meta.get("paired_physiological_views") is not False):
            raise ValueError("Recorded wet/dry sessions are unpaired full-token observations")
    for role, arrays in records.items():
        if dataset == "recorded_ssvep":
            if "x" in arrays:
                raise ValueError("Do not fabricate wet/dry physiological pairs")
            pooled(arrays["x_observed"])
        else:
            if arrays["x"].shape[1:] != (2, width):
                raise ValueError("Paired source shape changed")
            if dataset == "temporal_n170" and set(arrays["subject"]) != {"flex:"+p for p in N170_PEOPLE[role]}:
                raise ValueError("Temporal N170 participant roles changed")
    return records, state, manifest


@dataclass
class FitBank:
    """Adapter observations, source labels, and optional physical endpoint pairs."""
    x: np.ndarray
    source: np.ndarray
    pairs: np.ndarray | None


def fit_bank(dataset, role):
    """Expose source-only observations while rejecting downstream labels."""
    if any(k == "y" or k.startswith("y_") or k.startswith("task_label") for k in role):
        raise ValueError("Downstream labels are not adapter-fitting inputs")
    if dataset == "controlled":
        # One assigned endpoint per trial retains the prescribed 0.9 association.
        return FitBank(role["x_observed"], role["u"], role["x"])
    if dataset == "temporal_n170":
        x = role["x"]
        return FitBank(x.reshape(-1, 240), np.tile([0, 1], len(x)), x)
    if dataset == "recorded_ssvep":
        return FitBank(pooled(role["x_observed"]), role["u"], None)
    raise ValueError("Unknown experiment")


def recorded_calibration_pairs(fit):
    """Orient six same-trial transport families as dry-like/wet-like pairs."""
    ids = np.asarray(fit["event_id_observed"]).astype(str)
    cal_ids = np.asarray(fit["calibration_event_id"]).astype(str)
    if len(set(ids)) != len(ids) or len(set(cal_ids)) != len(cal_ids):
        raise ValueError("Calibration event identities must be unique")
    lookup = {event: i for i, event in enumerate(ids)}
    if any(event not in lookup for event in cal_ids):
        raise ValueError("Calibration contains a non-FIT event")
    rows = np.asarray([lookup[event] for event in cal_ids], int)
    reference, alternatives = fit["calibration_reference"], fit["calibration_alternatives"]
    if (alternatives.shape != (len(rows), 6, 15, 512)
            or tuple(fit["calibration_view_names"].astype(str)) != VIEW_NAMES
            or not np.array_equal(reference, fit["x_observed"][rows])):
        raise ValueError("Recorded calibration no longer matches its original FIT recordings")
    source = np.asarray(fit["u"])[rows]
    r = pooled(reference)
    a = alternatives.mean(axis=2, dtype=np.float64).astype(np.float32)
    dry = np.where((source == 0)[:, None, None], r[:, None, :], a)
    wet = np.where((source == 1)[:, None, None], r[:, None, :], a)
    return np.stack((dry, wet), axis=2).reshape(-1, 2, 512)


def transform_role(model, dataset, role, *, endpoint=None, source_aware=False):
    """Replay a fixed map; only explicitly routed CORAL/FEATMAP receive IDs."""
    if dataset == "recorded_ssvep":
        x = np.asarray(role["x_observed"], np.float32)
        mean = pooled(x)
        corrected = model.transform(mean, endpoint=endpoint,
            source_ids=role["u"] if source_aware else None)
        result = (x + (corrected-mean)[:, None, :]).astype(np.float32)
    else:
        x = role["x"]
        source = np.tile([0, 1], len(x)) if source_aware else None
        # Preserve each map's released output precision. Closed-form/affine
        # maps can return float64; iterative trajectories store float32.
        result = model.transform(x.reshape(-1, x.shape[-1]), endpoint=endpoint,
                                 source_ids=source).reshape(x.shape)
    if result.shape != x.shape or not np.isfinite(result).all():
        raise ValueError("Adapter returned invalid downstream-interface coordinates")
    return result
