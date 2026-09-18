"""Portable, hash-checked JSON plus numeric NPZ model state.

Only the public release estimators and explicitly declared stage records are
supported. Loading never imports a path from an artifact and never uses pickle.
"""
from dataclasses import fields
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = "mgpa-release-model-v1"


def _types():
    """Return the fixed estimator and stage-record allowlists for portable artifacts."""
    from .baselines import CORAL, FEATMAP, Identity, LEACE
    from .closed_form import ClosedFormMGPA
    from .core import Endpoint, ScoreStage
    from .critics import CriticState
    from .igbp import IGBP, IGBPStage
    from .iterative import IterativeMGPA
    from .tokens import TokenOffsetAdapter
    models = {c.__name__: c for c in (ClosedFormMGPA, IterativeMGPA, Identity,
                                    LEACE, CORAL, FEATMAP, IGBP, TokenOffsetAdapter)}
    records = {c.__name__: c for c in (Endpoint, ScoreStage, CriticState, IGBPStage)}
    return models, records


_STATE = {
    "ClosedFormMGPA": ("n_features_in_", "basis_", "complement_", "covariance_",
        "metric_inverse_", "direction_", "anchor_intercept_", "anchor_coef_",
        "fallback_threshold_", "metadata_", "history_"),
    "IterativeMGPA": ("n_features_in_", "basis_", "anchor_basis_", "frozen_anchor_",
        "stages_", "history_", "metadata_"),
    "IGBP": ("n_features_in_", "stages_", "history_", "metadata_", "endpoints_"),
}
_AFFINE_STATE = ("n_features_in_", "weight_", "bias_", "metadata_", "history_")


def _encode(value, arrays):
    """Encode allowed model state while storing numeric arrays in the NPZ payload."""
    _, records = _types()
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
            raise ValueError("Only finite numeric arrays may be serialized")
        key = f"array_{len(arrays):05d}"
        arrays[key] = value.copy()
        return {"__array__": key}
    if type(value).__name__ in records and type(value) is records[type(value).__name__]:
        return {"__record__": type(value).__name__,
                "fields": {f.name: _encode(getattr(value, f.name), arrays) for f in fields(value)}}
    if isinstance(value, dict):
        if any(not isinstance(k, str) or k in ("__array__", "__record__") for k in value):
            raise ValueError("Invalid metadata dictionary key")
        return {k: _encode(v, arrays) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(v, arrays) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported model-state type: {type(value).__name__}")


def _decode(value, arrays):
    """Reconstruct allowed records and arrays without artifact-directed imports."""
    _, records = _types()
    if isinstance(value, dict):
        if "__array__" in value:
            if set(value) != {"__array__"} or value["__array__"] not in arrays:
                raise ValueError("Invalid numerical-state reference")
            return arrays[value["__array__"]].copy()
        if "__record__" in value:
            if set(value) != {"__record__", "fields"} or value["__record__"] not in records:
                raise ValueError("Unknown portable record")
            cls = records[value["__record__"]]
            if set(value["fields"]) != {f.name for f in fields(cls)}:
                raise ValueError("Portable record fields changed")
            return cls(**{k: _decode(v, arrays) for k, v in value["fields"].items()})
        return {k: _decode(v, arrays) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v, arrays) for v in value]
    return value


def _pack(model, arrays):
    """Collect constructor parameters and the declared fitted state of one estimator."""
    models, _ = _types()
    kind = type(model).__name__
    if kind not in models or type(model) is not models[kind]:
        raise TypeError("Unsupported estimator class")
    model._require_fitted()
    if kind == "TokenOffsetAdapter":
        return dict(kind=kind, inner=_pack(model.inner, arrays), token_shape=model.token_shape_)
    if kind == "ClosedFormMGPA":
        parameters = dict(ridge_alpha=model.ridge_alpha, shrinkage=model.shrinkage)
    elif kind in ("IterativeMGPA", "IGBP"):
        parameters = model.params
    elif kind == "Identity":
        parameters = {}
    elif kind == "LEACE":
        parameters = dict(ridge=model.ridge)
    elif kind == "CORAL":
        parameters = dict(reference=model.reference, ridge=model.ridge, moments=model.moments)
    else:
        parameters = dict(reference=model.reference, ridge=model.ridge)
    names = _STATE.get(kind, _AFFINE_STATE)
    return dict(kind=kind, parameters=_encode(parameters, arrays),
                state={name: _encode(getattr(model, name), arrays) for name in names})


def _unpack(record, arrays):
    """Reconstruct an estimator from its closed schema and validate stored dimensions."""
    models, _ = _types()
    kind = record.get("kind")
    if kind not in models:
        raise ValueError("Unknown estimator class")
    if kind == "TokenOffsetAdapter":
        if set(record) != {"kind", "inner", "token_shape"}:
            raise ValueError("Invalid token wrapper schema")
        return models[kind](_unpack(record["inner"], arrays), token_shape=record["token_shape"])
    if set(record) != {"kind", "parameters", "state"}:
        raise ValueError("Invalid model schema")
    names = _STATE.get(kind, _AFFINE_STATE)
    if set(record["state"]) != set(names):
        raise ValueError("Unexpected estimator state fields")
    model = models[kind](**_decode(record["parameters"], arrays))
    for name, value in record["state"].items():
        setattr(model, name, _decode(value, arrays))
    width = model.n_features_in_
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError("Invalid representation width")
    if kind in ("Identity", "LEACE", "CORAL", "FEATMAP"):
        if model.weight_.shape != (width, width) or model.bias_.shape != (width,):
            raise ValueError("Affine state dimensions disagree")
    elif kind == "ClosedFormMGPA":
        rank = model.basis_.shape[1]
        shapes = {"basis_": (width, rank), "complement_": (width, width-rank),
            "covariance_": (rank, rank), "metric_inverse_": (rank, rank),
            "direction_": (rank,), "anchor_intercept_": (rank,), "anchor_coef_": (width-rank, rank)}
        if any(getattr(model, name).shape != shape for name, shape in shapes.items()):
            raise ValueError("Closed-form state dimensions disagree")
    return model


def save_model(model, directory):
    """Create model.json/model.npz, preserving dtype and refusing an overwrite."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    index, state = directory/"model.json", directory/"model.npz"
    if index.exists() or state.exists():
        raise FileExistsError(f"Model artifact already exists: {directory}")
    arrays = {}
    record = _pack(model, arrays)
    # Validate JSON before writing either file.
    json.dumps(record, allow_nan=False)
    np.savez_compressed(state, **arrays)
    payload = dict(schema=SCHEMA, model=record, state_file="model.npz",
                   state_sha256=hashlib.sha256(state.read_bytes()).hexdigest())
    index.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)+"\n")
    return index


def load_model(directory):
    """Load a declared estimator and verify its numeric-state SHA256."""
    directory = Path(directory)
    if directory.name == "model.json":
        directory = directory.parent
    payload = json.loads((directory/"model.json").read_text())
    if payload.get("schema") != SCHEMA or payload.get("state_file") != "model.npz":
        raise ValueError("Unknown portable model schema")
    state = directory/"model.npz"
    if hashlib.sha256(state.read_bytes()).hexdigest() != payload["state_sha256"]:
        raise ValueError("Numerical-state SHA256 mismatch")
    with np.load(state, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    if any(a.dtype.kind not in "biuf" or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Only finite numeric model state is allowed")
    return _unpack(payload["model"], arrays)
