from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .ssvep_data import load_trials

from .ssvep_encoder import EMBED_DIM, N_PATCHES, FrozenEEGPTEncoder


REFERENCE_VIEW_NAMES = ("car", "poz_reference", "oz_reference")
REFERENCE_VIEW_VERSION = "eegpt-exact-algebraic-reference-v1"


def _reference_weights(name: str) -> np.ndarray:
    """Return weights for a reference waveform formed from recorded channels."""

    weights = np.zeros(8, dtype=np.float32)
    if name == "car":
        weights[:] = 1.0 / 8.0
    elif name == "poz_reference":
        weights[0] = 1.0
    elif name == "oz_reference":
        weights[5] = 1.0
    else:
        raise ValueError(f"Unknown reference view: {name}")
    if not np.isclose(weights.sum(), 1.0):
        raise AssertionError("A re-reference waveform must be an affine channel average")
    return weights


def apply_reference_view(trials: np.ndarray, name: str) -> np.ndarray:
    """Re-reference stored trials while preserving all channel contrasts.

    The separately recorded forehead reference is not included in the released
    eight-channel tensors.  These are therefore exact *algebraic* views of the
    stored recording, not claims of a second physical acquisition.  For any
    channels ``i`` and ``j``, ``y_i - y_j == x_i - x_j``.
    """

    values = np.asarray(trials, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (8, 710):
        raise ValueError(f"Expected [trial,8,710], got {values.shape}")
    weights = _reference_weights(name)
    reference = np.einsum("c,nct->nt", weights, values, optimize=True)
    output = values - reference[:, None, :]
    if output.shape != values.shape or not np.isfinite(output).all():
        raise AssertionError("Invalid re-referenced trial tensor")
    return output.astype(np.float32)


def exact_reference_views(trials: np.ndarray) -> dict[str, np.ndarray]:
    """Construct CAR, POz, and Oz algebraic views of each recording."""
    return {
        name: apply_reference_view(trials, name)
        for name in REFERENCE_VIEW_NAMES
    }


def _hash(values: np.ndarray) -> str:
    """Hash the ordered string identities of a calibration or reference set."""
    return hashlib.sha256("\n".join(values.astype(str)).encode()).hexdigest()


def load_or_create_reference_grid(
    *,
    cache_path: Path,
    index: pd.DataFrame,
    encoder: FrozenEEGPTEncoder,
) -> np.ndarray:
    """Encode deterministic reference views once, outside any fold fitting.

    The returned memory map has shape ``[trial, view, 15, 512]``.  It is safe
    to share across folds because no statistic, source label, or target label
    is fitted when constructing an algebraic re-reference.
    """

    cache_path = Path(cache_path)
    manifest_path = cache_path.with_suffix(".json")
    sample_ids = index["sample_id"].astype(str).to_numpy()
    expected_shape = (
        len(index),
        len(REFERENCE_VIEW_NAMES),
        N_PATCHES,
        EMBED_DIM,
    )
    sample_hash = _hash(sample_ids)
    if cache_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("version") != REFERENCE_VIEW_VERSION
            or manifest.get("sample_hash") != sample_hash
            or tuple(manifest.get("view_names", ())) != REFERENCE_VIEW_NAMES
            or tuple(manifest.get("shape", ())) != expected_shape
        ):
            raise AssertionError(f"Stale reference-view cache: {cache_path}")
        values = np.load(cache_path, mmap_mode="r")
        if values.shape != expected_shape or not np.isfinite(values[:1]).all():
            raise AssertionError("Invalid reference-view embedding cache")
        return values

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    values = np.lib.format.open_memmap(
        cache_path,
        mode="w+",
        dtype=np.float16,
        shape=expected_shape,
    )
    positions = pd.Series(np.arange(len(index)), index=index.index)
    groups = list(index.groupby("subject", sort=True))
    for number, (subject, metadata) in enumerate(groups, start=1):
        rows = positions.loc[metadata.index].to_numpy(dtype=np.int64)
        trials = load_trials(metadata)
        views = exact_reference_views(trials)
        for view_number, name in enumerate(REFERENCE_VIEW_NAMES):
            encoded = encoder.encode_patch_mean_grid(views[name]).reshape(
                len(metadata), N_PATCHES, EMBED_DIM
            )
            values[rows, view_number] = encoded.astype(np.float16)
        if number == 1 or number % 10 == 0 or number == len(groups):
            print(
                f"EEGPT exact-reference extraction {number:3d}/{len(groups)} "
                f"S{int(subject):03d}",
                flush=True,
            )
    values.flush()
    manifest_path.write_text(
        json.dumps(
            {
                "version": REFERENCE_VIEW_VERSION,
                "sample_hash": sample_hash,
                "view_names": list(REFERENCE_VIEW_NAMES),
                "shape": list(expected_shape),
                "dtype": "float16",
                "construction": {
                    "car": "subtract the mean of all eight recorded channels",
                    "poz_reference": "subtract recorded POz from every channel",
                    "oz_reference": "subtract recorded Oz from every channel",
                },
                "uses_source_labels": False,
                "uses_target_labels": False,
                "interpretation": "exact algebraic views, not repeat acquisitions",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return np.load(cache_path, mmap_mode="r")
