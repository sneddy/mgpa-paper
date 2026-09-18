from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .ssvep_data import deterministic_subsample, load_trials
from .ssvep_transport import fit_source_transport, source_transport_views

from .ssvep_encoder import FrozenEEGPTEncoder


VIEW_NAMES = tuple(
    f"transport_{kind}_opposite@{alpha:g}"
    for kind in ("spectral", "spatial", "combined")
    for alpha in (0.5, 1.0)
)
VIEW_CACHE_VERSION = "eegpt-fold-local-source-transport-v1"


def _hash(values: list[str] | np.ndarray) -> str:
    """Hash the ordered string identities of a calibration or reference set."""
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


@dataclass(frozen=True)
class ViewPartition:
    """One role's original token grid and matching synthetic views."""
    sample_ids: np.ndarray
    raw: np.ndarray
    views: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class ViewBundle:
    """Verified calibration partitions and their transport provenance."""
    parts: dict[str, ViewPartition]
    view_names: tuple[str, ...]
    transport_fit_hash: str
    cache_path: Path


def load_or_create_view_bundle(
    *,
    cache_path: Path,
    raw_sequence: np.ndarray,
    transport_fit_metadata: pd.DataFrame,
    partitions: dict[str, pd.DataFrame],
    encoder: FrozenEEGPTEncoder,
    seed: int,
    max_transport_trials: int = 4096,
) -> ViewBundle:
    """Create fold-local controlled views with train-only transport fitting."""

    cache_path = Path(cache_path)
    manifest_path = cache_path.with_suffix(".json")
    transport_metadata = deterministic_subsample(
        transport_fit_metadata,
        min(max_transport_trials, len(transport_fit_metadata)),
        seed=seed,
        group_columns=("subject", "source"),
    )
    transport_hash = _hash(transport_metadata["sample_id"].astype(str).tolist())
    expected_ids = {
        name: metadata["sample_id"].astype(str).to_numpy(dtype=str)
        for name, metadata in partitions.items()
    }
    if cache_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("version") != VIEW_CACHE_VERSION
            or manifest.get("transport_fit_hash") != transport_hash
            or tuple(manifest.get("view_names", ())) != VIEW_NAMES
        ):
            raise AssertionError(f"Stale or incompatible controlled-view cache: {cache_path}")
        # Early v1 caches wrote pandas string columns as NumPy ``object``
        # arrays.  They are local, ignored artifacts; accept them only after
        # proving every recovered element is a plain string.  New caches are
        # written as fixed-width Unicode above and need no pickle payload.
        with np.load(cache_path, allow_pickle=False) as payload:
            output: dict[str, ViewPartition] = {}
            for name, ids in expected_ids.items():
                cached_raw_ids = payload[f"{name}_sample_ids"]
                if cached_raw_ids.dtype.hasobject and not all(
                    isinstance(value, str) for value in cached_raw_ids.tolist()
                ):
                    raise AssertionError("Legacy controlled-view IDs are not strings")
                cached_ids = cached_raw_ids.astype(str)
                if not np.array_equal(cached_ids, ids):
                    raise AssertionError(f"Controlled-view IDs mismatch for {name}")
                raw = payload[f"{name}_raw"].astype(np.float32)
                views = tuple(
                    payload[f"{name}_view_{number}"].astype(np.float32)
                    for number in range(len(VIEW_NAMES))
                )
                output[name] = ViewPartition(cached_ids, raw, views)
        return ViewBundle(output, VIEW_NAMES, transport_hash, cache_path)

    transport_trials = load_trials(transport_metadata)
    transport = fit_source_transport(
        transport_trials,
        transport_metadata["source"].to_numpy(dtype=int),
    )
    del transport_trials
    arrays: dict[str, np.ndarray] = {}
    output: dict[str, ViewPartition] = {}
    for part_name, metadata in partitions.items():
        ids = expected_ids[part_name]
        raw = np.asarray(
            raw_sequence[metadata.index.to_numpy(dtype=int)], dtype=np.float32
        )
        trials = load_trials(metadata)
        generated = source_transport_views(
            trials,
            metadata["source"].to_numpy(dtype=int),
            transport,
            alphas=(0.5, 1.0),
            target="opposite",
        )
        if tuple(generated) != VIEW_NAMES:
            raise AssertionError("Controlled-view ordering changed")
        encoded: list[np.ndarray] = []
        for view_name, transformed in generated.items():
            encoded_view = encoder.encode_patch_mean_grid(transformed).reshape(raw.shape)
            encoded.append(encoded_view.astype(np.float32))
            print(
                f"encoded {cache_path.stem} {part_name} {view_name} n={len(metadata)}",
                flush=True,
            )
        output[part_name] = ViewPartition(ids, raw, tuple(encoded))
        arrays[f"{part_name}_sample_ids"] = ids
        arrays[f"{part_name}_raw"] = raw.astype(np.float16)
        for number, value in enumerate(encoded):
            arrays[f"{part_name}_view_{number}"] = value.astype(np.float16)
        del trials, generated, encoded

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **arrays)
    manifest = {
        "version": VIEW_CACHE_VERSION,
        "transport_fit_hash": transport_hash,
        "transport_fit_n": int(len(transport_metadata)),
        "transport_fit_subjects": sorted(
            map(int, transport_metadata["subject"].unique())
        ),
        "view_names": list(VIEW_NAMES),
        "partitions": {
            name: {
                "n": int(len(metadata)),
                "sample_hash": _hash(ids.tolist()),
                "subjects": sorted(map(int, metadata["subject"].unique())),
            }
            for (name, metadata), ids in zip(partitions.items(), expected_ids.values())
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return ViewBundle(output, VIEW_NAMES, transport_hash, cache_path)
