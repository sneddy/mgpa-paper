"""The FIT-only coordinate and gate recipes used by the published experiments.

Operation order and precision intentionally follow their respective frozen
protocols; unifying them numerically would change the experiment.
"""
from __future__ import annotations

import numpy as np


def fit_standardizer(fit, floor):
    """Fit the released population mean and scale on FIT feature rows only."""
    x = np.asarray(fit, np.float32).reshape(-1, fit.shape[-1])
    center = x.mean(0, dtype=np.float64)
    scale = np.sqrt(np.mean((x.astype(np.float64) - center) ** 2, axis=0, dtype=np.float64))
    return {"center": center.astype(np.float32), "scale": np.maximum(scale, floor).astype(np.float32)}


def standardize(state, x):
    """Apply saved float32 centering and scaling in the original operation order."""
    return ((np.asarray(x, np.float32) - state["center"]) / state["scale"]).astype(np.float32)


def controlled_gate(full_pairs, state, config):
    """Estimate a normalized full-reference contrast span from FIT calibration."""
    transformed = standardize(state, full_pairs)
    delta = (transformed[:, 1] - transformed[:, 0]).astype(np.float64)
    available = len(delta)
    if len(delta) > config["measurement_max_rows"]:
        delta = delta[np.linspace(0, len(delta) - 1, config["measurement_max_rows"], dtype=np.int64)]
    norms = np.linalg.norm(delta, axis=1)
    threshold = max(float(np.quantile(norms, config["measurement_norm_floor_quantile"])), 1e-8)
    keep = norms > threshold
    if not keep.any():
        keep = norms > 1e-8
    if not keep.any():
        return np.empty((delta.shape[1], 0), np.float32), {"null_gate": True, "rank": 0}
    matrix = delta[keep] / norms[keep, None]
    _, singular, right = np.linalg.svd(matrix, full_matrices=False)
    cumulative = np.cumsum(singular ** 2) / np.sum(singular ** 2)
    rank = min(int(np.searchsorted(cumulative, config["retained_measurement_energy"]) + 1), right.shape[0])
    basis = np.linalg.qr(right[:rank].T, mode="reduced")[0].astype(np.float32)
    return basis, {"null_gate": False, "rank": rank, "rows_available": available, "rows_used": int(keep.sum()),
                   "direction_threshold": config["retained_measurement_energy"], "max_rows": config["measurement_max_rows"],
                   "row_subsampling": "deterministic linspace before norm quantile", "norm_selection": "strictly above max(q05,1e-8)"}


def natural_gate(reference, alternatives, *, energy=.99):
    """Estimate the gate from normalized individual-token virtual contrasts."""
    if not 0 < energy <= 1:
        raise ValueError("Natural gate energy must lie in (0,1]")
    width = reference.shape[-1]
    covariance = np.zeros((width, width), np.float64)
    count = 0
    for alternative in alternatives:
        delta = (alternative - reference).reshape(-1, width).astype(np.float64)
        delta /= np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-8)
        covariance += delta.T @ delta
        count += len(delta)
    covariance /= max(count, 1)
    eig, vec = np.linalg.eigh(.5 * (covariance + covariance.T))
    order = np.argsort(eig)[::-1]
    eig, vec = np.maximum(eig[order], 0.), vec[:, order]
    if eig.sum() <= 1e-8:
        return np.empty((width, 0), np.float32), {"rank": 0, "null_gate": True}
    cumulative = np.cumsum(eig) / eig.sum()
    rank = min(int(np.searchsorted(cumulative, energy) + 1), width)
    return vec[:, :rank].astype(np.float32), {"rank": rank, "null_gate": False, "token_contrasts": count,
        "energy_threshold": energy, "normalization": "each token delta / max(norm,1e-8); no q05 filtering"}


def transform_coordinates(state, values):
    """Apply fitted coordinates without estimating new input statistics."""
    x = np.asarray(values, dtype=np.float64)
    if "pca_components" in state:
        x = (x - state["pca_mean"]) @ state["pca_components"].T
    return ((x - state["center"]) / state["scale"]).astype(np.float32)


def fit_coordinates(fit, *, pca_rank=None):
    """Fit temporal paired coordinates and their directional-energy gate."""
    fit = np.asarray(fit)
    if fit.ndim != 3 or fit.shape[1] != 2 or not np.isfinite(fit).all() or not len(fit):
        raise ValueError("Need finite paired [event,2,feature] FIT")
    flat = np.concatenate([fit[:, 0], fit[:, 1]]).astype(np.float64)
    state = {}
    if pca_rank is not None:
        from sklearn.decomposition import PCA
        if not 0 < pca_rank <= min(flat.shape):
            raise ValueError("Infeasible PCA rank; no silent reduction")
        pca = PCA(n_components=pca_rank, svd_solver="full")
        projected = pca.fit_transform(flat)
        state.update(pca_mean=pca.mean_, pca_components=pca.components_, pca_explained_variance_ratio=pca.explained_variance_ratio_)
    else:
        projected = flat
    state.update(center=projected.mean(0, dtype=np.float64), scale=np.maximum(projected.std(0, dtype=np.float64), 1e-8))
    x = transform_coordinates(state, fit)
    delta = (x[:, 1] - x[:, 0]).astype(np.float64)
    norms = np.linalg.norm(delta, axis=1)
    nonzero = norms > 1e-10
    if not nonzero.any():
        basis, used = np.empty((x.shape[-1], 0), dtype=np.float64), np.zeros(len(x), bool)
        energy = np.zeros(x.shape[-1])
    else:
        used = nonzero & (norms >= np.quantile(norms[nonzero], .05))
        directions = delta[used] / norms[used, None]
        eig, vec = np.linalg.eigh(directions.T @ directions)
        order = np.argsort(eig)[::-1]
        eig, vec = np.maximum(eig[order], 0), vec[:, order]
        energy = eig / eig.sum()
        rank = int(np.searchsorted(np.cumsum(energy), .9) + 1)
        basis = vec[:, :rank]
    state.update(basis=np.asarray(basis, np.float64), gate_energy=energy)
    return state, {"rank": basis.shape[1], "null_gate": basis.shape[1] == 0,
        "fit_pairs": len(x), "used_contrasts": int(used.sum()), "energy_threshold": .9,
        "contrast_floor": 1e-10, "nonzero_norm_quantile": .05, "scale_floor": 1e-8,
        "pca_rank": pca_rank, "fit_only": True, "task_labels_used": False}
