"""Numerical primitives for the published measurement-gated correction.

Inputs to these kernels are already in FIT-standardized coordinates. A basis
contains orthonormal columns, not a dense projection matrix. Fitted stages
store numerical critic weights, so replay never imports an experiment runner.
"""
from dataclasses import dataclass

import numpy as np
import torch

from .critics import matrix
from .solvers import euclidean_trust_region


class EndpointUnavailableError(ValueError):
    """A requested movement was not reached by the fitted trajectory."""


@dataclass
class Endpoint:
    """A fixed replay choice after selection using validation inputs.

    ``stages`` includes the fractional final stage. ``strength`` is used only
    for affine maps. An unattained budget cannot silently become a native map.
    """
    stages: int = 0
    fraction: float = 1.
    attained: bool = True
    requested_movement: float | None = None
    validation_movement: float = 0.
    reason: str = ""
    strength: float = 1.


def relative_movement(original, current):
    """Return ||current-original||F / ||original||F in float64."""
    original = np.asarray(original, dtype=np.float64)
    denominator = float(np.linalg.norm(original))
    if denominator <= 0:
        raise ValueError("Movement is undefined for an all-zero reference")
    return float(np.linalg.norm(np.asarray(current, dtype=np.float64)-original)/denominator)


def quotient(x, basis):
    """Original ambient complement Q = X - X B Bᵀ, preserving float32 order."""
    return np.asarray(x, dtype=np.float32) - (x @ basis) @ basis.T


def validate_basis(basis, width):
    """Validate a supplied float32 gate without silently repairing its QR."""
    b = np.asarray(basis, dtype=np.float32).copy()
    if b.ndim != 2 or b.shape[0] != width or b.shape[1] > width or not np.isfinite(b).all():
        raise ValueError("Require finite [width, rank] basis")
    if not np.allclose(b.T @ b, np.eye(b.shape[1]), atol=2e-6):
        raise ValueError("Basis columns must be orthonormal")
    return b


def permission_basis(arm, x_fit, measurement_basis, *, random_seed=None):
    """Construct the published measurement/PCA/random/ungated edit permission.

    PCA uses only observed FIT rows. Random frames use sign-corrected Gaussian
    QR. The caller passes the original measurement basis as ``anchor_basis``
    when fitting a PCA, random, or ungated control; all condition on original Q₀.
    """
    x = matrix(x_fit, "x_fit")
    measurement = validate_basis(measurement_basis, x.shape[1])
    width, rank = measurement.shape
    if arm == "measurement":
        return measurement.copy()
    if arm == "ungated":
        return np.eye(width, dtype=np.float32)
    if arm == "random":
        if isinstance(random_seed, bool) or not isinstance(random_seed, (int, np.integer)) or random_seed < 0:
            raise ValueError("random_seed must be a nonnegative integer")
        frame, triangular = np.linalg.qr(np.random.default_rng(random_seed).normal(size=(width, rank)), mode="reduced")
        return validate_basis(frame*np.where(np.diag(triangular) < 0, -1., 1.), width)
    if arm != "pca":
        raise ValueError("arm must be measurement, pca, random, or ungated")
    if not rank:
        return np.empty((width, 0), dtype=np.float32)
    centered = x.astype(np.float64)-x.mean(0, dtype=np.float64)
    _, _, right = np.linalg.svd(centered, full_matrices=len(x) < width)
    frame = right[:rank].T.copy()
    pivots = np.argmax(np.abs(frame), axis=0)
    frame *= np.where(frame[pivots, np.arange(rank)] < 0, -1., 1.)
    return validate_basis(frame, width)


def crossing_fraction(original, before, after, budget):
    """Published 48-bisection interpolation at the first upward VAL crossing."""
    left, right = 0., 1.
    for _ in range(48):
        middle = (left+right)/2
        if relative_movement(original, before+middle*(after-before)) < budget:
            left = middle
        else:
            right = middle
    return (left+right)/2


@dataclass
class ScoreStage:
    """One smooth, measurement-gated, Euclidean trust-region update.

    Conditional anchors predict each current critic score from original FIT Q.
    ``frozen_original_q`` detaches Q₀ for edit-permission controls. Ordinary
    measurement replay retains the original quotient-gradient expression.
    """
    full: tuple
    anchors: tuple
    edit_basis: np.ndarray
    anchor_basis: np.ndarray
    target: str = "conditional"
    floor: float = .1
    damping: float = .001
    max_step: float = .05
    frozen_original_q: bool = False

    def apply(self, x, fraction=1., *, original_q0=None):
        """Apply this frozen stage, interpolating only the correction displacement."""
        if not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("fraction must be in [0,1]")
        if self.target not in ("conditional", "fit_mean", "zero"):
            raise ValueError("Unknown target")
        x = matrix(x)
        if self.frozen_original_q:
            q0 = matrix(original_q0, "original_q0")
            if q0.shape != x.shape:
                raise ValueError("Original Q0 must align with current rows")
        b = torch.as_tensor(self.edit_basis, dtype=torch.float32)
        qb = torch.as_tensor(self.anchor_basis, dtype=torch.float32)
        output = []
        for start in range(0, len(x), 512):
            current = torch.tensor(x[start:start+512], dtype=torch.float32, requires_grad=True)
            q = (torch.as_tensor(q0[start:start+512], dtype=torch.float32).detach()
                 if self.frozen_original_q else current-(current @ qb) @ qb.T)
            rows, errors = [], []
            for full, anchor in zip(self.full, self.anchors):
                score = full.score_tensor(current)
                if self.target == "zero":
                    error = score
                elif self.frozen_original_q:
                    with torch.no_grad():
                        target = anchor.score_tensor(q)
                    error = score-target
                else:
                    error = score-anchor.score_tensor(q)
                grad = torch.autograd.grad(error.sum(), current, retain_graph=True)[0] @ b
                weight = torch.rsqrt(torch.sum(grad*grad, dim=1)+self.floor**2)
                rows.append(grad*weight[:, None])
                errors.append(error*weight)
            a, e = torch.stack(rows, 1), torch.stack(errors, 1)
            coefficient = torch.linalg.solve(a @ a.transpose(1, 2)+self.damping*torch.eye(len(rows)), e[..., None])
            delta = -(a.transpose(1, 2) @ coefficient).squeeze(-1) @ b.T
            lengths = torch.linalg.vector_norm(delta, dim=1)
            cap = self.max_step*np.sqrt(x.shape[1])
            chosen = delta.detach().clone()
            constrained = lengths > cap
            if constrained.any():
                z, _ = euclidean_trust_region(a[constrained], e[constrained], self.damping, cap)
                transformed = z @ b.T
                guard = (cap/torch.linalg.vector_norm(transformed, dim=1).clamp(min=1e-12)).clamp(max=1.)
                chosen[constrained] = transformed*guard[:, None]
            output.append((current+fraction*chosen).detach().numpy())
        result = np.concatenate(output).astype(np.float32)
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite score-map output")
        return result


class PortableEstimator:
    """Common numerical-state serialization for public estimators."""
    def save(self, directory):
        """Write a portable model artifact without overwriting an existing model."""
        from .serialization import save_model
        return save_model(self, directory)

    @classmethod
    def load(cls, directory):
        """Load and verify an artifact containing this estimator class."""
        from .serialization import load_model
        result = load_model(directory)
        if not isinstance(result, cls):
            raise TypeError(f"Artifact does not contain {cls.__name__}")
        return result

    def _require_fitted(self):
        """Reject inference on an estimator with no fitted coordinate width."""
        if not hasattr(self, "n_features_in_"):
            raise ValueError("Call fit before transform or save")
