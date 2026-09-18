"""Published Closed-form MGPA: one paired direction and a pooled Ridge anchor."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .core import Endpoint, EndpointUnavailableError, PortableEstimator, relative_movement
from .critics import binary, matrix


def _pairs(value, width):
    """Validate finite, ordered calibration endpoints of the fitted feature width."""
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 3 or not len(result) or result.shape[1:] != (2, width) or not np.isfinite(result).all():
        raise ValueError("pairs must be finite [pair, 2, width] endpoints")
    return result


def _ridge(q, targets, alpha):
    """Sum-loss Ridge with an unpenalized intercept and unstandardized Q."""
    qbar, ybar = q.mean(0), targets.mean(0)
    qc, yc = q-qbar, targets-ybar
    if q.shape[1]:
        gram, rhs = qc.T @ qc+alpha*np.eye(q.shape[1]), qc.T @ yc
        beta = (np.linalg.lstsq(gram, rhs, rcond=None)[0] if alpha == 0 else
                cho_solve(cho_factor(gram, lower=True), rhs))
    else:
        beta = np.zeros((0, targets.shape[1]))
    return ybar-qbar @ beta, beta


def _joint_metric(z, q, u, ridge_alpha, shrinkage):
    """Observed-FIT residual second moment after joint Z ~ intercept + Q + U.

    Source and intercept coefficients are unpenalized; only Q receives Ridge.
    Operation ordering matches the published fit, including its FWL solve.
    """
    rank = z.shape[1]
    zbar, qbar, p = z.mean(0), q.mean(0), float(u.mean())
    zc, qc, dc = z-zbar, q-qbar, u-p
    if q.shape[1]:
        gram = qc.T @ qc+ridge_alpha*np.eye(q.shape[1])
        rhs = np.column_stack((qc.T @ zc, qc.T @ dc))
        solved = (np.linalg.lstsq(gram, rhs, rcond=None)[0] if ridge_alpha == 0 else
                  cho_solve(cho_factor(gram, lower=True), rhs))
        pooled_beta, source_beta = solved[:, :-1], solved[:, -1]
        residual_d = dc-qc @ source_beta
        overlap = float(dc @ residual_d)
        delta = (dc @ (zc-qc @ pooled_beta))/overlap if overlap > 1e-12 else np.zeros(rank)
        beta = pooled_beta-np.outer(source_beta, delta)
    else:
        beta = np.zeros((0, rank))
        overlap = float(dc @ dc)
        delta = dc @ zc/overlap
    if overlap <= max(1e-12, 1e-12*float(dc @ dc)):
        raise ValueError("Conditional source contrast is not identified")
    intercept = zbar-qbar @ beta-p*delta
    residual = z-intercept-q @ beta-u[:, None]*delta
    covariance = residual.T @ residual/len(z)
    scale = float(np.trace(covariance)/rank)
    covariance = (1.-shrinkage)*covariance+shrinkage*scale*np.eye(rank)
    inverse = (cho_solve(cho_factor(covariance, lower=True), np.eye(rank))
               if shrinkage > 0 and scale > 0 else np.linalg.pinv(covariance, hermitian=True))
    if not np.isfinite(inverse).all():
        raise ValueError("Nonfinite covariance metric")
    return covariance, inverse, dict(covariance_ddof=0, covariance_isotropic_scale=scale,
        effective_residual_source_sum_squares=overlap, source_coefficient_penalized=False)


class ClosedFormMGPA(PortableEstimator):
    """Fit the constant paired direction with the pooled conditional-mean target.

    Parameters
    ----------
    ridge_alpha : float, default=10
        Sum-loss penalty for Q coefficients in the anchor and covariance fits.
    shrinkage : float, default=.01
        Residual-covariance shrinkage toward mean-diagonal identity.

    ``fit(x, source, basis, pairs)`` sees FIT representations/source labels and
    ordered calibration endpoints only; downstream labels are never accepted.
    ``basis`` is retained as supplied. Complete QR constructs its complement.
    ``transform`` accepts strength > 1 because the published reuse map uses 1.1.
    All calculations are float64; standardization belongs to the data pipeline.
    """
    def __init__(self, ridge_alpha=10., shrinkage=.01):
        """Set the pooled-anchor Ridge penalty and residual-covariance shrinkage."""
        if not np.isfinite(ridge_alpha) or ridge_alpha < 0:
            raise ValueError("ridge_alpha must be finite and nonnegative")
        if not np.isfinite(shrinkage) or not 0 <= shrinkage <= 1:
            raise ValueError("shrinkage must be in [0,1]")
        self.ridge_alpha, self.shrinkage = float(ridge_alpha), float(shrinkage)

    def fit(self, x, source, basis, pairs, *, validation_x=None):
        """Estimate the paired direction, pooled anchor, and covariance on FIT."""
        x = matrix(x, "x", dtype=np.float64)
        source = binary(source, len(x)).astype(np.float64)
        b = np.asarray(basis, dtype=np.float64)
        width = x.shape[1]
        if b.ndim != 2 or b.shape[0] != width or b.shape[1] > width or not np.isfinite(b).all():
            raise ValueError("basis must be finite [width,rank]")
        rank = b.shape[1]
        if rank and np.max(np.abs(b.T @ b-np.eye(rank))) > 2e-5:
            raise ValueError("Supplied frozen gate is not approximately orthonormal")
        pairs = _pairs(pairs, width)
        complete, _ = np.linalg.qr(b, mode="complete")
        self.n_features_in_, self.basis_, self.complement_ = width, b.copy(), complete[:, rank:]
        self.metadata_ = dict(source_blind=True, downstream_labels_used=False,
            target="pooled_conditional_mean", direction="mean_calibration_contrast",
            fit_rows=len(x), calibration_pairs=len(pairs), gate_rank=rank,
            ridge_alpha=self.ridge_alpha, covariance_shrinkage=self.shrinkage)
        if rank:
            z, q = x @ b, x @ self.complement_
            self.covariance_, self.metric_inverse_, details = _joint_metric(z, q, source, self.ridge_alpha, self.shrinkage)
            self.direction_ = ((pairs[:, 1]-pairs[:, 0]) @ b).mean(0)
            self.anchor_intercept_, self.anchor_coef_ = _ridge(q, z, self.ridge_alpha)
            norm2 = float(self.direction_ @ self.metric_inverse_ @ self.direction_)
            if not np.isfinite(norm2) or norm2 <= np.finfo(float).tiny:
                raise ValueError("Paired mean has no identifiable correction direction")
            self.fallback_threshold_ = 1e-16*norm2
            self.metadata_.update(details)
        else:
            self.covariance_, self.metric_inverse_ = np.empty((0, 0)), np.empty((0, 0))
            self.direction_, self.anchor_intercept_ = np.empty(0), np.empty(0)
            self.anchor_coef_ = np.empty((width, 0))
            self.fallback_threshold_ = 0.
            self.metadata_["rank_zero_identity"] = True
        self.history_ = []
        if validation_x is not None:
            self.metadata_["native_validation_movement"] = relative_movement(validation_x, self.transform(validation_x))
        return self

    def transform(self, x, strength=1., source_ids=None, *, endpoint=None):
        """Apply one source-blind correction to new standardized rows."""
        self._require_fitted()
        x = matrix(x, "x", dtype=np.float64)
        if x.shape[1] != self.n_features_in_:
            raise ValueError("Representation width changed")
        if endpoint is not None:
            if not endpoint.attained:
                raise EndpointUnavailableError("Movement endpoint was not attained")
            strength = endpoint.strength
        if not np.isfinite(strength) or strength < 0:
            raise ValueError("strength must be finite and nonnegative")
        if strength == 0 or not self.basis_.shape[1]:
            return x.copy()
        z, q = x @ self.basis_, x @ self.complement_
        # Materialize row-major directions as in the published constant arm;
        # a zero-stride broadcast can dispatch a different BLAS operation.
        delta = np.broadcast_to(self.direction_, z.shape).copy()
        dual = delta @ self.metric_inverse_
        norm2 = np.einsum("ij,ij->i", delta, dual)
        valid = norm2 > self.fallback_threshold_
        v = np.zeros_like(delta)
        v[valid] = dual[valid]/norm2[valid, None]
        target = self.anchor_intercept_+q @ self.anchor_coef_
        residual = np.einsum("ij,ij->i", v, z-target)
        result = x-strength*(delta*residual[:, None]) @ self.basis_.T
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite closed-form correction")
        return result

    def select_movement_budget(self, validation_x, budget):
        """Return a VAL-only identity-to-native chord; never extrapolate a budget."""
        if not np.isfinite(budget) or budget < 0:
            raise ValueError("budget must be finite and nonnegative")
        movement = relative_movement(validation_x, self.transform(validation_x))
        strength = 0. if budget == 0 else min(1., budget/movement) if movement else 1.
        return Endpoint(strength=strength, attained=budget <= movement,
            requested_movement=float(budget), validation_movement=strength*movement,
            reason="identity-to-native chord")
