"""FIT-only affine comparison methods, with explicit source routing where needed."""
import numpy as np
from scipy.linalg import pinvh

from .closed_form import _pairs
from .core import Endpoint, EndpointUnavailableError, PortableEstimator, relative_movement
from .critics import binary, matrix


def _spd_power(covariance, power):
    """Apply a spectral matrix power to a positive-definite covariance."""
    values, vectors = np.linalg.eigh((covariance+covariance.T)*.5)
    if not np.isfinite(values).all() or values.min() <= 0:
        raise FloatingPointError("Regularized covariance must be positive definite")
    return (vectors*np.power(values, power)[None, :]) @ vectors.T


class _AffineEstimator(PortableEstimator):
    """Common transform and validation-only movement selection."""
    reference = None

    def transform(self, x, strength=1., source_ids=None, *, endpoint=None):
        """Apply the fitted affine map, preserving the reference route if present."""
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
        u = None
        if self.reference is not None:
            if source_ids is None:
                raise ValueError("CORAL and FEATMAP require actual source_ids at deployment")
            u = binary(source_ids, len(x), both=False)
        if strength == 0:
            return x.copy()
        result = x @ self.weight_+self.bias_
        if u is not None:
            result = np.where((u != self.reference)[:, None], result, x)
        result = result if strength == 1 else x+strength*(result-x)
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite affine output")
        return result

    def select_movement_budget(self, validation_x, budget, source_ids=None):
        """Freeze a VAL identity-to-native chord; budget selection never extrapolates."""
        if not np.isfinite(budget) or budget < 0:
            raise ValueError("budget must be finite and nonnegative")
        movement = relative_movement(validation_x, self.transform(validation_x, source_ids=source_ids))
        strength = 0. if budget == 0 else min(1., budget/movement) if movement else 1.
        return Endpoint(strength=strength, attained=budget <= movement,
            requested_movement=float(budget), validation_movement=strength*movement,
            reason="identity-to-native chord")

    def _record(self, x):
        """Initialize shared shape and information-access metadata."""
        self.n_features_in_ = x.shape[1]
        self.history_ = []
        self.metadata_ = dict(source_blind=self.reference is None,
            downstream_labels_used=False, fit_rows=len(x),
            source_ids_required_at_deployment=self.reference is not None)


class Identity(_AffineEstimator):
    """The unchanged representation, in the shared float64 affine interface."""
    def fit(self, x, source=None, *, pairs=None):
        """Record width; no statistic, label, or correction is estimated."""
        x = matrix(x, dtype=np.float64)
        self._record(x)
        self.weight_, self.bias_ = np.eye(x.shape[1]), np.zeros(x.shape[1])
        return self


class LEACE(_AffineEstimator):
    """Covariance-weighted affine erasure of a binary source, fitted on observed FIT.

    ``ridge=0`` is the component-study recipe. Reuse searches the declared
    covariance-ridge grid and ultimately uses zero. This is an oblique eraser;
    no output rescaling or orthogonal-projection substitution is performed.
    """
    def __init__(self, ridge=0.):
        """Set the FIT covariance regularizer used by the affine eraser."""
        if not np.isfinite(ridge) or ridge < 0:
            raise ValueError("ridge must be finite and nonnegative")
        self.ridge = float(ridge)

    def fit(self, x, source, *, pairs=None):
        """Estimate the observed-FIT mean, covariance, and source direction."""
        x = matrix(x, dtype=np.float64)
        u = binary(source, len(x))
        self._record(x)
        mean = x.mean(0)
        centered = x-mean
        direction = centered.T @ (u-u.mean())/len(x)
        covariance = centered.T @ centered/len(x)+self.ridge*np.eye(x.shape[1])
        dual = pinvh(covariance) @ direction
        removed = np.outer(dual/max(float(direction @ dual), 1e-15), direction)
        self.weight_, self.bias_ = np.eye(x.shape[1])-removed, mean @ removed
        self.metadata_.update(covariance_ridge=self.ridge, covariance_ddof=0)
        return self


class CORAL(_AffineEstimator):
    """Centered FIT covariance alignment toward a specified reference source.

    ``moments='observed'`` uses the source-observed FIT bank. For paired
    component studies choose ``moments='calibration'`` and pass all paired
    FIT endpoints; their matching is not used. Covariance uses ddof=1 and
    identity regularization ``ridge``. The reference source stays unchanged.
    """
    def __init__(self, reference=0, ridge=1., moments="observed"):
        """Declare the reference source, covariance ridge, and FIT moment population."""
        if isinstance(reference, bool) or reference not in (0, 1):
            raise ValueError("reference must be 0 or 1")
        if not np.isfinite(ridge) or ridge < 0:
            raise ValueError("ridge must be finite and nonnegative")
        if moments == "paired":
            moments = "calibration"
        if moments not in ("observed", "calibration"):
            raise ValueError("moments must be observed, calibration, or paired")
        self.reference, self.ridge, self.moments = int(reference), float(ridge), moments

    def fit(self, x, source, *, pairs=None):
        """Estimate source-domain means and covariances from declared FIT inputs."""
        x = matrix(x, dtype=np.float64)
        u = binary(source, len(x))
        self._record(x)
        if self.moments == "calibration":
            p = _pairs(pairs, x.shape[1])
            source, target = p[:, 1-self.reference], p[:, self.reference]
        else:
            source, target = x[u != self.reference], x[u == self.reference]
        if min(len(source), len(target)) < 2:
            raise ValueError("CORAL needs at least two FIT observations per source")
        mu_s, mu_t = source.mean(0), target.mean(0)
        cs = np.atleast_2d(np.cov(source, rowvar=False, ddof=1))+self.ridge*np.eye(x.shape[1])
        ct = np.atleast_2d(np.cov(target, rowvar=False, ddof=1))+self.ridge*np.eye(x.shape[1])
        self.weight_ = _spd_power(cs, -.5) @ _spd_power(ct, .5)
        self.bias_ = mu_t-mu_s @ self.weight_
        self.metadata_.update(reference=self.reference, covariance_ridge=self.ridge,
            covariance_ddof=1, moments=self.moments, pair_matching_used=False)
        return self


class FEATMAP(_AffineEstimator):
    """Centered paired affine regression toward a specified reference source.

    ``ridge=0`` uses numpy's minimum-norm least-squares slope (component study).
    Positive ridge uses the published reuse sum-loss regularization. Pair
    endpoints must be actual corresponding measurements; source IDs route
    deployment and the reference device stays unchanged.
    """
    def __init__(self, reference=0, ridge=0.):
        """Declare the unchanged reference source and paired-regression penalty."""
        if isinstance(reference, bool) or reference not in (0, 1):
            raise ValueError("reference must be 0 or 1")
        if not np.isfinite(ridge) or ridge < 0:
            raise ValueError("ridge must be finite and nonnegative")
        self.reference, self.ridge = int(reference), float(ridge)

    def fit(self, x, source=None, *, pairs=None):
        """Fit the centered regression using the supplied corresponding pairs."""
        x = matrix(x, dtype=np.float64)
        if source is not None:
            binary(source, len(x))
        self._record(x)
        p = _pairs(pairs, x.shape[1])
        source, target = p[:, 1-self.reference], p[:, self.reference]
        mu_s, mu_t = source.mean(0), target.mean(0)
        centered_s, centered_t = source-mu_s, target-mu_t
        if self.ridge == 0:
            self.weight_, _, rank, _ = np.linalg.lstsq(centered_s, centered_t, rcond=None)
        else:
            self.weight_ = np.linalg.solve(centered_s.T @ centered_s+self.ridge*np.eye(x.shape[1]),
                                           centered_s.T @ centered_t)
            rank = np.linalg.matrix_rank(centered_s)
        self.bias_ = mu_t-mu_s @ self.weight_
        self.metadata_.update(reference=self.reference, ridge=self.ridge, pair_matching_used=True,
                              fit_pairs=len(p), centered_design_rank=int(rank))
        return self
