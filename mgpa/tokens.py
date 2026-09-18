"""Common-offset deployment on recorded EEGPT tokens."""
import numpy as np

from .core import PortableEstimator


def pooled(tokens):
    """Mean over tokens, accumulated in float64 and stored as float32."""
    x = np.asarray(tokens, dtype=np.float32)
    if x.ndim != 3 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError("Expected finite [record, token, width] values")
    return x.mean(axis=1, dtype=np.float64).astype(np.float32)


class TokenOffsetAdapter(PortableEstimator):
    """Lift a fitted vector estimator to a shared offset for every token.

    The inner map acts on the token mean. Adding its displacement to every
    token preserves centered token residuals to floating-point precision.
    This adapter can wrap an already fitted estimator or fit it via ``fit``.
    """
    def __init__(self, inner, token_shape=None):
        """Wrap a vector estimator and optionally declare its fixed token layout."""
        self.inner = inner
        self.token_shape_ = None if token_shape is None else tuple(token_shape)
        if hasattr(inner, "n_features_in_"):
            self.n_features_in_ = inner.n_features_in_
            self.metadata_ = dict(inner.metadata_, interface="common_token_offset")

    def fit(self, tokens, source, basis=None, pairs=None, *, validation_x=None, **kwargs):
        """Fit the inner estimator on token means using the supplied FIT inputs."""
        x = pooled(tokens)
        self.token_shape_ = tuple(np.asarray(tokens).shape[1:])
        if validation_x is not None:
            kwargs["validation_x"] = pooled(validation_x)
        if basis is not None:
            kwargs["basis"] = basis
        if pairs is not None:
            kwargs["pairs"] = pairs
        self.inner.fit(x, source, **kwargs)
        self.n_features_in_ = self.inner.n_features_in_
        self.metadata_ = dict(self.inner.metadata_, interface="common_token_offset")
        return self

    def transform(self, tokens, source_ids=None, **kwargs):
        """Add the inner mean correction equally to every token of each recording."""
        self._require_fitted()
        x = np.asarray(tokens, dtype=np.float32)
        mean = pooled(x)
        if mean.shape[1] != self.n_features_in_ or (self.token_shape_ is not None and x.shape[1:] != self.token_shape_):
            raise ValueError("Token shape changed")
        correction = self.inner.transform(mean, source_ids=source_ids, **kwargs)-mean
        return (x+correction[:, None, :]).astype(np.float32)
