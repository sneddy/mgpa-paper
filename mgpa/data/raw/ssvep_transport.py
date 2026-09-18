from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .ssvep_data import prestimulus


@dataclass(frozen=True)
class SourceTransport:
    """Source-conditioned spectral gains and spatial covariance factors."""
    frequency_grid: np.ndarray
    log_psd: np.ndarray
    spatial_covariance: np.ndarray
    spatial_opposite: np.ndarray
    spatial_common: np.ndarray
    sfreq: int


def _matrix_power(matrix: np.ndarray, power: float, floor: float = 1e-6) -> np.ndarray:
    """Apply a floored symmetric matrix power for covariance transport."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    scale = max(float(eigenvalues.max()), floor)
    powered = np.maximum(eigenvalues, floor * scale) ** power
    return (eigenvectors * powered) @ eigenvectors.T


def fit_source_transport(
    trials: np.ndarray,
    source: np.ndarray,
    *,
    sfreq: int = 250,
    n_fft: int = 1024,
    covariance_shrinkage: float = 0.1,
) -> SourceTransport:
    """Fit wet/dry measurement transports from pre-stimulus samples only."""

    values = np.asarray(trials, dtype=np.float32)
    source = np.asarray(source, dtype=np.int64)
    if values.ndim != 3 or values.shape[1:] != (8, 710):
        raise ValueError(f"Expected [trial,8,710], got {values.shape}.")
    if set(np.unique(source)) != {0, 1}:
        raise ValueError("Source transport requires both electrode types.")
    baseline = prestimulus(values)
    baseline = baseline - baseline.mean(axis=-1, keepdims=True)
    window = np.hanning(baseline.shape[-1]).astype(np.float32)
    frequency_grid = np.fft.rfftfreq(n_fft, d=1.0 / sfreq)
    log_psd = np.empty((2, 8, len(frequency_grid)), dtype=np.float64)
    covariance = np.empty((2, 8, 8), dtype=np.float64)
    for domain in (0, 1):
        selected = baseline[source == domain]
        spectrum = np.fft.rfft(selected * window, n=n_fft, axis=-1)
        psd = np.mean(np.abs(spectrum) ** 2, axis=0)
        log_psd[domain] = gaussian_filter1d(np.log(psd + 1e-8), sigma=2.0, axis=-1)
        flat = selected.transpose(0, 2, 1).reshape(-1, 8).astype(np.float64)
        empirical = np.cov(flat, rowvar=False)
        isotropic = np.trace(empirical) / 8.0 * np.eye(8)
        covariance[domain] = (
            (1.0 - covariance_shrinkage) * empirical
            + covariance_shrinkage * isotropic
        )

    common_covariance = 0.5 * (covariance[0] + covariance[1])
    opposite = np.empty_like(covariance)
    common = np.empty_like(covariance)
    for domain in (0, 1):
        opposite[domain] = _matrix_power(covariance[1 - domain], 0.5) @ _matrix_power(
            covariance[domain], -0.5
        )
        common[domain] = _matrix_power(common_covariance, 0.5) @ _matrix_power(
            covariance[domain], -0.5
        )
    return SourceTransport(
        frequency_grid=frequency_grid,
        log_psd=log_psd,
        spatial_covariance=covariance,
        spatial_opposite=opposite,
        spatial_common=common,
        sfreq=sfreq,
    )


def _spectral_transport(
    values: np.ndarray,
    source: np.ndarray,
    transport: SourceTransport,
    *,
    alpha: float,
    target: str,
) -> np.ndarray:
    """Interpolate the fitted spectral transfer toward the target source."""
    length = values.shape[-1]
    frequency_grid = np.fft.rfftfreq(length, d=1.0 / transport.sfreq)
    centered = values - values.mean(axis=-1, keepdims=True)
    spectrum = np.fft.rfft(centered, axis=-1)
    output_spectrum = spectrum.copy()
    common_log_psd = transport.log_psd.mean(axis=0)
    for domain in (0, 1):
        selector = source == domain
        target_log_psd = (
            transport.log_psd[1 - domain] if target == "opposite" else common_log_psd
        )
        log_gain_grid = 0.5 * (target_log_psd - transport.log_psd[domain])
        log_gain = np.vstack(
            [
                np.interp(frequency_grid, transport.frequency_grid, channel_gain)
                for channel_gain in log_gain_grid
            ]
        )
        gain = np.clip(np.exp(float(alpha) * log_gain), 0.25, 4.0)
        gain[:, frequency_grid < 0.5] = 1.0
        output_spectrum[selector] *= gain[None, :, :]
    output = np.fft.irfft(output_spectrum, n=length, axis=-1)
    output += values.mean(axis=-1, keepdims=True)
    return output.astype(np.float32)


def _spatial_transport(
    values: np.ndarray,
    source: np.ndarray,
    transport: SourceTransport,
    *,
    alpha: float,
    target: str,
) -> np.ndarray:
    """Interpolate fitted source whitening and target coloring."""
    means = values.mean(axis=-1, keepdims=True)
    centered = values - means
    output = np.empty_like(centered)
    matrices = transport.spatial_opposite if target == "opposite" else transport.spatial_common
    identity = np.eye(8)
    for domain in (0, 1):
        selector = source == domain
        matrix = (1.0 - float(alpha)) * identity + float(alpha) * matrices[domain]
        output[selector] = np.einsum("ij,njt->nit", matrix, centered[selector], optimize=True)
    return (output + means).astype(np.float32)


def apply_source_transport(
    trials: np.ndarray,
    source: np.ndarray,
    transport: SourceTransport,
    *,
    kind: str,
    alpha: float,
    target: str = "opposite",
) -> np.ndarray:
    """Apply a source-aware intervention; source is needed only in calibration."""

    values = np.asarray(trials, dtype=np.float32)
    source = np.asarray(source, dtype=np.int64)
    if target not in {"opposite", "common"}:
        raise ValueError(target)
    if kind == "spectral":
        output = _spectral_transport(values, source, transport, alpha=alpha, target=target)
    elif kind == "spatial":
        output = _spatial_transport(values, source, transport, alpha=alpha, target=target)
    elif kind == "combined":
        spectral = _spectral_transport(values, source, transport, alpha=alpha, target=target)
        output = _spatial_transport(spectral, source, transport, alpha=alpha, target=target)
    else:
        raise ValueError(kind)
    if output.shape != values.shape or not np.isfinite(output).all():
        raise AssertionError("Source transport produced invalid signal views.")
    return output


def source_transport_views(
    trials: np.ndarray,
    source: np.ndarray,
    transport: SourceTransport,
    *,
    alphas: tuple[float, ...] = (0.5, 1.0),
    target: str = "opposite",
) -> dict[str, np.ndarray]:
    """Construct the six ordered same-trial opposite-source transport views."""
    return {
        f"transport_{kind}_{target}@{alpha:g}": apply_source_transport(
            trials,
            source,
            transport,
            kind=kind,
            alpha=alpha,
            target=target,
        )
        for kind in ("spectral", "spatial", "combined")
        for alpha in alphas
    }
