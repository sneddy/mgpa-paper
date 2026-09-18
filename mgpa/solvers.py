"""Batched Euclidean trust-region solver, faithful float64 dual bisection.

Port of the project's MIT-licensed trust_region.euclidean_trust_region.
Minimize .5||e + Az||² + .5*damping*||z||² subject to ||z|| <= radius.
"""
import math
import torch


def euclidean_trust_region(matrix, error, damping, radius, *, iterations=32):
    """Solve batched damped least squares inside the declared Euclidean radius."""
    if not isinstance(matrix, torch.Tensor) or not isinstance(error, torch.Tensor):
        raise TypeError("matrix and error must be torch tensors")
    if matrix.ndim != 3 or error.shape != matrix.shape[:2] or matrix.shape[1] == 0:
        raise ValueError("Expected matrix[n,m,r] and error[n,m], m>=1")
    if matrix.device != error.device or not matrix.is_floating_point() or not error.is_floating_point():
        raise ValueError("Inputs must be floating point on the same device")
    if not torch.isfinite(matrix).all() or not torch.isfinite(error).all():
        raise ValueError("Nonfinite trust-region input")
    damping = float(damping)
    if not math.isfinite(damping) or damping <= 0:
        raise ValueError("damping must be finite and positive")
    if int(iterations) != iterations or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    a, e = matrix.detach().to(torch.float64), error.detach().to(torch.float64)
    radii = torch.as_tensor(radius, dtype=torch.float64, device=matrix.device).expand(matrix.shape[0])
    if not torch.isfinite(radii).all() or (radii < 0).any():
        raise ValueError("radius must be finite and nonnegative")
    eye = torch.eye(a.shape[1], dtype=torch.float64, device=matrix.device)
    gram = a @ a.transpose(1, 2)

    def solve(aa, gg, ee, dual):
        """Solve the batched dual-regularized normal equations in float64."""
        coefficient = torch.linalg.solve(gg + (damping + dual)[:, None, None] * eye, ee[..., None])
        return -(aa.transpose(1, 2) @ coefficient).squeeze(-1)

    dual = torch.zeros(a.shape[0], dtype=torch.float64, device=matrix.device)
    result = solve(a, gram, e, dual)
    lengths = torch.linalg.vector_norm(result, dim=1)
    zero_radius = radii == 0
    active = (lengths > radii) & ~zero_radius
    result[zero_radius] = 0
    if active.any():
        aa, gg, ee, rr = a[active], gram[active], e[active], radii[active]
        rhs = (aa.transpose(1, 2) @ ee[..., None]).squeeze(-1)
        high = torch.linalg.vector_norm(rhs, dim=1) / rr
        low = torch.zeros_like(high)
        for _ in range(int(iterations)):
            middle = .5 * (low + high)
            candidate = solve(aa, gg, ee, middle)
            outside = torch.linalg.vector_norm(candidate, dim=1) > rr
            low = torch.where(outside, middle, low)
            high = torch.where(outside, high, middle)
        result[active] = solve(aa, gg, ee, high)
        dual[active] = high
    if not torch.isfinite(result).all():
        raise FloatingPointError("Nonfinite trust-region solution")
    return result.to(matrix.dtype), dual
