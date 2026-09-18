"""Deterministic CPU critics shared by fitting and independent evaluation.

Adapted from the project's MIT-licensed critic_backend numerical routines.
Hyperparameters are explicit rather than mutable module globals. Independent
readers must supply their own training horizon; method critics default to 12.
"""
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


def matrix(values, name="values", dtype=np.float32):
    """Copy a finite two-dimensional observation matrix into the declared dtype."""
    x = np.asarray(values, dtype=dtype)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError(f"{name} must be a nonempty finite matrix")
    return x.copy()


def binary(values, n, name="source", both=True):
    """Validate aligned 0/1 source labels, optionally requiring both groups."""
    u = np.asarray(values)
    if u.shape != (n,) or not np.isin(u, (0, 1)).all():
        raise ValueError(f"{name} must be aligned binary source IDs")
    if both and len(np.unique(u)) != 2:
        raise ValueError(f"{name} must include both sources")
    return u.astype(np.int64)


@contextmanager
def isolated_torch_seed(seed):
    """Use a reproducible CPU seed without advancing the caller's RNG stream."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed) % (2**31 - 1))
        yield


class ShallowBinaryCritic(nn.Module):
    """Published width→128→2 GELU source classifier."""
    def __init__(self, width, hidden_width=128):
        """Build the binary GELU network for the supplied representation width."""
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(int(width), int(hidden_width)),
                                    nn.GELU(), nn.Linear(int(hidden_width), 2))

    def forward(self, values):
        """Return two logits per observation."""
        return self.layers(values)


class AffineBinaryCritic(nn.Module):
    """Portable balanced-logistic score as symmetric binary logits."""
    def __init__(self, weight, bias):
        """Store a fitted logistic coefficient vector and intercept as fixed tensors."""
        super().__init__()
        self.register_buffer("weight", torch.from_numpy(np.asarray(weight, np.float32).reshape(-1).copy()))
        self.register_buffer("bias", torch.tensor(float(bias), dtype=torch.float32))

    def forward(self, values):
        """Return logits whose class difference is the fitted affine score."""
        score = values @ self.weight + self.bias
        return torch.stack((-.5 * score, .5 * score), dim=1)


def centered_logit(model, values):
    """Extract class-1 minus class-0 logits for binary score matching."""
    logits = model(values)
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("Binary critics must return two logits per row")
    return logits[:, 1] - logits[:, 0]


def fit_affine_critic(values, source, *, seed, c=1.):
    """Fit the published balanced logistic source critic on supplied FIT rows."""
    x = matrix(values)
    u = binary(source, len(x))
    classifier = LogisticRegression(C=float(c), max_iter=2500, class_weight="balanced",
                                    random_state=int(seed) % (2**31 - 1)).fit(x, u)
    return AffineBinaryCritic(classifier.coef_.reshape(-1), classifier.intercept_[0]).eval()


def fit_shallow_critic(values, source, *, seed, epochs=12, hidden_width=128,
                       batch_size=256, learning_rate=.001, weight_decay=.001):
    """Fit the fixed-epoch GELU critic with isolated, deterministic minibatches."""
    x = matrix(values)
    u = binary(source, len(x))
    value = int(seed) % (2**31 - 1)
    with isolated_torch_seed(value):
        model = ShallowBinaryCritic(x.shape[1], hidden_width)
        loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(u)),
                            batch_size=int(batch_size), shuffle=True,
                            generator=torch.Generator().manual_seed(value))
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate),
                                      weight_decay=float(weight_decay))
        for _ in range(int(epochs)):
            model.train()
            for bx, bu in loader:
                loss = F.cross_entropy(model(bx), bu)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        model.requires_grad_(False)
    return model


def predict_score(model, values):
    """Evaluate binary logit differences in fixed-size inference batches."""
    x = matrix(values)
    with torch.no_grad():
        return np.concatenate([centered_logit(model, torch.from_numpy(x[i:i+2048])).numpy()
                               for i in range(0, len(x), 2048)]).astype(np.float64)


@dataclass
class CriticState:
    """Portable affine/GELU weights; no pickled estimator or torch module."""
    kind: str
    weights: tuple

    @classmethod
    def from_model(cls, model):
        """Extract affine or GELU numerical tensors without retaining a module."""
        if hasattr(model, "weight") and hasattr(model, "bias"):
            return cls("affine", (model.weight.detach().cpu().numpy().copy(),
                                   np.asarray(model.bias.detach().cpu().numpy()).copy()))
        layers = model.layers
        return cls("gelu", tuple(w.detach().cpu().numpy().copy() for w in
                                (layers[0].weight, layers[0].bias, layers[2].weight, layers[2].bias)))

    def logits_tensor(self, x):
        """Reconstruct differentiable logits from the saved numerical weights."""
        w = [torch.as_tensor(v, dtype=x.dtype, device=x.device) for v in self.weights]
        if self.kind == "affine":
            score = x @ w[0] + w[1]
            return torch.stack((-.5 * score, .5 * score), 1)
        if self.kind == "gelu":
            return F.linear(F.gelu(F.linear(x, w[0], w[1])), w[2], w[3])
        raise ValueError(f"Unknown critic type: {self.kind}")

    def score_tensor(self, x):
        """Return differentiable class-logit differences."""
        logits = self.logits_tensor(x)
        return logits[:, 1] - logits[:, 0]

    def predict(self, x):
        """Predict scores as a float64 NumPy vector with no parameter gradients."""
        x = matrix(x)
        with torch.no_grad():
            return np.concatenate([self.score_tensor(torch.from_numpy(x[i:i+2048])).numpy()
                                   for i in range(0, len(x), 2048)]).astype(np.float64)


_fit_affine_critic = fit_affine_critic
_fit_shallow_critic = fit_shallow_critic
_isolated_torch_seed = isolated_torch_seed
_predict_score = predict_score
