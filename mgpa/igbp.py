"""Independent binary IGBP baseline, following Iskander et al. (ACL 2023).

The projection is Eq. (6)-(7): x <- x - f(x) grad f(x) / ||grad f(x)||^2,
where f is the difference of the two classifier logits. It does not use source
labels when applied. This is not an ungated MGPA/TR adapter.

Primary sources:
https://aclanthology.org/2023.findings-acl.369.pdf
https://github.com/technion-cs-nlp/igbp_nonlinear-removal
Upstream inspected: a61ea88112665012d9bdb19d3b172b4088e024fa,
src/debias_representation.py and src/run_igbp.py.

The architecture follows that script, with the paper's strong critic-training
preset. We compute the equivalent logit-space update directly, avoiding softmax
saturation and partial-batch scaling artefacts. Every native map retains all
100 projections; source-validation early stopping selects a critic's training
epoch only. All evaluation data enter only transform(), never fitting.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np
import torch
from torch import nn

from mgpa.core import Endpoint, EndpointUnavailableError, PortableEstimator, relative_movement
from mgpa.critics import binary, matrix


DEFAULTS = {
    "max_stages": 100,
    "critic_epochs": 50,
    "learning_rate": 0.0002,
    "weight_decay": 0.01,
    "batch_size": 256,
    "early_accuracy": 1.0,
    "patience": 10,
    "min_accuracy_gain": 0.002,
    "gradient_floor": 1e-12,
    "transform_batch_size": 512,
    "budgets": [0.10, 0.25],
}
UPSTREAM_COMMIT = "a61ea88112665012d9bdb19d3b172b4088e024fa"


def validate_config(supplied=None):
    """Merge declared IGBP settings with the strong preset and validate their ranges."""
    supplied = dict(supplied or {})
    unknown = set(supplied) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown IGBP options: {sorted(unknown)}")
    hp = {**DEFAULTS, **supplied}
    for key in ("max_stages", "critic_epochs", "batch_size", "patience",
                "transform_batch_size"):
        if isinstance(hp[key], bool) or int(hp[key]) != hp[key] or hp[key] < 1:
            raise ValueError(f"Invalid {key}")
        hp[key] = int(hp[key])
    for key in ("learning_rate", "gradient_floor"):
        if not np.isfinite(hp[key]) or hp[key] <= 0:
            raise ValueError(f"Invalid {key}")
    for key in ("weight_decay", "min_accuracy_gain"):
        if not np.isfinite(hp[key]) or hp[key] < 0:
            raise ValueError(f"Invalid {key}")
    if not 0 < hp["early_accuracy"] <= 1:
        raise ValueError("early_accuracy must be in (0,1]")
    hp["budgets"] = sorted(set(float(v) for v in hp["budgets"]))
    if any(not np.isfinite(v) or v < 0 for v in hp["budgets"]):
        raise ValueError("Invalid movement budgets")
    return hp


def build_probe(width):
    """The authors' binary one-hidden-layer, input-width ReLU architecture."""
    return nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 2))


def projection_step(probe, x, *, gradient_floor=1e-12, fraction=1.):
    """One label-free IGBP update; gradient_floor only handles zero gradients.

    No clipping, trust region, calibration gate, anchor, or line search is used.
    Fraction < 1 is only for predeclared movement-matched endpoint diagnostics.
    """
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("fraction must be in [0,1]")
    probe.eval()
    current = x.detach().clone().requires_grad_(True)
    logits = probe(current)
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("IGBP baseline requires exactly two output logits")
    score = logits[:, 1] - logits[:, 0]
    gradient = torch.autograd.grad(score.sum(), current)[0]
    norm2 = gradient.square().sum(1, keepdim=True)
    delta = score[:, None] * gradient / norm2.clamp_min(gradient_floor)
    # If f is locally constant, no local projection direction exists.
    delta = torch.where(norm2 > gradient_floor, delta, torch.zeros_like(delta))
    result = (current - fraction * delta).detach()
    if not torch.isfinite(result).all():
        raise FloatingPointError("Nonfinite IGBP projection")
    return result


@dataclass
class IGBPStage:
    """Portable ReLU classifier weights and one frozen projective update."""
    state: dict
    width: int
    gradient_floor: float = 1e-12
    batch_size: int = 512

    @classmethod
    def from_probe(cls, probe, *, gradient_floor=1e-12, batch_size=512):
        """Extract the trained probe's numerical weights and replay settings."""
        state = {k: v.detach().cpu().numpy().copy() for k, v in probe.state_dict().items()}
        return cls(state, int(probe[0].in_features), gradient_floor, batch_size)

    def probe(self):
        """Reconstruct the fixed classifier without advancing the caller's RNG."""
        # Replay must not advance global random state while constructing layers.
        with torch.random.fork_rng(devices=[]):
            probe = build_probe(self.width)
        probe.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in self.state.items()})
        for parameter in probe.parameters():
            parameter.requires_grad_(False)
        return probe.eval()

    def apply(self, x, fraction=1.):
        """Project observations with this fixed classifier in bounded memory batches."""
        x = matrix(x)
        if x.shape[1] != self.width:
            raise ValueError("Representation width changed")
        probe = self.probe()
        output = []
        for start in range(0, len(x), self.batch_size):
            inputs = torch.from_numpy(np.asarray(x[start:start+self.batch_size]).copy())
            output.append(projection_step(probe, inputs,
                                          gradient_floor=self.gradient_floor,
                                          fraction=fraction).numpy())
        return np.concatenate(output).astype(np.float32)

    def scores(self, x):
        """Return this stage's class-1 minus class-0 logits without fitting."""
        probe = self.probe()
        with torch.no_grad():
            logits = probe(torch.from_numpy(matrix(x).copy()))
        return (logits[:, 1] - logits[:, 0]).numpy()


def _summary(probe, x, labels):
    """Describe source-classifier loss, accuracy, and logit magnitude on supplied rows."""
    with torch.no_grad():
        logits = probe(x)
        loss = nn.functional.cross_entropy(logits, labels).item()
        accuracy = (logits.argmax(1) == labels).float().mean().item()
        absolute = (logits[:, 1] - logits[:, 0]).abs()
    return {"bce": float(loss), "accuracy": float(accuracy),
            "absolute_logit_q95": float(torch.quantile(absolute, .95))}


def _fit_probe(x, y, xv, yv, hp, seed):
    """Mirror upstream's last-epoch VAL-accuracy stopping, not MGPA training."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        probe = build_probe(x.shape[1])
        tx, ty = torch.from_numpy(x.copy()), torch.from_numpy(y.astype(np.int64))
        vx, vy = torch.from_numpy(xv.copy()), torch.from_numpy(yv.astype(np.int64))
        generator = torch.Generator().manual_seed(int(seed) + 1)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tx, ty),
                                            batch_size=hp["batch_size"], shuffle=True,
                                            generator=generator)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=hp["learning_rate"],
                                      weight_decay=hp["weight_decay"])
        high_accuracy = 0.
        stale = 0
        epochs = []
        reason = "max_epochs"
        for epoch in range(1, hp["critic_epochs"] + 1):
            probe.train()
            for xb, yb in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(probe(xb), yb)
                loss.backward()
                optimizer.step()
            probe.eval()
            dev = _summary(probe, vx, vy)
            epochs.append({"epoch": epoch, "validation": dev})
            accuracy = dev["accuracy"]
            if high_accuracy == 0:
                high_accuracy = accuracy
            if accuracy > hp["early_accuracy"]:
                reason = "validation_accuracy_threshold"
                break
            if (accuracy > high_accuracy + hp["min_accuracy_gain"]
                    and accuracy < hp["early_accuracy"]):
                stale = 0
                high_accuracy = accuracy
            else:
                stale += 1
                if stale >= hp["patience"]:
                    reason = "validation_accuracy_patience"
                    break
        probe.eval()
        return probe, {"epochs": epochs, "stop_reason": reason,
                       "fit": _summary(probe, tx, ty), "validation": _summary(probe, vx, vy)}


def _fraction(original, before, after, budget):
    """First radius crossing on the line between consecutive native states."""
    original = np.asarray(original, dtype=np.float64)
    residual = np.asarray(before, dtype=np.float64) - original
    delta = np.asarray(after, dtype=np.float64) - np.asarray(before, dtype=np.float64)
    a = np.sum(delta * delta)
    b = 2 * np.sum(residual * delta)
    c = np.sum(residual * residual) - budget**2 * np.sum(original * original)
    if a <= 0:
        raise ValueError("No movement crossing in a constant stage")
    root = (-b + np.sqrt(max(0., b*b-4*a*c))) / (2*a)
    if not -1e-7 <= root <= 1+1e-7:
        raise ValueError("Movement budget does not cross this stage")
    return float(np.clip(root, 0., 1.))


@dataclass
class _IGBPState:
    """Internal fitted trajectory with its validation-selected replay endpoints."""
    width: int
    stages: list = field(default_factory=list)
    endpoints: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    fit_trace: list = field(default_factory=list)
    method_id: str = "igbp"

    @property
    def trace(self):
        """Expose the recorded source-critic training diagnostics."""
        return self.fit_trace

    def transform(self, x, endpoint="native", source_ids=None):
        """Replay an attained named endpoint without source routing."""
        # source_ids is accepted solely to match the common adapter interface.
        original = matrix(x)
        if original.shape[1] != self.width:
            raise ValueError("Representation width changed")
        if endpoint not in self.endpoints or not self.endpoints[endpoint].attained:
            raise EndpointUnavailableError(endpoint)
        selected = self.endpoints[endpoint]
        current = original
        for k, stage in enumerate(self.stages[:selected.stages]):
            current = stage.apply(current, fraction=selected.fraction
                                  if k == selected.stages-1 else 1.)
        return current


def _fit_igbp(x_fit, u_fit, x_val, u_val, *, seed, config=None,
             progress: Callable | None = None):
    """Fit the full strong trajectory and declared VAL movement endpoints.

    Native always replays every projection. A movement endpoint interpolates
    only the first crossing's last update; unattained budgets are unavailable.
    """
    hp = validate_config(config)
    x, xv = matrix(x_fit).copy(), matrix(x_val).copy()
    if x.shape[1] != xv.shape[1]:
        raise ValueError("FIT/VAL feature widths differ")
    y, yv = binary(u_fit, len(x)), binary(u_val, len(xv))
    original_val = xv.copy()
    model = _IGBPState(width=x.shape[1], metadata={
        "algorithm": "IGBP", "paper": "10.18653/v1/2023.findings-acl.369",
        "upstream_commit": UPSTREAM_COMMIT, "config": hp, "seed": int(seed),
        "architecture": [int(x.shape[1]), int(x.shape[1]), 2], "activation": "ReLU",
        "fit_source_labels": True, "task_labels_used": False,
        "source_ids_at_application": False, "measurement_pairs_used": False,
        "differences_from_upstream": [
            "Exact logit-space formula avoids softmax saturation and partial-batch scaling artefacts.",
            "Source-only VAL critic-training stopping; native retains every projection.",
            "Pre-standardized EEG input is shared with all other experiment adapters.",
            "Independent deterministic stage/batch seeds replace a shared global PRNG stream.",
            "Squared-gradient values <= gradient_floor produce no update; no movement cap is imposed.",
        ],
    })
    if 0. in hp["budgets"]:
        model.endpoints["budget_0"] = Endpoint(0, requested_movement=0.)
    for k in range(hp["max_stages"]):
        stage_seed = int(seed) + 1009*k
        probe, training = _fit_probe(x, y, xv, yv, hp, stage_seed)
        stage = IGBPStage.from_probe(probe, gradient_floor=hp["gradient_floor"],
                                    batch_size=hp["transform_batch_size"])
        old_val = xv
        x, xv = stage.apply(x), stage.apply(xv)
        model.stages.append(stage)
        movement = relative_movement(original_val, xv)
        for budget in hp["budgets"]:
            key = f"budget_{budget:g}"
            if key not in model.endpoints and movement >= budget:
                fraction = _fraction(original_val, old_val, xv, budget)
                model.endpoints[key] = Endpoint(k+1, fraction=fraction,
                    requested_movement=budget,
                    validation_movement=relative_movement(original_val, old_val+fraction*(xv-old_val)))
        item = {"stage": k+1, "seed": stage_seed, "critic_training": training,
                "validation_movement": movement,
                "post_projection_frozen_critic_accuracy": float(np.mean((stage.scores(xv)>0) == yv))}
        model.fit_trace.append(item)
        if progress is not None:
            progress(item)
    count = len(model.stages)
    model.endpoints["native"] = Endpoint(count, validation_movement=relative_movement(original_val, xv))
    for budget in hp["budgets"]:
        key = f"budget_{budget:g}"
        if key not in model.endpoints:
            model.endpoints[key] = Endpoint(count, attained=False, requested_movement=budget,
                validation_movement=model.endpoints["native"].validation_movement,
                reason="VAL trajectory never reached the requested movement")
    model.metadata["endpoints"] = {k: asdict(v) for k, v in model.endpoints.items()}
    return model


class IGBP(PortableEstimator):
    """The published strong IGBP baseline, with a sklearn-style fit/transform API.

    Default native output applies 100 projections. Each width→width→2 ReLU
    source classifier trains for at most 50 epochs, AdamW lr=.0002, with
    source-validation accuracy patience 10 and minimum improvement .002.
    These defaults are the single IGBP configuration reported in the paper.
    No measurement gate, source identity, or downstream labels enter replay.

    ``settings`` accepts the named fields in DEFAULTS. Movement budgets affect
    reporting only and never shorten the native trajectory. Serialization
    stores numerical tensors and JSON, with no fitted estimator pickle.
    """
    def __init__(self, **settings):
        """Initialize the strong published preset with explicit setting overrides."""
        self.params = validate_config(settings)

    def fit(self, x, source, *, validation_x, validation_source, seed=17, on_stage=None):
        """Fit source classifiers/projections using FIT and source-VAL data."""
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        model = _fit_igbp(x, source, validation_x, validation_source,
                          seed=seed, config=self.params, progress=on_stage)
        self.n_features_in_, self.stages_ = model.width, model.stages
        self.history_, self.metadata_ = model.fit_trace, model.metadata
        self.endpoints_ = model.endpoints
        return self

    def transform(self, x, stages=None, fraction=1., source_ids=None, *, endpoint=None):
        """Replay a fixed prefix/final fraction; source_ids are intentionally unused."""
        self._require_fitted()
        current = matrix(x)
        if current.shape[1] != self.n_features_in_:
            raise ValueError("Representation width changed")
        if endpoint is not None:
            if not endpoint.attained:
                raise EndpointUnavailableError("Movement endpoint was not attained")
            stages, fraction = endpoint.stages, endpoint.fraction
        count = len(self.stages_) if stages is None else stages
        if isinstance(count, bool) or int(count) != count or not 0 <= count <= len(self.stages_):
            raise ValueError("stages must be a valid fitted prefix")
        if not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("fraction must be in [0,1]")
        for k, stage in enumerate(self.stages_[:int(count)]):
            current = stage.apply(current, fraction=fraction if k == count-1 else 1.)
        return current

    def iter_transforms(self, x):
        """Yield each native prefix, including identity, without repeated replay."""
        self._require_fitted()
        current = matrix(x)
        if current.shape[1] != self.n_features_in_:
            raise ValueError("Representation width changed")
        yield 0, current.copy()
        for k, stage in enumerate(self.stages_, 1):
            current = stage.apply(current)
            yield k, current.copy()

    def select_movement_budget(self, validation_x, budget):
        """Freeze the original analytic first-crossing interpolation on VAL."""
        if not np.isfinite(budget) or budget < 0:
            raise ValueError("budget must be finite and nonnegative")
        original = matrix(validation_x)
        relative_movement(original, original)
        if budget == 0:
            return Endpoint(0, requested_movement=0.)
        previous, before, movement = original, 0., 0.
        for k, current in self.iter_transforms(original):
            if k == 0:
                continue
            movement = relative_movement(original, current)
            if before < budget <= movement:
                fraction = _fraction(original, previous, current, budget)
                return Endpoint(k, fraction, True, float(budget),
                    relative_movement(original, previous+fraction*(current-previous)), "first upward VAL crossing")
            previous, before = current, movement
        return Endpoint(len(self.stages_), attained=False, requested_movement=float(budget),
                        validation_movement=movement, reason="VAL trajectory did not attain budget")
