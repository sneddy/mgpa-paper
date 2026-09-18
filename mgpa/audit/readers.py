"""HEAD-only fitted readers. EVAL labels are absent from every fitting API."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mgpa.critics import ShallowBinaryCritic, isolated_torch_seed, predict_score
from .metrics import arrays_hash

SOURCE_EPOCHS = (12, 50, 200)
SOURCE_FAMILIES = ("linear", "whitened_linear") + tuple(
    f"gelu_e{epoch}_r{restart}" for restart in range(3) for epoch in SOURCE_EPOCHS)


def fit_task_model(x, y):
    """Fit the HEAD-standardized balanced Ridge downstream readout."""
    return make_pipeline(StandardScaler(), RidgeClassifier(alpha=10.0, class_weight="balanced", solver="lsqr")).fit(x, y)


def fit_whitening(train):
    """Estimate the regularized whitening map from HEAD features only."""
    train = np.asarray(train, dtype=np.float64)
    center = train.mean(axis=0)
    centered = train - center
    covariance = centered.T @ centered / len(centered)
    scale = float(np.trace(covariance) / train.shape[1])
    regularized = .99 * covariance + (.01 * scale + 1e-8) * np.eye(train.shape[1])
    values, vectors = np.linalg.eigh(.5 * (regularized + regularized.T))
    if not np.isfinite(values).all() or values.min() <= 0:
        raise FloatingPointError("Invalid HEAD-only whitening covariance")
    whitening = (vectors * (1. / np.sqrt(values))[None, :]) @ vectors.T
    return center, whitening


def source_linear_scores(train, source, test, *, family, seed):
    """Fit an ordinary or whitened source reader and score EVAL features."""
    train, test = np.asarray(train, dtype=np.float64), np.asarray(test, dtype=np.float64)
    if family == "whitened_linear":
        center, whitening = fit_whitening(train)
        train, test = (train - center) @ whitening, (test - center) @ whitening
    elif family != "linear":
        raise ValueError("Unknown linear reader family")
    model = LogisticRegression(C=1., max_iter=2500, class_weight="balanced", random_state=int(seed)).fit(train, source)
    return np.asarray(model.decision_function(test), dtype=np.float64), {
        "family": family, "fitting_role": "HEAD_only", "task_labels_used": False,
        "seed": int(seed), "C": 1., "max_iter": 2500,
        "n_iter": int(np.max(model.n_iter_)), "converged_before_limit": bool(np.max(model.n_iter_) < 2500)}


def gelu_checkpoint_scores(train, source, test, *, seed, checkpoints=SOURCE_EPOCHS):
    """Exact archived AdamW stream: hidden128, batch256, no gradient clipping.

    Checkpoints are scored only after HEAD-only training has completed. The
    evaluation feature array has no route into optimizer or stopping logic.
    """
    checkpoints = tuple(sorted(checkpoints))
    if not checkpoints or any(e not in SOURCE_EPOCHS for e in checkpoints):
        raise ValueError("Only fixed 12/50/200 source checkpoints are supported")
    x = np.asarray(train, dtype=np.float32)
    y = np.asarray(source, dtype=np.int64)
    if x.ndim != 2 or y.shape != (len(x),) or set(y.tolist()) != {0, 1} or not np.isfinite(x).all():
        raise ValueError("Finite HEAD features and both source classes required")
    value = int(seed) % (2**31 - 1)
    states, history = {}, []
    with isolated_torch_seed(value):
        model = ShallowBinaryCritic(x.shape[1], hidden_width=128)
        xt, yt = torch.from_numpy(x), torch.from_numpy(y)
        loader = DataLoader(TensorDataset(xt, yt), batch_size=256, shuffle=True,
                            generator=torch.Generator().manual_seed(value))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
        for epoch in range(1, max(checkpoints) + 1):
            model.train()
            loss_sum, correct, seen = 0., 0, 0
            for features, labels in loader:
                logits = model(features)
                loss = nn.functional.cross_entropy(logits, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite source reader HEAD training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(labels)
                correct += int((logits.detach().argmax(dim=1) == labels).sum())
                seen += len(labels)
            # Original risk diagnostics make no RNG draws and no loader pass.
            model.eval()
            train_loss, train_correct = 0., 0
            with torch.no_grad():
                for start in range(0, len(xt), 256):
                    logits = model(xt[start:start + 256])
                    labels = yt[start:start + 256]
                    train_loss += float(nn.functional.cross_entropy(logits, labels, reduction="sum"))
                    train_correct += int((logits.argmax(dim=1) == labels).sum())
            history.append({"epoch": epoch, "training_rows": seen, "train_loss": train_loss / len(xt),
                            "train_accuracy": train_correct / len(xt), "minibatch_loss": loss_sum / seen,
                            "minibatch_accuracy": correct / seen})
            if epoch in checkpoints:
                states[epoch] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        optimizer.zero_grad(set_to_none=True)
        model.eval().requires_grad_(False)
    scores, hashes = {}, {}
    for epoch, state in states.items():
        model.load_state_dict(state)
        scores[epoch] = np.asarray(predict_score(model, test), dtype=np.float64)
        hashes[epoch] = arrays_hash(*(state[k].numpy() for k in sorted(state)))
    return {"scores": scores, "checkpoint_sha256": hashes, "training_history": history}


def source_bank(train, source, test, *, seed=98, smoke=False):
    """Yield the full fixed bank, or an explicitly marked smoke-only subset."""
    for family in ("linear", "whitened_linear"):
        score, details = source_linear_scores(train, source, test, family=family, seed=seed)
        yield family, score, details, None, None, None
    checkpoints, restarts = ((12,), 1) if smoke else (SOURCE_EPOCHS, 3)
    for restart in range(restarts):
        family_seed = seed + 1001 + 10000 * restart
        output = gelu_checkpoint_scores(train, source, test, seed=family_seed, checkpoints=checkpoints)
        for epoch in checkpoints:
            details = {"fitting_role": "HEAD_only", "task_labels_used": False, "seed": family_seed,
                       "epochs": epoch, "trained_to_epochs": max(checkpoints), "hidden_width": 128,
                       "learning_rate": .001, "weight_decay": .001, "batch_size": 256,
                       "gradient_clipping": None, "checkpoint_sha256": output["checkpoint_sha256"][epoch]}
            yield f"gelu_e{epoch}_r{restart}", output["scores"][epoch], details, epoch, restart, output["training_history"]


class TemporalGridHead(nn.Module):
    """Recorded SSVEP native 12-class token-grid head, not a binary surrogate."""
    def __init__(self, tokens, width, classes=12):
        """Create the shared token projection and concatenated classifier."""
        super().__init__()
        self.tokens, self.width = int(tokens), int(width)
        self.patch = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 64), nn.GELU())
        self.output = nn.Linear(tokens * 64, classes)

    def forward(self, values):
        """Return logits for one recording per leading batch element."""
        values = values.reshape(len(values), self.tokens, self.width)
        return self.output(self.patch(values).flatten(1))


def fit_token_task_path(train, labels, *, seed, checkpoints=(4, 12, 50)):
    """Fit the fixed HEAD-only token task network and retain its checkpoints."""
    train = np.asarray(train, dtype=np.float32)
    tokens, width = train.shape[1:]
    flat = train.reshape(len(train), -1)
    center, scale = flat.mean(axis=0), np.maximum(flat.std(axis=0), 1e-5)
    standardized = (flat - center) / scale
    models, history = {}, []
    with isolated_torch_seed(seed):
        model = TemporalGridHead(tokens, width, 12)
        loader = DataLoader(TensorDataset(torch.from_numpy(standardized), torch.from_numpy(np.asarray(labels, dtype=np.int64))),
                            batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(int(seed)))
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
        for epoch in range(1, max(checkpoints) + 1):
            model.train()
            total, count = 0., 0
            for batch, truth in loader:
                loss = nn.functional.cross_entropy(model(batch), truth)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite recorded task HEAD training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                total += float(loss.detach()) * len(batch)
                count += len(batch)
            history.append({"epoch": epoch, "head_training_cross_entropy": total / count, "head_records": count})
            if epoch in checkpoints:
                snapshot = deepcopy(model).eval()
                snapshot.register_buffer("input_mean", torch.from_numpy(center.copy()), persistent=True)
                snapshot.register_buffer("input_scale", torch.from_numpy(scale.copy()), persistent=True)
                models[epoch] = snapshot
    return models, {"seed": int(seed), "training_history": history, "head_records": len(train)}


def predict_token_task(model, values):
    """Apply the frozen model and its original HEAD input standardizer."""
    flat = np.asarray(values, dtype=np.float32).reshape(len(values), -1)
    standardized = (flat - model.input_mean.numpy()) / model.input_scale.numpy()
    with torch.inference_mode():
        return np.concatenate([model(torch.from_numpy(standardized[start:start + 256])).numpy()
                               for start in range(0, len(flat), 256)])
