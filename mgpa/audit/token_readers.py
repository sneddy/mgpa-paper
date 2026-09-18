"""HEAD-only source readers for complete recorded token representations.

The pooled and token banks share the same GELU and TemporalGridHead kernels.
One recording is one sample: flattening never turns tokens into extra records.
EVAL features are used for prediction only; EVAL labels are not accepted.
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mgpa.critics import isolated_torch_seed
from .metrics import arrays_hash
from .readers import SOURCE_EPOCHS, TemporalGridHead, gelu_checkpoint_scores


RESTART_SEEDS = (1099, 11099, 21099)
JOBS = ("linear_raw", "linear_standardized") + tuple(
    f"{family}_r{restart}"
    for family in ("flat_gelu", "grid_gelu") for restart in range(3)
)
RESIDUAL_JOBS = ("linear_standardized",) + tuple(f"grid_gelu_r{restart}" for restart in range(3))
STD_FLOOR = 1e-5


def _tokens(values, name):
    """Validate recording-major token arrays without flattening the sample axis."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite nonempty [records, tokens, width] features")
    return values


def residual_tokens(values):
    """Remove each record's token mean without modifying input arrays."""
    values = _tokens(values, "tokens").astype(np.float64)
    return (values - values.mean(axis=1, keepdims=True)).astype(np.float32)


def standardize_head(train_flat, eval_flat):
    """Fit float32 coordinate normalization only on HEAD (native task recipe).

    The third return value contains numerical mean/scale arrays for testing.
    ``fit_reader`` stores their hashes, not arrays, in its JSON metadata.
    """
    train = np.asarray(train_flat, dtype=np.float32)
    test = np.asarray(eval_flat, dtype=np.float32)
    if (train.ndim != 2 or test.ndim != 2 or min(train.shape) < 1
            or min(test.shape) < 1 or train.shape[1] != test.shape[1]
            or not np.isfinite(train).all() or not np.isfinite(test).all()):
        raise ValueError("HEAD/EVAL must be finite nonempty matrices of equal feature width")
    center = train.mean(axis=0)
    scale = np.maximum(train.std(axis=0), np.float32(STD_FLOOR))
    return (train - center) / scale, (test - center) / scale, {"center": center, "scale": scale}


def _grid_checkpoint_scores(train, source, test, *, tokens, width, seed, checkpoints):
    """Fixed native token-head optimizer, extended to source checkpoints.

    All snapshots are trained before any EVAL forward pass. Training diagnostics
    are deterministic extra HEAD-only passes, not selection/stopping criteria.
    """
    x, y = np.asarray(train, np.float32), np.asarray(source, np.int64)
    states, history = {}, []
    with isolated_torch_seed(seed):
        model = TemporalGridHead(tokens, width, classes=2)
        xt, yt = torch.from_numpy(x), torch.from_numpy(y)
        loader = DataLoader(TensorDataset(xt, yt), batch_size=128, shuffle=True,
                            generator=torch.Generator().manual_seed(int(seed)))
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
        for epoch in range(1, max(checkpoints) + 1):
            model.train()
            loss_sum, correct, seen = 0., 0, 0
            for features, labels in loader:
                logits = model(features)
                loss = nn.functional.cross_entropy(logits, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite grid source reader HEAD training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                loss_sum += float(loss.detach()) * len(labels)
                correct += int((logits.detach().argmax(dim=1) == labels).sum())
                seen += len(labels)
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
                states[epoch] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        optimizer.zero_grad(set_to_none=True)
        model.eval().requires_grad_(False)
    scores, hashes = {}, {}
    for epoch, state in states.items():
        model.load_state_dict(state)
        with torch.inference_mode():
            predictions = []
            for start in range(0, len(test), 256):
                logits = model(torch.from_numpy(test[start:start + 256]))
                predictions.append((logits[:, 1] - logits[:, 0]).numpy())
        scores[epoch] = np.concatenate(predictions).astype(np.float64)
        hashes[epoch] = arrays_hash(*(state[key].numpy() for key in sorted(state)))
    return {"scores": scores, "checkpoint_sha256": hashes, "training_history": history,
            "states": {epoch: {key: value.numpy().copy() for key, value in state.items()}
                       for epoch, state in states.items()}}


def fit_reader(train_tokens, source_labels, eval_tokens, *, reader, smoke=False, retain_states=False):
    """Fit one fixed source-only reader job and return all fixed checkpoints.

    ``reader`` names a JOBS entry; scope prefixes are added by the caller.
    Smoke mode uses the same architecture/optimizer but only epoch12 and must
    never be reported as a completed paper run.
    """
    if reader not in JOBS:
        raise ValueError(f"Unknown fixed reader job: {reader!r}")
    if retain_states and reader.startswith("flat_gelu"):
        raise ValueError("Portable residual-reader states support linear/grid jobs only")
    train, test = _tokens(train_tokens, "HEAD"), _tokens(eval_tokens, "EVAL")
    if train.shape[1:] != test.shape[1:]:
        raise ValueError("HEAD and EVAL token dimensions must match")
    labels = np.asarray(source_labels)
    if labels.shape != (len(train),) or not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2:
        raise ValueError("HEAD source labels must align with records and include both binary classes")
    labels = labels.astype(np.int64)
    tokens, width = train.shape[1:]
    xf, xe = train.reshape(len(train), -1), test.reshape(len(test), -1)
    metadata = {
        "reader": reader, "fitting_role": "HEAD_only", "task_labels_used": False,
        "eval_labels_used": False, "eval_features_used_in_fit": False,
        "early_stopping": False, "smoke": bool(smoke), "paper_eligible": not bool(smoke),
        "head_records": len(train), "eval_records": len(test), "tokens_per_record": tokens,
        "token_width": width, "flattened_width": tokens * width,
        "head_features_sha256": arrays_hash(train), "head_source_sha256": arrays_hash(labels),
        "eval_features_sha256": arrays_hash(test),
        "input_dtype": "float32", "prediction": "class1_minus_class0_logit",
    }
    normalization = {"center": None, "scale": None}
    if reader != "linear_raw":
        xf, xe, stats = standardize_head(xf, xe)
        normalization = {key: value.copy() for key, value in stats.items()}
        metadata["normalization"] = {"kind": "per_coordinate_standardization", "fitting_role": "HEAD_only",
                                     "dtype": "float32", "std_floor": STD_FLOOR,
                                     "center_sha256": arrays_hash(stats["center"]),
                                     "scale_sha256": arrays_hash(stats["scale"])}
    else:
        metadata["normalization"] = {"kind": "none"}
    if reader.startswith("linear_"):
        c = 1. if reader == "linear_raw" else .01
        model = LogisticRegression(C=c, max_iter=10000, tol=1e-4, class_weight="balanced",
                                   solver="lbfgs", random_state=RESTART_SEEDS[0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(np.asarray(xf, np.float64), labels)
        iterations = int(np.max(model.n_iter_))
        if iterations >= 10000 or any(issubclass(item.category, ConvergenceWarning) for item in caught):
            raise RuntimeError(f"{reader} failed the prespecified convergence requirement ({iterations} iterations)")
        scores = {reader: np.asarray(model.decision_function(np.asarray(xe, np.float64)), np.float64)}
        metadata.update({"seed": RESTART_SEEDS[0], "C": c, "solver": "lbfgs", "tol": 1e-4,
                         "max_iter": 10000, "class_weight": "balanced", "n_iter": iterations,
                         "converged_before_limit": True,
                         "checkpoint_sha256": {reader: arrays_hash(model.coef_, model.intercept_, model.classes_)}})
        if retain_states:
            states = {reader: {"kind": "linear", "normalization": normalization,
                               "token_shape": [tokens, width], "coef": model.coef_.copy(),
                               "intercept": model.intercept_.copy(), "classes": model.classes_.copy()}}
        history = []
    else:
        match = re.fullmatch(r"(flat_gelu|grid_gelu)_r([0-2])", reader)
        family, restart = match.group(1), int(match.group(2))
        seed, checkpoints = RESTART_SEEDS[restart], (12,) if smoke else SOURCE_EPOCHS
        if family == "flat_gelu":
            output = gelu_checkpoint_scores(xf, labels, xe, seed=seed, checkpoints=checkpoints)
            metadata.update({"architecture": "ShallowBinaryCritic", "hidden_width": 128,
                             "learning_rate": 1e-3, "batch_size": 256, "gradient_clipping": None,
                             "parameter_count": (tokens * width + 1) * 128 + 128 * 2 + 2})
        else:
            output = _grid_checkpoint_scores(xf, labels, xe, tokens=tokens, width=width,
                                             seed=seed, checkpoints=checkpoints)
            metadata.update({"architecture": "TemporalGridHead", "patch_width": 64,
                             "learning_rate": 5e-4, "batch_size": 128, "gradient_clipping": 5.,
                             "parameter_count": 2 * width + (width + 1) * 64 + tokens * 64 * 2 + 2})
        scores = {f"{family}_e{epoch}_r{restart}": np.asarray(output["scores"][epoch], np.float64)
                  for epoch in checkpoints}
        metadata.update({"seed": seed, "restart": restart, "classes": 2, "optimizer": "AdamW",
                         "weight_decay": 1e-3, "trained_to_epochs": max(checkpoints),
                         "checkpoints": list(checkpoints),
                         "checkpoint_sha256": {f"{family}_e{epoch}_r{restart}": output["checkpoint_sha256"][epoch]
                                               for epoch in checkpoints}})
        history = output["training_history"]
        if retain_states:
            states = {f"{family}_e{epoch}_r{restart}": {
                "kind": "grid", "normalization": normalization, "token_shape": [tokens, width],
                "state_dict": output["states"][epoch]} for epoch in checkpoints}
    for family, score in scores.items():
        if score.shape != (len(test),) or score.dtype != np.float64 or not np.isfinite(score).all():
            raise FloatingPointError(f"{family}: invalid source reader predictions")
    metadata["prediction_sha256"] = {family: arrays_hash(score) for family, score in scores.items()}
    result = {"scores": scores, "metadata": metadata, "training_history": history}
    if retain_states:
        result["states"] = states
    return result


def predict_states(states, eval_tokens):
    """Replay portable linear/grid snapshots, including fitted HEAD statistics.

    Values contain only dictionaries, lists, strings, scalars, and numerical
    arrays, so the caller can persist them using NPZ plus JSON without pickle.
    """
    test = _tokens(eval_tokens, "EVAL")
    scores = {}
    for family, state in states.items():
        if tuple(state["token_shape"]) != test.shape[1:]:
            raise ValueError("Portable state and EVAL token dimensions must match")
        flat = test.reshape(len(test), -1)
        center, scale = state["normalization"]["center"], state["normalization"]["scale"]
        if center is not None:
            center, scale = np.asarray(center, np.float32), np.asarray(scale, np.float32)
            if (center.shape != (flat.shape[1],) or scale.shape != center.shape
                    or not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0)):
                raise ValueError("Invalid portable HEAD normalization")
            flat = (flat - center) / scale
        elif scale is not None:
            raise ValueError("Portable normalization needs both center and scale")
        if state["kind"] == "linear":
            coef, intercept = np.asarray(state["coef"], np.float64), np.asarray(state["intercept"], np.float64)
            if coef.shape != (1, flat.shape[1]) or intercept.shape != (1,) or list(state["classes"]) != [0, 1]:
                raise ValueError("Invalid portable binary linear state")
            prediction = (np.asarray(flat, np.float64) @ coef.T + intercept).ravel()
        elif state["kind"] == "grid":
            with isolated_torch_seed(0):
                model = TemporalGridHead(*test.shape[1:], classes=2).eval().requires_grad_(False)
            model.load_state_dict({key: torch.from_numpy(np.asarray(value).copy())
                                   for key, value in state["state_dict"].items()})
            with torch.inference_mode():
                values = []
                for start in range(0, len(test), 256):
                    logits = model(torch.from_numpy(flat[start:start + 256]))
                    values.append((logits[:, 1] - logits[:, 0]).numpy())
            prediction = np.concatenate(values).astype(np.float64)
        else:
            raise ValueError("Unknown portable reader kind")
        if prediction.shape != (len(test),) or not np.isfinite(prediction).all():
            raise FloatingPointError("Invalid portable reader predictions")
        scores[family] = np.asarray(prediction, np.float64)
    return scores
