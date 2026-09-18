"""Pure scoring primitives; no fitting, file access, or checkpoint selection."""
from __future__ import annotations

import hashlib
import json

import numpy as np
from scipy.special import logsumexp, softmax
from sklearn.metrics import roc_auc_score


def arrays_hash(*arrays):
    """Hash array shapes, dtypes and contiguous numerical values."""
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise ValueError("Object arrays are not portable numerical identities")
        digest.update(json.dumps({"shape": array.shape, "dtype": array.dtype.str}, sort_keys=True).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def json_compatible(value):
    """Convert NumPy containers to finite-JSON-compatible Python objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_compatible(item) for item in value]
    return value


def strict_mean(values):
    """A missing participant is not silently removed from the headline mean."""
    values = list(values)
    return float(np.mean(values)) if values and all(v is not None and np.isfinite(v) for v in values) else None


def available_mean(values):
    """Average available auxiliary values, retaining None when none exist."""
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def binary_metrics(y, score):
    """Score AUROC and threshold-zero balanced accuracy without refitting."""
    y, score = np.asarray(y), np.asarray(score, dtype=np.float64)
    if y.shape != score.shape or not np.isfinite(score).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Aligned finite binary predictions required")
    prediction = (score > 0.0).astype(np.int8)
    recalls = [float(np.mean(prediction[y == label] == label)) if np.any(y == label) else None for label in (0, 1)]
    complete = all(v is not None for v in recalls)
    return {"n": len(y), "class0_n": int(np.sum(y == 0)), "class1_n": int(np.sum(y == 1)),
            "auroc": float(roc_auc_score(y, score)) if complete else None,
            "balanced_accuracy": float(np.mean(recalls)) if complete else None,
            "recall0": recalls[0], "recall1": recalls[1],
            "status": "evaluated" if complete else "one_class_auc_ba_unestimable"}


def multiclass_metrics(y, logits, classes=12):
    """Compute the complete twelve-frequency task metrics from fixed logits."""
    y, logits = np.asarray(y, dtype=np.int64), np.asarray(logits, dtype=np.float64)
    if logits.shape != (len(y), classes) or not np.isfinite(logits).all():
        raise ValueError("Aligned finite multiclass logits required")
    if not np.isin(y, np.arange(classes)).all():
        raise ValueError("Invalid multiclass label")
    prediction = logits.argmax(axis=1)
    present = np.unique(y)
    complete = len(present) == classes
    recalls = [float(np.mean(prediction[y == label] == label)) for label in present]
    return {"n": len(y), "classes_present": len(present),
            "accuracy": float(np.mean(prediction == y)),
            "balanced_accuracy": float(np.mean(recalls)) if complete else None,
            "macro_auroc": float(roc_auc_score(y, softmax(logits, axis=1), labels=np.arange(classes),
                                               multi_class="ovr", average="macro")) if complete else None,
            "cross_entropy": float(np.mean(logsumexp(logits, axis=1) - logits[np.arange(len(y)), y])),
            "status": "evaluated_12_classes" if complete else "missing_class_12way_auc_ba_unestimable"}


def binary_log_loss(score, y):
    """Evaluate stable logistic loss for a class-one-minus-class-zero score."""
    return np.logaddexp(0.0, np.asarray(score, dtype=np.float64)) - np.asarray(y, dtype=np.float64) * score


def movement(original, output, *, observed=False, basis=None):
    """Measure paired-vector or full-token movement and preserved-space drift."""
    original, output = np.asarray(original, dtype=np.float64), np.asarray(output, dtype=np.float64)
    if original.shape != output.shape or original.ndim != 3:
        raise ValueError("Movement requires aligned paired vectors or observed token arrays")
    delta = output - original
    flat = delta.reshape(-1, delta.shape[-1])
    energy = float(np.square(original).sum())
    denominator = np.sqrt(energy) if observed else float(np.linalg.norm(original.reshape(-1, original.shape[-1])))
    result = {"relative_movement_l2": float(np.linalg.norm(delta) / denominator) if denominator else None,
              "original_energy": energy}
    if observed:
        result.update(movement_mse_per_record=float(np.square(delta).sum(axis=(1, 2)).mean()),
                      movement_mse_per_token=float(np.square(flat).sum(axis=1).mean()),
                      observed_records=len(original),
                      shared_offset_residual_max_abs=float(np.abs(delta - delta.mean(axis=1, keepdims=True)).max()))
    else:
        result.update(movement_mse_per_view=float(np.square(flat).sum(axis=1).mean()),
                      movement_l2_mean_per_view=float(np.linalg.norm(flat, axis=1).mean()),
                      paired_squared_discrepancy=float(np.square(output[:, 1] - output[:, 0]).sum(axis=1).mean()),
                      per_view_relative_movement=[float(np.linalg.norm(delta[:, u]) / np.linalg.norm(original[:, u]))
                                                 if np.linalg.norm(original[:, u]) else None for u in (0, 1)])
    if basis is not None:
        qdelta = flat - (flat @ basis) @ basis.T
        result.update(q_drift_relative_l2=float(np.linalg.norm(qdelta) / np.sqrt(energy)) if energy else None,
                      q_drift_max_abs=float(np.abs(qdelta).max()))
    return result


def summarize(main_rows, native_rows, people, *, recorded=False, source_families=(), task_epoch=50):
    """Retain every reader and participant, so bootstrap recomputes the bank maximum."""
    people = list(people)
    participants = []
    for person in people:
        rows = [r for r in main_rows if r["participant"] == person]
        sources = {r["family"]: r["auroc"] for r in rows if r["kind"] == "source"}
        if set(sources) != set(source_families):
            raise ValueError("Incomplete source reader bank")
        task = {(str(r["mode"]), str(r["train_view"]), str(r["test_view"])): r
                for r in rows if r["kind"] == "task" and (not recorded or r["task_epoch"] == task_epoch)}
        move = [r for r in rows if r["kind"] == "movement"]
        if len(move) != 1:
            raise ValueError("Expected exactly one movement row per participant")
        item = {"participant": person, "source_auroc_by_family": sources, "M": move[0]["relative_movement_l2"]}
        for short, mode in (("Tf", "frozen_original"), ("Tr", "refit")):
            for field, suffix in ((("macro_auroc" if recorded else "auroc"), ""), ("balanced_accuracy", "_BA")):
                values = [task.get((mode, str(a), str(b)), {}).get(field) for a, b in ((0, 1), (1, 0))]
                item[short + suffix] = strict_mean(values)
                item[short + suffix + "_01"], item[short + suffix + "_10"] = values
        participants.append(item)
    source_means = {family: strict_mean(r["source_auroc_by_family"][family] for r in participants) for family in source_families}
    fields = ("Tf", "Tr", "Tf_BA", "Tr_BA", "M")
    summary = {key: strict_mean(r.get(key) for r in participants) for key in fields}
    summary.update(S=max(source_means.values()) if all(v is not None for v in source_means.values()) else None,
                   source_auroc_by_family=source_means, participants=len(people),
                   estimable_counts={key: sum(r.get(key) is not None for r in participants) for key in fields},
                   S_definition="maximum over fixed-bank whole-participant-mean source AUROCs; not mean participant max",
                   T_definition="equal mean of 0->1 and 1->0 AUROC; frozen original HEAD or refitted transformed HEAD",
                   M_definition="equal participant mean of relative L2 movement, not pooled ratio")
    return summary, participants
