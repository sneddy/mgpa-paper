"""Fixed-marginal device-selection stress tests on *actual* paired recordings.

Each event has its two observed device endpoints.  Selection never interpolates,
scales, synthesizes or drops a recording.  FIT/HEAD use one endpoint per event;
evaluation integrates over the same two endpoints with exact coupling weights.
The empirical target prevalence is preserved, not silently balanced to one half.
Task labels enter this experimental assignment and task-head fitting only: these
utilities do not fit a provenance adapter or pass target labels to one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


DEFAULT_ASSOCIATIONS = {"aligned": 1.0, "independent": 0.0, "reversed": -1.0}


def _metadata(y, subjects):
    """Validate binary labels and return aligned string participant identifiers."""
    labels, people = np.asarray(y), np.asarray(subjects)
    if labels.ndim != 1 or people.ndim != 1 or labels.shape != people.shape or not len(labels):
        raise ValueError("y and subjects must be nonempty equal-length vectors")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("y must contain binary 0/1 labels")
    # Explicit string ids make saved results JSON-compatible and grouping stable.
    people = people.astype(str)
    return labels.astype(np.int8, copy=False), people


def source_probabilities(y, subjects, association, max_probability=0.9):
    """Return P(device=1 | target, participant), preserving P(device=1)=.5.

    For participant target prevalence ``p``, the coupling is
    ``.5 + rho * (max_probability-.5) * (y-p) / max(p,1-p)``.
    Thus |rho|=1 brings the minority class to the requested probability cap;
    the majority-class probability compensates to retain the device marginal.
    ``rho`` is a coupling-strength parameter, not a Pearson correlation.  A
    one-class participant has no manipulable association and receives .5.
    """
    labels, people = _metadata(y, subjects)
    rho, cap = float(association), float(max_probability)
    if not np.isfinite(rho) or not -1 <= rho <= 1:
        raise ValueError("association must be finite and within [-1, 1]")
    if not np.isfinite(cap) or not 0.5 <= cap <= 1:
        raise ValueError("max_probability must be finite and within [0.5, 1]")
    result = np.empty(len(labels), dtype=np.float64)
    for person in np.unique(people):
        mask = people == person
        prevalence = float(labels[mask].mean())
        result[mask] = 0.5 + rho * (cap - 0.5) * (labels[mask] - prevalence) / max(prevalence, 1 - prevalence)
    return result


def assign_endpoints(y, subjects, association, seed, max_probability=0.9):
    """Select one actual device per event using reproducible classwise quotas.

    Every participant gets exactly floor(n/2) device-1 endpoints, at every rho.
    The target/device-1 quota is nearest-integer to its expected value, subject
    to feasibility; remaining device-1 endpoints come from non-target events.
    Identical random permutations are used across rho values for paired designs.
    With odd n an exactly .5 empirical device fraction is impossible; the .5
    marginal remains exact for ``pair_weights`` used in evaluation.
    """
    labels, people = _metadata(y, subjects)
    probabilities = source_probabilities(labels, people, association, max_probability)
    rng = np.random.default_rng(int(seed))
    result = np.zeros(len(labels), dtype=np.int8)
    for person in np.unique(people):
        group = np.flatnonzero(people == person)
        positives = rng.permutation(group[labels[group] == 1])
        negatives = rng.permutation(group[labels[group] == 0])
        total = len(group) // 2
        expected = probabilities[positives].sum()
        positive_quota = int(np.clip(np.floor(expected + 0.5), max(0, total - len(negatives)), min(total, len(positives))))
        result[positives[:positive_quota]] = 1
        result[negatives[:total - positive_quota]] = 1
    return result


def pair_weights(y, subjects, association, max_probability=0.9):
    """Exact weights [event, device]; rows sum to one, no fake observations."""
    probability = source_probabilities(y, subjects, association, max_probability)
    return np.column_stack((1.0 - probability, probability))


def select_endpoints(pairs, endpoint):
    """Copy selected actual feature vectors, without scaling or interpolation."""
    values, selected = np.asarray(pairs), np.asarray(endpoint)
    if values.ndim < 3 or values.shape[1] != 2:
        raise ValueError("pairs must have shape [event, 2, ...features]")
    if selected.shape != (len(values),) or not np.isin(selected, [0, 1]).all():
        raise ValueError("endpoint must have one binary device index per event")
    return values[np.arange(len(values)), selected.astype(np.intp)].copy()


@dataclass(frozen=True)
class FrozenTaskHead:
    """Train-only feature scaling plus an immutable-by-interface task head."""
    scaler: StandardScaler
    classifier: LogisticRegression
    feature_shape: tuple[int, ...]

    def predict_pairs(self, pairs):
        """Apply the frozen scaler and classifier to both observed endpoints."""
        values = np.asarray(pairs)
        if values.ndim < 3 or values.shape[1] != 2 or tuple(values.shape[2:]) != self.feature_shape:
            raise ValueError("paired evaluation features do not match the fitted head")
        if not np.isfinite(values).all():
            raise ValueError("paired evaluation features must be finite")
        flat = values.reshape(len(values) * 2, -1)
        return self.classifier.predict_proba(self.scaler.transform(flat))[:, 1].reshape(len(values), 2)


def fit_task_head(pairs, y, subjects, endpoint, *, C=1.0, class_weight="balanced", seed=0,
                  subject_equal=True, max_iter=2000):
    """Fit logistic task prediction on one actual endpoint per HEAD event.

    ``class_weight='balanced'`` changes task-loss weights, not the dataset or
    device assignment; report it because the effective training loss then has
    balanced target mass.  The scaler is fit only on selected HEAD features.
    Fit the independent control by calling this function on the same pairs and
    rho=0 endpoint assignment, with identical C/weighting settings.
    """
    labels, people = _metadata(y, subjects)
    selected = select_endpoints(pairs, endpoint)
    if len(selected) != len(labels) or not np.isfinite(selected).all():
        raise ValueError("HEAD features must be finite and align with metadata")
    if len(np.unique(labels)) != 2:
        raise ValueError("HEAD requires both task classes")
    if not np.isfinite(C) or C <= 0:
        raise ValueError("C must be finite and positive")
    weights = np.ones(len(labels), dtype=np.float64)
    if subject_equal:
        for person in np.unique(people):
            mask = people == person
            weights[mask] = 1 / mask.sum()
        weights *= len(weights) / weights.sum()
    flat = selected.reshape(len(selected), -1)
    scaler = StandardScaler().fit(flat, sample_weight=weights)
    classifier = LogisticRegression(C=float(C), class_weight=class_weight, random_state=int(seed),
                                    solver="lbfgs", max_iter=int(max_iter))
    classifier.fit(scaler.transform(flat), labels, sample_weight=weights)
    return FrozenTaskHead(scaler, classifier, tuple(selected.shape[1:]))


def _mean_valid(rows, key):
    """Average defined participant values, returning None when none are defined."""
    values = [row[key] for row in rows if row[key] is not None]
    return float(np.mean(values)) if values else None


def evaluate_pair_predictions(scores, y, subjects, associations=None, *, max_probability=0.9):
    """Absolute frozen-task outcomes under association stress, JSON serializable.

    ``scores`` are positive-class *probabilities*, not logits, for both actual
    endpoints.  Each AUROC uses all real paired observations and the exact
    coupling weights; participants are subsequently averaged equally.  AUC for
    a one-class participant is None and ``n_auc_subjects`` exposes its exclusion.
    Subject-cluster paired intervals should use ``paired_bootstrap``; neither
    the two endpoints nor individual events are independent inference units.
    """
    labels, people = _metadata(y, subjects)
    probability = np.asarray(scores, dtype=np.float64)
    if probability.shape != (len(labels), 2) or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("scores must be finite probabilities with shape [event, 2]")
    if associations is None:
        associations = DEFAULT_ASSOCIATIONS
    elif not isinstance(associations, Mapping):
        associations = {str(float(rho)): float(rho) for rho in associations}
    if not associations:
        raise ValueError("associations cannot be empty")
    repeated_labels = np.repeat(labels, 2)
    association_metrics = {}
    for name, rho in associations.items():
        weights = pair_weights(labels, people, rho, max_probability)
        rows = []
        for person in np.unique(people):
            mask = people == person
            pair_mask = np.repeat(mask, 2)
            event_y, pred, weight = repeated_labels[pair_mask], probability[mask].ravel(), weights[mask].ravel()
            auc = float(roc_auc_score(event_y, pred, sample_weight=weight)) if len(np.unique(event_y)) == 2 else None
            rows.append({"subject": str(person), "n_events": int(mask.sum()), "n_positive": int(labels[mask].sum()),
                         "auroc": auc, "brier": float(np.average((pred - event_y) ** 2, weights=weight)),
                         "source1_fraction": float(weights[mask, 1].mean()),
                         "target_fraction": float(labels[mask].mean())})
        association_metrics[str(name)] = {
            "association": float(rho), "auroc": _mean_valid(rows, "auroc"),
            "brier": _mean_valid(rows, "brier"), "n_auc_subjects": sum(row["auroc"] is not None for row in rows),
            "per_subject": rows,
        }
    paired_rows = []
    difference = probability[:, 1] - probability[:, 0]
    flips = (probability[:, 1] >= 0.5) != (probability[:, 0] >= 0.5)
    for person in np.unique(people):
        mask = people == person
        row = {"subject": str(person), "n_events": int(mask.sum()),
               "mean_absolute_probability_difference": float(np.abs(difference[mask]).mean()),
               "mean_signed_probability_difference": float(difference[mask].mean()),
               "prediction_flip_fraction": float(flips[mask].mean())}
        for target in (0, 1):
            subset = mask & (labels == target)
            row[f"class{target}_absolute_probability_difference"] = float(np.abs(difference[subset]).mean()) if subset.any() else None
            row[f"class{target}_prediction_flip_fraction"] = float(flips[subset].mean()) if subset.any() else None
        paired_rows.append(row)
    keys = [key for key in paired_rows[0] if key not in ("subject", "n_events")]
    return {"n_events": len(labels), "n_subjects": len(paired_rows), "max_probability": float(max_probability),
            "target_prevalence": "empirical within participant", "aggregation": "equal participant mean",
            "association_metrics": association_metrics,
            "paired_device_metrics": {**{key: _mean_valid(paired_rows, key) for key in keys}, "per_subject": paired_rows}}
